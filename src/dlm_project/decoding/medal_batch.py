"""Cooperative MEDAL evaluations preserving ordinary UCB sibling rounds.

Only the forward boundary is suspended for cross-question batching. Selection,
action order, reward, backup, budget checks and final selection use MEDAL's
existing primitives. No GEN/CONT decisions or Bayesian posteriors are created.
"""

import torch

from dlm_project.decoding import mcts_decoding as medal
from dlm_project.decoding.ab_medal import EvaluationRequest


class CooperativeMEDALSearch(medal.MCTSSearch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.stop_mode != 'budget':
            raise ValueError('Cooperative MEDAL requires fixed-budget search.')
        self._pending = self._parent = None
        self._remaining = []
        self._root_cycle = False

    def next_evaluation(self):
        if self._pending is not None:
            return self._pending
        if self.done:
            return None
        if self.root.analysis is None:
            self.cycle += 1
            self._root_cycle = True
            self._pending = EvaluationRequest(self.root, None, [], [], 'tree_analysis')
            return self._pending
        if self._at_cap() or self._at_simulation_cap():
            self._finish('model_call_budget' if self._at_cap() else 'simulation_budget')
            return None
        while self._parent is None:
            if not self._root_cycle:
                self.cycle += 1
            self._root_cycle = False
            with self._counted.phase('selection'):
                parent, decisions = medal.select_node(self.root, self.config.exploration_c)
            if parent is None:
                self._finish('frontier_exhausted')
                return None
            self._emit('selection', node_id=parent.node_id,
                       path=[0] + [hop['chosen_id'] for hop in decisions], decisions=decisions)
            parent.expanded = True
            self._remaining = list(parent.analysis.actions)
            if not self._remaining:
                self._close_exhausted(parent)
                if self.root.closed:
                    self._finish('frontier_exhausted')
                    return None
                continue
            self._parent = parent
        parent = self._parent
        action = self._remaining.pop(0)
        if action.token_id in self.search_excluded_token_ids:
            raise RuntimeError('Excluded termination token reached search expansion.')
        with self._counted.phase('expansion'):
            child = medal.expand_node(parent, action, len(self.nodes), self.prompt_len,
                                      self.mask_id, self.config.init_tokens)
            self.nodes.append(child)
        self._emit('expansion', parent_id=parent.node_id, child_id=child.node_id,
                   depth=child.depth, action=medal.action_record(action, self.tokenizer))
        self.simulations_started += 1
        self._emit('simulation_started', child_id=child.node_id, simulations_started=self.simulations_started)
        self._pending = EvaluationRequest(child, parent, [], [], 'simulation')
        return self._pending

    def record_external_forward(self, request, *, invocation_id, batch_size):
        if request is not self._pending or request.attempted:
            raise ValueError('Expected the current, undispatched evaluation request.')
        if self._at_cap():
            raise medal.BudgetExhausted('Search forward cap reached.')
        request.attempted = True
        self._counted.calls += 1
        self._counted.counts[request.phase] += 1
        self._emit('forward', phase=request.phase, invocation_id=invocation_id,
                   batch_size=batch_size, node_id=request.node.node_id, status='attempted',
                   **self._counted.costs())

    def submit_evaluation(self, request, *, logits):
        if request is not self._pending or not request.attempted:
            raise ValueError('Expected the current dispatched evaluation request.')
        analysis = medal.distribution_from_logits(logits, request.node.state, self.prompt_len,
            self.mask_id, self.config, search_excluded_token_ids=self.search_excluded_token_ids)[0]
        child, parent = request.node, request.parent
        if parent is None:
            self.root.analysis = analysis
            self._record_evaluated(self.root)
            self._emit('initialization', node_id=0, entropy=analysis.total_entropy,
                       actions=[medal.action_record(a, self.tokenizer) for a in analysis.actions])
            self._pending = None
            return
        evaluation = medal.apply_child_analysis(parent, child, analysis, self.prompt_len,
            self.mask_id, self.config, signal_nfe=self.model_calls)
        self.simulations_completed += 1
        self._record_evaluated(child)
        self._emit('evaluation', parent_id=parent.node_id, child_id=child.node_id, depth=child.depth,
                   action=medal.action_record(child.action, self.tokenizer), **evaluation)
        with self._counted.phase('backpropagation'):
            updates = medal.backpropagate(child, evaluation['reward'])
        self._emit('backpropagation', child_id=child.node_id, reward=evaluation['reward'], updates=updates)
        if child.depth == self.config.init_tokens:
            key = medal._state_key(child.state)
            duplicate = key in self.candidates
            self.candidates.setdefault(key, child)
            self._emit('candidate', node_id=child.node_id, duplicate=duplicate,
                       candidate_count=len(self.candidates), score=evaluation['reward'])
        self._pending = None
        if self._at_cap() or self._at_simulation_cap():
            self._finish('model_call_budget' if self._at_cap() else 'simulation_budget')
            return
        if not self._remaining:
            self._close_exhausted(parent)
            self._parent = None
            if self.root.closed:
                self._finish('frontier_exhausted')

    @torch.inference_mode()
    def step(self):
        start = len(self.trace)
        while not self.done:
            request = self.next_evaluation()
            if request is None:
                break
            self.record_external_forward(request, invocation_id=self.model_calls + 1, batch_size=1)
            prediction = self.model(request.node.state.to(medal.model_device(self.model)))
            self.submit_evaluation(request, logits=prediction.logits[0])
            del prediction
            if request.parent is not None:
                break
        return self.trace[start:]
