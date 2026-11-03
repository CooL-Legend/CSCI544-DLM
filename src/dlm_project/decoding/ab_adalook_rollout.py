"""AdaLook-inspired comparison groups for one-child AB traversals.

Only the selected real child receives a reward and a posterior update. Other
parent actions are detached hypotheses, never outer nodes or consumed actions.
Cached analyses are immutable base-state evidence, not previous rollout scores.
"""

from .adalook_rollout import ConfidenceRefinement, Request, score
from .mcts_decoding import Node


class ABConfidenceRefinement(ConfidenceRefinement):
    def __init__(self, search, request, analysis):
        self.search, self.request, self.analysis = search, request, analysis

    def run(self):
        s, cfg = self.search, self.search.config
        child, parent = self.request.node, self.request.parent
        base_score = score(child.action.probability, self.analysis.confidence_mass, cfg)
        metadata = dict(rollout_depth=1, extra_nfes=0, comparison_nfes=0,
                        stop_reason='depth_limit', score_variance=0.0,
                        branch_actions=[], comparison_candidates=[], cached_comparisons=0)
        # Exact one-step ablation: do not buy comparison evidence if no deeper
        # refinement is possible, including a gate that must always stop.
        if cfg.adalook_max_depth == 1 or cfg.adalook_variance_threshold == 0:
            metadata['stop_reason'] = 'depth_limit' if cfg.adalook_max_depth == 1 else 'score_variance'
            return base_score, metadata

        boundary = self._boundary()
        if s.model_calls in s.checkpoint_budgets or s._at_cap():
            metadata['stop_reason'] = 'base_budget'
            return base_score, metadata

        cached = {(n.action.output_position, n.action.token_id): n
                  for n in parent.children if n.analysis is not None}
        alternatives = []
        for action in parent.analysis.actions:
            if (action.output_position, action.token_id) == (child.action.output_position, child.action.token_id):
                continue
            state = parent.state.clone()
            state[0, s.prompt_len + action.output_position] = action.token_id
            temporary = Node(-2 - len(alternatives), state, parent=parent,
                             action=action, depth=child.depth)
            previous = cached.get((action.output_position, action.token_id))
            alternatives.append((temporary, previous))
        if not alternatives:
            metadata['stop_reason'] = 'single_branch'
            return base_score, metadata
        required = sum(previous is None for _, previous in alternatives)
        # Never refine an arbitrary prefix of the comparison group.
        if s.model_calls + required > boundary:
            metadata['stop_reason'] = 'comparison_budget'
            return base_score, metadata

        children, analyses = [child], [self.analysis]
        for temporary, previous in alternatives:
            descriptor = dict(temporary_id=temporary.node_id,
                              output_position=temporary.action.output_position,
                              token_id=temporary.action.token_id,
                              cached_node_id=previous.node_id if previous is not None else None)
            metadata['comparison_candidates'].append(descriptor)
            s._emit('adalook_comparison', real_child_id=child.node_id,
                    parent_id=parent.node_id, **descriptor)
            if previous is None:
                analysis = yield Request(temporary, child, 'adalook_comparison')
                metadata['comparison_nfes'] += 1
            else:
                analysis = previous.analysis
                metadata['cached_comparisons'] += 1
            children.append(temporary)
            analyses.append(analysis)

        if s.model_calls == boundary:
            metadata['stop_reason'] = 'comparison_budget'
            return base_score, metadata
        # Keep parent action order for deterministic tie-breaking, regardless
        # of which member is the real child of this particular AB traversal.
        order = {(a.output_position, a.token_id): i for i, a in enumerate(parent.analysis.actions)}
        group = sorted(zip(children, analyses), key=lambda pair:
                       order[(pair[0].action.output_position, pair[0].action.token_id)])
        children, analyses = zip(*group)
        values, refined = yield from self._refine(children, analyses)
        metadata.update(refined)
        # Pruned origins keep their last valid score in the shared evaluator.
        return values[child.node_id], metadata
