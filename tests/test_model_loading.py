import torch
import pytest
from transformers import BertConfig, BertModel, BertTokenizerFast

from Llada import LLaDA
from model_profiles import get_profile


def test_wrapper_loads_a_local_checkpoint_without_hub_access(tmp_path):
    # A tiny real Transformers checkpoint exercises device_map/local loading on CPU.
    # It is only a loader test; the 8B LLaDA inference check belongs on Colab.
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\npuzzle\n")
    BertTokenizerFast(vocab_file=str(vocab)).save_pretrained(tmp_path)
    BertModel(BertConfig(vocab_size=6, hidden_size=16, num_hidden_layers=1,
                         num_attention_heads=2, intermediate_size=32)).save_pretrained(tmp_path)
    wrapper = LLaDA(device="cpu", model_path=str(tmp_path), dtype="float32", local_files_only=True)
    assert wrapper.model.device.type == "cpu"
    assert next(wrapper.model.parameters()).dtype == torch.float32
    assert wrapper.model.training is False
    assert wrapper.tokenizer.encode("puzzle", add_special_tokens=False) == [5]


@pytest.mark.parametrize("family", ["llada", "illada"])
def test_generation_uses_selected_mask_and_chat_template(monkeypatch, family):
    import Llada

    calls = {}

    class Tokenizer:
        chat_template = "checkpoint template"

        def convert_tokens_to_ids(self, token):
            return 5

        def apply_chat_template(self, messages, **kwargs):
            calls["messages"] = messages
            assert kwargs == {"add_generation_prompt": True, "tokenize": False}
            return "formatted prompt"

        def __call__(self, text, **kwargs):
            assert text == "formatted prompt"
            assert kwargs["add_special_tokens"] is False
            return type("Encoded", (), {"input_ids": torch.tensor([[10, 11]])})()

        def batch_decode(self, ids, **kwargs):
            assert ids.tolist() == [[1, 2, 3, 4]]
            assert kwargs["skip_special_tokens"] is True
            return ["1234"]

    class Model:
        def eval(self):
            return self

    def load_tokenizer(path, **kwargs):
        calls["tokenizer_path"] = path
        calls["tokenizer_options"] = kwargs
        return Tokenizer()

    def load_model(path, **kwargs):
        calls["model_path"] = path
        calls["model_options"] = kwargs
        return Model()

    def generate(model, ids, **kwargs):
        calls["generation"] = kwargs
        return torch.cat((ids, torch.tensor([[1, 2, 3, 4]])), dim=1)

    monkeypatch.setattr(Llada.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(Llada.AutoModel, "from_pretrained", load_model)
    monkeypatch.setattr(Llada, "generate", generate)
    wrapper = LLaDA(device="cpu", family=family)
    assert wrapper.predict("Sudoku", steps=4, gen_length=4, block_length=2) == "1234"
    profile = get_profile(family)
    assert calls["generation"]["mask_id"] == profile["mask_id"]
    assert calls["tokenizer_options"]["revision"] == calls["model_options"]["revision"] == profile["revision"]
    assert calls["messages"] == [{"role": "user", "content": "Sudoku"}]
    assert wrapper.identity["revision"] == profile["revision"]


def test_illada_rejects_wrong_mask_token_before_model_loading(monkeypatch):
    import Llada
    tokenizer = type("Tokenizer", (), {"chat_template": "template", "convert_tokens_to_ids": lambda self, token: 126336})()
    monkeypatch.setattr(Llada.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    with pytest.raises(ValueError, match="token 5"):
        LLaDA(device="cpu", family="illada")
