import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from Llada import LLaDA


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
