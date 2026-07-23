import re

from torch import nn


class HFTokenizer(nn.Module):
    """Tensor-returning HF tokenizer compatible with MobileViCLIP tasks."""

    def __init__(self, cfg, context_length=None):
        super().__init__()
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install transformers to use an HF tokenizer") from exc

        tokenizer_name = cfg.get("hf_tokenizer_name", cfg["hf_model_name"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=cfg.get("cache_dir"),
            local_files_only=cfg.get("local_files_only", False),
        )
        self.context_length = context_length or cfg.get("context_length", 77)

    def forward(self, texts, context_length=None, max_length=None, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        texts = [re.sub(r"\s+", " ", text).strip() for text in texts]
        length = max_length or context_length or self.context_length
        encoded = self.tokenizer.batch_encode_plus(
            texts,
            return_tensors="pt",
            max_length=length,
            padding=kwargs.get("padding", "max_length"),
            truncation=kwargs.get("truncation", True),
        )
        return encoded.input_ids

    def get_vocab_size(self):
        return len(self.tokenizer)

    def get_eot_token(self):
        return self.tokenizer.sep_token_id

    def get_sot_token(self):
        return self.tokenizer.cls_token_id
