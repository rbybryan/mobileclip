import torch
from torch import nn


class HFTextEncoder(nn.Module):
    """OpenCLIP-compatible Hugging Face text tower."""

    def __init__(self, cfg, projection_dim):
        super().__init__()
        try:
            from transformers import AutoConfig, AutoModel
        except ImportError as exc:
            raise RuntimeError("Install transformers to use an HF text encoder") from exc

        model_name = cfg["hf_model_name"]
        local_files_only = cfg.get("local_files_only", False)
        cache_dir = cfg.get("cache_dir")
        hf_config = AutoConfig.from_pretrained(
            cfg.get("hf_config_name", model_name),
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        try:
            self.transformer = AutoModel.from_config(hf_config, add_pooling_layer=False)
        except TypeError:
            self.transformer = AutoModel.from_config(hf_config)
            if hasattr(self.transformer, "pooler"):
                self.transformer.pooler = None

        self.config = hf_config
        self.vocab_size = hf_config.vocab_size
        self.context_length = hf_config.max_position_embeddings
        self.pooler_type = cfg.get("hf_pooler_type", "mean_pooler")

        model_dim = hf_config.hidden_size
        projection_type = cfg.get("hf_proj_type", "mlp")
        if projection_type == "mlp":
            hidden_dim = (model_dim + projection_dim) // 2
            self.proj = nn.Sequential(
                nn.Linear(model_dim, hidden_dim, bias=False),
                nn.GELU(),
                nn.Linear(hidden_dim, projection_dim, bias=False),
            )
        elif projection_type == "linear":
            self.proj = nn.Linear(model_dim, projection_dim, bias=False)
        elif projection_type in (None, "none") and model_dim == projection_dim:
            self.proj = nn.Identity()
        else:
            raise ValueError(f"Unsupported HF projection type: {projection_type}")

    def forward(self, text_tokens):
        if isinstance(text_tokens, dict):
            input_ids = text_tokens["input_ids"]
            attention_mask = text_tokens.get("attention_mask")
        else:
            input_ids = text_tokens
            attention_mask = None
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()

        output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        if self.pooler_type == "mean_pooler":
            mask = attention_mask.unsqueeze(-1).to(output.last_hidden_state.dtype)
            pooled = (output.last_hidden_state * mask).sum(dim=1)
            pooled = pooled / mask.sum(dim=1).clamp_min(1)
        elif self.pooler_type == "cls_pooler":
            pooled = output.last_hidden_state[:, 0]
        else:
            raise ValueError(f"Unsupported HF pooler: {self.pooler_type}")
        return self.proj(pooled)

    def set_grad_checkpointing(self, enable=True):
        if enable:
            self.transformer.gradient_checkpointing_enable()
        else:
            self.transformer.gradient_checkpointing_disable()
