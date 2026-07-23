import torch

from models.checkpoint_utils import convert_openclip_mobileclip_state_dict


def test_openclip_fastvit_and_hf_key_conversion():
    checkpoint = {
        "state_dict": {
            "module.logit_scale": torch.tensor(4.0),
            "module.visual.trunk.stem.0.conv_kxk.0.conv.weight": torch.ones(1),
            "module.visual.trunk.stages.2.blocks.3.mlp.fc1.weight": torch.ones(2),
            "module.visual.trunk.stages.3.pos_emb.pos_enc.weight": torch.ones(3),
            "module.visual.trunk.head.fc.weight": torch.ones(2, 3),
            "module.text.transformer.embeddings.word_embeddings.weight": torch.ones(4),
        }
    }

    converted = convert_openclip_mobileclip_state_dict(checkpoint)

    assert "vision_encoder.patch_embed.0.rbr_conv.0.conv.weight" in converted
    assert "vision_encoder.network.4.3.convffn.fc1.weight" in converted
    assert "vision_encoder.network.6.pe.weight" in converted
    assert converted["vision_encoder.head.proj"].shape == (3, 2)
    assert "text_encoder.transformer.embeddings.word_embeddings.weight" in converted
    assert torch.allclose(converted["temp"], torch.exp(torch.tensor(-4.0)))


if __name__ == "__main__":
    test_openclip_fastvit_and_hf_key_conversion()
    print("checkpoint conversion test passed")
