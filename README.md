# multi-lora-medical
多个lora适配器切换



test_model_dir = {
    "lora_adaptor_1": "path/to/lora1",
    "lora_adaptor_2": "path/to/lora2",
    "lora_adaptor_3": "path/to/lora3",
    "lora_adaptor_4": "path/to/lora4",
    "lora_adaptor_5": "path/to/lora5"
}

for domain, path in test_model_dir.items():
    model.load_adapter(path, adapter_name=domain)

model:
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): lora.Linear(
            (base_layer): Linear(in_features=4096, out_features=4096, bias=False)
            (lora_dropout): ModuleDict(
              (骨科康复): Identity()
              (脊髓损伤康复): Identity()
              (内科康复): Identity()
              (言语、吞咽康复): Identity()
              (卒中康复): Identity()
            )
            (lora_A): ModuleDict(
              (骨科康复): Linear(in_features=4096, out_features=16, bias=False)
              (脊髓损伤康复): Linear(in_features=4096, out_features=16, bias=False)
              (内科康复): Linear(in_features=4096, out_features=16, bias=False)
              (言语、吞咽康复): Linear(in_features=4096, out_features=16, bias=False)
              (卒中康复): Linear(in_features=4096, out_features=16, bias=False)
            )
            (lora_B): ModuleDict(
              (骨科康复): Linear(in_features=16, out_features=4096, bias=False)
              (脊髓损伤康复): Linear(in_features=16, out_features=4096, bias=False)
              (内科康复): Linear(in_features=16, out_features=4096, bias=False)
              (言语、吞咽康复): Linear(in_features=16, out_features=4096, bias=False)
              (卒中康复): Linear(in_features=16, out_features=4096, bias=False)
            )
            (lora_embedding_A): ParameterDict()
            (lora_embedding_B): ParameterDict()
            (lora_magnitude_vector): ModuleDict()
          )
          (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (v_proj): lora.Linear(
            (base_layer): Linear(in_features=4096, out_features=1024, bias=False)
            (lora_dropout): ModuleDict(
              (骨科康复): Identity()
              (脊髓损伤康复): Identity()
              (内科康复): Identity()
              (言语、吞咽康复): Identity()
              (卒中康复): Identity()
            )
            (lora_A): ModuleDict(
              (骨科康复): Linear(in_features=4096, out_features=16, bias=False)
              (脊髓损伤康复): Linear(in_features=4096, out_features=16, bias=False)
              (内科康复): Linear(in_features=4096, out_features=16, bias=False)
              (言语、吞咽康复): Linear(in_features=4096, out_features=16, bias=False)
              (卒中康复): Linear(in_features=4096, out_features=16, bias=False)
            )
            (lora_B): ModuleDict(
              (骨科康复): Linear(in_features=16, out_features=1024, bias=False)
              (脊髓损伤康复): Linear(in_features=16, out_features=1024, bias=False)
              (内科康复): Linear(in_features=16, out_features=1024, bias=False)
              (言语、吞咽康复): Linear(in_features=16, out_features=1024, bias=False)
              (卒中康复): Linear(in_features=16, out_features=1024, bias=False)
            )
            (lora_embedding_A): ParameterDict()
            (lora_embedding_B): ParameterDict()
            (lora_magnitude_vector): ModuleDict()
          )
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((4096,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=4096, out_features=128256, bias=False)
)

for domain, path in test_model_dir.items():
    model.set_adapter(domain)

