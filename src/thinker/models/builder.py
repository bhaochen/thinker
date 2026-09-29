from unsloth import FastLanguageModel


def build_unsloth_model(cfg):
    """加载量化基座模型并挂载 LoRA 适配器"""
    print(f"Loading base model: {cfg.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        load_in_8bit=False,
        full_finetuning=False,
    )

    print(f"Attaching LoRA adapters (Rank: {cfg.lora_rank})")
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "out_proj",
        ],
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=cfg.use_gradient_checkpointing,
        random_state=3407,
    )
    return model, tokenizer
