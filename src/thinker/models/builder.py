from unsloth import FastLanguageModel


def build_unsloth_model(cfg):
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
        r=cfg.lora_rank,  # 矩阵秩 Rank 控制低秩分解矩阵大小
        target_modules=[  # 指定将 LoRA 应用于 Transformer 哪些投影层 升降维
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "out_proj",
        ],
        lora_alpha=cfg.lora_alpha,  # 缩放系数 Alpha, 与 r 共同决定 LoRA 权重的缩放比例
        lora_dropout=0,  # Dropout 比例, unsloth 推荐 0 以获得最优性能, 省略随机掩码 Mask 的生成和计算
        bias="none",  # 偏置项优化策略, 设为 "none" 不训练偏置
        use_gradient_checkpointing=cfg.use_gradient_checkpointing,  # 梯度检查点, 支持 unsloth, 用于极限显存优化
        random_state=3407,  # 随机种子, 确保初始化权重可复现
        use_rslora=False,
        loftq_config=None,
    )
    return model, tokenizer
