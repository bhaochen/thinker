from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only


class DistillationTrainer:
    """封装推理微调的训练逻辑"""

    def __init__(self, model, tokenizer, train_dataset, cfg, output_dir):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg

        sft_config = SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum_steps,
            warmup_ratio=cfg.warmup_ratio,
            num_train_epochs=cfg.epochs,
            learning_rate=cfg.learning_rate,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            save_steps=cfg.save_steps,
            save_total_limit=1,
            save_strategy="steps",
            report_to="wandb",
            output_dir=output_dir,
        )

        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=None,
            args=sft_config,
        )

        # 【论文关键技巧】Masking 机制：仅对回答计算 Loss
        self.trainer = train_on_responses_only(
            self.trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n<think>",
        )

    def train(self):
        """执行训练并保存本地权重"""
        print("Starting Efficient Reasoning Distillation...")
        self.trainer.train()

        save_path = f"{self.trainer.args.output_dir}/final_lora"
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Local LoRA weights saved to {save_path}")
