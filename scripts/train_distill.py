from thinker.config import get_configs
from thinker.utils.env import setup_environment
from thinker.data.loader import load_raw_datasets
from thinker.data.processor import ReasoningDataPipeline
from thinker.models.builder import build_unsloth_model
from thinker.trainer.sft import DistillationTrainer
from thinker.utils.export import push_to_hub


def main():
    # 1. 加载所有配置 (DataConfig, ModelConfig, TrainConfig)
    cfg = get_configs()
    setup_environment(cfg.env)

    # 2. 构建模型与分词器 (Unsloth + LoRA)
    model, tokenizer = build_unsloth_model(cfg.model)

    # 3. 数据处理流 (加载 -> 蒸馏标准化为 <think> 范式 -> 模板化 -> 过滤)
    raw_datasets = load_raw_datasets(cfg.data)
    pipeline = ReasoningDataPipeline(tokenizer, cfg.data)
    train_dataset = pipeline.process(raw_datasets)

    # 4. 初始化训练器 (并应用 Response-only Loss Masking)
    trainer = DistillationTrainer(model, tokenizer, train_dataset, cfg.train)

    # 5. 开始高效训练
    trainer.train()

    # 6. 导出与发布
    push_to_hub(model, tokenizer, cfg.export)


if __name__ == "__main__":
    main()
