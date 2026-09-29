from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnvConfig:
    output_dir: str = "/kaggle/working/"
    wandb_project: str = "thinker-distillation"
    seed: int = 1234


@dataclass
class DataConfig:
    ds1_name: str = "Jackrong/Competitive-Programming-python-blend"
    ds1_samples: int = 700
    ds2_repo_id: str = "stepfun-ai/Step-3.5-Flash-SFT"
    ds2_samples: int = 100000
    step_random_json_files: int = 5
    max_context_window: int = 16384
    num_proc: int = 21


@dataclass
class ModelConfig:
    model_name: str = "Unsloth/Qwen3.5-9B"
    max_seq_length: int = 16384
    load_in_4bit: bool = True
    lora_rank: int = 64
    lora_alpha: int = 64
    use_gradient_checkpointing: str = "unsloth"


@dataclass
class TrainConfig:
    batch_size: int = 6
    grad_accum_steps: int = 6
    learning_rate: float = 2e-4
    epochs: int = 2
    warmup_ratio: float = 0.04
    save_steps: int = 100


@dataclass
class ExportConfig:
    push_to_hub: bool = True
    repo_id: str = "Qwopus-3.5-9B-neo"
    export_gguf: bool = True


@dataclass
class ThinkerConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    export: ExportConfig = field(default_factory=ExportConfig)


def get_configs() -> ThinkerConfig:
    """获取全局配置，后续可扩展为从 YAML 或 argparse 读取"""
    return ThinkerConfig()
