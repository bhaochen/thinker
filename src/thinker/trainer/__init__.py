try:
    from .sft import DistillationTrainer
except ImportError:
    pass

__all__ = ["DistillationTrainer"]
