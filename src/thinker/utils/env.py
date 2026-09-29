import os
import gc
import wandb


def setup_environment(cfg):
    """初始化 W&B 和输出目录"""
    try:
        from kaggle_secrets import UserSecretsClient

        user_secrets = UserSecretsClient()
        wandb_key = user_secrets.get_secret("WANDB_API_KEY")
        wandb.login(key=wandb_key)
        print("Successfully logged into W&B via Kaggle Secrets.")
    except Exception:
        print("Not running in Kaggle or WANDB_API_KEY not found. Skipping auto-login.")

    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    print(f"Checkpoints will be saved to: {cfg.output_dir}")


def force_cleanup(tag: str = ""):
    """强制垃圾回收以节省内存"""
    if tag:
        print(f"[Memory Cleanup] {tag}")
    gc.collect()
