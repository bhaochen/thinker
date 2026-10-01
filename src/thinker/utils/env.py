import os
import gc
import wandb


def _resolve_secret(name: str) -> str | None:
    """按 .env -> 环境变量 -> Kaggle Secrets 的顺序查找密钥"""
    value = os.environ.get(name)
    if value:
        return value

    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret(name)
    except Exception:
        return None


def login_services():
    """登录 W&B 与 Hugging Face（.env 由 import thinker 时加载）"""
    wandb_key = _resolve_secret("WANDB_API_KEY")
    if wandb_key:
        wandb.login(key=wandb_key)
        print("Successfully logged into W&B.")
    else:
        print("WANDB_API_KEY not found. Skipping auto-login.")

    hf_token = _resolve_secret("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login

        login(token=hf_token)
        print("Successfully logged into Hugging Face.")
    else:
        print("HF_TOKEN not found. Skipping auto-login.")


def setup_environment(cfg):
    """登录各服务并准备 checkpoint 输出目录"""
    login_services()

    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    print(f"Checkpoints will be saved to: {cfg.output_dir}")


def force_cleanup(tag: str = ""):
    """强制垃圾回收以节省内存"""
    if tag:
        print(f"[Memory Cleanup] {tag}")
    gc.collect()
