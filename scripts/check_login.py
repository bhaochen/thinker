"""检查 .env 中的 WANDB_API_KEY / HF_TOKEN 能否成功登录。"""

from dotenv import load_dotenv

from thinker.utils.env import _resolve_secret, setup_environment

load_dotenv()


def check_wandb() -> bool:
    try:
        import wandb

        api = wandb.Api()
        print(f"W&B 登录成功: {api.viewer.entity}")
        return True
    except Exception as e:
        print(f"W&B 登录失败: {e}")
        return False


def check_hf() -> bool:
    try:
        from huggingface_hub import whoami

        print(f"Hugging Face 登录成功: {whoami()['name']}")
        return True
    except Exception as e:
        print(f"Hugging Face 登录失败: {e}")
        return False


def main():
    for name in ("WANDB_API_KEY", "HF_TOKEN"):
        value = _resolve_secret(name)
        print(f"{name}: {'已找到' if value else '未找到'}"
              f" (前 4 位 {value[:4] + '...' if value else '-'}, 长度 {len(value) if value else 0})")

    setup_environment(None)

    ok = [check_wandb(), check_hf()]
    print("\n结果:", "全部通过" if all(ok) else "存在失败项")


if __name__ == "__main__":
    main()
