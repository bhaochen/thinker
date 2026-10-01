from thinker.utils import resolve_secret, login_services

print(
    {
        k: (v[:4] + "..." if (v := resolve_secret(k)) else None)
        for k in ("WANDB_API_KEY", "HF_TOKEN")
    }
)
login_services()
