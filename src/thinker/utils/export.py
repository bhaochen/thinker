def push_to_hub(model, tokenizer, export_cfg):
    """合并 16-bit 模型并导出 GGUF 到 Hugging Face"""
    if not export_cfg.push_to_hub:
        return

    try:
        from huggingface_hub import whoami

        from .env import resolve_secret

        hf_token = resolve_secret("HF_TOKEN")
        username = whoami(token=hf_token)["name"]
        repo_id = f"{username}/{export_cfg.repo_id}"

        print(f"Uploading merged 16-bit model to {repo_id}...")
        model.push_to_hub_merged(
            repo_id, tokenizer, save_method="merged_16bit", token=hf_token
        )

        if export_cfg.export_gguf:
            gguf_repo = f"{repo_id}-GGUF"
            print(f"Converting and uploading GGUF to {gguf_repo}...")
            model.push_to_hub_gguf(
                gguf_repo, tokenizer, quantization_method=["q4_k_m"], token=hf_token
            )

        print("Export completed successfully!")
    except Exception as e:
        print(f"Export failed: {e}")
