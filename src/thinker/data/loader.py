import os
import json
import random
from datasets import load_dataset, Dataset
from huggingface_hub import snapshot_download


def load_standard_dataset(dataset_name, sample_count, seed=1234):
    """加载标准 HF 数据集并采样"""
    ds = load_dataset(dataset_name, split="train")
    if sample_count:
        ds = ds.shuffle(seed=seed).select(range(min(sample_count, len(ds))))
    return ds


def _iter_step_raw_examples(local_repo_dir, num_random_files, seed):
    """内部生成器：解析本地 JSON 缓存"""
    json_root = os.path.join(local_repo_dir, "json")
    json_files = [
        os.path.join(r, f)
        for r, _, fs in os.walk(json_root)
        for f in fs
        if f.endswith(".json")
    ]
    json_files.sort()

    sampled_files = random.Random(seed).sample(
        json_files, min(num_random_files, len(json_files))
    )
    for fp in sampled_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    yield from data
        except Exception:
            continue


def load_step_dataset(repo_id, sample_count, num_random_files, seed=1234):
    """手动下载并抽取 Step 格式数据集"""
    local_repo_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["json/*.json", "json/**/*.json"],
    )

    raw_data = []
    for ex in _iter_step_raw_examples(local_repo_dir, num_random_files, seed):
        raw_data.append(ex)
        if sample_count and len(raw_data) >= sample_count * 2:  # 冗余 buffer 供后续清洗
            break

    return Dataset.from_list(raw_data)


def load_raw_datasets(cfg):
    """对外暴露的主函数：拉取论文中声明的所有异构数据集"""
    ds1 = load_standard_dataset(cfg.ds1_name, cfg.ds1_samples)
    ds2 = load_step_dataset(
        cfg.ds2_repo_id, cfg.ds2_samples, cfg.step_random_json_files
    )
    return {"programming": ds1, "step_sft": ds2}
