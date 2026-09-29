import re
import hashlib
from datasets import concatenate_datasets
from unsloth.chat_templates import get_chat_template
from thinker.utils.env import force_cleanup


class ReasoningDataPipeline:
    """推理数据蒸馏与提纯管道"""

    def __init__(self, tokenizer, data_cfg):
        self.data_cfg = data_cfg
        self.tokenizer = get_chat_template(tokenizer, chat_template="qwen3-thinking")
        self.think_block_re = re.compile(r"<think>.*?</think>", flags=re.DOTALL)

    def _normalize_think_solution(self, text: str) -> str:
        """强制转化为 <think>...</think>\n{solution} 格式"""
        text = (text or "").strip()
        if not text:
            return "<think></think>\n"

        m = self.think_block_re.search(text)
        if m:
            think_block = m.group(0).strip()
            rest = text[m.end() :].lstrip()
            return f"{think_block}\n{rest}".rstrip() if rest else f"{think_block}\n"
        return f"<think></think>\n{text}".rstrip()

    def _canonicalize_conversation(self, msgs: list) -> list:
        """规范化对话回合结构"""
        cleaned = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = m.get("role", "")
            content = (m.get("content", "") or "").strip()

            if role not in {"user", "assistant"} or not content:
                continue

            if role == "assistant":
                content = self._normalize_think_solution(content)

            if cleaned and cleaned[-1]["role"] == role:
                prev = cleaned[-1]["content"].rstrip()
                merged = prev + "\n" + content
                cleaned[-1]["content"] = (
                    self._normalize_think_solution(merged)
                    if role == "assistant"
                    else merged
                )
            else:
                cleaned.append({"role": role, "content": content})

        if (
            len(cleaned) < 2
            or cleaned[0]["role"] != "user"
            or cleaned[-1]["role"] != "assistant"
        ):
            return None
        return cleaned

    def process(self, raw_datasets: dict):
        """执行完整的数据清洗流水线"""
        processed_splits = []

        # 1. 结构化清洗 (映射为你自己的数据集适配逻辑)
        for name, ds in raw_datasets.items():

            def format_func(ex):
                msgs = ex.get("messages") or ex.get("conversations")
                if not msgs:
                    return {"conversations": None}
                canonical = self._canonicalize_conversation(msgs)
                return {"conversations": canonical}

            ds = ds.map(
                format_func, remove_columns=ds.column_names, load_from_cache_file=False
            )
            ds = ds.filter(lambda x: x["conversations"] is not None)

            # 2. 应用 Chat Template 转换为 text
            ds = ds.map(
                lambda ex: {
                    "text": [
                        self.tokenizer.apply_chat_template(
                            c, tokenize=False, add_generation_prompt=False
                        )
                        for c in ex["conversations"]
                    ]
                },
                batched=True,
                remove_columns=["conversations"],
            )
            processed_splits.append(ds)

        # 3. 合并异构数据
        merged_ds = concatenate_datasets(processed_splits).shuffle(
            seed=self.data_cfg.seed
        )
        force_cleanup("Data Merged")

        # 4. Hash 精确去重
        seen = set()
        keep_indices = []
        for i, text in enumerate(merged_ds["text"]):
            h = hashlib.md5(text.encode("utf-8")).hexdigest()
            if h not in seen:
                seen.add(h)
                keep_indices.append(i)
        merged_ds = merged_ds.select(keep_indices)

        # 5. Token 长度过滤
        _text_tok = getattr(self.tokenizer, "tokenizer", self.tokenizer)

        def filter_len(examples):
            tokenized = _text_tok(
                examples["text"],
                truncation=True,
                max_length=self.data_cfg.max_context_window + 1,
                padding=False,
                add_special_tokens=False,
            )["input_ids"]
            return [len(t) <= self.data_cfg.max_context_window for t in tokenized]

        final_ds = merged_ds.filter(
            filter_len, batched=True, num_proc=self.data_cfg.num_proc
        )
        force_cleanup("Pipeline Completed")

        return final_ds
