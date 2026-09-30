"""评测指标计算 — 全套推理性能、思考链统计、效率指标、per-category 分析"""

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class EvaluationMetrics:
    """计算学术论文全套推理性能、思考链统计与效率指标"""

    @staticmethod
    def compute_all_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算所有指标"""
        total_count = len(results)
        if total_count == 0:
            return {}

        passed_count = sum(1 for r in results if r["is_correct"])
        rescued_count = sum(1 for r in results if r.get("is_rescued", False))
        truncated_count = sum(1 for r in results if r["is_truncated"])

        # 非截断任务的 CoT 长度统计
        non_truncated = [r for r in results if not r["is_truncated"]]
        think_chars = [r.get("think_char_len", 0) for r in non_truncated]
        think_words = [r.get("think_word_len", 0) for r in non_truncated]
        total_think_chars = sum(think_chars)

        # 核心推理效率指标
        passes_per_10k = (
            (passed_count / total_think_chars * 10000) if total_think_chars > 0 else 0.0
        )
        chars_per_correct = (
            (total_think_chars / passed_count) if passed_count > 0 else 0
        )

        # 错误分解
        completed_but_wrong = sum(
            1 for r in results if not r["is_correct"] and not r["is_truncated"]
        )

        return {
            "total": total_count,
            "passed": passed_count,
            "pass_at_1": round(passed_count / total_count * 100, 2),
            "rescues": rescued_count,
            "truncated": truncated_count,
            "truncation_rate": round(truncated_count / total_count * 100, 2),
            # CoT 统计
            "mean_chars": int(np.mean(think_chars)) if think_chars else 0,
            "mean_words": int(np.mean(think_words)) if think_words else 0,
            "median_chars": int(np.median(think_chars)) if think_chars else 0,
            "max_chars": int(np.max(think_chars)) if think_chars else 0,
            # 推理效率
            "passes_per_10k_chars": round(passes_per_10k, 2),
            "chars_per_correct_pass": int(chars_per_correct),
            "total_think_chars": total_think_chars,
            # 错误分解
            "completed_wrong": completed_but_wrong,
            "truncated_error": truncated_count,
        }

    @staticmethod
    def compute_per_category(
        results: List[Dict[str, Any]],
        categories: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """按类别计算指标"""
        if categories is None:
            categories = sorted(set(r["category"] for r in results))

        per_cat = {}
        for cat in categories:
            cat_results = [r for r in results if r["category"] == cat]
            if cat_results:
                per_cat[cat] = EvaluationMetrics.compute_all_metrics(cat_results)
        return per_cat

    @staticmethod
    def compute_accuracy_cost_tradeoff(
        base_results: List[Dict[str, Any]],
        neo_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """计算 accuracy-cost tradeoff 数据"""
        base_metrics = EvaluationMetrics.compute_all_metrics(base_results)
        neo_metrics = EvaluationMetrics.compute_all_metrics(neo_results)

        return {
            "base": {
                "accuracy": base_metrics["pass_at_1"],
                "mean_think_chars": base_metrics["mean_chars"],
                "label": "Qwen3.5-4B",
            },
            "neo": {
                "accuracy": neo_metrics["pass_at_1"],
                "mean_think_chars": neo_metrics["mean_chars"],
                "label": "Qwen3.5-4B-Neo",
            },
            "delta_accuracy": round(neo_metrics["pass_at_1"] - base_metrics["pass_at_1"], 2),
            "delta_cost_pct": round(
                (neo_metrics["mean_chars"] - base_metrics["mean_chars"])
                / base_metrics["mean_chars"]
                * 100,
                1,
            ),
        }

    @staticmethod
    def compute_error_breakdown(
        results: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """错误类型分解"""
        completed_wrong = sum(
            1 for r in results if not r["is_correct"] and not r["is_truncated"]
        )
        truncated = sum(1 for r in results if r["is_truncated"])
        correct = sum(1 for r in results if r["is_correct"])

        return {
            "correct": correct,
            "completed_but_wrong": completed_wrong,
            "truncated": truncated,
        }

    @staticmethod
    def compute_length_distribution(
        results: List[Dict[str, Any]],
        bin_size: int = 2000,
        max_len: int = 40000,
    ) -> Dict[str, List[int]]:
        """计算思考链长度分布（用于直方图）"""
        non_truncated = [r for r in results if not r["is_truncated"]]
        lengths = [r.get("think_char_len", 0) for r in non_truncated]

        bins = list(range(0, max_len + bin_size, bin_size))
        hist = [0] * (len(bins) - 1)

        for length in lengths:
            for i in range(len(bins) - 1):
                if bins[i] <= length < bins[i + 1]:
                    hist[i] += 1
                    break

        return {
            "bins": bins[:-1],
            "counts": hist,
            "lengths": lengths,
        }


class BenchmarkMetrics:
    """计算 Standard Error (SE) 并生成规范的 Table 2 表格"""

    @staticmethod
    def calculate_se(acc_percentage: float, sample_size: int) -> float:
        """计算准确率的标准差 SE = sqrt(p * (1 - p) / N) * 100%"""
        if sample_size <= 0:
            return 0.0
        p = acc_percentage / 100.0
        se = math.sqrt(p * (1.0 - p) / sample_size) * 100.0
        return round(se, 2)

    @staticmethod
    def generate_table_2(
        base_data: Dict[str, Dict],
        neo_data: Dict[str, Dict],
    ) -> str:
        """根据 Base 与 Neo 数据自动生成 Markdown Table 2"""
        headers = ["Subset", "Base Score", "Neo Score", "Δ (pp)", "SE base", "SE Neo"]
        rows = []

        for subset, b_info in base_data.items():
            if subset not in neo_data:
                continue
            n_info = neo_data[subset]

            b_score = b_info["score"]
            n_score = n_info["score"]
            delta = n_score - b_score

            b_se = BenchmarkMetrics.calculate_se(b_score, b_info.get("n_samples", 1000))
            n_se = BenchmarkMetrics.calculate_se(n_score, n_info.get("n_samples", 1000))

            b_str = f"**{b_score:.2f}%**" if b_score > n_score else f"{b_score:.2f}%"
            n_str = f"**{n_score:.2f}%**" if n_score > b_score else f"{n_score:.2f}%"
            delta_str = f"**+{delta:.2f}**" if delta > 0 else f"{delta:.2f}"

            rows.append(
                f"| {subset} | {b_str} | {n_str} | {delta_str} | {b_se:.2f} | {n_se:.2f} |"
            )

        table_md = "| " + " | ".join(headers) + " |\n"
        table_md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        table_md += "\n".join(rows)
        return table_md

    @staticmethod
    def generate_cot_table(
        base_metrics: Dict[str, Any],
        neo_metrics: Dict[str, Any],
    ) -> str:
        """生成 CoT 长度统计表 (Table 2 in paper)"""
        headers = ["Model", "Mean chars", "Mean words", "Median chars", "Max chars"]
        rows = []

        for label, m in [("Qwen3.5-4B", base_metrics), ("Qwen3.5-4B-Neo", neo_metrics)]:
            rows.append(
                f"| {label} | {m['mean_chars']:,} | {m['mean_words']:,} | "
                f"{m['median_chars']:,} | {m['max_chars']:,} |"
            )

        # Delta row
        if base_metrics["mean_chars"] > 0:
            d_mean = round(
                (neo_metrics["mean_chars"] - base_metrics["mean_chars"])
                / base_metrics["mean_chars"]
                * 100,
                1,
            )
            d_median = round(
                (neo_metrics["median_chars"] - base_metrics["median_chars"])
                / base_metrics["median_chars"]
                * 100,
                1,
            )
            d_max = round(
                (neo_metrics["max_chars"] - base_metrics["max_chars"])
                / base_metrics["max_chars"]
                * 100,
                1,
            )
            rows.append(f"| Δ | {d_mean}% | {d_mean}% | {d_median}% | {d_max}% |")

        table_md = "| " + " | ".join(headers) + " |\n"
        table_md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        table_md += "\n".join(rows)
        return table_md

    @staticmethod
    def generate_efficiency_table(
        base_metrics: Dict[str, Any],
        neo_metrics: Dict[str, Any],
    ) -> str:
        """生成推理效率表 (Table 3 in paper)"""
        headers = ["Model", "Passes / 10k chars", "Chars / correct pass", "Total think chars"]
        rows = []

        for label, m in [("Qwen3.5-4B", base_metrics), ("Qwen3.5-4B-Neo", neo_metrics)]:
            rows.append(
                f"| {label} | {m['passes_per_10k_chars']:.2f} | "
                f"{m['chars_per_correct_pass']:,} | {m['total_think_chars']:,} |"
            )

        # Delta
        if base_metrics["passes_per_10k_chars"] > 0:
            d_passes = round(
                (neo_metrics["passes_per_10k_chars"] - base_metrics["passes_per_10k_chars"])
                / base_metrics["passes_per_10k_chars"]
                * 100,
                1,
            )
            d_chars = round(
                (neo_metrics["chars_per_correct_pass"] - base_metrics["chars_per_correct_pass"])
                / base_metrics["chars_per_correct_pass"]
                * 100,
                1,
            )
            rows.append(f"| Δ | +{d_passes}% | {d_chars}% | — |")

        table_md = "| " + " | ".join(headers) + " |\n"
        table_md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        table_md += "\n".join(rows)
        return table_md
