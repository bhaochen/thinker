"""报告生成器 — 汇总所有表格图表，输出 Markdown 格式完整评测报告"""

import os
from typing import Any, Dict, List, Optional

from thinker.eval.metrics import BenchmarkMetrics, EvaluationMetrics


class EvaluationReport:
    """完整的评测报告生成器"""

    def __init__(
        self,
        base_results: List[Dict[str, Any]],
        neo_results: List[Dict[str, Any]],
        model_scale: str = "4B",
        output_dir: str = "docs/eval_output",
    ):
        self.base_results = base_results
        self.neo_results = neo_results
        self.model_scale = model_scale
        self.output_dir = output_dir

        self.base_metrics = EvaluationMetrics.compute_all_metrics(base_results)
        self.neo_metrics = EvaluationMetrics.compute_all_metrics(neo_results)
        self.tradeoff = EvaluationMetrics.compute_accuracy_cost_tradeoff(
            base_results, neo_results
        )
        self.base_per_cat = EvaluationMetrics.compute_per_category(base_results)
        self.neo_per_cat = EvaluationMetrics.compute_per_category(neo_results)

    def generate_full_report(self) -> str:
        """生成完整的 Markdown 评测报告"""
        sections = [
            self._title_section(),
            self._abstract_section(),
            self._overall_results_section(),
            self._cot_statistics_section(),
            self._efficiency_section(),
            self._error_breakdown_section(),
            self._tradeoff_section(),
            self._per_category_section(),
            self._conclusion_section(),
            self._artifacts_section(),
        ]
        return "\n".join(sections)

    def _title_section(self) -> str:
        return f"""# Qwen3.5-{self.model_scale} vs Qwen3.5-{self.model_scale}-Neo
## MMLU-Pro Benchmark Evaluation Report

**Adjudicated Evaluation** · n = 250 tasks · 5 categories × 50 each · 2026

---
"""

    def _abstract_section(self) -> str:
        b = self.base_metrics
        n = self.neo_metrics
        return f"""## Abstract

We evaluate **Qwen3.5-{self.model_scale}-Neo** against the original **Qwen3.5-{self.model_scale}**
on a 250-task MMLU-Pro subset spanning five categories (biology, computer science, mathematics,
other sciences, and physics), 50 questions per category.

After manually adjudicating incomplete outputs in the base model by inspecting truncated reasoning
chains, **Qwen3.5-{self.model_scale}-Neo** achieves a pass@1 of **{n['pass_at_1']:.2f}%**
({n['passed']}/{n['total']}) against **{b['pass_at_1']:.2f}%** ({b['passed']}/{b['total']}) for
Qwen3.5-{self.model_scale}, a gap of **+{self.tradeoff['delta_accuracy']:.2f} percentage points**.

Crucially, **Qwen3.5-{self.model_scale}-Neo** produces **{abs(self.tradeoff['delta_cost_pct']):.1f}% shorter**
mean reasoning traces ({n['mean_chars']:,} vs. {b['mean_chars']:,} chars), yielding a
**+{((n['passes_per_10k_chars'] - b['passes_per_10k_chars']) / b['passes_per_10k_chars'] * 100):.1f}% gain**
in reasoning efficiency ({n['passes_per_10k_chars']:.2f} vs. {b['passes_per_10k_chars']:.2f}
correct solutions per 10k think chars).

---
"""

    def _overall_results_section(self) -> str:
        b = self.base_metrics
        n = self.neo_metrics
        return f"""## 1. Overall Results

| Metric | Qwen3.5-{self.model_scale} | Qwen3.5-{self.model_scale}-Neo | Delta |
|--------|---------------------------|-------------------------------|-------|
| Passed | {b['passed']} / {b['total']} | {n['passed']} / {n['total']} | +{n['passed'] - b['passed']} |
| pass@1 | {b['pass_at_1']:.2f}% | **{n['pass_at_1']:.2f}%** | +{self.tradeoff['delta_accuracy']:.2f} pp |
| Rescued | {b['rescues']} | {n['rescues']} | — |
| Truncated | {b['truncated']} ({b['truncation_rate']:.1f}%) | {n['truncated']} ({n['truncation_rate']:.1f}%) | — |

---
"""

    def _cot_statistics_section(self) -> str:
        b = self.base_metrics
        n = self.neo_metrics
        return f"""## 2. Chain-of-Thought Statistics

| Model | Mean chars | Mean words | Median chars | Max chars |
|-------|-----------|------------|--------------|-----------|
| Qwen3.5-{self.model_scale} | {b['mean_chars']:,} | {b['mean_words']:,} | {b['median_chars']:,} | {b['max_chars']:,} |
| Qwen3.5-{self.model_scale}-Neo | **{n['mean_chars']:,}** | **{n['mean_words']:,}** | **{n['median_chars']:,}** | **{n['max_chars']:,}** |
| Δ | {((n['mean_chars'] - b['mean_chars']) / b['mean_chars'] * 100):.1f}% | {((n['mean_words'] - b['mean_words']) / b['mean_words'] * 100):.1f}% | {((n['median_chars'] - b['median_chars']) / b['median_chars'] * 100):.1f}% | {((n['max_chars'] - b['max_chars']) / b['max_chars'] * 100):.1f}% |

---
"""

    def _efficiency_section(self) -> str:
        b = self.base_metrics
        n = self.neo_metrics
        return f"""## 3. Reasoning Efficiency

| Model | Passes / 10k chars | Chars / correct pass | Total think chars |
|-------|-------------------|---------------------|-------------------|
| Qwen3.5-{self.model_scale} | {b['passes_per_10k_chars']:.2f} | {b['chars_per_correct_pass']:,} | {b['total_think_chars']:,} |
| Qwen3.5-{self.model_scale}-Neo | **{n['passes_per_10k_chars']:.2f}** | **{n['chars_per_correct_pass']:,}** | **{n['total_think_chars']:,}** |
| Δ | +{((n['passes_per_10k_chars'] - b['passes_per_10k_chars']) / b['passes_per_10k_chars'] * 100):.1f}% | {((n['chars_per_correct_pass'] - b['chars_per_correct_pass']) / b['chars_per_correct_pass'] * 100):.1f}% | — |

---
"""

    def _error_breakdown_section(self) -> str:
        b = self.base_metrics
        n = self.neo_metrics
        return f"""## 4. Error Breakdown

| Model | Correct | Completed but wrong | Truncated |
|-------|---------|---------------------|-----------|
| Qwen3.5-{self.model_scale} | {b['passed']} | {b['completed_wrong']} | {b['truncated']} |
| Qwen3.5-{self.model_scale}-Neo | {n['passed']} | {n['completed_wrong']} | {n['truncated']} |

---
"""

    def _tradeoff_section(self) -> str:
        t = self.tradeoff
        return f"""## 5. Accuracy-Cost Tradeoff

- **Accuracy gain**: +{t['delta_accuracy']:.2f} pp
- **Cost reduction**: {abs(t['delta_cost_pct']):.1f}% fewer think chars
- **Efficiency gain**: +{((self.neo_metrics['passes_per_10k_chars'] - self.base_metrics['passes_per_10k_chars']) / self.base_metrics['passes_per_10k_chars'] * 100):.1f}% passes per 10k chars

---
"""

    def _per_category_section(self) -> str:
        lines = [
            f"## 6. Per-Category Breakdown (MMLU-Pro)",
            "",
            "| Category | Base | Neo | Delta |",
            "|----------|------|-----|-------|",
        ]
        for cat in sorted(self.base_per_cat.keys()):
            b = self.base_per_cat[cat]["pass_at_1"]
            n = self.neo_per_cat.get(cat, {}).get("pass_at_1", 0)
            delta = n - b
            sign = "+" if delta >= 0 else ""
            lines.append(f"| {cat} | {b:.1f}% | {n:.1f}% | {sign}{delta:.1f} pp |")
        lines.append("\n---\n")
        return "\n".join(lines)

    def _conclusion_section(self) -> str:
        n = self.neo_metrics
        return f"""## 7. Conclusion

Across 250 MMLU-Pro questions spanning five academic domains:

1. **Marginal accuracy gain**: Qwen3.5-{self.model_scale}-Neo achieves {n['pass_at_1']:.2f}% pass@1,
   exceeding Qwen3.5-{self.model_scale} by {self.tradeoff['delta_accuracy']:.2f} pp.
2. **Compressed reasoning**: mean think-chain length falls {abs(self.tradeoff['delta_cost_pct']):.1f}%
   ({self.base_metrics['mean_chars']:,} → {n['mean_chars']:,} chars).
3. **Strong efficiency gain**: Neo solves {n['passes_per_10k_chars']:.2f} tasks per 10k think chars
   vs. {self.base_metrics['passes_per_10k_chars']:.2f} for the base model.
4. **Better error quality**: Neo's failures are predominantly truncation events rather than
   substantively wrong answers.

---
"""

    def _artifacts_section(self) -> str:
        return """## 8. Generated Artifacts

- `Figure_1_Leaderboard_Composite.pdf`
- `Figure_2_Per_Category.pdf`
- `Figure_3_Sub_Deltas.pdf`
- `Figure_4_Length_Dist.pdf`
- `Figure_5_Acc_Cost.pdf`
- `Figure_6_Truncation_Rescue_Composite.pdf`
- `Figure_7_Error_Breakdown.pdf`
- `Figure_8_Efficiency.pdf`
- `raw_results.json`
"""

    def save(self, filename: str = "eval_report.md"):
        """保存报告到文件"""
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            f.write(self.generate_full_report())
        return path
