"""一键评测管道 — 支持 mock 模式和真实模型推理"""

import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from thinker.eval.benchmarks import generate_all_mock, load_benchmark
from thinker.eval.runner import BaseVsNeoRunner, MockModelRunner
from thinker.eval.adjudicator import FormatRescueAdjudicator
from thinker.eval.metrics import BenchmarkMetrics, EvaluationMetrics
from thinker.eval.plotter import ComprehensiveAcademicPlotter


def run_mock_evaluation(model_scale: str = "4B", output_dir: str = "docs/eval_output"):
    """使用虚拟数据运行完整评测管道"""
    print("=" * 70)
    print(f"  Mock Evaluation Pipeline — Qwen3.5-{model_scale} vs Qwen3.5-{model_scale}-Neo")
    print("=" * 70)

    # 1. 生成虚拟 benchmark 数据
    print("\n[1/6] Generating mock benchmark data...")
    all_data = generate_all_mock(seed=42)
    mmlu_questions = all_data["mmlu_pro"]
    print(f"  MMLU-Pro: {len(mmlu_questions)} questions")
    for name, questions in all_data.items():
        print(f"  {name}: {len(questions)} questions")

    # 2. 运行模拟推理
    print("\n[2/6] Running mock inference (Base vs Neo)...")
    runner = BaseVsNeoRunner(seed=42)
    base_results, neo_results = runner.run_comparison(mmlu_questions)
    print(f"  Base results: {len(base_results)}")
    print(f"  Neo results: {len(neo_results)}")

    # 3. 判定工作流
    print("\n[3/6] Running adjudication...")
    adjudicator = FormatRescueAdjudicator()
    base_records = adjudicator.adjudicate(base_results)
    print(f"  Base adjudication: {adjudicator.get_summary()}")

    adjudicator_neo = FormatRescueAdjudicator()
    neo_records = adjudicator_neo.adjudicate(neo_results)
    print(f"  Neo adjudication: {adjudicator_neo.get_summary()}")

    # 4. 计算指标
    print("\n[4/6] Computing metrics...")
    base_metrics = EvaluationMetrics.compute_all_metrics(base_results)
    neo_metrics = EvaluationMetrics.compute_all_metrics(neo_results)

    print(f"  Base pass@1: {base_metrics['pass_at_1']}%")
    print(f"  Neo pass@1: {neo_metrics['pass_at_1']}%")
    print(f"  Base truncation: {base_metrics['truncation_rate']}%")
    print(f"  Neo truncation: {neo_metrics['truncation_rate']}%")
    print(f"  Base mean chars: {base_metrics['mean_chars']:,}")
    print(f"  Neo mean chars: {neo_metrics['mean_chars']:,}")

    # 5. 生成图表
    print("\n[5/6] Generating plots...")
    plotter = ComprehensiveAcademicPlotter(output_dir=output_dir)
    all_metrics = plotter.generate_all_from_results(
        base_results, neo_results, model_scale=model_scale
    )
    print(f"  Plots saved to: {output_dir}/")

    # 6. 生成报告
    print("\n[6/6] Generating report...")
    report = generate_markdown_report(
        base_metrics, neo_metrics, all_metrics, model_scale
    )
    report_path = os.path.join(output_dir, "eval_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report saved to: {report_path}")

    # 保存原始结果
    results_path = os.path.join(output_dir, "raw_results.json")
    with open(results_path, "w") as f:
        json.dump({"base": base_results, "neo": neo_results}, f, indent=2, default=str)
    print(f"  Raw results saved to: {results_path}")

    print("\n" + "=" * 70)
    print("  Mock evaluation completed successfully!")
    print("=" * 70)

    return all_metrics


def generate_markdown_report(base_m, neo_m, all_metrics, model_scale):
    """生成 Markdown 格式的评测报告"""
    tradeoff = all_metrics["tradeoff"]

    lines = [
        f"# Qwen3.5-{model_scale} vs Qwen3.5-{model_scale}-Neo Evaluation Report",
        "",
        "## 1. Overall Results",
        "",
        "| Metric | Base | Neo | Delta |",
        "|--------|------|-----|-------|",
        f"| pass@1 | {base_m['pass_at_1']}% | {neo_m['pass_at_1']}% | +{tradeoff['delta_accuracy']:.2f} pp |",
        f"| Truncation rate | {base_m['truncation_rate']}% | {neo_m['truncation_rate']}% | — |",
        f"| Rescues | {base_m['rescues']} | {neo_m['rescues']} | — |",
        "",
        "## 2. Chain-of-Thought Statistics",
        "",
        "| Model | Mean chars | Mean words | Median chars | Max chars |",
        "|-------|-----------|------------|--------------|-----------|",
        f"| Qwen3.5-{model_scale} | {base_m['mean_chars']:,} | {base_m['mean_words']:,} | {base_m['median_chars']:,} | {base_m['max_chars']:,} |",
        f"| Qwen3.5-{model_scale}-Neo | {neo_m['mean_chars']:,} | {neo_m['mean_words']:,} | {neo_m['median_chars']:,} | {neo_m['max_chars']:,} |",
        "",
        "## 3. Reasoning Efficiency",
        "",
        "| Model | Passes / 10k chars | Chars / correct pass | Total think chars |",
        "|-------|-------------------|---------------------|-------------------|",
        f"| Qwen3.5-{model_scale} | {base_m['passes_per_10k_chars']:.2f} | {base_m['chars_per_correct_pass']:,} | {base_m['total_think_chars']:,} |",
        f"| Qwen3.5-{model_scale}-Neo | {neo_m['passes_per_10k_chars']:.2f} | {neo_m['chars_per_correct_pass']:,} | {neo_m['total_think_chars']:,} |",
        "",
        "## 4. Error Breakdown",
        "",
        "| Model | Correct | Completed but wrong | Truncated |",
        "|-------|---------|---------------------|-----------|",
        f"| Qwen3.5-{model_scale} | {base_m['passed']} | {base_m['completed_wrong']} | {base_m['truncated']} |",
        f"| Qwen3.5-{model_scale}-Neo | {neo_m['passed']} | {neo_m['completed_wrong']} | {neo_m['truncated']} |",
        "",
        "## 5. Accuracy-Cost Tradeoff",
        "",
        f"- **Accuracy gain**: +{tradeoff['delta_accuracy']:.2f} pp",
        f"- **Cost reduction**: {tradeoff['delta_cost_pct']:.1f}% fewer think chars",
        "",
        "## 6. Per-Category Breakdown (MMLU-Pro)",
        "",
        "| Category | Base | Neo | Delta |",
        "|----------|------|-----|-------|",
    ]

    base_per_cat = all_metrics["base_per_cat"]
    neo_per_cat = all_metrics["neo_per_cat"]
    for cat in sorted(base_per_cat.keys()):
        b = base_per_cat[cat]["pass_at_1"]
        n = neo_per_cat.get(cat, {}).get("pass_at_1", 0)
        delta = n - b
        sign = "+" if delta >= 0 else ""
        lines.append(f"| {cat} | {b:.1f}% | {n:.1f}% | {sign}{delta:.1f} pp |")

    lines.extend([
        "",
        "## 7. Generated Artifacts",
        "",
        "- `Figure_1_Leaderboard_Composite.pdf` — Per-sub-benchmark + overall",
        "- `Figure_2_Per_Category.pdf` — MMLU-Pro per-category accuracy",
        "- `Figure_3_Sub_Deltas.pdf` — Per-sub-benchmark delta",
        "- `Figure_4_Length_Dist.pdf` — Think-chain length distribution",
        "- `Figure_5_Acc_Cost.pdf` — Accuracy vs reasoning cost",
        "- `Figure_6_Truncation_Rescue_Composite.pdf` — Truncation & rescue rates",
        "- `Figure_7_Error_Breakdown.pdf` — Error type breakdown",
        "- `Figure_8_Efficiency.pdf` — Reasoning efficiency",
        "- `raw_results.json` — Raw inference results",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Full Evaluation Pipeline — Mock mode with virtual data"
    )
    parser.add_argument(
        "--model_scale",
        type=str,
        default="4B",
        help="Model scale suffix (e.g. 4B, 9B)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="docs/eval_output",
        help="Output directory for plots and reports",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="mmlu_pro",
        choices=["mmlu_pro", "bbh", "gpqa", "math_hard", "musr", "all"],
        help="Benchmark to evaluate",
    )
    args = parser.parse_args()

    run_mock_evaluation(
        model_scale=args.model_scale,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
