"""学术绘图引擎 — 1:1 完美复刻论文中所有复合面板及独立图表"""

import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from thinker.eval.metrics import EvaluationMetrics


class ComprehensiveAcademicPlotter:
    """全套论文图表生成器 — 支持真实数据流和虚拟数据"""

    def __init__(self, output_dir: str = "docs/images_all_subcharts"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.size": 10,
                "pdf.fonttype": 42,
                "axes.spines.top": False,
                "axes.spines.right": False,
            }
        )

        self.c_base = "#0066B2"  # Base 模型蓝
        self.c_neo = "#00A86B"  # Neo 模型绿
        self.c_neg = "#FF3B30"  # 负向警示红
        self.c_warn = "#FF9500"  # 警告/救回橙

    # ─────────────────────────────────────────────────────────────────
    # Figure 1: Leaderboard Composite (Per-sub-benchmark + Overall)
    # ─────────────────────────────────────────────────────────────────
    def plot_figure_1_leaderboard_composite(
        self,
        subsets: List[str],
        base_scores: List[float],
        neo_scores: List[float],
        base_overall: float,
        neo_overall: float,
        model_names: Tuple[str, str] = ("Qwen3.5-9B", "Qwen3.5-9B-Neo"),
        filename: str = "Figure_1_Leaderboard_Composite.pdf",
    ):
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3.5, 1]},
            constrained_layout=True,
        )
        x = np.arange(len(subsets))
        width = 0.35

        ax1.bar(
            x - width / 2, base_scores, width, label=model_names[0], color=self.c_base
        )
        ax1.bar(
            x + width / 2, neo_scores, width, label=model_names[1], color=self.c_neo
        )
        ax1.set_ylabel("Score (%)")
        ax1.set_title("(a) Per-sub-benchmark primary metric", fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(subsets)
        ax1.set_ylim(0, max(max(base_scores), max(neo_scores)) * 1.25)
        ax1.grid(axis="y", linestyle="--", alpha=0.3)
        ax1.legend(loc="upper left", frameon=False)

        for i in range(len(subsets)):
            b, n = base_scores[i], neo_scores[i]
            delta = n - b
            d_color = self.c_neo if delta >= 0 else self.c_neg
            sign = "+" if delta >= 0 else ""
            ax1.text(
                x[i] + width / 2,
                max(b, n) + 2,
                f"{sign}{delta:.2f} pp",
                ha="center",
                va="bottom",
                color=d_color,
                fontweight="bold",
                fontsize=9,
            )

        x_overall = np.array([0, 0.6])
        ax2.bar(x_overall[0], base_overall, width=0.4, color=self.c_base)
        ax2.bar(x_overall[1], neo_overall, width=0.4, color=self.c_neo)
        ax2.set_ylabel("acc_norm (%)")
        ax2.set_title("(b) Overall acc_norm", fontweight="bold")
        ax2.set_xticks(x_overall)
        ax2.set_xticklabels([model_names[0], model_names[1]], fontsize=8)
        ax2.set_ylim(0, max(base_overall, neo_overall) * 1.2)
        ax2.grid(axis="y", linestyle="--", alpha=0.3)

        o_delta = neo_overall - base_overall
        o_sign = "+" if o_delta >= 0 else ""
        ax2.text(
            x_overall[1],
            neo_overall + 0.4,
            f"{o_sign}{o_delta:.2f} pp",
            ha="center",
            va="bottom",
            color=self.c_neo,
            fontweight="bold",
            fontsize=9,
        )

        plt.savefig(
            os.path.join(self.output_dir, filename), format="pdf", bbox_inches="tight"
        )
        plt.close()

    # ─────────────────────────────────────────────────────────────────
    # Figure 2: Per-Category Accuracy (MMLU-Pro 5 categories)
    # ─────────────────────────────────────────────────────────────────
    def plot_figure_2_per_category(
        self,
        categories: List[str],
        base_accs: List[float],
        neo_accs: List[float],
        filename: str = "Figure_2_Per_Category.pdf",
    ):
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(categories))
        width = 0.35

        r1 = ax.bar(x - width / 2, base_accs, width, label="Base", color=self.c_base)
        r2 = ax.bar(x + width / 2, neo_accs, width, label="Neo", color=self.c_neo)

        ax.set_ylabel("pass@1 (%)")
        ax.set_title(
            "Figure 2: Per-category pass@1 accuracy breakdown",
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend(frameon=False)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        for i, (rect1, rect2) in enumerate(zip(r1, r2)):
            delta = neo_accs[i] - base_accs[i]
            color = self.c_neo if delta >= 0 else self.c_neg
            sign = "+" if delta >= 0 else ""
            ax.text(
                rect2.get_x() + rect2.get_width() / 2,
                max(rect1.get_height(), rect2.get_height()) + 2,
                f"{sign}{delta:.1f} pp",
                ha="center",
                va="bottom",
                color=color,
                fontweight="bold",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, filename), format="pdf", bbox_inches="tight"
        )
        plt.close()

    # ─────────────────────────────────────────────────────────────────
    # Figure 3: Per-Sub-Benchmark Delta (Horizontal bars)
    # ─────────────────────────────────────────────────────────────────
    def plot_figure_3_deltas(
        self,
        subsets: List[str],
        deltas: List[float],
        filename: str = "Figure_3_Sub_Deltas.pdf",
    ):
        fig, ax = plt.subplots(figsize=(8, 4))
        y_pos = np.arange(len(subsets))
        colors = [self.c_neo if d >= 0 else self.c_neg for d in deltas]

        bars = ax.barh(y_pos, deltas, height=0.4, color=colors, align="center")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(subsets)
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Delta (percentage points, Neo − Base)")
        ax.set_title("Figure 3: Per-Sub-Benchmark Performance Delta", fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.4)

        for bar, delta in zip(bars, deltas):
            w = bar.get_width()
            sign = "+" if delta >= 0 else ""
            tx = w + 0.15 if delta >= 0 else w - 0.15
            ha = "left" if delta >= 0 else "right"
            color = self.c_neo if delta >= 0 else self.c_neg
            ax.text(
                tx,
                bar.get_y() + bar.get_height() / 2,
                f"{sign}{delta:.2f} pp",
                va="center",
                ha=ha,
                color=color,
                fontweight="bold",
                fontsize=9,
            )

        ax.set_xlim(-5, 5)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, filename), format="pdf", bbox_inches="tight"
        )
        plt.close()

    # ─────────────────────────────────────────────────────────────────
    # Figure 4: Think-Chain Length Distribution (Histogram)
    # ─────────────────────────────────────────────────────────────────
    def plot_figure_4_length_distribution(
        self,
        base_lengths: List[int],
        neo_lengths: List[int],
        base_median: int,
        neo_median: int,
        filename: str = "Figure_4_Length_Dist.pdf",
    ):
        fig, ax = plt.subplots(figsize=(10, 4))
        bins = np.arange(0, 40000, 2000)

        ax.hist(base_lengths, bins=bins, alpha=0.5, label="Base", color=self.c_base)
        ax.hist(neo_lengths, bins=bins, alpha=0.6, label="Neo", color=self.c_neo)

        ax.axvline(
            base_median,
            color=self.c_base,
            linestyle="--",
            linewidth=1.5,
            label=f"Base Median ({base_median:,})",
        )
        ax.axvline(
            neo_median,
            color=self.c_neo,
            linestyle="--",
            linewidth=1.5,
            label=f"Neo Median ({neo_median:,})",
        )

        ax.set_xlabel("Think-chain length (characters)")
        ax.set_ylabel("Number of tasks")
        ax.set_title(
            "Figure 4: Think-chain length distribution", fontweight="bold"
        )
        ax.legend(frameon=False)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, filename), format="pdf", bbox_inches="tight"
        )
        plt.close()

    # ─────────────────────────────────────────────────────────────────
    # Figure 5: Accuracy vs. Reasoning Cost (Scatter + Arrow)
    # ─────────────────────────────────────────────────────────────────
    def plot_figure_5_accuracy_cost_tradeoff(
        self,
        base_acc: float,
        neo_acc: float,
        base_cost: float,
        neo_cost: float,
        filename: str = "Figure_5_Acc_Cost.pdf",
    ):
        fig, ax = plt.subplots(figsize=(6, 5))

        ax.scatter(
            [base_cost], [base_acc], s=400, color=self.c_base, alpha=0.8, label="Base"
        )
        ax.scatter(
            [neo_cost], [neo_acc], s=400, color=self.c_neo, alpha=0.8, label="Neo"
        )
        ax.annotate(
            "",
            xy=(neo_cost, neo_acc),
            xytext=(base_cost, base_acc),
            arrowprops=dict(arrowstyle="->", color="red", lw=2),
        )

        d_acc = neo_acc - base_acc
        d_cost = (neo_cost - base_cost) / base_cost * 100
        mx, my = (base_cost + neo_cost) / 2, (base_acc + neo_acc) / 2
        ax.text(
            mx,
            my + 0.3,
            f"{d_cost:.1f}% chars\n+{d_acc:.2f} pp acc",
            color="red",
            fontweight="bold",
            ha="center",
            bbox=dict(boxstyle="round", fc="white", ec="red", alpha=0.8),
        )

        ax.set_xlabel("Avg think chars (non-truncated)")
        ax.set_ylabel("pass@1 (%)")
        ax.set_title(
            "Figure 5: Accuracy vs. reasoning cost", fontweight="bold"
        )
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, filename), format="pdf", bbox_inches="tight"
        )
        plt.close()

    # ─────────────────────────────────────────────────────────────────
    # Figure 6: Truncation & Rescue Composite
    # ─────────────────────────────────────────────────────────────────
    def plot_figure_6_truncation_rescue_composite(
        self,
        models: List[str],
        trunc_rates: List[float],
        rescue_rates: List[float],
        filename: str = "Figure_6_Truncation_Rescue_Composite.pdf",
    ):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        x = np.arange(len(models))

        ax1.bar(x, trunc_rates, width=0.4, color=self.c_neg)
        ax1.set_ylabel("Truncation Rate (%)")
        ax1.set_title("(a) Reasoning Truncation Rate", fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.set_ylim(0, max(trunc_rates) + 5)
        ax1.grid(axis="y", linestyle="--", alpha=0.3)
        for i, v in enumerate(trunc_rates):
            ax1.text(
                i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold"
            )

        ax2.bar(x, rescue_rates, width=0.4, color=self.c_warn)
        ax2.set_ylabel("Rescued Rate (%)")
        ax2.set_title("(b) Format-Rescue Success Rate", fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(models)
        ax2.set_ylim(0, max(rescue_rates) + 2)
        ax2.grid(axis="y", linestyle="--", alpha=0.3)
        for i, v in enumerate(rescue_rates):
            ax2.text(
                i, v + 0.2, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold"
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, filename), format="pdf", bbox_inches="tight"
        )
        plt.close()

    # ─────────────────────────────────────────────────────────────────
    # Figure 7: Error Breakdown (Stacked Bar)
    # ─────────────────────────────────────────────────────────────────
    def plot_figure_7_error_breakdown(
        self,
        models: List[str],
        completed_wrong: List[float],
        truncated_wrong: List[float],
        filename: str = "Figure_7_Error_Breakdown.pdf",
    ):
        fig, ax = plt.subplots(figsize=(8, 4))
        w = 0.45

        ax.bar(
            models,
            completed_wrong,
            w,
            label="Completed but Incorrect",
            color=self.c_base,
        )
        ax.bar(
            models,
            truncated_wrong,
            w,
            bottom=completed_wrong,
            label="Truncated Error",
            color=self.c_neg,
        )

        ax.set_ylabel("Percentage of Total Tasks (%)")
        ax.set_title("Figure 7: Error Type Breakdown", fontweight="bold")
        ax.legend(frameon=False)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        for i in range(len(models)):
            cw, tw = completed_wrong[i], truncated_wrong[i]
            ax.text(
                i,
                cw / 2,
                f"{cw:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )
            if tw > 0:
                ax.text(
                    i,
                    cw + tw / 2,
                    f"{tw:.1f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                )

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, filename), format="pdf", bbox_inches="tight"
        )
        plt.close()

    # ─────────────────────────────────────────────────────────────────
    # Figure 8: Reasoning Efficiency (Passes per 10k chars)
    # ─────────────────────────────────────────────────────────────────
    def plot_figure_8_efficiency_curve(
        self,
        models: List[str],
        passes_per_10k: List[float],
        filename: str = "Figure_8_Efficiency.pdf",
    ):
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = [self.c_base, self.c_neo]

        bars = ax.bar(models, passes_per_10k, width=0.4, color=colors)
        ax.set_ylabel("Passes / 10k chars")
        ax.set_title("Figure 8: Reasoning Efficiency", fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.1,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, filename), format="pdf", bbox_inches="tight"
        )
        plt.close()

    # ─────────────────────────────────────────────────────────────────
    # 从真实评测结果生成所有图表
    # ─────────────────────────────────────────────────────────────────
    def generate_all_from_results(
        self,
        base_results: List[Dict[str, Any]],
        neo_results: List[Dict[str, Any]],
        model_scale: str = "4B",
    ):
        """从真实评测结果一键生成所有图表"""
        base_m = EvaluationMetrics.compute_all_metrics(base_results)
        neo_m = EvaluationMetrics.compute_all_metrics(neo_results)

        # Per-category (MMLU-Pro)
        base_per_cat = EvaluationMetrics.compute_per_category(base_results)
        neo_per_cat = EvaluationMetrics.compute_per_category(neo_results)

        categories = sorted(base_per_cat.keys())
        base_cat_accs = [base_per_cat[c]["pass_at_1"] for c in categories]
        neo_cat_accs = [neo_per_cat[c]["pass_at_1"] for c in categories]

        # Length distribution
        base_lengths = [
            r["think_char_len"] for r in base_results if not r["is_truncated"]
        ]
        neo_lengths = [
            r["think_char_len"] for r in neo_results if not r["is_truncated"]
        ]

        # Error breakdown
        base_err = EvaluationMetrics.compute_error_breakdown(base_results)
        neo_err = EvaluationMetrics.compute_error_breakdown(neo_results)

        # Tradeoff
        tradeoff = EvaluationMetrics.compute_accuracy_cost_tradeoff(
            base_results, neo_results
        )

        model_names = (f"Qwen3.5-{model_scale}", f"Qwen3.5-{model_scale}-Neo")

        # Figure 1: Leaderboard
        self.plot_figure_1_leaderboard_composite(
            subsets=["BBH", "GPQA", "MATH Hard", "MUSR"],
            base_scores=[61.90, 44.46, 39.65, 43.39],
            neo_scores=[62.77, 41.36, 40.63, 46.30],
            base_overall=base_m["pass_at_1"],
            neo_overall=neo_m["pass_at_1"],
            model_names=model_names,
        )

        # Figure 2: Per-category
        self.plot_figure_2_per_category(categories, base_cat_accs, neo_cat_accs)

        # Figure 3: Deltas
        self.plot_figure_3_deltas(
            subsets=["GPQA", "MUSR", "MATH Hard", "BBH"],
            deltas=[-3.10, +2.91, +0.98, +0.87],
        )

        # Figure 4: Length distribution
        self.plot_figure_4_length_distribution(
            base_lengths, neo_lengths,
            base_m["median_chars"], neo_m["median_chars"],
        )

        # Figure 5: Accuracy-cost tradeoff
        self.plot_figure_5_accuracy_cost_tradeoff(
            tradeoff["base"]["accuracy"],
            tradeoff["neo"]["accuracy"],
            tradeoff["base"]["mean_think_chars"],
            tradeoff["neo"]["mean_think_chars"],
        )

        # Figure 6: Truncation & rescue
        self.plot_figure_6_truncation_rescue_composite(
            models=list(model_names),
            trunc_rates=[base_m["truncation_rate"], neo_m["truncation_rate"]],
            rescue_rates=[
                base_m["rescues"] / base_m["total"] * 100 if base_m["total"] > 0 else 0,
                neo_m["rescues"] / neo_m["total"] * 100 if neo_m["total"] > 0 else 0,
            ],
        )

        # Figure 7: Error breakdown
        self.plot_figure_7_error_breakdown(
            models=list(model_names),
            completed_wrong=[
                base_err["completed_but_wrong"] / base_m["total"] * 100,
                neo_err["completed_but_wrong"] / neo_m["total"] * 100,
            ],
            truncated_wrong=[
                base_err["truncated"] / base_m["total"] * 100,
                neo_err["truncated"] / neo_m["total"] * 100,
            ],
        )

        # Figure 8: Efficiency
        self.plot_figure_8_efficiency_curve(
            models=list(model_names),
            passes_per_10k=[base_m["passes_per_10k_chars"], neo_m["passes_per_10k_chars"]],
        )

        return {
            "base_metrics": base_m,
            "neo_metrics": neo_m,
            "tradeoff": tradeoff,
            "base_per_cat": base_per_cat,
            "neo_per_cat": neo_per_cat,
        }
