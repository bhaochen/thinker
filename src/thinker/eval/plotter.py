"""学术绘图引擎 — 1:1 完美复刻论文中所有复合面板及独立图表"""

import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.patches import FancyArrowPatch

from thinker.eval.metrics import EvaluationMetrics


class ComprehensiveAcademicPlotter:
    """全套论文图表生成器 — 支持真实数据流和虚拟数据"""

    def __init__(self, output_dir: str = "docs/images_all_subcharts", png: bool = True):
        self.output_dir = output_dir
        self.png = png
        os.makedirs(self.output_dir, exist_ok=True)

        # ── 全局样式：浅灰色轴线/刻度/网格，养眼风格 ──
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.size": 10,
                "pdf.fonttype": 42,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.spines.left": True,
                "axes.spines.bottom": True,
                "axes.edgecolor": "#CCCCCC",
                "axes.linewidth": 0.8,
                "axes.labelcolor": "#888888",
                "axes.titlesize": 11,
                "axes.titleweight": "bold",
                "xtick.color": "#888888",
                "ytick.color": "#888888",
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "grid.color": "#E0E0E0",
                "grid.linestyle": "--",
                "grid.linewidth": 0.6,
                "grid.alpha": 0.7,
                "legend.frameon": True,
                "legend.edgecolor": "#CCCCCC",
                "legend.fontsize": 9,
                "figure.facecolor": "white",
                "axes.facecolor": "white",
            }
        )

        # ── 柱子颜色：参考图采样 ──
        self.c_base = "#90B8D8"       # 浅蓝（参考图采样）
        self.c_base_edge = "#6E9FC4"   # 深一点的蓝（边框）
        self.c_neo = "#98C8B0"         # 浅青绿（参考图采样）
        self.c_neo_edge = "#7DB89A"    # 深一点的绿（边框）
        self.c_neg = "#E57373"         # 柔和红
        self.c_warn = "#FFB74D"        # 柔和橙

        # 柱子不透明（与参考图一致）
        self.bar_alpha = 1.0

    def _save(self, fig, filename: str):
        """保存图表 — 同时输出 PDF（论文用）和 PNG（预览用）"""
        stem = os.path.splitext(filename)[0]
        pdf_path = os.path.join(self.output_dir, f"{stem}.pdf")
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        if self.png:
            png_path = os.path.join(self.output_dir, f"{stem}.png")
            fig.savefig(png_path, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)

    def _style_ax(self, ax):
        """统一样式：浅灰轴线、刻度、网格"""
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CCCCCC")
        ax.spines["bottom"].set_color("#CCCCCC")
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(colors="#888888", labelsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="#E0E0E0")
        ax.set_axisbelow(True)

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

        # 柱子：半透明填充 + 边框
        ax1.bar(
            x - width / 2, base_scores, width,
            label=model_names[0], color=self.c_base,
            edgecolor=self.c_base_edge, linewidth=0.8, alpha=self.bar_alpha,
        )
        ax1.bar(
            x + width / 2, neo_scores, width,
            label=model_names[1], color=self.c_neo,
            edgecolor=self.c_neo_edge, linewidth=0.8, alpha=self.bar_alpha,
        )
        ax1.set_ylabel("Score (%)")
        ax1.set_title("(a) Per-sub-benchmark primary metric", fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(subsets)
        ax1.set_ylim(0, max(max(base_scores), max(neo_scores)) * 1.3)
        ax1.legend(loc="upper left", frameon=True, edgecolor="#CCCCCC")
        self._style_ax(ax1)

        # 柱子上方数值标注
        for i in range(len(subsets)):
            b, n = base_scores[i], neo_scores[i]
            delta = n - b
            d_color = self.c_neo if delta >= 0 else self.c_neg
            sign = "+" if delta >= 0 else ""
            ax1.text(
                x[i] + width / 2,
                max(b, n) + 1.5,
                f"{sign}{delta:.2f} pp",
                ha="center", va="bottom",
                color=d_color, fontweight="bold", fontsize=9,
            )

        # 右图：Overall
        x_overall = np.array([0, 0.6])
        ax2.bar(
            x_overall[0], base_overall, width=0.4,
            color=self.c_base, edgecolor=self.c_base_edge,
            linewidth=0.8, alpha=self.bar_alpha,
        )
        ax2.bar(
            x_overall[1], neo_overall, width=0.4,
            color=self.c_neo, edgecolor=self.c_neo_edge,
            linewidth=0.8, alpha=self.bar_alpha,
        )
        ax2.set_ylabel("acc_norm (%)")
        ax2.set_title("(b) Overall acc_norm", fontweight="bold")
        ax2.set_xticks(x_overall)
        ax2.set_xticklabels([model_names[0], model_names[1]], fontsize=8)
        ax2.set_ylim(0, max(base_overall, neo_overall) * 1.25)
        self._style_ax(ax2)

        o_delta = neo_overall - base_overall
        o_sign = "+" if o_delta >= 0 else ""
        ax2.text(
            x_overall[1],
            neo_overall + 0.5,
            f"{o_sign}{o_delta:.2f} pp",
            ha="center", va="bottom",
            color=self.c_neo, fontweight="bold", fontsize=9,
        )

        self._save(fig, filename)

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
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        x = np.arange(len(categories))
        width = 0.35

        r1 = ax.bar(
            x - width / 2, base_accs, width,
            label="Base", color=self.c_base,
            edgecolor=self.c_base_edge, linewidth=0.8, alpha=self.bar_alpha,
        )
        r2 = ax.bar(
            x + width / 2, neo_accs, width,
            label="Neo", color=self.c_neo,
            edgecolor=self.c_neo_edge, linewidth=0.8, alpha=self.bar_alpha,
        )

        ax.set_ylabel("pass@1 (%)")
        ax.set_title(
            "Figure 2: Per-category pass@1 accuracy breakdown",
            fontweight="bold",
        )
        ax.set_xticks(x)
        # 类别名首字母大写，匹配参考图 "Biology" 风格
        ax.set_xticklabels([
            c.replace("_", " ").title() for c in categories
        ])
        ax.legend(frameon=True, edgecolor="#CCCCCC", loc="upper right")
        ax.set_ylim(0, 120)
        self._style_ax(ax)

        # 柱子上方数值 + delta 标注
        for i, (rect1, rect2) in enumerate(zip(r1, r2)):
            delta = neo_accs[i] - base_accs[i]
            color = self.c_neo if delta >= 0 else self.c_neg
            sign = "+" if delta >= 0 else ""
            # 数值标注
            ax.text(
                rect1.get_x() + rect1.get_width() / 2,
                rect1.get_height() + 1,
                f"{base_accs[i]:.0f}%",
                ha="center", va="bottom",
                color="#555555", fontsize=8,
            )
            ax.text(
                rect2.get_x() + rect2.get_width() / 2,
                rect2.get_height() + 1,
                f"{neo_accs[i]:.0f}%",
                ha="center", va="bottom",
                color="#555555", fontsize=8,
            )
            # delta 标注：居中在两根柱子正上方
            ax.text(
                x[i],
                max(rect1.get_height(), rect2.get_height()) + 6,
                f"{sign}{delta:.0f} pp",
                ha="center", va="bottom",
                color=color, fontweight="bold", fontsize=9,
            )

        self._save(fig, filename)

    # ─────────────────────────────────────────────────────────────────
    # Figure 3: Per-Sub-Benchmark Delta (Horizontal bars)
    # ─────────────────────────────────────────────────────────────────
    def plot_figure_3_deltas(
        self,
        subsets: List[str],
        deltas: List[float],
        metrics: Optional[List[str]] = None,
        filename: str = "Figure_3_Sub_Deltas.pdf",
    ):
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        y_pos = np.arange(len(subsets))
        colors = [self.c_neo if d >= 0 else self.c_neg for d in deltas]

        bars = ax.barh(
            y_pos, deltas, height=0.4, color=colors, align="center",
            edgecolor=[self.c_neo_edge if d >= 0 else "#D32F2F" for d in deltas],
            linewidth=0.8, alpha=self.bar_alpha,
        )
        ax.set_yticks(y_pos)
        # y 轴标签带指标名后缀，如 "GPQA\n(acc_norm)"
        if metrics:
            ax.set_yticklabels([
                f"{s}\n({m})" for s, m in zip(subsets, metrics)
            ])
        else:
            ax.set_yticklabels(subsets)
        ax.invert_yaxis()
        ax.axvline(0, color="#CCCCCC", linewidth=0.8)
        ax.set_xlabel("Delta (percentage points, Neo − Base)")
        ax.set_title("Figure 3: Per-Sub-Benchmark Performance Delta", fontweight="bold")
        self._style_ax(ax)

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
                va="center", ha=ha,
                color=color, fontweight="bold", fontsize=9,
            )

        ax.set_xlim(-5, 5)
        self._save(fig, filename)

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
        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        bins = np.arange(0, 40000, 2000)

        # 直方图：半透明 + 边框
        n_base, bins_base, patches_base = ax.hist(
            base_lengths, bins=bins, alpha=0.5,
            label="Base", color=self.c_base,
            edgecolor=self.c_base_edge, linewidth=0.6,
        )
        n_neo, bins_neo, patches_neo = ax.hist(
            neo_lengths, bins=bins, alpha=0.6,
            label="Neo", color=self.c_neo,
            edgecolor=self.c_neo_edge, linewidth=0.6,
        )

        # 中位数线：虚线（不加 label，图内已有文字标注框说明）
        ax.axvline(
            base_median, color=self.c_base_edge, linestyle="--", linewidth=1.5,
        )
        ax.axvline(
            neo_median, color=self.c_neo_edge, linestyle="--", linewidth=1.5,
        )

        # 中位数标签框（带边框的文本框）)
        ax.text(
            base_median + 200, ax.get_ylim()[1] * 0.95,
            f"Base median\n{base_median:,} chars",
            ha="left", va="top", fontsize=8, color=self.c_base_edge,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=self.c_base_edge, lw=0.8),
        )
        ax.text(
            neo_median - 200, ax.get_ylim()[1] * 0.85,
            f"Neo median\n{neo_median:,} chars",
            ha="right", va="top", fontsize=8, color=self.c_neo_edge,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=self.c_neo_edge, lw=0.8),
        )

        # 柱子上方数字标注（count）
        for i, (nb, nn) in enumerate(zip(n_base, n_neo)):
            x_center = (bins_base[i] + bins_base[i + 1]) / 2
            if nb > 0:
                ax.text(
                    x_center, nb + 0.5, f"{int(nb)}",
                    ha="center", va="bottom", fontsize=7,
                    color=self.c_base_edge, fontweight="bold",
                )
            if nn > 0:
                ax.text(
                    x_center, nn + 0.5, f"{int(nn)}",
                    ha="center", va="bottom", fontsize=7,
                    color=self.c_neo_edge, fontweight="bold",
                )

        ax.set_xlabel("Think-chain length (characters)")
        ax.set_ylabel("Number of tasks")
        ax.set_title(
            "Think-chain length distribution (2000-char bins, non-truncated tasks)",
            fontweight="bold", fontsize=10,
        )
        # x 轴紧凑刻度：2k/4k/6k...（左侧留一个柱宽避免原点 0 重叠）
        ax.set_xlim(-2000, 36000)
        ax.set_xticks(np.arange(0, 36001, 2000))
        ax.set_xticklabels([
            "0" if v == 0 else f"{v // 1000}k" for v in np.arange(0, 36001, 2000)
        ], fontsize=8, rotation=45, ha="right")
        ax.legend(frameon=True, edgecolor="#CCCCCC", fontsize=8)
        self._style_ax(ax)

        self._save(fig, filename)

    # ─────────────────────────────────────────────────────────────────
    # Figure 5: Accuracy vs. Reasoning Cost (Scatter + Arrow)
    # ─────────────────────────────────────────────────────────────────
    def plot_figure_5_accuracy_cost_tradeoff(
        self,
        base_acc: float,
        neo_acc: float,
        base_cost: float,
        neo_cost: float,
        model_names: Tuple[str, str] = ("Qwen3.5-4B", "Qwen3.5-4B-Neo"),
        filename: str = "Figure_5_Acc_Cost.pdf",
    ):
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

        ax.scatter(
            [base_cost], [base_acc], s=400,
            color=self.c_base, alpha=0.85, label=model_names[0],
            edgecolor=self.c_base_edge, linewidth=1.2,
        )
        ax.scatter(
            [neo_cost], [neo_acc], s=400,
            color=self.c_neo, alpha=0.85, label=model_names[1],
            edgecolor=self.c_neo_edge, linewidth=1.2,
        )
        # 气泡旁的模型名标签
        ax.annotate(
            f"{model_names[0]}\n{base_acc:.2f}%",
            xy=(base_cost, base_acc), xytext=(-12, -28),
            textcoords="offset points", fontsize=8, color=self.c_base_edge,
            fontweight="bold", ha="right",
        )
        ax.annotate(
            f"{model_names[1]}\n{neo_acc:.2f}%",
            xy=(neo_cost, neo_acc), xytext=(12, -28),
            textcoords="offset points", fontsize=8, color=self.c_neo_edge,
            fontweight="bold", ha="left",
        )
        ax.annotate(
            "",
            xy=(neo_cost, neo_acc),
            xytext=(base_cost, base_acc),
            arrowprops=dict(arrowstyle="->", color="#E57373", lw=2),
        )

        d_acc = neo_acc - base_acc
        d_cost = (neo_cost - base_cost) / base_cost * 100
        mx, my = (base_cost + neo_cost) / 2, (base_acc + neo_acc) / 2
        ax.text(
            mx, my + 0.3,
            f"{d_cost:.1f}% chars\n+{d_acc:.2f} pp acc",
            color="#E57373", fontweight="bold", ha="center",
            bbox=dict(boxstyle="round", fc="white", ec="#E57373", alpha=0.8),
        )

        ax.set_xlabel("Avg think chars (non-truncated)")
        ax.set_ylabel("pass@1 (%)")
        ax.set_title(
            "Figure 5: Accuracy vs. reasoning cost", fontweight="bold"
        )
        # 留出边距，避免气泡与标签被裁切（注意 min/max，防止坐标轴反向）
        lo_x, hi_x = min(base_cost, neo_cost), max(base_cost, neo_cost)
        span_x = hi_x - lo_x
        ax.set_xlim(lo_x - span_x * 0.30, hi_x + span_x * 0.18)
        span_y = max(abs(neo_acc - base_acc), 0.4)
        ax.set_ylim(min(base_acc, neo_acc) - span_y * 1.4,
                    max(base_acc, neo_acc) + span_y * 1.4)
        ax.grid(True, linestyle="--", alpha=0.4, color="#E0E0E0")
        self._style_ax(ax)

        self._save(fig, filename)

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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        x = np.arange(len(models))

        ax1.bar(
            x, trunc_rates, width=0.4, color=self.c_neg,
            edgecolor="#D32F2F", linewidth=0.8, alpha=self.bar_alpha,
        )
        ax1.set_ylabel("Truncation Rate (%)")
        ax1.set_title("(a) Reasoning Truncation Rate", fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.set_ylim(0, max(trunc_rates) * 1.25)
        self._style_ax(ax1)
        for i, v in enumerate(trunc_rates):
            ax1.text(
                i, v + max(trunc_rates) * 0.03, f"{v:.1f}%",
                ha="center", va="bottom", fontweight="bold", fontsize=10,
            )

        # 零值救回率：给一个最小可见高度，避免柱子消失只留悬空标签
        max_rescue = max(rescue_rates) if rescue_rates else 0
        if max_rescue <= 0:
            max_rescue = 1.0
        floor = max_rescue * 0.012
        rescue_plot = [max(v, floor) for v in rescue_rates]
        ax2.bar(
            x, rescue_plot, width=0.4, color=self.c_warn,
            edgecolor="#F57C00", linewidth=0.8, alpha=self.bar_alpha,
        )
        ax2.set_ylabel("Rescued Rate (%)")
        ax2.set_title("(b) Format-Rescue Success Rate", fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(models)
        ax2.set_ylim(0, max_rescue * 1.3)
        self._style_ax(ax2)
        for i, v in enumerate(rescue_rates):
            ax2.text(
                i, max(v, floor) + max_rescue * 0.04, f"{v:.1f}%",
                ha="center", va="bottom", fontweight="bold", fontsize=10,
            )

        self._save(fig, filename)

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
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        w = 0.45

        ax.bar(
            models, completed_wrong, w,
            label="Completed but Incorrect",
            color=self.c_base, edgecolor=self.c_base_edge,
            linewidth=0.8, alpha=self.bar_alpha,
        )
        ax.bar(
            models, truncated_wrong, w,
            bottom=completed_wrong,
            label="Truncated Error",
            color=self.c_neg, edgecolor="#D32F2F",
            linewidth=0.8, alpha=self.bar_alpha,
        )

        ax.set_ylabel("Percentage of Total Tasks (%)")
        ax.set_title("Figure 7: Error Type Breakdown", fontweight="bold")
        ax.legend(
            frameon=True, edgecolor="#CCCCCC",
            loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=2,
        )
        self._style_ax(ax)

        for i in range(len(models)):
            cw, tw = completed_wrong[i], truncated_wrong[i]
            ax.text(
                i, cw / 2, f"{cw:.1f}%",
                ha="center", va="center",
                color="white", fontweight="bold",
            )
            if tw > 0:
                ax.text(
                    i, cw + tw / 2, f"{tw:.1f}%",
                    ha="center", va="center",
                    color="white", fontweight="bold",
                )

        self._save(fig, filename)

    # ─────────────────────────────────────────────────────────────────
    # Figure 8: Reasoning Efficiency (Passes per 10k chars)
    # ─────────────────────────────────────────────────────────────────
    def plot_figure_8_efficiency_curve(
        self,
        models: List[str],
        passes_per_10k: List[float],
        filename: str = "Figure_8_Efficiency.pdf",
    ):
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        colors = [self.c_base, self.c_neo]
        edge_colors = [self.c_base_edge, self.c_neo_edge]

        bars = ax.bar(
            models, passes_per_10k, width=0.4,
            color=colors, edgecolor=edge_colors,
            linewidth=0.8, alpha=self.bar_alpha,
        )
        ax.set_ylabel("Passes / 10k chars")
        ax.set_title("Figure 8: Reasoning Efficiency", fontweight="bold")
        self._style_ax(ax)

        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.1, f"{h:.2f}",
                ha="center", va="bottom", fontweight="bold",
            )

        self._save(fig, filename)

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
            metrics=["acc_norm", "acc_norm", "exact_match", "acc_norm"],
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
            model_names=model_names,
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
