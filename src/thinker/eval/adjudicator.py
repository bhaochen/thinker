"""判定工作流 — 截断检测、Format-rescue 救回、生成 adjudication 报告"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class AdjudicationRecord:
    """单条判定记录"""
    qid: str
    benchmark: str
    category: str
    golden_answer: str
    original_prediction: Optional[str]
    rescued_prediction: Optional[str]
    is_truncated: bool
    is_rescued: bool
    rescue_reason: str = ""
    think_char_len: int = 0


class FormatRescueAdjudicator:
    """Format-rescue 判定器 — 对截断输出进行救回判定"""

    def __init__(self):
        self.records: List[AdjudicationRecord] = []

    def adjudicate(
        self,
        results: List[Dict[str, Any]],
    ) -> List[AdjudicationRecord]:
        """对所有结果进行判定"""
        self.records = []
        for r in results:
            record = self._adjudicate_single(r)
            self.records.append(record)
        return self.records

    def _adjudicate_single(self, result: Dict[str, Any]) -> AdjudicationRecord:
        """对单条结果进行判定"""
        qid = result["qid"]
        benchmark = result["benchmark"]
        category = result["category"]
        golden = result["golden_answer"]
        is_truncated = result["is_truncated"]
        prediction = result["prediction"]
        think_char_len = result.get("think_char_len", 0)

        # 如果已经正确，不需要救回
        if result["is_correct"] and not is_truncated:
            return AdjudicationRecord(
                qid=qid,
                benchmark=benchmark,
                category=category,
                golden_answer=golden,
                original_prediction=prediction,
                rescued_prediction=None,
                is_truncated=False,
                is_rescued=False,
                think_char_len=think_char_len,
            )

        # 截断情况下的救回判定
        if is_truncated:
            raw_output = result.get("raw_output", "")
            rescued_pred = self._try_rescue(raw_output, golden)

            if rescued_pred is not None:
                return AdjudicationRecord(
                    qid=qid,
                    benchmark=benchmark,
                    category=category,
                    golden_answer=golden,
                    original_prediction=None,
                    rescued_prediction=rescued_pred,
                    is_truncated=True,
                    is_rescued=True,
                    rescue_reason="Think chain contains explicit answer matching gold",
                    think_char_len=think_char_len,
                )
            else:
                return AdjudicationRecord(
                    qid=qid,
                    benchmark=benchmark,
                    category=category,
                    golden_answer=golden,
                    original_prediction=None,
                    rescued_prediction=None,
                    is_truncated=True,
                    is_rescued=False,
                    think_char_len=think_char_len,
                )

        # 非截断但错误
        return AdjudicationRecord(
            qid=qid,
            benchmark=benchmark,
            category=category,
            golden_answer=golden,
            original_prediction=prediction,
            rescued_prediction=None,
            is_truncated=False,
            is_rescued=False,
            think_char_len=think_char_len,
        )

    def _try_rescue(self, raw_output: str, golden: str) -> Optional[str]:
        """尝试从截断输出中救回答案"""
        # 从思考链最后 200 字符中搜索答案
        tail = raw_output[-200:] if len(raw_output) > 200 else raw_output

        patterns = [
            r"the correct answer is\s*([A-J])",
            r"answer is\s*([A-J])",
            r"I will go with\s*([A-J])",
            r"choose\s*([A-J])",
            r"select\s*([A-J])",
            r"\b([A-J])\)",
        ]

        for pattern in patterns:
            match = re.search(pattern, tail, re.IGNORECASE)
            if match:
                candidate = match.group(1).upper()
                if candidate == golden.upper():
                    return candidate

        return None

    def generate_report(self) -> str:
        """生成 Markdown 格式的判定报告"""
        total = len(self.records)
        truncated = sum(1 for r in self.records if r.is_truncated)
        rescued = sum(1 for r in self.records if r.is_rescued)
        completed_wrong = sum(
            1 for r in self.records
            if not r.is_rescued and not r.is_truncated
        )

        lines = [
            "# Adjudication Report",
            "",
            f"- **Total tasks**: {total}",
            f"- **Truncated**: {truncated} ({truncated/total*100:.1f}%)" if total > 0 else "- **Truncated**: 0",
            f"- **Rescued**: {rescued}",
            f"- **Rescue rate**: {rescued/truncated*100:.1f}%" if truncated > 0 else "- **Rescue rate**: N/A",
            "",
            "## Rescued Cases",
            "",
        ]

        for r in self.records:
            if r.is_rescued:
                lines.append(
                    f"- **{r.qid}** ({r.benchmark}/{r.category}): "
                    f"gold={r.golden_answer}, rescued={r.rescued_prediction}"
                )

        lines.extend(["", "## Truncated but Not Rescued", ""])

        for r in self.records:
            if r.is_truncated and not r.is_rescued:
                lines.append(
                    f"- **{r.qid}** ({r.benchmark}/{r.category}): "
                    f"gold={r.golden_answer}, think_len={r.think_char_len}"
                )

        return "\n".join(lines)

    def get_summary(self) -> Dict[str, Any]:
        """获取判定摘要"""
        total = len(self.records)
        if total == 0:
            return {}

        truncated = sum(1 for r in self.records if r.is_truncated)
        rescued = sum(1 for r in self.records if r.is_rescued)

        return {
            "total": total,
            "truncated": truncated,
            "truncation_rate": round(truncated / total * 100, 2),
            "rescued": rescued,
            "rescue_rate": round(rescued / truncated * 100, 2) if truncated > 0 else 0.0,
        }
