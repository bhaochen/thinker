import re
from typing import Any, Dict, Optional


class ReasoningExtractor:
    """解析推理大模型输出的文本结构，支持截断检测与 Format-rescue 救回逻辑"""

    @staticmethod
    def parse_response(raw_output: str, golden_answer: str) -> Dict[str, Any]:
        """解析模型输出，提取 <think> 链、检测截断并判定最终答案"""
        # 1. 截断检测：缺少闭合的 </think> 标签[cite: 9]
        is_truncated = "</think>" not in raw_output

        # 2. 提取 think 内容与最终回答[cite: 9]
        think_content = ""
        prediction_text = ""

        if "<think>" in raw_output:
            if not is_truncated:
                think_match = re.search(r"<think>(.*?)</think>", raw_output, re.DOTALL)
                if think_match:
                    think_content = think_match.group(1).strip()
                prediction_text = raw_output.split("</think>")[-1].strip()
            else:
                # 截断情况：提取 <think> 之后未闭合的所有文本
                think_content = raw_output.split("<think>")[-1].strip()
                prediction_text = ""
        else:
            prediction_text = raw_output.strip()

        # 3. 从最终回答中提取选项字母
        extracted_pred = ReasoningExtractor._extract_option(prediction_text)

        # 4. 救回逻辑 (Format-rescue adjudication)[cite: 9]
        is_rescued = False
        if not extracted_pred and think_content:
            # 检索思考链最后 200 个字符进行救回匹配[cite: 9]
            rescued_option = ReasoningExtractor._extract_option(think_content[-200:])
            if rescued_option == golden_answer.upper():
                extracted_pred = rescued_option
                is_rescued = True

        is_correct = (
            (extracted_pred == golden_answer.upper()) if extracted_pred else False
        )

        return {
            "think_content": think_content,
            "think_char_len": len(think_content),
            "think_word_len": len(think_content.split()),
            "is_truncated": is_truncated,
            "is_rescued": is_rescued,
            "prediction": extracted_pred,
            "is_correct": is_correct,
        }

    @staticmethod
    def _extract_option(text: str) -> Optional[str]:
        """从文本中正则提取 A-J 的选项字母"""
        patterns = [
            r"the correct answer is\s*([A-J])",
            r"answer is\s*([A-J])",
            r"([A-J])\)",
            r"\b([A-J])\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        return None
