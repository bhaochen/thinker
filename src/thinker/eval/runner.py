"""模拟推理运行器 — 生成带截断/救回/正确/错误的评测结果"""

import random
import re
from typing import Any, Dict, List, Optional, Tuple

from thinker.eval.benchmarks import BenchmarkQuestion


class MockModelRunner:
    """模拟模型推理，生成逼真的评测结果"""

    def __init__(
        self,
        model_name: str,
        pass_at_1: float = 0.80,
        truncation_rate: float = 0.088,
        rescue_rate: float = 0.012,
        mean_think_chars: int = 6962,
        think_std: int = 2000,
        seed: int = 42,
    ):
        """
        Args:
            model_name: 模型名称
            pass_at_1: 目标准确率 (0-1)
            truncation_rate: 截断率 (0-1)
            rescue_rate: 救回率 (0-1)
            mean_think_chars: 平均思考链字符数
            think_std: 思考链标准差
            seed: 随机种子
        """
        self.model_name = model_name
        self.pass_at_1 = pass_at_1
        self.truncation_rate = truncation_rate
        self.rescue_rate = rescue_rate
        self.mean_think_chars = mean_think_chars
        self.think_std = think_std
        self.rng = random.Random(seed)

        # 计算非截断情况下的准确率
        # pass_at_1 = (1 - truncation_rate) * non_trunc_acc + rescue_rate
        # => non_trunc_acc = (pass_at_1 - rescue_rate) / (1 - truncation_rate)
        non_trunc_rate = 1.0 - truncation_rate
        if non_trunc_rate > 0:
            self.non_trunc_accuracy = (pass_at_1 - rescue_rate) / non_trunc_rate
            self.non_trunc_accuracy = max(0.0, min(1.0, self.non_trunc_accuracy))
        else:
            self.non_trunc_accuracy = 0.0

    def run_inference(self, question: BenchmarkQuestion) -> Dict[str, Any]:
        """对单道题进行模拟推理"""
        # 1. 判定是否截断
        is_truncated = self.rng.random() < self.truncation_rate

        # 2. 生成思考链长度
        if is_truncated:
            # 截断的思考链通常更长（模型陷入循环）
            think_len = int(self.rng.gauss(self.mean_think_chars * 1.3, self.think_std * 0.8))
        else:
            think_len = int(self.rng.gauss(self.mean_think_chars, self.think_std))
        think_len = max(100, think_len)

        # 3. 判定正确性
        if is_truncated:
            # 截断的题目：部分可以救回
            rescue_prob = self.rescue_rate / self.truncation_rate if self.truncation_rate > 0 else 0
            is_rescued = self.rng.random() < rescue_prob
            is_correct = is_rescued
        else:
            # 非截断：按非截断准确率判定
            is_correct = self.rng.random() < self.non_trunc_accuracy
            is_rescued = False

        # 4. 生成预测答案
        if is_correct:
            prediction = question.answer
        else:
            # 错误答案：随机选一个不同的
            wrong_options = [k for k in question.options.keys() if k != question.answer]
            prediction = self.rng.choice(wrong_options) if wrong_options else question.answer

        # 5. 生成模拟输出文本
        raw_output = self._generate_mock_output(
            question, think_len, is_truncated, prediction, is_rescued
        )

        return {
            "qid": question.qid,
            "benchmark": question.benchmark,
            "category": question.category,
            "golden_answer": question.answer,
            "prediction": prediction,
            "is_correct": is_correct,
            "is_truncated": is_truncated,
            "is_rescued": is_rescued,
            "think_char_len": think_len,
            "think_word_len": think_len // 5,  # 英文平均 5 字符/词
            "raw_output": raw_output,
        }

    def _generate_mock_output(
        self,
        question: BenchmarkQuestion,
        think_len: int,
        is_truncated: bool,
        prediction: str,
        is_rescued: bool,
    ) -> str:
        """生成模拟的模型输出文本"""
        # 生成思考链内容
        base_text = "Let me think about this step by step. "
        think_content = base_text * (think_len // len(base_text) + 1)
        think_content = think_content[:think_len]

        if is_truncated:
            # 截断：没有 </think> 闭合标签
            # 如果是救回的，在思考链末尾包含答案信息
            if is_rescued:
                # 添加答案线索（模拟模型在循环中反复选择同一答案）
                answer_hint = f" The answer is {prediction}."
                think_content = think_content[:think_len - len(answer_hint)] + answer_hint
            return f"<think>{think_content}"
        else:
            # 正常闭合
            return f"<think>{think_content}</think>\nThe answer is {prediction}."

    def run_batch(
        self, questions: List[BenchmarkQuestion]
    ) -> List[Dict[str, Any]]:
        """批量推理"""
        results = []
        for q in questions:
            result = self.run_inference(q)
            results.append(result)
        return results


class BaseVsNeoRunner:
    """Base vs Neo 对比运行器 — 使用论文真实数据参数"""

    # 论文中的真实数据参数
    # Base: pass@1=80.4%, truncation=8.8%, rescues=3/250=1.2%
    # Neo: pass@1=82.0%, truncation=13.2%, rescues=0/250=0%
    BASE_CONFIG = {
        "model_name": "Qwen3.5-4B",
        "pass_at_1": 0.804,
        "truncation_rate": 0.088,
        "rescue_rate": 0.012,
        "mean_think_chars": 6962,
        "think_std": 2000,
    }

    NEO_CONFIG = {
        "model_name": "Qwen3.5-4B-Neo",
        "pass_at_1": 0.820,
        "truncation_rate": 0.132,
        "rescue_rate": 0.0,
        "mean_think_chars": 3955,
        "think_std": 1200,
    }

    def __init__(self, seed: int = 42):
        self.base_runner = MockModelRunner(seed=seed, **self.BASE_CONFIG)
        self.neo_runner = MockModelRunner(seed=seed + 1, **self.NEO_CONFIG)

    def run_comparison(
        self, questions: List[BenchmarkQuestion]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """运行 base vs neo 对比"""
        base_results = self.base_runner.run_batch(questions)
        neo_results = self.neo_runner.run_batch(questions)
        return base_results, neo_results
