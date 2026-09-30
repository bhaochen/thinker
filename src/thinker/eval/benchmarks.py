"""Benchmark 数据加载器 — 支持真实 HF 数据集和虚拟数据生成"""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BenchmarkQuestion:
    """统一的 benchmark 题目格式"""
    qid: str
    question: str
    options: Dict[str, str]  # {"A": "...", "B": "...", ...}
    answer: str  # "A" - "J"
    category: str  # "biology", "computer_science", "mathematics", "other_sciences", "physics"
    benchmark: str  # "mmlu_pro", "bbh", "gpqa", "math_hard", "musr"
    difficulty: str = "medium"  # "easy", "medium", "hard"


# MMLU-Pro 5 个类别 × 50 题的类别名
MMLU_CATEGORIES = [
    "biology",
    "computer_science",
    "mathematics",
    "other_sciences",
    "physics",
]

# 每个类别的题目模板（用于虚拟数据生成）
CATEGORY_TEMPLATES = {
    "biology": [
        "Which of the following best describes the mechanism of {topic}?",
        "In cellular biology, what is the primary function of {topic}?",
        "Which statement about {topic} is correct?",
        "The role of {topic} in biological systems is primarily:",
        "Which of the following is true regarding {topic}?",
    ],
    "computer_science": [
        "What is the time complexity of {topic}?",
        "Which data structure is most efficient for {topic}?",
        "In the context of {topic}, which algorithm is preferred?",
        "Which of the following best describes {topic}?",
        "The primary advantage of {topic} is:",
    ],
    "mathematics": [
        "What is the value of {topic}?",
        "Which of the following is the correct solution for {topic}?",
        "The derivative of {topic} with respect to x is:",
        "Which theorem applies to {topic}?",
        "The integral of {topic} equals:",
    ],
    "other_sciences": [
        "Which of the following best explains {topic}?",
        "In the context of {topic}, which principle applies?",
        "The relationship between {topic} and observed phenomena suggests:",
        "Which statement about {topic} is scientifically accurate?",
        "The primary factor influencing {topic} is:",
    ],
    "physics": [
        "According to the laws of physics, {topic} implies:",
        "Which equation correctly describes {topic}?",
        "The physical quantity associated with {topic} is:",
        "In the context of {topic}, which conservation law applies?",
        "Which of the following correctly describes {topic}?",
    ],
}

CATEGORY_TOPICS = {
    "biology": [
        "protein folding", "DNA replication", "mitosis", "enzyme kinetics",
        "photosynthesis", "cellular respiration", "gene expression", "immune response",
        "neural signaling", "evolutionary adaptation", "homeostasis", "metabolic pathways",
        "membrane transport", "hormonal regulation", "ecological succession",
    ],
    "computer_science": [
        "binary search", "hash tables", "dynamic programming", "graph traversal",
        "sorting algorithms", "cache coherence", "deadlock prevention", "B-trees",
        "Dijkstra's algorithm", "recursion", "greedy algorithms", "NP-completeness",
        "amortized analysis", "balanced trees", "memoization",
    ],
    "mathematics": [
        "the chain rule", "integration by parts", "Taylor series", "eigenvalues",
        "Fourier transforms", "partial derivatives", "matrix inversion",
        "differential equations", "probability distributions", "Bayes' theorem",
        "linear regression", "gradient descent", "Lagrange multipliers",
        "vector calculus", "complex analysis",
    ],
    "other_sciences": [
        "climate feedback loops", "plate tectonics", "chemical equilibrium",
        "thermodynamic entropy", "atomic structure", "electromagnetic induction",
        "wave-particle duality", "nuclear decay", "stellar evolution",
        "ocean acidification", "atmospheric pressure", "quantum tunneling",
        "relativistic effects", "photoelectric effect", "superconductivity",
    ],
    "physics": [
        "Newton's second law", "conservation of energy", "electromagnetic waves",
        "quantum entanglement", "special relativity", "thermodynamic cycles",
        "harmonic oscillation", "magnetic flux", "photon momentum",
        "gravitational lensing", "nuclear fusion", "entropy increase",
        "wave interference", "particle acceleration", "field theory",
    ],
}

# BBH 子任务
BBH_SUBTASKS = [
    "boolean_expressions", "causal_judgement", "date_understanding",
    "disambiguation_qa", "dyck_languages", "formal_fallacies",
    "geometric_shapes", "hyperbaton", "logical_deduction_five_objects",
    "logical_deduction_seven_objects", "logical_deduction_three_objects",
    "movie_recommendation", "multistep_arithmetic_two", "navigate",
    "object_counting", "penguins_in_a_table", "reasoning_about_colored_objects",
    "ruin_names", "salient_translation_error_detection", "snarks",
    "sports_understanding", "temporal_sequences", "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects", "tracking_shuffled_objects_three_objects",
    "web_of_lies", "word_sorting",
]

# GPQA 领域
GPQA_DOMAINS = ["biology", "physics", "chemistry"]

# MATH-Hard 子领域
MATH_HARD_SUBFIELDS = [
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
    "number_theory", "prealgebra", "precalculus",
]

# MUSR 子任务
MUSR_SUBTASKS = ["murder_mysteries", "object_placements", "team_allocation"]


def generate_mmlu_pro_mock(n_per_category: int = 50, seed: int = 42) -> List[BenchmarkQuestion]:
    """生成 MMLU-Pro 250 题虚拟数据（5 类 × 50 题）"""
    rng = random.Random(seed)
    questions = []
    qid_counter = 10000

    for category in MMLU_CATEGORIES:
        templates = CATEGORY_TEMPLATES[category]
        topics = CATEGORY_TOPICS[category]

        for i in range(n_per_category):
            qid_counter += 1
            template = rng.choice(templates)
            topic = rng.choice(topics)
            question_text = template.format(topic=topic)

            # 随机生成 10 个选项 (A-J)
            options = {}
            option_letters = "ABCDEFGHIJ"
            for letter in option_letters:
                options[letter] = f"Option {letter} for {topic}"

            # 随机正确答案
            answer = rng.choice(option_letters)

            questions.append(BenchmarkQuestion(
                qid=f"QID_{qid_counter}",
                question=question_text,
                options=options,
                answer=answer,
                category=category,
                benchmark="mmlu_pro",
                difficulty=rng.choice(["easy", "medium", "hard"]),
            ))

    return questions


def generate_bbh_mock(n_per_subtask: int = 40, seed: int = 42) -> List[BenchmarkQuestion]:
    """生成 BBH 虚拟数据"""
    rng = random.Random(seed)
    questions = []
    qid_counter = 20000

    for subtask in BBH_SUBTASKS:
        for i in range(n_per_subtask):
            qid_counter += 1
            options = {letter: f"Option {letter}" for letter in "ABCDE"}
            answer = rng.choice("ABCDE")
            questions.append(BenchmarkQuestion(
                qid=f"BBH_{qid_counter}",
                question=f"[{subtask}] Question {i+1}: What is the correct answer?",
                options=options,
                answer=answer,
                category=subtask,
                benchmark="bbh",
                difficulty=rng.choice(["easy", "medium", "hard"]),
            ))

    return questions


def generate_gpqa_mock(n: int = 448, seed: int = 42) -> List[BenchmarkQuestion]:
    """生成 GPQA 虚拟数据"""
    rng = random.Random(seed)
    questions = []
    qid_counter = 30000

    for i in range(n):
        qid_counter += 1
        domain = rng.choice(GPQA_DOMAINS)
        options = {letter: f"Option {letter}" for letter in "ABCD"}
        answer = rng.choice("ABCD")
        questions.append(BenchmarkQuestion(
            qid=f"GPQA_{qid_counter}",
            question=f"[{domain}] Graduate-level question {i+1}",
            options=options,
            answer=answer,
            category=domain,
            benchmark="gpqa",
            difficulty="hard",
        ))

    return questions


def generate_math_hard_mock(n: int = 500, seed: int = 42) -> List[BenchmarkQuestion]:
    """生成 MATH-Hard 虚拟数据"""
    rng = random.Random(seed)
    questions = []
    qid_counter = 40000

    for i in range(n):
        qid_counter += 1
        subfield = rng.choice(MATH_HARD_SUBFIELDS)
        options = {letter: f"Option {letter}" for letter in "ABCDE"}
        answer = rng.choice("ABCDE")
        questions.append(BenchmarkQuestion(
            qid=f"MATH_{qid_counter}",
            question=f"[{subfield}] Hard math problem {i+1}",
            options=options,
            answer=answer,
            category=subfield,
            benchmark="math_hard",
            difficulty="hard",
        ))

    return questions


def generate_musr_mock(n: int = 500, seed: int = 42) -> List[BenchmarkQuestion]:
    """生成 MUSR 虚拟数据"""
    rng = random.Random(seed)
    questions = []
    qid_counter = 50000

    for i in range(n):
        qid_counter += 1
        subtask = rng.choice(MUSR_SUBTASKS)
        options = {letter: f"Option {letter}" for letter in "ABCDE"}
        answer = rng.choice("ABCDE")
        questions.append(BenchmarkQuestion(
            qid=f"MUSR_{qid_counter}",
            question=f"[{subtask}] Multi-step reasoning question {i+1}",
            options=options,
            answer=answer,
            category=subtask,
            benchmark="musr",
            difficulty="hard",
        ))

    return questions


def generate_all_mock(seed: int = 42) -> Dict[str, List[BenchmarkQuestion]]:
    """生成所有 benchmark 的虚拟数据"""
    return {
        "mmlu_pro": generate_mmlu_pro_mock(seed=seed),
        "bbh": generate_bbh_mock(seed=seed),
        "gpqa": generate_gpqa_mock(seed=seed),
        "math_hard": generate_math_hard_mock(seed=seed),
        "musr": generate_musr_mock(seed=seed),
    }


def load_benchmark(name: str, seed: int = 42) -> List[BenchmarkQuestion]:
    """加载指定 benchmark 的数据（当前为虚拟数据，未来可替换为真实 HF 数据集）"""
    loaders = {
        "mmlu_pro": generate_mmlu_pro_mock,
        "bbh": generate_bbh_mock,
        "gpqa": generate_gpqa_mock,
        "math_hard": generate_math_hard_mock,
        "musr": generate_musr_mock,
    }
    if name not in loaders:
        raise ValueError(f"Unknown benchmark: {name}. Available: {list(loaders.keys())}")
    return loaders[name](seed=seed)
