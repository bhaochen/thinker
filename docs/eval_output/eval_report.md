# Qwen3.5-4B vs Qwen3.5-4B-Neo Evaluation Report

## 1. Overall Results

| Metric | Base | Neo | Delta |
|--------|------|-----|-------|
| pass@1 | 78.8% | 80.0% | +1.20 pp |
| Truncation rate | 9.2% | 14.4% | — |
| Rescues | 2 | 0 | — |

## 2. Chain-of-Thought Statistics

| Model | Mean chars | Mean words | Median chars | Max chars |
|-------|-----------|------------|--------------|-----------|
| Qwen3.5-4B | 7,122 | 1,424 | 7,087 | 12,470 |
| Qwen3.5-4B-Neo | 3,946 | 788 | 3,958 | 6,834 |

## 3. Reasoning Efficiency

| Model | Passes / 10k chars | Chars / correct pass | Total think chars |
|-------|-------------------|---------------------|-------------------|
| Qwen3.5-4B | 1.22 | 8,207 | 1,616,788 |
| Qwen3.5-4B-Neo | 2.37 | 4,223 | 844,631 |

## 4. Error Breakdown

| Model | Correct | Completed but wrong | Truncated |
|-------|---------|---------------------|-----------|
| Qwen3.5-4B | 197 | 32 | 23 |
| Qwen3.5-4B-Neo | 200 | 14 | 36 |

## 5. Accuracy-Cost Tradeoff

- **Accuracy gain**: +1.20 pp
- **Cost reduction**: -44.6% fewer think chars

## 6. Per-Category Breakdown (MMLU-Pro)

| Category | Base | Neo | Delta |
|----------|------|-----|-------|
| biology | 82.0% | 72.0% | -10.0 pp |
| computer_science | 80.0% | 80.0% | +0.0 pp |
| mathematics | 78.0% | 82.0% | +4.0 pp |
| other_sciences | 80.0% | 88.0% | +8.0 pp |
| physics | 74.0% | 78.0% | +4.0 pp |

## 7. Generated Artifacts

- `Figure_1_Leaderboard_Composite.pdf` — Per-sub-benchmark + overall
- `Figure_2_Per_Category.pdf` — MMLU-Pro per-category accuracy
- `Figure_3_Sub_Deltas.pdf` — Per-sub-benchmark delta
- `Figure_4_Length_Dist.pdf` — Think-chain length distribution
- `Figure_5_Acc_Cost.pdf` — Accuracy vs reasoning cost
- `Figure_6_Truncation_Rescue_Composite.pdf` — Truncation & rescue rates
- `Figure_7_Error_Breakdown.pdf` — Error type breakdown
- `Figure_8_Efficiency.pdf` — Reasoning efficiency
- `raw_results.json` — Raw inference results