# Supplementary experiment results

These are the results on the 1,529 scorable BIRD dev questions. Field definitions are in `02_结果表/区间字段说明.md` and `03_逐题结果/字段说明.md`. `文件清单_SHA256.txt` lists checksums; verify with `shasum -a 256 -c 文件清单_SHA256.txt`.

| Experiment | Setting | Files |
|---|---|---|
| E1 | Initial answers from Qwen2.5-7B-Instruct; repair with an API model. ReLink-SQL vs. full-schema retry | `E1_*` |
| E1b | Same as E1, vs. CRITIC | `E1b_*` |
| E2 | Initial answers and repair both from the API model. ReLink-SQL vs. full-schema retry | `E2_*` |
| E2b | Same as E2, vs. CRITIC | `E2b_*` |
| E3 | Parameter sensitivity: lexical column count k ∈ {0, 2, 4, 8} and direct-revision budget ∈ {0, 1, 2} | `E3_k*`, `E3_direct*`, `E3_*_combined.csv` |
| E4 | Calls and tokens per stage | `E4_stage_tokens.csv` |
| E5 | Stage of first acceptance × outcome, plus failure status | `E5_*` |

- `02_结果表/`: summary tables. `*_results.csv` gives accuracy and token use per method. `*_bootstrap.csv` and `cluster_bootstrap_all.csv` give paired question-level and database-clustered bootstrap intervals (10,000 resamples).
- `03_逐题结果/`: per-question outcomes per method. Join files on `task_id`.
