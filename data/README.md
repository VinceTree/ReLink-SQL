# Data

Datasets are not redistributed here.

- **Spider 1.0**: https://yale-lily.github.io/spider. You need the dev JSON and `database/`.
- **BIRD dev**: https://bird-bench.github.io. You need `dev.json` and `dev_databases/`. These 11 databases are used: california_schools, card_games, codebase_community, debit_card_specializing, european_football_2, financial, formula_1, student_club, superhero, thrombosis_prediction and toxicology. Their expected SHA-256 values are listed in `code/relink_sql_cuda_20261110/bird_cuda.py`.

Suggested layout: `data/bird_dev_databases/<db>/<db>.sqlite`.

The full run logs are not included because of their size (about 2.6 GB).
