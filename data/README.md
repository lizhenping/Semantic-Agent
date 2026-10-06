# Datasets

Section §IV.A of the paper uses **8 benchmarks** in total. This directory holds the **5 benchmarks** whose databases are used for synthetic training-data generation. The other 3 Spider variants—**Spider-Syn**, **Spider-Realistic**, and **Spider-DK**—are evaluation-only and derived from Spider, so they do not need separate database downloads.

> ⚠️ This directory is excluded by `.gitignore` (~35 GB) and is **not** tracked in Git (this README is the only tracked placeholder). Download it via Alibaba Cloud Drive (see below).

---

## Download

All datasets are packaged as `data.tar` and uploaded to Alibaba Cloud Drive:

- **Link**: https://www.alipan.com/s/KL9QnhGA6qZ
- **Extraction code**: `l6c4`

```bash
# 1. Open the link above, enter the extraction code l6c4, and download data.tar
# 2. Extract it to the project root as data/
tar xf data.tar -C .
mv datasets data   # the archive extracts a datasets/ directory; rename it to data/
```

After extraction, verify the directory structure:

```bash
ls datasets/
# bird  ehrsql  science_benchmark  spider  spider2
```

---

## Unified Directory Layout

All benchmarks follow the same path convention:

```
data/<benchmark>/databases/<db_name>/<db_name>.sqlite
data/<benchmark>/qa/                                     # official QA pairs (evaluation only; not read during synthesis)
```

The code locates databases automatically using this rule:

```bash
# Internal path construction
{DB_ROOT}/{benchmark}/databases/{db}/{db}.sqlite
# DB_ROOT defaults to data/ (inside the project) and can be overridden
# by SEMANTICSQL_DB_ROOT or the --db-root CLI option
```

---

## Directory Structure

```
data/
├── spider/                          # Spider 1.0 (206 databases)
│   ├── databases/
│   │   └── <db_name>/<db_name>.sqlite
│   └── qa/                          # train / dev / test QA files
│       ├── train/train_spider.json
│       ├── eval/dev.json
│       └── tables.json
│
├── bird/                            # BIRD (81 databases)
│   ├── databases/
│   │   └── <db_name>/
│   │       ├── <db_name>.sqlite
│   │       └── database_description/  # BIRD column-description CSVs
│   └── qa/
│       ├── train/train.json
│       └── eval/dev.json
│
├── spider2/                         # Spider 2.0 SQLite subset (30 databases)
│   ├── databases/
│   └── qa/
│       └── eval/spider2-lite.jsonl
│
├── ehrsql/                          # EHRSQL (2 databases: MIMIC-III + eICU)
│   ├── databases/
│   │   ├── mimic_iii/mimic_iii.sqlite
│   │   └── eicu/eicu.sqlite
│   └── qa/
│
└── science_benchmark/               # ScienceBenchmark (3 databases)
    ├── databases/
    │   ├── cordis_temporary/cordis_temporary.sqlite
    │   ├── oncomx/oncomx.sqlite
    │   └── sdss/sdss.sqlite
    └── qa/
```

---

## Statistics

| Benchmark | Databases | Domains | Avg Tables / DB | Synth. Target | Role |
|-----------|-----------|---------|-----------------|---------------|------|
| Spider 1.0 | 206 | 138 | 5.11 | 40,000 | Cross-domain parsing |
| Spider 2.0-SQLite | 30 | 8 | 13.8 | 20,000 | Enterprise-level complexity |
| BIRD | 81 | 37 | 7.64 | 30,000 | Knowledge-intensive reasoning |
| EHRSQL | 2 | 1 | 31.55 | 5,000 | Clinical domain |
| ScienceBenchmark | 3 | 3 | 16.67 | 5,000 | Scientific domain |

**Total: 322 SQLite databases, ~35 GB**

---

## Data-Leakage Prevention

During synthesis, **evaluation databases (dev / test) are automatically excluded** to prevent data leakage:

| Benchmark | Eval DBs blocked | DBs available for synthesis |
|-----------|------------------|---------------------------|
| Spider | 59 (dev 20 + test 39) | 163 |
| BIRD | 11 (dev) | 69 |
| Spider2 | none (SQLite subset is intended for synthesis) | 30 |
| EHRSQL | database-level isolation is not applied (train/eval share DBs; leakage is prevented by never reading official QA files) | 2 |
| ScienceBenchmark | same as above | 3 |

To see which databases are allowed or blocked for a benchmark:

```bash
python cli.py list-dbs -b spider
```

Isolation is enforced by `app/adapters/dataset_split.py`. Any command—`run`, `run-benchmark`, `analyze`, or `generate`—will refuse an evaluation database with a clear error.

---

## Official Sources (for reference)

If the Alibaba Cloud Drive link is unavailable, you can download each benchmark from its official source:

| Benchmark | Official Download |
|-----------|-------------------|
| Spider 1.0 | https://github.com/taoyds/spider |
| BIRD | https://bird-bench.github.io/ |
| Spider 2.0 | https://github.com/xlang-ai/Spider2 |
| EHRSQL | https://github.com/glee4810/EHRSQL |
| ScienceBenchmark | https://github.com/yizhang-unifr/nl-ql-data-augmentation |
