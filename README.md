# ARGOS

## Privacy-Preserving and Permissive Access Realignment for LLM-Generated Database Queries

This repository is the working research codebase for ARGOS, an ontology-backed access realignment pipeline for LLM-generated SQL. It combines three things in one place:

- core query parsing, AST transformation, ontology instantiation, and pruning logic
- benchmark construction and evaluation utilities for privacy- and permission-aware text-to-SQL tasks
- experiment runners that integrate ARGOS with external text-to-SQL pipelines such as Agentar-Scale-SQL and DIN-SQL style baselines

The repository is organized as a research artifact rather than a polished package. The main runnable entry points are the scripts under `p3t2q_benchmark_building/` and `experiment/`; `main.py` is currently only a placeholder.

## What Is In This Repo

```text
ARGOS/
├── src/                           # Core ARGOS operators
│   ├── argos_abox_operator.py     # High-level schema/policy ABOX builder + query evaluator
│   ├── prune.py                   # AST pruning after ontology reasoning
│   ├── operators/
│   │   ├── astObject.py           # sqlglot wrapper and AST utilities
│   │   ├── astTree.py             # Custom AST tree projection
│   │   └── ontologyInstance.py    # OWLReady-backed ontology runtime
│   └── data/ASTTree.py            # Tree node data structure
├── data/
│   ├── P3T2Q_benchmark/           # Benchmark versions (v0, v1, v2)
│   │   └── v2/<db_id>/            # schema.json, access_control.json, qa.json, bird_qa.json, sqlite DB
│   ├── ontology_file/             # ARGOS ontologies and generated ontology artifacts
│   └── financial_benchmark/       # Earlier CSV-based benchmark assets
├── p3t2q_benchmark_building/      # Dataset generation and evaluation scripts
├── experiment/
│   ├── agentar_scale_sql/         # Vendored Agentar-Scale-SQL code + ARGOS experiment runners
│   ├── din_sql/                   # Alternative text-to-SQL baseline wrappers
│   ├── outputs/                   # Raw run artifacts
│   └── results/                   # Aggregated metrics and plots
├── notebooks/                     # Exploratory analysis and paper-support notebooks
├── build_full_qa_dataset.sh       # End-to-end dataset build pipeline for benchmark v2
├── requirements.txt               # Core Python dependencies
├── pyproject.toml                 # Project metadata
└── uv.lock                        # Locked dependency snapshot
```

## Core Pipeline

At a high level, the ARGOS flow is:

1. Parse generated SQL with `sqlglot`.
2. Convert the parsed tree into a simplified internal AST representation.
3. Materialize schema, policy, and query references as ontology individuals.
4. Run ontology reasoning to determine which tables/columns are aligned, denied, or row-constrained.
5. Prune or rewrite the SQL into a privacy-preserving and permission-preserving query.

The most direct high-level interface in the current codebase is `src/argos_abox_operator.py`, which builds schema/policy ABOXes from benchmark assets and evaluates queries role-by-role.

## Key Directories

### `src/`

The `src/` tree contains the actual ARGOS logic:

- `operators/astObject.py`: wraps `sqlglot`, parses SQL, assigns node IDs, and supports AST editing.
- `operators/astTree.py`: projects the raw SQL AST into a custom tree that is easier to reason over.
- `operators/ontologyInstance.py`: loads the ontology with Owlready2 and instantiates query references.
- `prune.py`: removes or rewrites AST nodes after ontology reasoning.
- `argos_abox_operator.py`: orchestrates schema ABOX creation, policy ABOX creation, reasoning, and query refinement.

### `data/`

The benchmark assets are already checked in.

- `data/P3T2Q_benchmark/v2/` is the main current benchmark layout used by the newer experiment runners.
- Each database folder (for example `financial`, `formula_1`, `superhero`) includes:
  - `schema.json`
  - `access_control.json`
  - `qa.json`
  - `bird_qa.json`
  - `<db_id>.sqlite`
- `data/ontology_file/` contains versioned ARGOS ontology files (`argos_v3.x.rdf`) plus generated ontology outputs.

### `p3t2q_benchmark_building/`

This folder contains the dataset and evaluation toolchain:

- build access-control policies from benchmark schemas
- generate QA configs for view-only, filter-only, and combined tracks
- create QA/PQ pairs
- naturalize question text via a model endpoint
- evaluate model outputs against the benchmark with per-sample and aggregate metrics

### `experiment/`

This folder contains experimental integrations and saved outputs:

- `experiment/agentar_scale_sql/` includes a vendored Agentar-Scale-SQL tree plus wrappers such as:
  - `run_access_control_experiment.py`
  - `run_access_control_experiment_llamacpp_batch.py`
  - `run_access_control_experiment_v2.py`
- `experiment/din_sql/` contains comparable baseline layers (`base`, DBMS enforcement, prompt filtering, view filtering, ARGOS).
- `experiment/outputs/` stores raw JSON/log artifacts from runs.
- `experiment/results/` stores summarized benchmark results and paper plots.

## Setup

### Python

The checked-in `pyproject.toml` targets Python `>=3.13`.

### Install Core Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or, if you use `uv`:

```bash
uv sync
```

### Optional / Experiment-Specific Dependencies

The root dependency files cover the core ARGOS stack (`owlready2`, `pandas`, `sqlglot`, `langchain`, `litellm`). Some experiment paths rely on additional libraries:

- `experiment/agentar_scale_sql/requirements.txt` for the full Agentar-Scale-SQL integration
- `rdflib` for RDF graph serialization paths used by `src/argos_abox_operator.py`
- `apted` for optional tree-edit-distance style evaluation in `p3t2q_benchmark_building/evaluate_qa_pipeline.py`

If you plan to use Pellet-based reasoning through Owlready2, make sure a Java runtime is available as well.

## Common Workflows

### 1. Build or Refresh the Benchmark Dataset

The root helper script builds the full v2 benchmark pipeline (QA configs, QA/PQ pairs, naturalized questions, and consolidated per-database `qa.json` / `bird_qa.json` files):

```bash
./build_full_qa_dataset.sh \
  --model "qwen/qwen3-4b-2507" \
  --base-dir data/P3T2Q_benchmark/v2
```

Useful options:

- `--db <db_id>` to limit the run to one or more databases
- `--no-bootstrap` to skip copying base assets from `v1`
- `--skip-access-control` to reuse existing `access_control.json`
- `--skip-consolidation` to skip rewriting `qa.json` / `bird_qa.json`

### 2. Run Agentar-Scale-SQL With ARGOS

The main experiment runner evaluates multiple access-control modes and writes raw and aggregate outputs to `experiment/outputs/`.

Example:

```bash
python experiment/agentar_scale_sql/run_access_control_experiment.py \
  --db financial \
  --start-index 0 \
  --end-index 10 \
  --output-prefix agenta_financial_full_v1
```

To run only the ARGOS enforcement layer:

```bash
python experiment/agentar_scale_sql/run_access_control_experiment.py \
  --access-control-mode argos_access_control \
  --db financial \
  --start-index 450 \
  --end-index 460 \
  --output-prefix agenta_financial_full_v6 \
  --save-argos-failures \
  --save-argos-db-abox
```

There are two additional variants:

- `run_access_control_experiment_llamacpp_batch.py` for llama.cpp batch inference
- `run_access_control_experiment_v2.py` for expanded mode combinations (baseline, prompt-filtered, view-filtered, DBMS, ARGOS)

Most of these scripts expect an OpenAI-compatible API endpoint (LM Studio is the local workflow implied by the current defaults).

### 3. Evaluate Predictions Against the Benchmark

Use the modular evaluator to score generated SQL against the benchmark:

```bash
python p3t2q_benchmark_building/evaluate_qa_pipeline.py \
  --dataset data/P3T2Q_benchmark/v2/financial/qa.json \
  --predictions path/to/predictions.json \
  --db-root data/P3T2Q_benchmark/v2 \
  --output-summary experiment/outputs/financial_eval_summary.json
```

The evaluator parses candidate SQL with the same AST utilities in `src/` and reports per-sample plus aggregate privacy/permissiveness metrics.

## Notes For Reproducibility

- `main.py` is not the primary entry point for this project.
- The repository includes many checked-in outputs under `experiment/outputs/` and `experiment/results/`; treat them as saved artifacts, not source code.
- The `notebooks/` directory contains exploratory work and figure-generation support for the paper.
- The current repository state mixes reusable framework code, benchmark assets, and in-progress experiment logs. That is normal for this artifact.

## Citation

If you are using this repository as the implementation artifact for the paper, add the final publication metadata here once the paper record is finalized.

Proposed title:

```text
Privacy-Preserving and Permissive Access Realignment for LLM-Generated Database Queries
```

## License

This repository contains research code and a vendored third-party subtree under `experiment/agentar_scale_sql/` that includes its own `LICENSE` and `LEGAL.md`. Review those files before redistributing or packaging the project.
