"""Utilities to generate a lifecycle simulation notebook."""

from __future__ import annotations

import json
from pathlib import Path


def _markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def _code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.splitlines(keepends=True),
    }


def build_simulation_notebook(output_path: str | Path = "notebooks/ews_simulation.ipynb") -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            _markdown_cell(
                "# EWS Lifecycle Simulation\n\n"
                "Bu notebook yeni Oracle-first anomaly lifecycle kurgusunu adım adım simüle etmek için hazırlandı.\n\n"
                "Aynı notebook içinde:\n"
                "- aktif config ve Oracle tablolarını görebilirsin\n"
                "- local runtime state'i temizleyebilirsin\n"
                "- demo data yükleyebilirsin\n"
                "- batch pipeline'ı tek komutta koşturabilirsin\n"
                "- Oracle output tablolarını ve local registry dosyalarını okuyabilirsin\n"
            ),
            _markdown_cell(
                "## Kullanım Modu\n\n"
                "Aşağıdaki flag hücrelerini `True` yaparak çalıştır.\n\n"
                "- `RESET_RUNTIME`: local `logs/`, `meta/`, `artifacts/` temizlenir\n"
                "- `LOAD_DEMO_DATA`: sentetik Oracle input/outcome tabloları yeniden yüklenir\n"
                "- `RUN_BATCH`: config-driven batch orchestration çalışır\n"
                "- `RUN_MANUAL_STEPS`: `develop -> tune -> evaluate -> promote -> score-live` tek tek çalışır\n"
            ),
            _code_cell(
                "from pathlib import Path\n"
                "import json\n"
                "import subprocess\n"
                "import sys\n"
                "\n"
                "import pandas as pd\n"
                "from IPython.display import display\n"
                "\n"
                "from engine.config_loader import load_config, load_secrets\n"
                "from engine.lifecycle import LifecycleManager\n"
                "from engine.oracle_io import OracleConnector\n"
                "\n"
                "config = load_config()\n"
                "secrets = load_secrets()\n"
                "manager = LifecycleManager()\n"
            ),
            _code_cell(
                "SEGMENT = 'ALL'\n"
                "RESET_RUNTIME = False\n"
                "LOAD_DEMO_DATA = False\n"
                "RUN_BATCH = False\n"
                "RUN_MANUAL_STEPS = False\n"
            ),
            _code_cell(
                "tables = config['oracle']['tables']\n"
                "runtime_layout = pd.DataFrame([\n"
                "    {'group': 'oracle_input', 'name': 'input_features', 'value': tables['input_features']},\n"
                "    {'group': 'oracle_input', 'name': 'outcomes', 'value': tables['outcomes']},\n"
                "    {'group': 'oracle_output', 'name': 'results', 'value': tables['results']},\n"
                "    {'group': 'oracle_output', 'name': 'details', 'value': tables['details']},\n"
                "    {'group': 'oracle_output', 'name': 'full_effects', 'value': tables['full_effects']},\n"
                "    {'group': 'local_runtime', 'name': 'meta_dir', 'value': config['registry']['meta_dir']},\n"
                "    {'group': 'local_runtime', 'name': 'artifacts_dir', 'value': config['registry']['artifacts_dir']},\n"
                "    {'group': 'local_runtime', 'name': 'logs_dir', 'value': config['registry']['logs_dir']},\n"
                "])\n"
                "display(runtime_layout)\n"
            ),
            _code_cell(
                "if RESET_RUNTIME:\n"
                "    reset_summary = manager.reset_runtime()\n"
                "    display(pd.DataFrame([reset_summary]))\n"
                "else:\n"
                "    print('RESET_RUNTIME = False, atlandi.')\n"
            ),
            _code_cell(
                "if LOAD_DEMO_DATA:\n"
                "    subprocess.run([sys.executable, 'cli.py', 'prepare-demo-data'], check=True)\n"
                "    print('Demo data Oracle tablolara yuklendi.')\n"
                "else:\n"
                "    print('LOAD_DEMO_DATA = False, atlandi.')\n"
            ),
            _code_cell(
                "batch_summary = None\n"
                "if RUN_BATCH:\n"
                "    batch_summary = manager.run_batch(segment=SEGMENT)\n"
                "    display(pd.json_normalize(batch_summary, sep=' -> '))\n"
                "else:\n"
                "    print('RUN_BATCH = False, atlandi.')\n"
            ),
            _code_cell(
                "manual_summary = {}\n"
                "if RUN_MANUAL_STEPS:\n"
                "    develop = manager.develop(segment=SEGMENT)\n"
                "    model_version = develop['model_version']\n"
                "    manual_summary['develop'] = develop\n"
                "    manual_summary['tune_weights'] = manager.tune_weights(segment=SEGMENT, model_version=model_version, apply=True)\n"
                "    manual_summary['evaluate_outcomes'] = manager.evaluate_outcomes(segment=SEGMENT, model_version=model_version)\n"
                "    manual_summary['promote'] = manager.promote(segment=SEGMENT, model_version=model_version)\n"
                "    manual_summary['score_live'] = manager.score_live(segment=SEGMENT)\n"
                "    display(pd.json_normalize(manual_summary, sep=' -> '))\n"
                "else:\n"
                "    print('RUN_MANUAL_STEPS = False, atlandi.')\n"
            ),
            _code_cell(
                "with OracleConnector(config, secrets) as ora:\n"
                "    row_counts = []\n"
                "    for key in ('input_features', 'outcomes', 'results', 'details', 'full_effects'):\n"
                "        table_name = ora._qualified_table_name(key)\n"
                "        count_df = ora._read_query(f'SELECT COUNT(*) AS row_count FROM {table_name}')\n"
                "        row_counts.append({'table_key': key, 'table_name': table_name, 'row_count': int(count_df.iloc[0, 0])})\n"
                "display(pd.DataFrame(row_counts))\n"
            ),
            _code_cell(
                "with OracleConnector(config, secrets) as ora:\n"
                "    results_table = ora._qualified_table_name('results')\n"
                "    details_table = ora._qualified_table_name('details')\n"
                "    effects_table = ora._qualified_table_name('full_effects')\n"
                "    results_sample = ora._read_query(f'SELECT * FROM {results_table} FETCH FIRST 10 ROWS ONLY')\n"
                "    details_sample = ora._read_query(f'SELECT * FROM {details_table} FETCH FIRST 10 ROWS ONLY')\n"
                "    effects_sample = ora._read_query(f'SELECT * FROM {effects_table} FETCH FIRST 10 ROWS ONLY')\n"
                "\n"
                "display(results_sample)\n"
                "display(details_sample)\n"
                "display(effects_sample)\n"
            ),
            _code_cell(
                "meta_files = {\n"
                "    'run_registry': Path(config['registry']['run_registry_file']),\n"
                "    'model_registry': Path(config['registry']['model_registry_file']),\n"
                "    'champions': Path(config['registry']['champion_registry_file']),\n"
                "}\n"
                "for name, path in meta_files.items():\n"
                "    print(f'--- {name}: {path} ---')\n"
                "    if path.exists():\n"
                "        payload = json.loads(path.read_text(encoding='utf-8'))\n"
                "        if isinstance(payload, list):\n"
                "            display(pd.json_normalize(payload, sep=' -> ').tail(10))\n"
                "        else:\n"
                "            display(pd.DataFrame([payload]))\n"
                "    else:\n"
                "        print('Dosya bulunamadi.')\n"
            ),
            _markdown_cell(
                "## Beklenen Oracle Akışı\n\n"
                "1. Feature generator `EWS_INPUT_FEATURES` tablosuna append eder.\n"
                "2. Outcome feed `EWS_OUTCOME_LABELS` tablosunu doldurur.\n"
                "3. Development / retraining aynı input tablonun zaman pencerelerini kullanır.\n"
                "4. Live scoring latest snapshot'ı okuyup Oracle output tablolarına yazar.\n"
                "5. Local `meta/` ve `artifacts/` altında run ve model metadata tutulur.\n"
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    return output
