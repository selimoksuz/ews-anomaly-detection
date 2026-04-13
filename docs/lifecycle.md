# Lifecycle Architecture

This project now supports a config-driven lifecycle on top of the original train/score pipeline.

## Core Concepts

- `sources`: define where development and live scoring data come from
- `development.windows`: define train/dev/calibration/OOT slices from one append-only input table
- `registry`: stores run manifests, model registry, and champion pointers
- `retraining`: controls challenger creation and comparison
- `retention`: cleans old logs, run manifests, and artifacts

## Runtime Folders

- `artifacts/<segment>/<run_id>/`
  Stores model artifacts and stability reports
- `meta/runs/<run_id>/manifest.json`
  Stores one run manifest per lifecycle execution
- `meta/model_registry.json`
  Stores candidate/champion model entries
- `meta/champions.json`
  Stores the active champion per segment

## CLI Commands

- `python cli.py prepare-demo-data`
- `python cli.py develop [segment]`
- `python cli.py retrain [segment]`
- `python cli.py compare [segment] [challenger_version]`
- `python cli.py promote [segment] [model_version]`
- `python cli.py score-live [segment]`
- `python cli.py cleanup`

## Expected Workflow

1. Append new feature rows into the input source table.
2. Run `develop` or `retrain` to produce a candidate model.
3. Run `compare` to compare champion and challenger stability.
4. Run `promote` if the challenger should become live.
5. Run `score-live` on the latest snapshot using the champion model.
6. Run `cleanup` on a schedule for retention.

## Notes

- Calibration is intentionally left as a separate next phase.
- Stability is currently tracked through train vs dev/OOT distribution summaries.
- The same architecture can use CSV or Oracle by changing `config/pipeline_config.yaml`.
