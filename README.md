# Biomedical Extraction Pipeline

A lightweight NLP project for biomedical named entity recognition and relation extraction.

## Project overview

This repository contains code and notebooks for building a biomedical extraction pipeline using NLP model training, dataset preparation, and evaluation.

## Contents

- `src/` - Python modules for data processing, model utilities, and evaluation.
- `notebooks/` - Jupyter notebooks for data preparation, NER training, weak supervision, re-training, and pipeline evaluation.
- `data/` - Processed datasets stored in Hugging Face Arrow format.
- `requirements/` - Dependency files for each stage of the workflow.

## Getting started

1. Create a Python environment.
2. Install dependencies from `requirements/00_core.txt` and any stage-specific requirements as needed.
3. Use the notebooks in `notebooks/` to run data preparation, training, and evaluation.

## Notes

- This is an initial project scaffold and can be extended with additional preprocessing, model training, and evaluation scripts.
- The repository is organized to keep data prep, model utilities, and experiments separated.
