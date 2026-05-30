# 100k

Code repository for the 100k project.

## Overview

This repository is a development sandbox for digital pathology and whole slide image (WSI) workflows. It focuses on data cleaning, pathology metadata review, tissue and artifact segmentation, tile generation, feature extraction, and ABMIL training and inference.

Most of the work lives in notebooks, with scripts supporting the notebook-based workflows and batch processing steps.

## Repository layout

- `notebooks/` - exploratory and analysis notebooks for data cleaning, WSI mapping, metadata review, segmentation, feature extraction, and ABMIL experiments.
- `scripts/` - reusable Python modules that support notebook workflows and batch processing.

## What to find here

- Data cleaning and mapping notebooks for slide metadata, labels, and folder organization.
- Digital pathology workflow notebooks for tissue detection, artifact segmentation, tile generation, and slide QC.
- Feature extraction and embedding notebooks/scripts for turning WSI tiles into model embeddings.
- ABMIL training and inference notebooks/scripts for slide-level classification from tile embeddings.
- Utility code for dataset handling, caching, overlap checks, visualization, and result summaries.

## Notebook order

- Notebooks are organized in a rough sequence: `00_*`, `01_*`, `02_*`, and so on.
- Experiment notebooks use numbered prefixes: `EXP1_*`, `EXP2_*`, and so on.

## Important note

This code is written for a remote server workflow, so direct references between scripts here do not always reflect how execution works in practice. Local imports, paths, and script-to-script links may not behave the same way as they do in the server environment.