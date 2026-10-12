# Predicting Retrieval Failure in Consumer Health Question Answering

Jeremy Childress · Programming for Health Data Science, Dartmouth

## Overview

Many AI tools answer health questions by first searching a set of documents for relevant passages. If the search returns the wrong passage, the tool can still give a confident but wrong answer. This project predicts when that search step fails, using four search methods and seven supervised learning models.

## Dataset

**MedQuAD (Medical Question Answering Dataset)**, consumer health questions paired with answers from NIH websites. Released under CC BY 4.0.
Source: https://github.com/abachaa/MedQuAD

Ben Abacha, A., & Demner-Fushman, D. (2019). A question-entailment approach to question answering. *BMC Bioinformatics, 20*(1), 511.

MedQuAD is not on the UCI repository. I requested an exception to use it. My backup is the UCI Phishing Websites dataset (https://archive.ics.uci.edu/dataset/327/phishing+websites).

## Prediction Problem

- **Target:** `failure` (1 = the correct answer is not in the top 10 search results, 0 = it is). MedQuAD does not include this variable. The notebook creates it by running four search methods (BM25, TF-IDF, LSA, and a hybrid) on 2,000 questions.
- **Observations:** 8,000 (2,000 questions × 4 search methods)
- **Predictors:** 11 (question length, topic length, word rarity, standardized top score and score gap, same-page results, template flag, search method, question type, source)
- **Primary metric:** ROC-AUC, with PR-AUC, recall, precision, F1, balanced accuracy, and Brier score

## Files

| File | Description |
|---|---|
| `Final_Project_Checkpoint_Childress.ipynb` | Full pipeline: import and validation, cleaning, descriptive statistics, search methods, feature engineering, visualizations, inferential statistics, seven models, evaluation, feature importance, interpretation, and documented enhancements |
| `Dataset_Selection_Lab_Childress.ipynb` | Week 3 dataset selection and exploration lab |
| `data/medquad_raw.csv.gz` | Full MedQuAD dataset (46,882 questions) converted from the original XML files |
| `data/retrieval_results.csv` | Saved search results (8,000 rows) produced by the checkpoint notebook |
| `requirements.txt` | Python packages used |

## How to Run

1. In Google Colab, choose File > Open notebook > GitHub, paste this repo's URL, and open `Final_Project_Checkpoint_Childress.ipynb`.
2. Choose Runtime > Run all. No uploads are needed. The notebook loads the data from this repo automatically, and rebuilds it from the original MedQuAD files if the repo copy is unavailable.

The full run takes about 2 to 4 minutes in Colab. To skip the search step, set `USE_CACHED_RETRIEVAL = True` in the configuration cell, which loads `data/retrieval_results.csv` instead.

## Key Results

- BM25 keyword search failed least often (16%). LSA failed most (72%).
- The random forest predicted failures best, with a test ROC-AUC of 0.914 and PR-AUC of 0.855.
- Within BM25 alone, the model ranks failures with a ROC-AUC of 0.938, so it learns more than which search method was used.
- The strongest single signal is whether the top results come from the same NIH page.

## Pipeline Enhancements

1. Central configuration cell
2. Modular, documented helper functions
3. Data validation with clear error messages
4. Feature engineering (within-method standardized scores, template flag, grouped question types)
5. Conditional data loading and cached search results
6. Leakage-safe grouped splitting and cross-validation
7. Additional metrics (PR-AUC, balanced accuracy, Brier score) and comparison visuals

Each enhancement is described in Section 15 of the notebook.
