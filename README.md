# Predicting Retrieval Failure in Consumer Health Question Answering

Final project dataset selection and exploratory analysis.
Jeremy Childress

## Overview

Many AI tools answer health questions by first searching a set of documents for relevant passages. If the search returns the wrong passage, the tool can still give a confident but wrong answer. This project predicts when that search step fails.

## Dataset

**MedQuAD (Medical Question Answering Dataset)**
https://github.com/abachaa/MedQuAD

Consumer health questions paired with answers from NIH websites such as cancer.gov, GARD, and MedlinePlus. Released under CC BY 4.0.

Ben Abacha, A., & Demner-Fushman, D. (2019). A question-entailment approach to question answering. *BMC Bioinformatics, 20*(1), 511.

MedQuAD is not on the UCI repository. I am requesting an exception to use it. My backup is the UCI Phishing Websites dataset (https://archive.ics.uci.edu/dataset/327/phishing+websites).

## Prediction Problem

Given a health question and a search method, predict whether the method fails to return the correct answer in its top 10 results.

- **Target:** `retrieval_success` (1 = correct answer in top 10, 0 = not), created by running BM25 search on each question
- **Observations:** 2,000 questions sampled from 13,675 cleaned questions (about 8,000 rows once four search methods are added)
- **Predictors:** 10 (question length, word rarity, topic length, question type, source, search method, top score, score gap, and same-page results)
- **Metric:** ROC-AUC, with PR-AUC as a second metric

## How to Run

1. Open `Dataset_Selection_Lab_Childress.ipynb` in Google Colab (File > Open notebook > GitHub, then paste this repo's URL).
2. Run all cells. The notebook downloads MedQuAD from GitHub and installs `rank_bm25` itself.

Runtime is about 3 to 5 minutes. Most of that is the BM25 search over 2,000 questions.

## Files

| File | Description |
|---|---|
| `Dataset_Selection_Lab_Childress.ipynb` | Data import, structure, data quality, target creation, descriptive statistics, visualizations, and project plan |
| `requirements.txt` | Python packages used |
