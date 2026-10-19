# Predicting Retrieval Failure in Consumer Health Question Answering

Jeremy Childress · Programming for Health Data Science, Dartmouth

## Overview

Patient-facing chatbots and clinical assistants often answer questions by first searching a set of trusted documents. If the search returns the wrong passage, the tool can still give a confident answer that is wrong. This project builds a reproducible pipeline that predicts, before any answer is written, when a search is likely to fail, so the question can be sent to a stronger method or to a person instead.

The notebook runs four search methods on consumer health questions, labels each search as a success or failure, tests whether failure differs by method and question type, and trains 10 classification models to predict failure.

## Quick Start

1. Open the notebook in Google Colab:
   https://colab.research.google.com/github/f007yss/dataset-selection-lab/blob/main/Final_Project_Childress.ipynb
2. Choose **Runtime > Run all**.

No uploads or setup are needed. The notebook loads the dataset from this repo and installs any missing packages. A full run takes about 3 to 5 minutes on a standard Colab CPU.

## Using Your Own Data

1. In the configuration cell (Section 1), set `DATA_SOURCE = 'upload'`.
2. Choose **Runtime > Run all**. A file picker appears in Section 4.
3. Choose a CSV, TSV, or Excel file.

The notebook profiles your file, checks it, and stops with a clear message if something needs fixing. It adapts to your data automatically and logs every change it makes (see Section 8 and Section 16 of the notebook).

| Column | Required | Description |
|---|---|---|
| `question` | Yes | The question text |
| `answer` | Yes | The passage that correctly answers the question |
| `qtype` (or `question_type`) | No | Question category. If missing, the question type analysis is skipped |
| `source` | No | Where the answer came from. If missing, it is not used as a predictor |
| `doc_id` | No | The page or document the answer belongs to. If missing, each answer is treated as its own page |
| `focus` | No | The topic or condition the question is about |

At least 100 rows with both a question and an answer are needed. To try it, upload `data/example_upload.csv`, which contains 5,029 MedQuAD questions from NIH sites other than the two largest ones. You can also set `DATA_SOURCE = 'path'` and `UPLOAD_PATH = 'data/example_upload.csv'` to run it without the file picker.

## Configuration

All settings are in the first code cell.

| Setting | Default | What it controls |
|---|---|---|
| `DATA_SOURCE` | `'medquad'` | `'medquad'`, `'upload'`, or `'path'` |
| `PRIMARY_METRIC` | `'roc_auc'` | How the best model is chosen: `'roc_auc'`, `'pr_auc'`, `'f1'`, `'recall'`, or `'balanced_accuracy'` |
| `TARGET_RECALL` | `0.90` | Share of failed searches the final cutoff should catch |
| `SAMPLE_SIZE` | `2000` | Questions to search (reduced automatically for smaller datasets) |
| `TOP_K` | `10` | A search succeeds if the correct answer is in the top K results |
| `RUN_TUNING` | `True` | Grid search for logistic regression and random forest |
| `USE_CACHED_RETRIEVAL` | `False` | Reuse saved MedQuAD search results to skip the search step |

## Pipeline

1. **Data input and validation.** Load MedQuAD or an uploaded file, standardize columns, profile every column, and validate.
2. **Cleaning.** Remove rows without answers and duplicate questions, and group rare question types.
3. **Descriptive statistics and visualizations.**
4. **Target creation.** Run BM25, TF-IDF, LSA, and a hybrid search on each question. A search fails if the correct answer is not in the top 10.
5. **Feature engineering.** 11 predictors, including word rarity, standardized search scores, and how many top results come from the same page.
6. **Inferential statistics.** Cochran's Q and McNemar tests across search methods, a chi-square test for question type, and a normality-based choice between Welch's t-test and the Mann-Whitney U test.
7. **Classification models.** Logistic regression, decision tree, SVM, random forest, bagging, AdaBoost, gradient boosting, XGBoost, stacking, and a neural network, with grouped cross-validation so no question appears in both training and test data.
8. **Best model selection** by cross-validated ROC-AUC, chosen in advance.
9. **Decision threshold** set to catch 90% of failed searches.
10. **Interpretation,** written automatically from the results.

## Key Results (MedQuAD, default settings)

- BM25 failed least often (16%) and LSA most often (72%). The methods differ significantly (Cochran's Q, p < 0.001).
- Failure depends on question type (chi-square p < 0.001, Cramér's V = 0.28). Research, inheritance, and outlook questions fail most.
- Stacking was the best model, with a test ROC-AUC of 0.915 and PR-AUC of 0.855, compared with 0.748 for a model that only knows the search method.
- At a cutoff of 0.23, the model catches 91% of failed searches while flagging about half of all questions.
- The strongest signal is whether the top results come from the same NIH page.

## Repository Structure

| File | Description |
|---|---|
| `Final_Project_Childress.ipynb` | **Final project notebook** |
| `data/medquad_raw.csv.gz` | Full MedQuAD dataset (46,882 questions), converted from the original XML files |
| `data/example_upload.csv` | Example file for testing the upload feature |
| `data/retrieval_results.csv` | Saved search results for MedQuAD, used when `USE_CACHED_RETRIEVAL = True` |
| `requirements.txt` | Python packages used |
| `Final_Project_Checkpoint_Childress.ipynb` | Week 4 checkpoint version |
| `Dataset_Selection_Lab_Childress.ipynb` | Week 3 dataset selection lab |

## Dependencies

Python 3 with the packages in `requirements.txt`. Google Colab includes all of them except possibly `xgboost` and `statsmodels`, which the notebook installs automatically if missing. To run locally: `pip install -r requirements.txt`.

## Data Source

Ben Abacha, A., & Demner-Fushman, D. (2019). A question-entailment approach to question answering. *BMC Bioinformatics, 20*(1), 511. https://github.com/abachaa/MedQuAD (CC BY 4.0)
