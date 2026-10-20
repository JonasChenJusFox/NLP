# SMS Spam Robustness

**NLP Final Project Proposal — Spam Detection under Character-Level Text Changes**

## Overview

This project will study how small changes in spelling and punctuation affect SMS spam detection. For example, a message might use `fr€e` instead of `free`, insert punctuation as in `f.r.e.e`, or repeat characters as in `freee`.

Our main research question is:

> Can text normalization improve the detection of modified spam messages without increasing false alarms on legitimate messages?

We will compare different text representations and implement a normalization component ourselves. We will also review related work on SMS spam detection and text normalization to guide our design and analysis.

## Dataset

We will use the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection), which contains messages labeled as **spam** or **ham** (legitimate messages).

After checking for duplicates, we plan to divide the data into approximately 70% training, 15% development, and 15% test data while maintaining the class proportions. Duplicate or closely related messages will stay in the same split.

Modified messages will be generated only after splitting. Each original message and its variants will remain in the same split. Vocabulary and TF-IDF weights will be learned only from the training set.

## Proposed Methods

The three main systems will use Logistic Regression so that we can focus on the effect of text representation and normalization.

| System | Method | Purpose |
| --- | --- | --- |
| Majority baseline | Always predict the most common training label | Provide a minimum reference |
| A: Word features | Word-level TF-IDF + Logistic Regression | Establish the main baseline |
| B: Character features | Character-level TF-IDF + Logistic Regression | Test an alternative representation |
| C: Custom normalization | Our normalization component + word-level TF-IDF + Logistic Regression | Evaluate our contribution |

Our normalization component will generate possible replacements for altered words and check them against a vocabulary built from both spam and ham training messages. It will use conservative rules and protect content such as phone numbers and URLs. Ambiguous cases will be left unchanged. The same normalization process will be applied during training and prediction.

## Robustness Experiments

We will create controlled text changes:

| Change | Example |
| --- | --- |
| Character substitution | `free → fr€e` |
| Punctuation insertion | `free → f.r.e.e` |
| Character repetition | `free → freee` |

In the main experiment, all models will be trained on the original training messages. We will apply the same modification rules to both spam and ham messages, use fixed random seeds, and check samples for readability and label preservation.

Parameters and normalization rules will be developed using the training and development sets. Once the setup is fixed, all systems will be evaluated on the same original and modified test messages.

## Evaluation

We will report:

- Spam precision, recall, and F1.
- The false-positive rate for legitimate messages.
- Changes in performance between original and modified text.
- The proportion of messages actually changed by each modification method.

Error analysis will examine which changes cause failures, when normalization helps, and when it incorrectly changes legitimate content. We will also compare normalization with and without its protective rules.

## Planned Technology Stack

- **Python and pandas:** data preparation and experiment scripts.
- **scikit-learn:** TF-IDF, classification pipelines, and evaluation.
- **Python `re` and `unicodedata`:** custom text modification and normalization.
- **Matplotlib or Seaborn:** result visualization.
- **Streamlit:** an optional interactive demonstration.
- **Git and GitHub:** version control and team collaboration.

## Team and Responsibilities

| Team Member | Role | Main Responsibilities |
| --- | --- | --- |
| [Jonas Chen](https://github.com/JonasChenJusFox) | Frontend | Design and implement the demo interface, including text input, modification controls, and side-by-side model predictions. |
| [Aarron Chen](https://github.com/kc5358-cpu) | Backend | Implement data processing, text modification and normalization, model training, and evaluation. |
| [Ophelia Liang](https://github.com/yl10837) | Paper Writing | Lead the literature review and paper writing, including the methods, results, discussion, and references. |

## Planned Deliverables

- Reproducible code for data preparation, model comparison, and robustness experiments.
- Evaluation tables, figures, and error analysis.
- A final research paper and an eight-minute presentation.
- A small interactive demo, if time permits.

The goal is to understand when normalization is useful and where it fails. Improved performance over the character-based system is a hypothesis to test, not an assumed result.
