# SMS Spam Robustness

**NLP Final Project Proposal — Spam Detection under Character-Level Text Changes**

**Team:** [Jonas Chen](https://github.com/JonasChenJusFox), [Aarron Chen](https://github.com/kc5358-cpu), [Ophelia Liang](https://github.com/yl10837)

## 1. Problem Statement

SMS spam filters may fail when a message contains small spelling or punctuation changes. For example, a spammer could write `fr€e`, `f.r.e.e`, or `freee` instead of `free`. These edits often leave the meaning understandable to a reader but may change the features used by a text classifier.

Our research question is: **Can conservative text normalization improve the detection of modified SMS spam without increasing false alarms on legitimate (ham) messages?** We will compare word-level features, character-level features, and a custom normalization step using the same classifier, Logistic Regression. The main contribution is an evaluation of the trade-off between recovering modified spam words and accidentally changing legitimate text. We do not assume that normalization will outperform character-level features.

## 2. Related Work

We have identified five published NLP papers that guide the project:

1. **Han and Baldwin (2011)** study lexical normalization of informal Twitter and SMS-style messages, including the problem of mapping unusual word forms to standard forms. Their work motivates treating normalization as a separate step rather than expecting a classifier to handle all spellings directly [1].

2. **Eger et al. (2019)** show that visually altered characters can disrupt NLP models and study defenses that include rule-based recovery. This motivates our character-substitution tests, such as `free` to `fr€e` [2].

3. **Pruthi et al. (2019)** place a word-recognition component before a downstream classifier to address adversarial misspellings. Their findings support our modular design and warn that correcting more words does not always improve classification [3].

4. **Bitton et al. (2022)** present a lightweight normalizer for character-level adversarial changes. Their approach is especially close to ours; we will focus our comparison on SMS spam, false-positive rates for ham, and the effect of conservative protection rules rather than claiming to invent normalization itself [4].

5. **Wang et al. (2021)** introduce TextFlint for controlled NLP robustness testing and analysis. We will follow the general idea of fixed, documented transformations and report model performance separately for each type of change [5].

We will also use the **Super SMS Dataset** introduced by Salman et al. (2024). That study examines SMS spam filtering and evasive strategies, providing the task-specific dataset and context for our work [6].

## 3. Data and Preparation

Our main dataset will be the [Super SMS Dataset](https://github.com/smspamresearch/spstudy), a public collection reported to contain **67,018 labeled messages** (26,181 spam and 40,837 ham) before our own cleaning. We will inspect its text and labels, check missing values and duplicate or conflicting records, and record the usable size after cleaning. Because this dataset combines earlier collections, we will not treat the smaller [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) as an independent test set without checking for overlap. We may use UCI only to test our code during early development.

We plan an approximately **70% training, 15% development, and 15% test** split. Identical or closely related messages will be grouped before splitting so that their variants cannot appear in different partitions. We will maintain spam/ham proportions as closely as grouping allows and report the final counts. All vocabulary construction, TF-IDF fitting, and model fitting will use **training data only**. Development data will be used to select rules and parameters; the held-out test set will be used only for final evaluation.

## 4. Systems and Implementation Strategy

All learned classifiers will use scikit-learn Logistic Regression, allowing us to focus on text representation.

| System | Method | Role |
|---|---|---|
| Majority baseline | Always predict the most frequent training label | Minimum comparison |
| A — Word | Word unigram/bigram TF-IDF + Logistic Regression | Standard baseline |
| B — Character | Character n-gram TF-IDF + Logistic Regression | Robust representation baseline |
| C — Normalized word | Custom normalization + word TF-IDF + Logistic Regression | Our proposed system |

**Custom normalizer.** We will implement explicit rules for visually similar character substitutions, punctuation inserted inside words, and repeated letters. The normalizer will generate possible standard forms and accept a replacement only if it has sufficiently strong evidence from a vocabulary built from *both* training classes. If several plausible replacements remain, it will leave the token unchanged. We will protect URLs, email addresses, phone numbers, numeric strings, and likely names or abbreviations whenever the rule could damage them. The same preprocessing will be applied when training and predicting. We will use Python `re` and `unicodedata` with small, explicit substitution maps; Unicode normalization alone will not be assumed to convert every symbol.

**Illustrative rule cases (not experimental results):** `fr€e → free`, `f.r.e.e → free`, `freee → free`; a URL or ambiguous abbreviation should remain unchanged. Whether a replacement is actually accepted will depend on our final rules and training vocabulary.

**Minimum version:** complete the four systems, a clean-data evaluation, three separate character-change tests, and development-set error analysis. If the larger dataset needs extra cleaning, we can first confirm the pipeline on UCI and then use the audited Super SMS data for the main study.

**Extended version (if time permits):** test combinations of changes; compare normalization with and without protection rules; try training-data augmentation; and build a small Streamlit interface. The demo is optional and will not replace the research evaluation.

## 5. Evaluation Plan

The task is **binary SMS classification**: predict *spam* or *ham* for a message. We will evaluate every system on the same clean messages and on matched modified versions of those messages.

**Robustness tests.** After splitting, we will generate controlled variants using fixed random seeds: (1) character substitution (`free → fr€e`), (2) punctuation insertion (`free → f.r.e.e`), and (3) character repetition (`free → freee`). The same label-independent rule and change budget will be applied to spam and ham. We will modify at most one eligible word per message for each single-change condition, keep the original and all its variants in the same partition, and record when a message could not be changed. A sample of development-set variants will be checked manually to verify readability and that the original label still applies.

**Measures.** Our primary measures will be **spam precision, recall, and F1**, and **ham false-positive rate** (the fraction of ham messages incorrectly predicted as spam). We will also report performance changes from clean to modified text, modification coverage by class and change type, and confusion matrices. For fairness, robustness comparisons will use matched original/modified messages that were actually changed, with full-set results shown separately. Classifier settings and any decision thresholds will be selected with development data, never the test labels.

**Error analysis.** We will inspect false negatives, false positives, and cases where normalization changes a correct prediction into a wrong one on the **development set**. We will categorize failures by edit type, normalization ambiguity, and protected content, then revise the rules using those findings. We will freeze all rules and settings before running the final test evaluation. A key comparison will be System C versus System B: normalization helps only if its spam-detection gains are not offset by unacceptable increases in false positives for ham.

## 6. Collaboration Plan

| Team member | Lead responsibility | Independent contribution |
|---|---|---|
| [Jonas Chen](https://github.com/JonasChenJusFox) | Evaluation and frontend | Define evaluation scripts, tables, plots, and error-analysis format; build the optional Streamlit demo |
| [Aarron Chen](https://github.com/kc5358-cpu) | Data and backend | Audit/split data; implement modifications, normalization, TF-IDF pipelines, and classifiers |
| [Ophelia Liang](https://github.com/yl10837) | Literature and paper | Review the five NLP papers; draft methods/background; organize manual checks, results, discussion, and references |

All members will review the experiments and co-author the final paper. To work **in parallel**, we will first agree on CSV interfaces: split data with `message_id`, `text`, `label`, and `split`; modified data with `message_id`, `change_type`, `modified_text`, and `changed`; predictions with `message_id`, `system`, `change_type`, `prediction`, and `score`. Jonas can build evaluation scripts and the demo using sample prediction files while Aarron develops the actual pipeline. Ophelia can draft the literature review and create a manual-review checklist independently. We will integrate components through GitHub pull requests. Our stack will be Python, pandas, scikit-learn, `re`, `unicodedata`, and Matplotlib, with Streamlit only for the optional demo.

## 7. Deliverables and Feasibility

We will submit reproducible Python code, documented data splits and transformation rules, evaluation tables/figures, development-set error analysis, a final research paper, and an **eight-minute presentation**. The optional Streamlit demo will allow a user to enter a message, apply text changes, and compare model predictions. The project is feasible without a large neural model because TF-IDF and Logistic Regression are lightweight; if normalization does not improve results, analyzing when and why it fails will still answer our research question.

## References

[1] Han, B., & Baldwin, T. (2011). *Lexical Normalisation of Short Text Messages: Makn Sens a #twitter*. In **Proceedings of ACL-HLT**, pp. 368–378. https://aclanthology.org/P11-1038/

[2] Eger, S., Şahin, G. G., Rücklé, A., Lee, J.-U., Schulz, C., Mesgar, M., Swarnkar, K., Simpson, E., & Gurevych, I. (2019). *Text Processing Like Humans Do: Visually Attacking and Shielding NLP Systems*. In **Proceedings of NAACL-HLT**, pp. 1634–1647. https://aclanthology.org/N19-1165/

[3] Pruthi, D., Dhingra, B., & Lipton, Z. C. (2019). *Combating Adversarial Misspellings with Robust Word Recognition*. In **Proceedings of ACL**, pp. 5582–5591. https://aclanthology.org/P19-1561/

[4] Bitton, J., Pavlova, M., & Evtimov, I. (2022). *Adversarial Text Normalization*. In **Proceedings of NAACL-HLT: Industry Track**, pp. 268–279. https://aclanthology.org/2022.naacl-industry.30/

[5] Wang, X., Liu, Q., Gui, T., et al. (2021). *TextFlint: Unified Multilingual Robustness Evaluation Toolkit for Natural Language Processing*. In **Proceedings of ACL-IJCNLP: System Demonstrations**, pp. 347–355. https://aclanthology.org/2021.acl-demo.41/

[6] Salman, M., Ikram, M., & Kaafar, M. A. (2024). *Investigating Evasive Techniques in SMS Spam Filtering: A Comparative Analysis of Machine Learning Models*. **IEEE Access, 12**, pp. 24306–24324. https://doi.org/10.1109/ACCESS.2024.3364671
