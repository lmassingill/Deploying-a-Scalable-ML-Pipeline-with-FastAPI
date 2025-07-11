# Model Card

---

## Model Details

- **Model Type**: Random Forest Classifier  
- **Library Used**: scikit-learn (`sklearn.ensemble.RandomForestClassifier`)  
- **Version**: sklearn 1.x  
- **Hyperparameters**:  
  - `n_estimators = 100`  
  - `max_depth = None`  
  - `random_state = 42`  
  - `n_jobs = -1` (parallelized across all CPU cores)  
- **Trained On**: Structured tabular data from the U.S. Census Bureau  
- **Encoding Used**:  
  - OneHotEncoder for categorical features  
  - LabelBinarizer for the target label  

---

## Intended Use

- **Goal**: Predict whether an individual earns more than $50K annually based on demographic attributes.  
- **Intended Users**: Data scientists, researchers, engineers evaluating ML fairness or deploying income prediction systems.  
- **Applications**:  
  - Educational demonstrations on fairness in ML  
  - Research on bias detection and mitigation  
  - Prototyping for decision-support systems  

> ⚠️ Not intended for use in high-stakes decisions like lending, hiring, or law enforcement without further validation.

---

## Training Data

- **Source**: UCI Adult Income Dataset (https://archive.ics.uci.edu/ml/datasets/adult)  
- **Sample Size**: ~32,000 rows (after preprocessing)  
- **Features**:  
  - Categorical: `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`  
  - Numerical: `age`, `fnlwgt`, `capital-gain`, `capital-loss`, `hours-per-week`  
- **Target**:  
  - `0`: Income <= 50K  
  - `1`: Income > 50K  

---

## Evaluation Data

- 20% split from the original dataset using `train_test_split` (random_state = 42)  
- Preprocessing consistent with training using saved encoder and label binarizer  
- Metrics computed globally and on **slices** of the data (e.g., by `race`, `sex`, `workclass`)  

---

## Metrics

**Global Performance** on test set:
- **Precision**: 0.81  
- **Recall**: 0.71  
- **F1-score**: 0.76  

Metrics were computed using `precision_score`, `recall_score`, and `fbeta_score` from scikit-learn (`beta=1`).

**Slice-Based Performance Example**:
- **workclass: Private**  
  - Precision: 0.82  
  - Recall: 0.72  
  - F1: 0.76  
- **sex: Female**  
  - Precision: 0.79  
  - Recall: 0.66  
  - F1: 0.72  

_Slice metrics were logged in `slice_output.txt` for further fairness analysis._

---

## Ethical Considerations

- **Bias & Fairness**:  
  - Dataset reflects real-world demographic and socioeconomic biases.  
  - Model performance varies across demographic groups.  

- **Privacy**:  
  - Data is anonymized and public, but production use must consider privacy safeguards.

- **Accountability**:  
  - This model is for educational and prototyping use only.  
  - Real-world use would require audits, fairness testing, and risk evaluation.

---

## Caveats and Recommendations

- Model is trained on historical U.S. data and may not generalize to other contexts.  
- Some features (e.g., `education`, `occupation`) may act as proxies for sensitive attributes.  
- Before production use:  
  - Retrain on updated, representative data  
  - Incorporate fairness-aware methods  
  - Monitor for performance and equity over time

