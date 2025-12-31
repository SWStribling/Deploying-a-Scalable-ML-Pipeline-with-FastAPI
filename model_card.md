# Model Card
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- **Model type:** Logistic Regression (scikit-learn, solver=`lbfgs`, max_iter=1000)
- **Framework:** scikit-learn 1.x
- **Preprocessing:** 
  - OneHotEncoder for categorical features
  - LabelBinarizer for target (`salary`)
  - StandardScaler for continuous features
  - Education mapped to ordinal `education_num`
- **Artifacts saved:** `model.pkl`, `encoder.pkl`, `lb.pkl`, `scaler.pkl`
- **Owner:** S. W. Stribling (student project for WGU D501 Machine Learning DevOps)

## Intended Use
- **Primary purpose:** Predict whether an individual earns `>50K` or `<=50K` annually based on census features.
- **Deployment context:** FastAPI service for inference, supporting slice-based performance analysis.
- **Users:** Educational / demonstration purposes in ML DevOps coursework. Not intended for production decision-making about individuals.

## Training Data
- **Source:** UCI Adult Census dataset (commonly used for income prediction tasks).
- **Features:** Workclass, education, marital-status, occupation, relationship, race, sex, native-country, plus continuous features (age, fnlwgt, hours-per-week, etc.).
- **Preprocessing:** Education labels mapped to ordinal values; categorical features one-hot encoded; continuous features scaled.

## Evaluation Data
- **Method:** 5-fold cross-validation on the full dataset.
- **Test folds:** Each fold held out ~20% of the data for evaluation.
- **Slice analysis:** Performance computed across subgroups of categorical features (e.g., race, sex, education).

## Metrics
- **Cross-validation averages:**
  - Precision: **0.7371**
  - Recall: **0.6023**
  - F1-score: **0.6629**
- **Per-fold results:**
  - Fold 1: Precision=0.7518, Recall=0.6149, F1=0.6765
  - Fold 2: Precision=0.7224, Recall=0.6010, F1=0.6562
  - Fold 3: Precision=0.7321, Recall=0.5944, F1=0.6561
  - Fold 4: Precision=0.7439, Recall=0.6013, F1=0.6650
  - Fold 5: Precision=0.7352, Recall=0.6001, F1=0.6608
- **Slice metrics:** Outputted to `slice_output.txt` for subgroup fairness analysis.

## Ethical Considerations
- **Bias risk:** Census data reflects historical socioeconomic patterns, which may encode bias across race, sex, and education. Predictions may perpetuate inequities if used in real-world decision-making.
- **Fairness:** Slice metrics are computed to monitor subgroup performance. Disparities across slices should be carefully examined before deployment.
- **Privacy:** Model trained on public dataset; no personal or sensitive user data involved.

## Caveats and Recommendations
- **Not for production use:** This model is for educational purposes only. It should not be used to make real financial, hiring, or policy decisions.
- **Convergence warnings:** Logistic regression may require scaling and higher iteration limits to converge fully.
- **Performance limitations:** Recall is lower than precision, meaning the model misses some positive cases. Further tuning or alternative models (e.g., Random Forest, Gradient Boosting) could improve balance.
- **Recommendations:** 
  - Always evaluate slice metrics before deployment.
  - Retrain periodically if applied to new data distributions.
  - Consider fairness-aware techniques if extending beyond coursework.
