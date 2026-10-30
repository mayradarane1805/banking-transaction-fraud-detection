# Machine-learning syllabus coverage

The syllabus is used as the project method guide, not as a separate dashboard-only page. The table maps the supplied six-unit syllabus to working project code and explains where a topic is a bounded teaching experiment rather than a valid production fraud policy.

| Unit | Syllabus topic | Project implementation | Status and interpretation |
|---|---|---|---|
| I | ML foundations and learning paradigms | `model_training.py`, `additional_experiments.py`, `sequence_experiments.py`, `rl_experiments.py` | Supervised, unsupervised, semi-supervised, and reinforcement-learning examples are implemented in their relevant workflows. |
| I | Data preparation and preprocessing | `data_processing.py`, `fraud_detection.py` | Required schema, target, numeric finiteness, and upload fields are validated. The source data is already numeric and has no required missing values, so the project does not invent an imputation step. |
| I | Feature engineering, selection, and extraction | `model_training.py`, `additional_experiments.py` | `SelectKBest` is fitted within a classifier pipeline. Existing `V1`-`V28` are PCA-derived; clustering also demonstrates a lower-dimensional PCA projection. |
| I | Train/test separation and cross-validation | `fraud_detection.py`, `advanced_models.py`, `model_training.py` | Stratified holdouts, three-fold model validation, a distinct validation set for threshold selection, and an untouched final test set are used in the applicable workflows. |
| I | Evaluation and class imbalance | `fraud_detection.py`, `model_training.py`, `advanced_models.py` | Class weights are used where supported. Accuracy, precision, recall, F1, balanced accuracy, average precision, ROC AUC, confusion matrices, and calibration are computed from real labels. |
| II | Linear regression | `additional_experiments.py` | Predicts recorded `Amount` as a separate regression objective; it is not fraud classification. |
| II | Polynomial regression | `additional_experiments.py` | Degree-2 polynomial features followed by Ridge regression predict `Amount`. |
| II | Ridge regression | `additional_experiments.py` | Regularized amount-prediction comparison. |
| II | Lasso regression | `additional_experiments.py` | L1-regularized amount-prediction comparison. |
| II | Logistic regression | `model_training.py`, `advanced_models.py`, `sequence_experiments.py` | Class-weighted fraud classification; also used as the base emission model in the sequence demonstration. |
| III | k-nearest neighbors | `model_training.py` | Euclidean and Manhattan distance variants are available with standardized inputs and distance weighting. |
| III | Support Vector Machine | `model_training.py` | Linear, polynomial, and RBF kernels are implemented. The interactive comparison caps SVM training size to keep runtimes manageable. |
| III | Decision Tree and Random Forest | `fraud_detection.py`, `model_training.py` | Primary dashboard models and additional comparison candidates; confusion matrices and feature importance are available. |
| IV | Hidden Markov Models (HMM) | `sequence_experiments.py` | Both ergodic and left-to-right two-state categorical HMMs are fitted using forward-backward/Baum-Welch calculations; posterior filtering and Viterbi decoding are implemented. |
| IV | Conditional Random Fields (CRF) | `sequence_experiments.py` | A trainable two-label linear-chain CRF models pairwise transitions on a bounded prefix of the training period and evaluates filtered probabilities on a chronological holdout. |
| IV | Sequence interpretation | `sequence_experiments.py`, `app_pages/advanced.py` | There are no account/customer IDs. The sequence experiment therefore treats records as one market-wide time-ordered stream, not customer histories; short temporal windows may have few fraud cases. |
| V | K-Means and K-Medoids | `additional_experiments.py` | Feature-only clustering with labels compared only after fitting. |
| V | Hierarchical clustering and MST/single linkage | `additional_experiments.py` | Ward, average, and single-linkage agglomerative variants; single linkage corresponds to a minimum-spanning-tree hierarchy. |
| V | DBSCAN and Spectral clustering | `additional_experiments.py` | Feature-only comparisons with noise and clustering metrics reported where defined. |
| V | BIRCH and CURE | `additional_experiments.py` | BIRCH is implemented with scikit-learn; the CURE-style teaching implementation is intentionally limited to at most 250 rows. |
| V | Outlier detection and cluster metrics | `additional_experiments.py` | Local Outlier Factor (LOF), silhouette, Davies-Bouldin, Calinski-Harabasz, homogeneity, completeness, and adjusted Rand index are calculated when meaningful. |
| VI | Bagging and Random Forest | `model_training.py` | Bagging and class-weighted Random Forest are fitted on the comparison training data. |
| VI | Boosting, AdaBoost, and XGBoost | `model_training.py` | Gradient Boosting, Histogram Gradient Boosting, AdaBoost, and XGBoost are implemented and evaluated on the common test sample. |
| VI | Stacking and blending | `model_training.py` | Stacking uses cross-validated base predictions; blending reserves a training-only meta-model split. |
| VI | Voting ensemble | `model_training.py` | Soft voting combines logistic and forest probabilities. |
| VI | Perceptron and neural networks | `model_training.py` | A calibrated Perceptron and MLP variants with ReLU, tanh, and sigmoid/logistic activations are implemented. |
| VI | MDP concepts and Q-learning | `rl_experiments.py` | Score bucket is the state, allow/review are actions, and explicit relative error costs define rewards. |
| VI | Deep Q-learning | `rl_experiments.py` | A small NumPy neural Q-network is trained as a DQN-style teaching experiment and measured on later held-out records. It is not a bank policy. |
| VI | Reinforcement-learning limitations | `rl_experiments.py`, `app_pages/simulator.py` | No actual bank actions, customer responses, or reward history exist. Transitions and costs are assumptions; Q-learning/DQN outputs are illustrative only. |

## Important safeguards

- Model comparisons use actual records from the OpenML dataset. Unit tests may use small fixtures to check validation logic, but fixture outputs are not presented as project evaluation results.
- Test labels are excluded from model fitting and validation-threshold selection.
- Clustering uses feature columns only; labels are revealed only to calculate post-fit comparison metrics.
- The simulator replays held-out historical rows with generated IDs/timestamps and clearly labels them **DEMO DATA**.
- Every experimental page states when the dataset does not support a stronger claim. None of the syllabus exercises establish production readiness or real bank performance.
