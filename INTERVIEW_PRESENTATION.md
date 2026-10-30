# Faculty demonstration and viva guide

**Project:** AI-Powered Banking Transaction Fraud Detection and Risk Monitoring System  
**Suggested presentation time:** 6-8 minutes  
**Local dashboard:** `http://localhost:8501`

## Live faculty demonstration script

### 1. Introduce the project (about 45 seconds)

> This is an interactive machine-learning prototype for transaction-fraud research. It uses a documented public dataset, trains and evaluates models, scores transactions immediately when I submit their features, and provides a review queue. The simulator is clearly labeled DEMO DATA and replays held-out historical rows. This project is not connected to a bank or a payment feed.

Open **Executive dashboard** and point out the fraud rate and the interactive filters. Explain that all cards and charts use observed dataset rows.

### 2. Explain the problem and data (about 60 seconds)

Open **Introduction and objectives**, then **Dataset, EDA, and search**.

> OpenML dataset 1597 has 284,807 recorded transactions and 492 fraud labels. Fraud is very rare, so a model that predicts every row as normal could have high accuracy while catching no fraud. The `Class` column is the outcome, `Time` is elapsed time, `Amount` is the recorded amount, and `V1` through `V28` are anonymized principal components. Their original business meanings are unknown.

Show the normalized amount distributions, hourly activity, amount segments, feature association chart, and historical row search. Explain that associations are descriptive, not causal.

### 3. Train and evaluate (about 90 seconds)

Open **Training and evaluation**. Submit the default Decision Tree and Random Forest training form, then review the actual test metrics.

> The core workflow uses a stratified holdout and class weights. I report precision, recall, F1-score, average precision, ROC AUC, and a confusion matrix alongside accuracy. Precision tells us how many alerts were actually fraud; recall tells us how much recorded fraud was caught. Which metric matters most depends on verified operational costs.

Show confusion matrices, curves, global Random Forest importance, then run a small selected set in the syllabus algorithm comparison. Emphasize that the displayed values are computed by each run, not copied from a slide.

### 4. Explain validation and batch scoring (about 60 seconds)

Open **Advanced comparison and batch scoring** and click **Run advanced model comparison** if time permits.

> Cross-validation compares candidate models on development data. A separate validation set chooses each decision threshold using explicit relative-cost assumptions. The final held-out test set is not used to fit the models or choose those thresholds. Batch uploads must contain exactly the feature columns and are limited to 10,000 rows.

If training time is limited, explain the split and review the architecture rather than waiting for the fit to finish.

### 5. Show scoring and investigation (about 60 seconds)

Open **Individual fraud prediction**, submit an unchanged held-out record, then edit one or more values and submit again.

> The app immediately returns a score and threshold-based prediction. An unchanged example can be compared with its known historical label. An edited row is a new input and has no known label. The feature context shown is global model importance, not a local causal explanation.

Open **Fraud alerts and investigation** to search and filter the scored queue; update one review status.

### 6. Demonstrate DEMO DATA and limits (about 45 seconds)

Open **Live-style DEMO DATA simulator**, start the stream, and show the label on the event table. Then stop the stream and open **Model performance monitoring**.

> The simulator replays held-out research data with demo IDs and current timestamps. It does not connect to a real transaction source. The metrics page separates held-out test performance from simulator counts.

Finish with **System architecture** and **Conclusion and future scope**.

## Syllabus topics to mention

- **Foundations and preprocessing:** schema validation, train/test separation, cross-validation, feature selection, and imbalance-aware metrics.
- **Regression:** Linear, Polynomial, Ridge, and Lasso models predict transaction Amount as a separate task.
- **Classification:** Decision Tree, Random Forest, Logistic Regression, k-NN distance choices, linear/polynomial/RBF SVM, Perceptron, and MLP activations.
- **Sequence methods:** ergodic and left-to-right HMM, forward-backward/Baum-Welch, Viterbi, and a linear-chain CRF. These use a global time sequence because there are no customer identifiers.
- **Unsupervised methods:** K-Means, K-Medoids, hierarchical/single-linkage, DBSCAN, Spectral, BIRCH, CURE-style clustering, and LOF.
- **Ensembles:** bagging, Random Forest, AdaBoost, gradient boosting, XGBoost, stacking, blending, and voting.
- **Reinforcement learning:** Q-learning and a small DQN-style neural Q-network with disclosed assumed rewards. This is a teaching experiment, not a learned bank policy.

Use [SYLLABUS_COVERAGE.md](./SYLLABUS_COVERAGE.md) for the full implementation checklist and limitations.

## Viva questions and suggested answers

### 1. Why is fraud detection an imbalanced classification problem?

Fraudulent records are only a small fraction of all transactions. The normal class dominates, so evaluation must look at how well the minority fraud class is detected rather than rely on accuracy alone.

### 2. Why is accuracy not enough?

A classifier that predicts every transaction as normal could achieve high accuracy on this dataset and still identify zero fraud. Precision, recall, F1-score, average precision, and the confusion matrix show the fraud-specific trade-offs.

### 3. What do precision and recall mean here?

Precision is the share of flagged transactions that are labeled fraud. Recall is the share of all labeled fraud transactions that the model flags. Raising recall can also increase false alarms.

### 4. What is data leakage, and how does the project limit it?

Leakage happens when information from held-out data influences training or model selection. Training data fits the model, validation data selects thresholds in the advanced workflow, and the final test set is reserved for final evaluation.

### 5. Why use a stratified split?

Stratification preserves approximately the same rare fraud proportion in the train and test partitions, making it more likely that each contains both classes.

### 6. Why use class weights?

Class weights make errors on the minority class count more during fitting for the supported models. They do not create new fraud examples, and they do not by themselves guarantee a useful decision threshold.

### 7. What is the difference between a probability and a decision threshold?

A model probability is a score. A threshold turns the score into a class decision. Changing the threshold changes the number of alerts without retraining the model.

### 8. Why include Decision Tree and Random Forest?

A Decision Tree is an interpretable baseline that can model nonlinear splits. A Random Forest averages many trees to reduce variance and often captures richer feature interactions. Both can provide feature importance, but impurity importance does not prove causation.

### 9. What does `V1` through `V28` represent?

They are anonymized principal components from a transformation of the original transaction features. The original meanings are unavailable, so the project does not assign them business interpretations.

### 10. Why is the HMM/CRF experiment limited?

The dataset has elapsed timestamps but no customer or account identifiers. The experiment treats records as one global stream, which is not the same as modeling an individual customer's transaction history. The held-out time windows can also contain few fraud cases.

### 11. Why include amount regression in a fraud project?

It demonstrates the regression topics in the syllabus on a real numeric target. It predicts recorded transaction Amount; it is a separate learning experiment and not a fraud-detection score.

### 12. Is this a real-time bank system?

It is a real-time **scoring prototype**: a submitted row is validated and scored immediately. It is not connected to a bank, payment gateway, or live feed, and it does not authorize or decline transactions.

### 13. What does the DEMO DATA simulator do?

It replays held-out records, assigns generated demonstration identifiers and current demo timestamps, scores them with a trained model, and places flagged examples in the in-memory investigation queue. It is not a real transaction stream.

### 14. Why include reinforcement learning if there are no historical decisions?

The syllabus includes reinforcement learning, but the dataset does not contain actions, responses, or reward history. The app therefore creates a small offline teaching experiment with explicit relative costs and labels the result as assumption-based, not an operational fraud policy.

### 15. What would be required before production use?

Current representative data, secure service integration, validated financial cost assumptions, latency and drift monitoring, privacy and fairness review, access controls, audit logs, persistent case management, and approved human oversight would all be necessary. This project is not production-ready.

## Presentation tips

- Train the core models before presenting so the scoring and investigation pages are available.
- If time is short, use a small comparison sample and do not wait for every advanced algorithm.
- Keep the dashboard's DEMO DATA and no-bank-connection statements visible when showing the simulator.
- Report the exact test size and measured outputs shown during the live run. Do not copy fixture or assumed-reward results as model performance.
- Explain the dataset's limitations before interpreting individual feature values.
