from sklearn.preprocessing import LabelEncoder
from archive.explore_data import load_data
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, f1_score
import matplotlib.pyplot as plt
import numpy as np


df = load_data()

# handling categorical data 
le = LabelEncoder()
df['encoded_type'] = le.fit_transform(df['type'])
print(df['encoded_type'] )

# features to include in model
features = ['step', 'amount', 'log_amount', 'oldbalanceOrg', 'newbalanceOrig','oldbalanceDest', 'newbalanceDest', 'encoded_type']

# train/test split
X = df[features]
y = df['isFraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=.2, random_state=42)


# train xgboost model - gradient boosting (weak -> strong)
non_fraud = (y_train == 0).sum()
fraud = (y_train == 1).sum()
model = XGBClassifier(
    scale_pos_weight=(non_fraud/fraud),
    max_depth=6,
    n_estimators=100,
    learning_rate=0.1,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# # eval performance
# y_pred = model.predict(X_test)
# print("confusion matrix")
# print(confusion_matrix(y_test, y_pred))
# print("classification report")
# print(classification_report(y_test, y_pred, digits=4))

# model's confidence for each prediction (fraud class probabilities)
y_pred_probs = model.predict_proba(X_test)[:, 1]

# only flag if model is x (threshold)%+ sure
y_pred_custom = (y_pred_probs>.899).astype(int)
print("confusion matrix")
print(confusion_matrix(y_test, y_pred_custom))
print("classification report")
print(classification_report(y_test, y_pred_custom, digits=4))


# precision: % of all flagged predictions that are correct
# recall: % of all actual frauds that the model caught
# f1: maximize to balance precision and recall (2 * (Precision * recall)/ precision + recall)
# equal error rate (EER): where precision = recall
# precision_recall_curve returns arrays of precision/recall vals, one less threshold, account for that in f1 score calc
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_probs)

f1_scores = 2 * ((precision * recall)/(precision +recall+ 1e-8))
best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]
print(f"Best Threshold: {best_threshold}")
print(f"Precision at best threshold: {precision[best_index]:.4f}")
print(f"Recall at best threshold: {recall[best_index]:.4f}")
print(f"F1 Score at best threshold: {f1_scores[best_index]:.4f}")

plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision-Recall Tradeoff")
plt.legend()
plt.grid()
plt.show()




plt.figure(figsize=(8, 5))
plt.plot(thresholds, f1_scores[:-1], label="F1 Score")
plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.title("F1 Score vs. Threshold")
plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Best Threshold: {best_threshold:.3f}')
plt.legend()
plt.grid()
plt.show()


