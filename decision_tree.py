import pandas as pd
from sklearn.datasets import load_heart_disease  # Replace with actual dataset if available
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np

heart = load_heart_disease()
X = pd.DataFrame(heart.data, columns=heart.feature_names)
y = pd.Series(heart.target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model.fit(X_train, y_train)
dt_acc = dt_model.score(X_test, y_test)
print(f"Decision Tree Accuracy: {dt_acc:.2f}")
plt.figure(figsize=(12,6))
plot_tree(dt_model, feature_names=X.columns, class_names=["No Disease", "Disease"], filled=True)
plt.title("Decision Tree (max_depth=4)")
plt.show()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_acc = rf_model.score(X_test, y_test)
print(f"Random Forest Accuracy: {rf_acc:.2f}")
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
print("\nFeature Importances:")
for idx in indices:
    print(f"{X.columns[idx]}: {importances[idx]:.3f}")
plt.figure(figsize=(10,6))
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.title("Random Forest Feature Importances")
plt.show()
cv_scores = cross_val_score(rf_model, X, y, cv=5)
print(f"Random Forest CV Accuracy: {cv_scores.mean():.2f}")
