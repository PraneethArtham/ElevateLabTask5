# ElevateLabTask5

---

## 📌 Steps in the Project
1. Load and prepare the dataset.  
2. Train a Decision Tree with controlled depth.  
3. Visualize the decision tree structure.  
4. Train a Random Forest and compare performance.  
5. Show feature importances.  
6. Evaluate using cross-validation.

---

Interview Questions & Answers
1. How does a decision tree work?
It splits the dataset into branches based on feature values, forming rules that lead to predictions.

2. What is entropy and information gain?

Entropy: Measures uncertainty in data.

Information Gain: The reduction in entropy after splitting a dataset.

3. How is random forest better than a single tree?
It builds many trees and averages their results, reducing overfitting and improving accuracy.

4. What is overfitting and how do you prevent it?
Overfitting means the model memorizes training data instead of generalizing. Prevent it by limiting depth, using pruning, or using ensembles.

5. What is bagging?
Bootstrap Aggregating — training multiple models on random subsets of data and averaging their predictions.

6. How do you visualize a decision tree?
Using sklearn.tree.plot_tree or exporting to Graphviz.

7. How do you interpret feature importance?
Higher importance means the feature plays a bigger role in splitting and making decisions.

8. What are the pros/cons of random forests?

High accuracy, handles missing values, less overfitting.
Slower to train, less interpretable than a single tree.
