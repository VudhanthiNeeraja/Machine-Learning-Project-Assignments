from sklearn import datasets
titanic_data = datasets.fetch_openml(data_id=1142)

# creating decision trees based on criterion of entropy and gini
from sklearn.tree import DecisionTreeClassifier
entropy_tree = DecisionTreeClassifier(criterion="entropy")
gini_tree = DecisionTreeClassifier()

# training the decision trees
entropy_tree.fit(titanic_data.data, titanic_data.target)
gini_tree.fit(titanic_data.data, titanic_data.target)


entropy_pred = entropy_tree.predict(titanic_data.data)
gini_pred = gini_tree.predict(titanic_data.data)


# evaluation of the trees
from sklearn import metrics

pred_prob_entropy = entropy_tree.predict_proba(titanic_data.data)
pred_prob_gini = gini_tree.predict_proba(titanic_data.data)


from sklearn import model_selection
dtc_entropy = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=3)
y_scores_entropy = model_selection.cross_val_predict(dtc_entropy, titanic_data.data, titanic_data.target, method="predict_proba", cv=10)

dtc_gini = DecisionTreeClassifier(criterion="gini", min_samples_leaf=3)
y_scores_gini = model_selection.cross_val_predict(dtc_gini, titanic_data.data, titanic_data.target, method="predict_proba", cv=10)


from sklearn.metrics import roc_curve
fpr, tpr, th = roc_curve(titanic_data.target,y_scores_entropy[:,1],pos_label="Endometrium")

from matplotlib import pyplot as plt
plt.xlabel("1 - Specificity")
plt.ylabel("Sensitivity")
plt.xlim(0,2)
plt.ylim(0,2)
plt.plot(fpr,tpr,label="Decision Tree-entropy")
plt.legend()
plt.show()

fpr, tpr, th = roc_curve(titanic_data.target,y_scores_gini[:,1],pos_label="Endometrium")

from matplotlib import pyplot as plt
plt.xlabel("1 - Specificity")
plt.ylabel("Sensitivity")
plt.xlim(0,2)
plt.ylim(0,2)
plt.plot(fpr,tpr,label="Decision Tree-gini")
plt.legend()
plt.show()

from sklearn.metrics import roc_auc_score





from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(titanic_data.data, titanic_data.target, test_size=0.2, random_state=42)
entropy_tree.fit(X_train, y_train)
gini_tree.fit(X_train, y_train)
entropy_pred = entropy_tree.predict(X_test)
gini_pred = gini_tree.predict(X_test)


# evaluation of the trees
print("Entropy Tree:")
print("Accuracy:", metrics.accuracy_score(y_test, entropy_pred))
print("F1 Score:", metrics.f1_score(y_test, entropy_pred, pos_label='Endometrium'))
print("Precision:", metrics.precision_score(y_test, entropy_pred, pos_label='Endometrium'))
print("Recall:", metrics.recall_score(y_test, entropy_pred, pos_label='Endometrium'))
print("ROC AUC Score:", roc_auc_score(y_test, entropy_tree.predict_proba(X_test)[:, 1]))

print("\nGini Tree:")
print("Accuracy:", metrics.accuracy_score(y_test, gini_pred))
print("F1 Score:", metrics.f1_score(y_test, gini_pred, pos_label='Endometrium'))
print("Precision:", metrics.precision_score(y_test, gini_pred, pos_label='Endometrium'))
print("Recall:", metrics.recall_score(y_test, gini_pred, pos_label='Endometrium'))
print("ROC AUC Score:", roc_auc_score(y_test, gini_tree.predict_proba(X_test)[:, 1]))


# Defining the parameter grid
parameters = [{"min_samples_leaf":[2,4,6,8,10]}]

# Tuning decision tree classifier with entropy criterion
tuned_dtc_entropy = model_selection.GridSearchCV(entropy_tree, parameters, scoring="roc_auc", cv=10)
tuned_dtc_entropy.fit(X_train, y_train)
print("entropy best parameters:", tuned_dtc_entropy.best_params_)

# Tuning decision tree classifier with gini criterion
tuned_dtc_gini = model_selection.GridSearchCV(gini_tree, parameters, scoring="roc_auc", cv=10)
tuned_dtc_gini.fit(X_train, y_train)
print("gini best parameters:", tuned_dtc_gini.best_params_)

# ***************************************** second data set analysis ************************************************


titanic_data = datasets.fetch_openml(data_id=1130)

# creating decision trees based on criterion of entropy and gini
entropy_tree = DecisionTreeClassifier(criterion="entropy")
gini_tree = DecisionTreeClassifier()

# training the decision trees
entropy_tree.fit(titanic_data.data, titanic_data.target)
gini_tree.fit(titanic_data.data, titanic_data.target)


entropy_pred = entropy_tree.predict(titanic_data.data)
gini_pred = gini_tree.predict(titanic_data.data)



pred_prob_entropy = entropy_tree.predict_proba(titanic_data.data)
pred_prob_gini = gini_tree.predict_proba(titanic_data.data)


dtc_entropy = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=3)
y_scores_entropy = model_selection.cross_val_predict(dtc_entropy, titanic_data.data, titanic_data.target, method="predict_proba", cv=10)

dtc_gini = DecisionTreeClassifier(criterion="gini", min_samples_leaf=3)
y_scores_gini = model_selection.cross_val_predict(dtc_gini, titanic_data.data, titanic_data.target, method="predict_proba", cv=10)

from sklearn.metrics import roc_curve
fpr, tpr, th = roc_curve(titanic_data.target,y_scores_entropy[:,1],pos_label="Lung")

from matplotlib import pyplot as plt
plt.xlabel("1 - Specificity")
plt.ylabel("Sensitivity")
plt.xlim(0,2)
plt.ylim(0,2)
plt.plot(fpr,tpr,label="Decision Tree-entropy")
plt.legend()
plt.show()

fpr, tpr, th = roc_curve(titanic_data.target,y_scores_gini[:,1],pos_label="Lung")

from matplotlib import pyplot as plt
plt.xlabel("1 - Specificity")
plt.ylabel("Sensitivity")
plt.xlim(0,2)
plt.ylim(0,2)
plt.plot(fpr,tpr,label="Decision Tree-gini")
plt.legend()
plt.show()

from sklearn.metrics import roc_auc_score


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(titanic_data.data, titanic_data.target, test_size=0.2, random_state=42)
entropy_tree.fit(X_train, y_train)
gini_tree.fit(X_train, y_train)
entropy_pred = entropy_tree.predict(X_test)
gini_pred = gini_tree.predict(X_test)


# evaluation of the trees
print("Entropy Tree:")
print("Accuracy:", metrics.accuracy_score(y_test, entropy_pred))
print("F1 Score:", metrics.f1_score(y_test, entropy_pred, pos_label='Lung'))
print("Precision:", metrics.precision_score(y_test, entropy_pred, pos_label='Lung'))
print("Recall:", metrics.recall_score(y_test, entropy_pred, pos_label='Lung'))
print("ROC AUC Score:", roc_auc_score(y_test, entropy_tree.predict_proba(X_test)[:, 1]))

print("\nGini Tree:")
print("Accuracy:", metrics.accuracy_score(y_test, gini_pred))
print("F1 Score:", metrics.f1_score(y_test, gini_pred, pos_label='Lung'))
print("Precision:", metrics.precision_score(y_test, gini_pred, pos_label='Lung'))
print("Recall:", metrics.recall_score(y_test, gini_pred, pos_label='Lung'))
print("ROC AUC Score:", roc_auc_score(y_test, gini_tree.predict_proba(X_test)[:, 1]))


# Defining the parameter grid
parameters = [{"min_samples_leaf":[2,4,6,8,10]}]

# Tuning decision tree classifier with entropy criterion
tuned_dtc_entropy = model_selection.GridSearchCV(entropy_tree, parameters, scoring="roc_auc", cv=10)
tuned_dtc_entropy.fit(X_train, y_train)
print("entropy best parameters:", tuned_dtc_entropy.best_params_)

# Tuning decision tree classifier with gini criterion
tuned_dtc_gini = model_selection.GridSearchCV(gini_tree, parameters, scoring="roc_auc", cv=10)
tuned_dtc_gini.fit(X_train, y_train)
print("gini best parameters:", tuned_dtc_gini.best_params_)
