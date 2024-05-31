from sklearn.preprocessing import OneHotEncoder
from sklearn import datasets
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

lis = datasets.fetch_openml(data_id=301)
idx_range = [x for x in range(72)]
ct = ColumnTransformer([("encoder", OneHotEncoder(sparse_output=False), idx_range)], remainder= "passthrough")
new_data = ct.fit_transform(lis.data)
# print(ct.get_feature_names_out())
# print(type(new_data))
lis_new_data = pd.DataFrame(new_data, columns=ct.get_feature_names_out(), index=lis.data.index)
# print(lis_new_data.info())
lr = LinearRegression()
# ct = ColumnTransformer([("encoder", OneHotEncoder(sparse_output=False),[lis.target])], remainder= "passthrough")
# target = ct.fit_transform(lis.target)


#print(len(lis.target))

# ct = ColumnTransformer([("encoder", OneHotEncoder(sparse_output=False),[])], remainder= "passthrough")
# new_lis_target = ct.fit_transform(lis.target)

lis_target_reshaped = lis.target.values.reshape(-1, 1)

ct = ColumnTransformer([
    ("encoder", OneHotEncoder(sparse_output=False), [0])  # assuming lis.target is the only column to transform
], remainder="passthrough")

new_lis_target = ct.fit_transform(lis_target_reshaped)
scores = cross_validate(lr, new_data, new_lis_target, cv=10, scoring="neg_root_mean_squared_error")
print(scores["test_score"])

parameter_grid = [{"n_neighbors":[5]}]
tuned_knn_5 = GridSearchCV(KNeighborsRegressor(), parameter_grid, scoring="neg_root_mean_squared_error", cv=10)
scores_knn_5 = cross_validate(tuned_knn_5, new_data, new_lis_target, cv = 10, scoring="neg_root_mean_squared_error")
print(scores_knn_5)

from sklearn.model_selection import learning_curve
training_examples, training_scores, test_scores = learning_curve(tuned_knn_5, new_data, new_lis_target, train_sizes=[0.1,0.2,0.4,0.6,0.8,1.0],cv=10,scoring="neg_root_mean_squared_error")

test_scores = 0-test_scores

parameter_grid = [{"n_neighbors":[7]}]
tuned_knn_7 = GridSearchCV(KNeighborsRegressor(), parameter_grid, scoring="neg_root_mean_squared_error", cv=10)
scores_knn_7 = cross_validate(tuned_knn_7, new_data, new_lis_target, cv = 10, scoring="neg_root_mean_squared_error")
print(scores_knn_7)

from sklearn.model_selection import learning_curve
training_examples1, training_scores1, test_scores1 = learning_curve(tuned_knn_7, new_data, new_lis_target, train_sizes=[0.1,0.2,0.4,0.6,0.8,1.0],cv=10,scoring="neg_root_mean_squared_error")

parameter_grid = [{"n_neighbors":[3]}]
tuned_knn_3 = GridSearchCV(KNeighborsRegressor(), parameter_grid, scoring="neg_root_mean_squared_error", cv=10)
scores_knn_3= cross_validate(tuned_knn_3, new_data, new_lis_target, cv = 10, scoring="neg_root_mean_squared_error")
print(scores_knn_3)

training_examples2, training_scores2, test_scores2 = learning_curve(tuned_knn_3, new_data, new_lis_target, train_sizes=[0.1,0.2,0.4,0.6,0.8,1.0],cv=10,scoring="neg_root_mean_squared_error")

test_scores1 = 0-test_scores1
test_scores2 = 0-test_scores2
import numpy as np
test_mean = np.mean(test_scores, axis=1)
test_mean1 = np.mean(test_scores1, axis=1)
test_mean2 = np.mean(test_scores2, axis=1)

import matplotlib.pyplot as plt
# Plot learning curves
plt.plot(training_examples.squeeze(), test_mean, label='rmse k=5')
plt.plot(training_examples.squeeze(), test_mean1, label='rmse k=7')
plt.plot(training_examples.squeeze(), test_mean2, label='rmse k=3')



# Add labels and title
plt.xlabel('Number of training examples')
plt.ylabel('RMSE')
plt.title('Learning Curve')
plt.legend()
plt.show()


training_examples, training_scores_lr, test_scores_lr = learning_curve(lr, new_data, new_lis_target, train_sizes=[0.1,0.2,0.4,0.6,0.8,1.0],cv=10,scoring="neg_root_mean_squared_error")
test_scores_lr = 0-test_scores_lr


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, cross_validate, learning_curve
import numpy as np

# Define parameter grid for Decision Tree
parameter_grid = [{"max_depth": [5]}]

# Initialize GridSearchCV
tuned_dt_3 = GridSearchCV(DecisionTreeRegressor(), parameter_grid, scoring="neg_root_mean_squared_error", cv=10)

# Perform cross-validation
scores_dt_3 = cross_validate(tuned_dt_3, new_data, new_lis_target, cv=10, scoring="neg_root_mean_squared_error")
print(scores_dt_3)

# Learning curve
training_sizes = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
training_examples_dt, training_scores_dt, test_scores_dt = learning_curve(tuned_dt_3, new_data, new_lis_target, train_sizes=training_sizes, cv=10, scoring="neg_root_mean_squared_error")


test_scores_dt = 0-test_scores_dt

test_mean_dt = np.mean(test_scores_dt, axis=1)

plt.plot([0]+training_examples.squeeze(),[0]+test_mean1, label='KNN k=7')
plt.plot([0]+training_examples.squeeze(),[0]+test_mean_lr, label='lr')
plt.plot([0]+training_examples.squeeze(),[0]+test_mean_dt, label='dt')
plt.legend(loc='best')
plt.xlabel('Number of training examples')
plt.ylabel('RMSE')
plt.title('Learning Curve')
plt.legend()
plt.show()