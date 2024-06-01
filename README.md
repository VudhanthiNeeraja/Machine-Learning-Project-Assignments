These assignment projects include a series of reports and corresponding Python scripts that document and implement a comprehensive machine learning workflow. 
<br>
__Project Assignment - 1:__ <br>
This assignment focuses on comparing decision tree classifiers created using two different criteria: entropy and Gini index. <br>
The goal is to evaluate and compare the performance of these classifiers on multiple datasets by plotting ROC curves and computing AUC values. <br>
The evaluation will be done using 10-fold cross-validation, and a parameter search will be conducted on the min_samples_leaf parameter using GridSearchCV with at least five different values. <br>
The entire process, including data loading, model training, and evaluation, is implemented in a self-contained Python program. <br>
The results, including ROC curves and AUC values, will be documented in a report along with a brief description of each dataset and a discussion of the findings. <br>
<br>
__Project Assignment - 2:__ <br>
This assignment aims to compare the performance of different regression models using root mean square error (RMSE) as the evaluation metric. The tasks are divided into two parts: <br>
<br>
Task 1: Compare the RMSE of k-nearest neighbor (KNN) regressors obtained using three different k values by plotting their learning curves from 10-fold cross-validation. A table will display the RMSE at the last point of the learning curves, representing the maximum training data. <br>
<br>
Task 2: Compare the RMSE of a tuned KNN regressor (with k parameter tuned using at least five values), a decision tree regressor (with min_samples_leaf parameter tuned using at least five values), and a linear regression model. Their learning curves will be plotted using 10-fold cross-validation, and the RMSE at the last point of the learning curves will be shown in a table. <br>

The assignment includes a self-contained Python program that loads the data, trains the models, performs cross-validation, plots learning curves, and calculates RMSE values. The results and discussion of both tasks will be documented in a report, providing insights into the models' performance and their comparison. <br>

__Project Assignment - 3:__ <br>
This assignment involves building and evaluating neural network models using Keras for both regression and classification tasks. <br>
<br>
Task 1: Regression <br>
For the regression task, four different neural network models will be developed using Keras. These models will vary in terms of the number of hidden nodes, the number of hidden layers, and the activation functions used in the dense layers. Only input and dense layers will be employed. For each model, training errors and validation errors will be plotted against the number of epochs. If the training error does not plateau, the number of epochs will be increased accordingly. The results will be presented in four graphs, one for each model, and a table will show the minimum validation error obtained by each model. A discussion will follow, analyzing and comparing the results. <br>
<br>
Task 2: Classification <br>
For the classification task, another set of four neural network models will be built using Keras. Similar to Task 1, these models will vary in terms of architecture, including hidden nodes, hidden layers, and activation functions. The models will be evaluated based on training accuracy and validation accuracy, plotted against the number of epochs. If the training accuracy does not plateau, additional epochs will be run. The results will be presented in four graphs, one for each model, and a table will display the maximum validation accuracy achieved by each model. The report will conclude with a discussion analyzing the performance of the models and comparing their effectiveness. <br>

__Project Assignment - 4:__ <br>
This assignment focuses on developing and evaluating Convolutional Neural Network (CNN) architectures for image classification, comparing custom-built models with a fine-tuned pre-trained model, and performing error analysis. <br>
<br>
Task 1: Custom CNN Architectures
Design and train two different CNN architectures, varying layers, filter sizes, and dropout. Train each model until the training accuracy plateaus, then test their performance on the test dataset. Compare the test accuracies, and generate a confusion matrix for the better-performing model. Save the best model.<br>
<br>
Task 2: Fine-tuning a Pre-trained CNN
Select and fine-tune a pre-trained CNN model by adding custom layers on top. Train this model until the training accuracy plateaus and test its performance on the test dataset. Compare its test accuracy with the better-performing model from Task 1 and generate a confusion matrix for the fine-tuned model.<br>
<br>
Task 3: Error Analysis
Analyze the performance of the better model from Task 1 by examining 10 test images where the model made incorrect predictions. Identify potential reasons for these errors and check if the fine-tuned model from Task 2 improves on these cases.
