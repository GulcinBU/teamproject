'''
from sklearn.svm import SVC
### Ensemble - Bagging , boosting and stacking
### hyperparameter tuning
'''
'''
- Accuracy score
- Confusion matrix
- Classification report
- Data prediction
- Table of accuracies
'''

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import fileuploading

## Split data in test and train
y = fileuploading.pred_target
X = fileuploading.features

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=fileuploading.trainingsize,random_state=0)

## Linear regression
model_lin = LinearRegression()
model_lin.fit(X_train, y_train)
y_pred = model_lin.predict(X_test)
confusion_matrix = confusion_matrix(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)
accuracy_score = accuracy_score(y_test, y_pred)
print("Confusion Matrix:", confusion_matrix)
print("Classification Report:", classification_report)
print("Accuracy Score:", accuracy_score)

## Logistic regression
model_log = LogisticRegression()
model_log.fit(X_train,y_train)
y_pred = model_log.predict(X_test)
confusion_matrix = confusion_matrix(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)
accuracy_score = accuracy_score(y_test, y_pred)
print("Confusion Matrix:", confusion_matrix)
print("Classification Report:", classification_report)
print("Accuracy Score:", accuracy_score)

## KNN
model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(x_train,y_train)
y_pred = model_knn.predict(x_test)
confusion_matrix = confusion_matrix(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)
accuracy_score = accuracy_score(y_test, y_pred)
print("Confusion Matrix:", confusion_matrix)
print("Classification Report:", classification_report)
print("Accuracy Score:", accuracy_score)

## Gaussian Naive Bayes
model_nb = GaussianNB()
model_nb.fit(X_train,y_train)
y_pred_nb = model_nb.predict(X_test)
confusion_matrix = confusion_matrix(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)
accuracy_score = accuracy_score(y_test, y_pred)
print("Confusion Matrix:", confusion_matrix)
print("Classification Report:", classification_report)
print("Accuracy Score:", accuracy_score)

## SVM


## Decision trees
from sklearn.tree import DecisionTreeClassifier

## Random forest
from sklearn.ensemble import RandomForestClassifier
