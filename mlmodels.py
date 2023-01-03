'''
from sklearn.svm import SVC
### Ensemble - Bagging , boosting and stacking
### hyperparameter tuning
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

## Logistic regression
model = LogisticRegression()
model.fit(X_train,y_train)

X_new = [2,100,60,20,30,30,0.7,25] ## new user input
y_new = model.predict([X_new])
print(y_new)

y_pred = model.predict(X_test)
cnfm = confusion_matrix(y_test,y_pred)
print(cnfm)

acc = accuracy_score(y_test,y_pred)
print(acc)

clf = classification_report(y_test,y_pred)
print('Metrics for logistic regression')
print(clf)

model_nb = GaussianNB()
model_nb.fit(X_train,y_train)
print('Metrics for GaussianNB')

y_pred_nb = model_nb.predict(X_test)
clf_nb = classification_report(y_test,y_pred_nb)
print(clf_nb)

## KNN
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
clf = classification_report(y_test, y_pred)
print(clf)

## Naive bayes

## SVM

## Decision trees
from sklearn.tree import DecisionTreeClassifier

## Random forest
from sklearn.ensemble import RandomForestClassifier
