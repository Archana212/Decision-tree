import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV

#load the iris dataset
iris =load_iris()
X,y=iris.data,iris.target

#split the data into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#scale the features using standardscaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#create design tree classifier with ID3 algorithm
clf=DecisionTreeClassifier(random_state=42)

#define hyperparameters and their possible values for tuning
param_grid={
    'criterion':['gini','entropy'],
    'max_depth':[None,5,10,15],
    'min_samples_split':[2,5,10],
    'min_samples_leaf':[1,2,4]
    }

#use GridSearchCV to find the best hyperparameters
grid_search=GridSearchCV(clf,param_grid,cv=5)
grid_search.fit(X_train,y_train)

#get the best hyperparameters
best_params=grid_search.best_params_
print("Best heperparameters:",best_params)

best_clf=DecisionTreeClassifier(**best_params,random_state=42)

best_clf.fit(X_train,y_train)

y_pred=best_clf.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)

target_names=iris.target_names
print("Classification Report:")
print(classification_report(y_test,y_pred,target_names=target_names))

#visualization
plt.figure(figsize=(6,4))
sns.countplot(x=y,palette='coolwarm')
plt.xticks(ticks=np.unique(y),labels=target_names,rotation=45)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.show()

conf_matrix=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap='coolwarm',xticklabels=target_names,yticklabels=target_names)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix')
plt.show()




