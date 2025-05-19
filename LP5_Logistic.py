import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix


class LogisticRegression:
  def __init__(self,learning_rate=0.01,iterations=1000):
    self.learning_rate=learning_rate
    self.iterations=iterations
    self.weights=None

  def add_intercept(self,X):
    intercept=np.ones((X.shape[0],1))
    return np.concatenate((intercept,X),axis=1)

  def sigmoid(self,z):
    return 1/(1+np.exp(-z))


  def fit(self,X,y):
    X=self.add_intercept(X)
    self.weights=np.zeros(X.shape[1])

    for _ in range(self.iterations):
      z=np.dot(X,self.weights)
      h=self.sigmoid(z)
      gradient=np.dot(X.T,(h-y))/len(y)
      self.weights-=self.learning_rate*gradient

  def predict(self,X,threshold=0.5):
    X=self.add_intercept(X)
    return self.sigmoid(np.dot(X,self.weights)) >= threshold


data=pd.read_csv("/content/Breastcancer_data.csv")
data.head()

#.values is not??? important
X=data.iloc[:,2:-1]
X=np.float64(X)

y=data.iloc[:,1]
y=np.where(y=='M',1,0)

X_train,X_test,y_train,y_test=train_test_split(X,y)

model=LogisticRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred)
recall=recall_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)
confMatrix=confusion_matrix(y_test,y_pred)

print(accuracy)
print(precision)
print(recall)
print(f1)
print(confMatrix)
