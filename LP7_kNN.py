import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

class KNN:
  def __init__(self,k=3):
    self.k=k
    self.X_train=None
    self.y_train=None

  def fit(self,X,y):
    self.X_train=X
    self.y_train=y

  def predict_single(self,x):
    distances=[np.sqrt(np.sum((x-x_train)**2)) for x_train in self.X_train]
    kn_indices=np.argsort(distances)[:self.k]
    kn_classes=self.y_train[kn_indices]
    return np.argmax(np.bincount(kn_classes))

  def predict(self,X_test):
    return [self.predict_single(x_test) for x_test in X_test]


data=pd.read_csv('/content/iris_csv (1).csv')
data.head()

X=data.iloc[:,:-1]
X=np.float64(X)
y=data.iloc[:,-1]
#y.unique()
y=np.where(y=='Iris-setosa',1,np.where(y=='Iris-versicolor',2,3))

model=KNN()

X_train,X_test,y_train,y_test=train_test_split(X,y)

model.fit(X_train,y_train)
y_pred=model.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred,average='macro')
recall=recall_score(y_test,y_pred,average='macro')
f1=f1_score(y_test,y_pred,average='macro')
confMatrix=confusion_matrix(y_test,y_pred)

print(accuracy)
print(precision)
print(recall)
print(f1)
print(confMatrix)
