import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

class NaiveBayes:
  def __init__(self):
    self.class_probs={} #P(class)
    self.feature_probs={} #P(feature|class)

  def fit(self,X,y):
    self.class_probs=y.value_counts(normalize=True).to_dict()

    for cls in self.class_probs:
      self.feature_probs[cls]={}
      class_data=X[y==cls]

      for feature in X.columns:
        self.feature_probs[cls][feature]=class_data[feature].value_counts(normalize=True).to_dict()

  def predict(self,X):
    predictions=[]
    for _,row in X.iterrows():
      best_class=None
      highest_prob=-1

      for cls in self.class_probs:
        prob=self.class_probs[cls]

        for feature in X.columns:
          feature_value=row[feature]
          if feature_value in self.feature_probs[cls][feature]:
            prob=prob*self.feature_probs[cls][feature][feature_value]
          #THIS IS VERY IMPORTANTTTTT
          else:
            prob=0
            break

        if prob > highest_prob:
          highest_prob=prob
          best_class=cls

      predictions.append(best_class)

    return predictions


data=pd.read_csv('/content/Social_Network_Ads.csv')
data.head()

#data['Gender'] is important for np.where
data['Gender']=np.where(data['Gender']=='Male',1,0)
# DO NOT PUT .VALUES
X=data.iloc[:,1:-1]
y=data['Purchased']

model=NaiveBayes()
X_train,X_test,y_train,y_test=train_test_split(X,y)
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
