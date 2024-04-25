# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Upload the dataset and check for any null or sum values using .isnull() and .sum() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Predict the values of array.

5.Calculate the accuracy by importing the required modules from sklearn.

6.Apply new unknown values. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Tirupathi Jayadeep
RegisterNumber:  212223240169
*/

import pandas as pd 
data = pd.read_csv('Employee.csv')
data.head()

data.info()

data.isnull().sum()

data["left"].value_counts

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
# Head()
![image](https://github.com/23004426/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144979327/88eda77e-9bc1-4b2f-a350-307396ba8f80)

# info()
![image](https://github.com/23004426/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144979327/31158613-1a90-4a93-9c07-65282fd16956)

# isnull().sum()
![image](https://github.com/23004426/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144979327/975a4935-571e-45f9-9413-c77a85968f4f)

# Left value counts
![image](https://github.com/23004426/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144979327/294fdfea-0700-4380-aad7-69dbc559fbd5)

# Head()(After transform of salary)
![image](https://github.com/23004426/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144979327/f5f32118-8db8-4324-bfb4-805a51a48651)

# After removing left and departments columns
![image](https://github.com/23004426/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144979327/2f9238a6-a9a9-4d21-8617-795e8054cac6)

# accuracy
![image](https://github.com/23004426/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144979327/8b017485-1503-4601-a67f-064347835979)

# prediction
![image](https://github.com/23004426/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144979327/08a9e819-7f80-4d93-8118-b43146187660)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
