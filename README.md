# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary packages using import statement. 
2. Read the given csv file using read_csv() method and print the number 
of contents to be displayed using df.head(). 
3. Import KMeans and use for loop to cluster the data.
4. Predict the cluster and plot data graphs.
5. Print the outputs and end the program

## Program:
```

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: GOWTHAM S
RegisterNumber: 2305002008 

```
```python
import pandas as pd 
import numpy as np 
data=pd.read_csv("Employee.csv") 
data.head() 
data.info() 
data.isnull().sum() 
data["left"].value_counts() 
from sklearn.preprocessing import LabelEncoder 
l=LabelEncoder() 
data["salary"]=l.fit_transform(data["salary"]) 
data.head() 
data["Departments "]=l.fit_transform(data["Departments "]) 
data.head() 
data.info() 
data.shape 
x=data[['satisfaction_level','last_evaluation','number_project','average_montly
 _hours','time_spend_company','Work_accident','promotion_last_5years','Depa
 rtments ','salary']] 
x.head() 
x.shape 
x.info() 
y=data['left'] 
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=
 100) 
from sklearn.tree import DecisionTreeClassifier 
dt=DecisionTreeClassifier(criterion="entropy") 
dt.fit(x_train,y_train) 
y_pred=dt.predict(x_test) 
print(y_pred) 
from sklearn import metrics 
accuracy=metrics.accuracy_score(y_test,y_pred) 
print("Accuracy = ",accuracy) 
dt.predict([[0.5,0.8,9,260,6,0,1,2,1]])
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
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
