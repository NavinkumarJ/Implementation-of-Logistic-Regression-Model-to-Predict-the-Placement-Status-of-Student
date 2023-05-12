# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary Libraries.

2. Read the CSV file using pd.read_csv.

3. Print the data status using LabelEncoder().

4. Find the accuracy , confusion ,classification report .

5. End the Program.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: NAVIN KUMAR J
RegisterNumber:  212222240071
*/

import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
### 1.Placement data
![Output1](https://github.com/SanthoshUthiraKumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477975/9bfcb977-5842-4bbd-91ea-0e4b8297211a)

### 2.Salary data
![Output2](https://github.com/SanthoshUthiraKumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477975/04b238c7-eb12-4241-9a99-b0da4b106a82)

### 3.Checking the null() function
![Output3](https://github.com/SanthoshUthiraKumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477975/1ed70348-1131-4b61-bff6-879d75a7d0d4)

### 4. Data Duplicate
![Output4](https://github.com/SanthoshUthiraKumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477975/b299d2b6-103a-4013-ba98-910c43d37889)

### 5. Print data
![Output5](https://github.com/SanthoshUthiraKumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477975/955e2990-5893-4431-b246-164c20b3a040)

![Output6](https://github.com/SanthoshUthiraKumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477975/78c826a9-af0a-4c68-bcef-dc1517e0bf84)

### 6. Data-status
![Output7](https://github.com/SanthoshUthiraKumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477975/0ec67fbb-dd2d-4eee-92f7-3bbb7dc4d951)

### 7. y_prediction array
![Output8](https://github.com/SanthoshUthiraKumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477975/a374c554-5354-4625-8709-0eb1e743b290)

### 8.Accuracy value
![Output9](https://github.com/SanthoshUthiraKumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477975/de8a4bc1-7188-4ed0-a127-c56fce26cd68)

### 9. Confusion array
![Output10](https://github.com/SanthoshUthiraKumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477975/3b3ef7d4-780f-466b-9274-10dc85912b67)

### 10. Classification report
![Output11](https://github.com/SanthoshUthiraKumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477975/b2757951-8ee1-49e9-ab85-29c6774e08fb)

### 11.Prediction of LR
![Output12](https://github.com/SanthoshUthiraKumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477975/a032c6bd-f8f1-43e6-bcd7-d2a152eae10d)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
