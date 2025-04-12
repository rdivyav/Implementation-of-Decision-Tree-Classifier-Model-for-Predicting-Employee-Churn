# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the Dataset
2. Data Preprocessing
3. Feature and Target Selection
4. Split the Data into Training and Testing Sets
5. Build and Train the Decision Tree Model
6. Make Predictions
7. Evaluate the Model

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Divya R V
RegisterNumber:212223100005  
*/
```
```
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head() #no departments and no left
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
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()

```

## Output:
![decision tree classifier model](sam.png)

![Screenshot 2025-04-12 040310](https://github.com/user-attachments/assets/1c4eb0b4-4ae1-4670-a8db-52cac21fb82e)

![Screenshot 2025-04-12 040330](https://github.com/user-attachments/assets/b64da0a9-6b7f-4687-ad65-6433d15aea59)

![Screenshot 2025-04-12 040337](https://github.com/user-attachments/assets/2d2e07b4-ddbb-46fb-8c69-f6f3dc1af113)

![Screenshot 2025-04-12 040346](https://github.com/user-attachments/assets/04484f52-b90f-4583-93ca-8caa48b46941)

![Screenshot 2025-04-12 040357](https://github.com/user-attachments/assets/b2209008-6019-41ec-8097-67d8d6c6c222)

![Screenshot 2025-04-12 040409](https://github.com/user-attachments/assets/e649fff1-8b84-4871-b045-e3fb7c08f262)

![Screenshot 2025-04-12 040427](https://github.com/user-attachments/assets/d55ff8ee-22b7-43b4-8924-c67faf151585)

![Screenshot 2025-04-12 040437](https://github.com/user-attachments/assets/669dfe9a-1b6e-4245-aa77-573ec57294d3)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
