# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the Iris dataset

2.Split the dataset into training and testing sets

3.Train the SGD Classifier model

4.Predict the Iris species

## Program:
```

Program to implement the prediction of iris species using SGD Classifier.
Developed by: T.Goshanrajan
RegisterNumber:  212225040098
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# Create SGD Classifier model
model = SGDClassifier(max_iter=1000, tol=1e-3)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict species name
for i in range(5):
    print("Predicted Species:", iris.target_names[y_pred[i]])
```

## Output:
<img width="503" height="160" alt="image" src="https://github.com/user-attachments/assets/8972775e-e532-4f7f-86b4-9916cf7bba8c" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
