# NN-EXP-6-Heart-attack-prediction-using-MLP
## Aim:
To construct a  Multi-Layer Perceptron to predict heart attack using Python
## Algorithm:
Step 1:Import the required libraries: numpy, pandas, MLPClassifier, train_test_split, StandardScaler, accuracy_score, and matplotlib.pyplot.<br>
Step 2:Load the heart disease dataset from a file using pd.read_csv().<br>
Step 3:Separate the features and labels from the dataset using data.iloc values for features (X) and data.iloc[:, -1].values for labels (y).<br>
Step 4:Split the dataset into training and testing sets using train_test_split().<br>
Step 5:Normalize the feature data using StandardScaler() to scale the features to have zero mean and unit variance.<br>
Step 6:Create an MLPClassifier model with desired architecture and hyperparameters, such as hidden_layer_sizes, max_iter, and random_state.<br>
Step 7:Train the MLP model on the training data using mlp.fit(X_train, y_train). The model adjusts its weights and biases iteratively to minimize the training loss.<br>
Step 8:Make predictions on the testing set using mlp.predict(X_test).<br>
Step 9:Evaluate the model's accuracy by comparing the predicted labels (y_pred) with the actual labels (y_test) using accuracy_score().<br>
Step 10:Print the accuracy of the model.<br>
Step 11:Plot the error convergence during training using plt.plot() and plt.show().<br>

## Program:
```
# Step 1: Import the required libraries
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Step 2: Load the heart disease dataset
data = pd.read_csv("/content/drive/MyDrive/Neural Networks Dataset/heart.csv")

# Step 3: Separate features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Normalize the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Create an MLPClassifier model
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

# Step 7: Train the MLP model
mlp.fit(X_train, y_train)

# Step 8: Make predictions on the testing set
y_pred = mlp.predict(X_test)

# Step 9: Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)

# Step 10: Print the accuracy
print(f"Accuracy: {accuracy}")

# Step 11: Plot the error convergence during training
plt.plot(mlp.loss_curve_)
plt.xlabel('Number of Iterations')
plt.ylabel('Loss')
plt.title('Training Error Convergence')
plt.show()
```
## Output:
```
Accuracy: 0.9853658536585366
```
![image](https://github.com/Siddarthan999/NN-EXP-6-Heart-attack-prediction-using-MLP/assets/91734840/28eee692-4813-42fb-867c-f264419c5aa8)

## Result:
Thus, an ANN with MLP is constructed and trained to predict the heart attack using python.
     
