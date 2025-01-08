import numpy as np
import pandas as pd
import sklearn
import scipy
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn .ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams
rcParams['figure.figsize'] = 14,8
RANDOM_SEED = 42
LABELS = ['normal','fraud']
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from google.colab import drive
drive.mount('/content/drive')
#loading the dataset to a pandas DataFrame
credit_card_data = pd.read_csv('/content/drive/MyDrive/creditcard.csv')
#dataset information
credit_card_data.info()
# First 5 rows of dataset
credit_card_data.head()
# last 5 rows of dataset
credit_card_data.tail()
#separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]
print(legit.shape)
print(fraud.shape)
legit_sample = legit.sample(n=492)
new_dataset = pd.concat([legit_sample, fraud], axis=0)
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape,X_train.shape,X_test.shape)
model = LogisticRegression()
# training the Logistic Regression Model with Training Data
model.fit(X_train, Y_train)
# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_data_accuracy)
# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on Test Data : ', test_data_accuracy)
from sklearn.ensemble import RandomForestClassifier
# Initialize the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
# Train the model
model.fit(X_train, Y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy:", accuracy)
# prompt: user input and result

# Get user input for transaction features
time = float(input("Enter the transaction time: "))
V1 = float(input("Enter the value for V1: "))
V2 = float(input("Enter the value for V2: "))
V3 = float(input("Enter the value for V3: "))
V4 = float(input("Enter the value for V4: "))
V5 = float(input("Enter the value for V5: "))
V6 = float(input("Enter the value for V6: "))
V7 = float(input("Enter the value for V7: "))
V8 = float(input("Enter the value for V8: "))
V9 = float(input("Enter the value for V9: "))
V10 = float(input("Enter the value for V10: "))
V11 = float(input("Enter the value for V11: "))
V12 = float(input("Enter the value for V12: "))
V13 = float(input("Enter the value for V13: "))
V14 = float(input("Enter the value for V14: "))
V15 = float(input("Enter the value for V15: "))
V16 = float(input("Enter the value for V16: "))
V17 = float(input("Enter the value for V17: "))
V18 = float(input("Enter the value for V18: "))
V19 = float(input("Enter the value for V19: "))
V20 = float(input("Enter the value for V20: "))
V21 = float(input("Enter the value for V21: "))
V22 = float(input("Enter the value for V22: "))
V23 = float(input("Enter the value for V23: "))
V24 = float(input("Enter the value for V24: "))
V25 = float(input("Enter the value for V25: "))
V26 = float(input("Enter the value for V26: "))
V27 = float(input("Enter the value for V27: "))
V28 = float(input("Enter the value for V28: "))
amount = float(input("Enter the transaction amount: "))

# Create a DataFrame with the user input
user_input = pd.DataFrame({
    'Time': [time],
    'V1': [V1], 'V2': [V2],'V3': [V3],'V4': [V4],'V5': [V5],'V6': [V6],'V7': [V7],'V8': [V8],'V9': [V9], 'V10': [V10], # Fixed: Included V10 and removed duplicate V1
    'V11': [V11],'V12': [V12],'V13': [V13],'V14': [V14],'V15': [V15],'V16': [V16],'V17': [V17],'V18': [V18],'V19': [V19],'V20': [V20],
    'V21': [V21],'V22': [V22],'V23': [V23],'V24': [V24],'V25': [V25],'V26': [V26],'V27': [V27],'V28': [V28],

    'Amount': [amount],
    # ... add other features
})

# Make a prediction
prediction = model.predict(user_input)

# Display the result
if prediction[0] == 0:
  print("This transaction is   normal.")
else:
  print("This transaction is  fraudulent!")
