# Credit-Card-Fraud-detection

The code you're working with is for building a machine learning model to detect fraudulent transactions in a credit card transaction dataset. Here's a breakdown of what each part of the code does:

### 1. **Importing Required Libraries**
   - `numpy`, `pandas`, `sklearn`, `scipy`, `matplotlib`, `seaborn`: These libraries are used for data manipulation, machine learning, statistical analysis, and plotting.
   - `IsolationForest`, `LocalOutlierFactor`, `OneClassSVM`: These are various machine learning algorithms for outlier detection, although they aren’t directly used in the final part of your code.
   - `LogisticRegression`, `RandomForestClassifier`: These are supervised learning algorithms used for classification tasks.
   - `joblib`: Used for saving and loading models (though not used in your code so far).
   - `google.colab.drive`: This is used for accessing files stored in Google Drive when running the code on Google Colab.

### 2. **Loading the Dataset**
   - The dataset is loaded from Google Drive using `pd.read_csv` into a DataFrame. The dataset contains credit card transactions, with a `Class` column indicating whether a transaction is legitimate (0) or fraudulent (1).

### 3. **Exploring the Data**
   - The `info()` method gives basic information about the dataset, such as the number of entries and column types.
   - `head()` and `tail()` are used to inspect the first and last 5 rows of the dataset, respectively.
   - The data is split into `legit` and `fraud` subsets based on the `Class` column.

### 4. **Balancing the Dataset**
   - The dataset is highly imbalanced (more legitimate transactions than fraudulent ones). To address this, a sample of legitimate transactions is taken to match the number of fraudulent transactions, ensuring balanced training data.

### 5. **Splitting Data into Features and Labels**
   - The `X` and `Y` variables are created, where:
     - `X` contains the features (all columns except `Class`).
     - `Y` contains the target variable (`Class`).
   - The data is then split into training and test sets using `train_test_split`, ensuring that the distribution of `Class` is maintained in both training and testing sets (`stratify=Y`).

### 6. **Training the Models**
   - **Logistic Regression**: This is a simple classification algorithm that predicts the probability of a binary outcome (fraud or not fraud in this case). It is trained on the `X_train` and `Y_train` data.
   - **Random Forest Classifier**: This is an ensemble method that uses multiple decision trees to make predictions. It is trained in a similar manner to the logistic regression model.

### 7. **Model Evaluation**
   - The models are evaluated on both the training and test sets using accuracy:
     - `accuracy_score` compares the predicted values to the true labels and computes the percentage of correct predictions.

### 8. **User Input for Predictions**
   - After training the models, the script prompts the user to input a set of transaction features (like `V1`, `V2`, ..., `V28` and `Amount`).
   - This user input is then converted into a DataFrame that matches the structure of the training data.
   - The trained model (in this case, `RandomForestClassifier`) makes a prediction on the user's input. If the prediction is `0`, the transaction is classified as normal, and if it's `1`, the transaction is classified as fraudulent.

### 9. **Prediction Output**
   - Depending on the model's prediction, the script outputs whether the transaction is classified as "normal" or "fraudulent".

### Key Considerations:
- **Imbalanced Data**: Credit card fraud detection is typically an imbalanced classification problem, where fraudulent transactions are much fewer than legitimate ones. Balancing the data, as done here, is crucial for improving model performance.
- **Feature Engineering**: The features `V1` to `V28` are anonymized, which suggests that the dataset has been pre-processed or sanitized for privacy. Further preprocessing (e.g., scaling or normalization) might be needed depending on the model used.
- **Evaluation Metrics**: In addition to accuracy, it would be beneficial to evaluate models using other metrics, such as precision, recall, and F1-score, especially in imbalanced datasets, to ensure the model is correctly identifying fraudulent transactions.

This approach demonstrates a typical machine learning pipeline for fraud detection: loading and preparing the data, training models, evaluating performance, and making predictions.
