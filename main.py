--2025-01-08 11:28:42--  https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv
Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.110.133, 185.199.111.133, 185.199.108.133, ...
Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.110.133|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 30202 (29K) [text/plain]
Saving to: ‘/content/parkinsons.csv’

/content/parkinsons 100%[===================>]  29.49K  --.-KB/s    in 0.003s  

2025-01-08 11:28:42 (11.3 MB/s) - ‘/content/parkinsons.csv’ saved [30202/30202]

# prompt: load the file as data frame

import pandas as pd
df = pd.read_csv('/content/parkinsons.csv')
print(df.head())

# prompt: Choose two features as inputs for the model.
# Identify one feature to use as the output for the model.

# Choose two features as inputs (X) and one as the output (y)
X = df[['MDVP:Flo(Hz)', 'MDVP:RAP']]  # Example: Using 'MDVP:Fo(Hz)' and 'MDVP:Fhi(Hz)' as input features
y = df['status']  # Example: Using 'status' as the output feature

print(X.head())
print(y.head())

# prompt: Apply the MinMaxScaler to scale the two input columns to a range between 0 and 1.

from sklearn.preprocessing import MinMaxScaler

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the input features (X)
X_scaled = scaler.fit_transform(X)

# Convert the scaled data back to a DataFrame (optional, but recommended)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print(X_scaled.head())

# prompt: Divide the dataset into a training set and a validation set.

from sklearn.model_selection import train_test_split

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42) # Example: 80% training, 20% validation

print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

# prompt: Select a model to train on the data.
# Advice:
# Consider using the model discussed in the paper from the GitHub repository as a reference.

from sklearn.linear_model import LogisticRegression

# Initialize the model (Logistic Regression as an example, based on the paper's suggestion)
model = LogisticRegression()

# prompt: Evaluate the model's accuracy on the test set. Ensure that the accuracy is at least 0.8.

# Train the model
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Evaluate the accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy}")

if accuracy < 0.8:
    print("Accuracy is below the target threshold of 0.8.")
    # You might want to add code here to adjust the model or data
    # For example, try different features or a different model

# prompt: After you are happy with your results, save the model with the .joblib extension and upload it to your GitHub repository main folder.
# Additionally, update the config.yaml file with the list of selected features and the model's joblib file name.

import joblib

# Save the model
joblib.dump(model, 'parkinsons_model.joblib')

# Upload the model to GitHub (replace with your actual GitHub credentials)
!git config --global user.email "your_email@example.com"
!git config --global user.name "Your Name"
!git add parkinsons_model.joblib
!git commit -m "Add trained model file"
!git push origin main

# Create or update config.yaml
with open('config.yaml', 'w') as f:
  f.write("selected_features: ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)']\n")
  f.write("path: 'parkinsons_model.joblib'\n")

!git add config.yaml
!git commit -m "Update config.yaml with features and model path"
!git push origin main
