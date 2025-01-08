%load_ext autoreload
%autoreload 2

# Download the data from your GitHub repository
!wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv

import pandas as pd

df = pd.read_csv('parkinsons.csv')
df.head()
x = df[['MDVP:Fo(Hz)', 'MDVP:Flo(Hz)']]
y = df['status']
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')  
x = imputer.fit_transform(x)

# Scale the data
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)
from sklearn.neighbors import KNeighborsClassifier

# Choose a KNN model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)

# Evaluate the model's accuracy on the test set
accuracy = model.score(x_test, y_test)
print(f"Accuracy: {accuracy}")

if accuracy < 0.8:
    print("Warning: Accuracy is below the target of 0.8. Consider trying different features, models, or hyperparameters.")
import joblib
joblib.dump(model, 'parkinsons_model.joblib')
print("Model saved as parkinsons_model.joblib")

# Create the config.yaml file
config = {
    'selected_features': ['MDVP:Fo(Hz)', 'MDVP:Flo(Hz)'],
    'path': 'parkinsons_model.joblib'}







