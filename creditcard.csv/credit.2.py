import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from geopy.distance import geodesic

# Step 1: Load the dataset
df = pd.read_csv(r"C:\Users\gowri\OneDrive\Desktop\me\eng\creditcard\creditcard.csv\creditcard.csv")

# Step 2: Feature Engineering

# 2.1: Convert the 'Time' feature into more meaningful components (e.g., hour of the day, weekday)
df['Hour'] = df['Time'].apply(lambda x: (x // 3600) % 24)  # Extract hour from time (in seconds)
df['Weekday'] = df['Time'].apply(lambda x: (x // (3600 * 24)) % 7)  # Extract weekday (0 = Monday, 6 = Sunday)

# 2.2: Scaling the Amount feature
scaler = StandardScaler()
df['Scaled_Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

# 2.3: Simulate Usual Behavior for Geofencing/Location Simulated (for example, let's use 'Amount' and 'Time')
# Create a feature that represents unusual behavior based on transaction amount and time
df['Unusual_Behavior'] = np.where((df['Amount'] > df['Scaled_Amount'].mean() + 2 * df['Scaled_Amount'].std()) | 
                                  (df['Hour'] < 6) | (df['Hour'] > 22), 1, 0)  # Heuristic for unusual behavior

# 2.4: Simulate Geolocation data (latitude, longitude) for each transaction
np.random.seed(42)
df['Latitude'] = np.random.uniform(low=40.0, high=42.0, size=len(df))  # Simulating latitude
df['Longitude'] = np.random.uniform(low=-74.0, high=-72.0, size=len(df))  # Simulating longitude

# 2.5: Calculate the distance between consecutive transactions using geodesic distance
def calculate_distance(row, prev_row):
    if prev_row is None:
        return 0
    current_location = (row['Latitude'], row['Longitude'])
    previous_location = (prev_row['Latitude'], prev_row['Longitude'])
    return geodesic(current_location, previous_location).km

df['Prev_Latitude'] = df['Latitude'].shift(1)
df['Prev_Longitude'] = df['Longitude'].shift(1)

# Apply the distance calculation
df['Distance'] = df.apply(lambda row: calculate_distance(row, df.iloc[row.name-1] if row.name > 0 else None), axis=1)

# Create a new feature for unusual location-based behavior (e.g., large distance between consecutive transactions)
df['Unusual_Location_Behavior'] = np.where(df['Distance'] > 50, 1, 0)  # Example: Distance > 50 km is considered unusual

# Step 3: Define features and target
X = df[['Scaled_Amount', 'Hour', 'Weekday', 'Unusual_Behavior', 'Unusual_Location_Behavior']]  # Features
y = df['Class']  # Target variable (fraudulent vs. non-fraudulent)

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Apply SMOTE for oversampling the minority class (fraudulent transactions)
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Step 6: Train a Random Forest Classifier on the resampled data
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Step 7: Predict and Evaluate the Model
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class (fraudulent)

# Step 8: Adjust the threshold for predicting fraud (Class 1)
threshold = 0.3  # Set a lower threshold to predict fraud more easily
y_pred_custom = (y_prob >= threshold).astype(int)

# Step 9: Evaluation - Classification Report and Confusion Matrix
print("Classification Report:\n", classification_report(y_test, y_pred_custom))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_custom))

# Step 10: Visualize the Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_custom), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Step 11: Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Step 12: Feature Importance (Optional) - To see which features are important for the classification
feature_importance = model.feature_importances_
features = X.columns

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(features, feature_importance, color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Feature Importance for Fraud Detection')
plt.show()

# Step 13: Cross-validation using Stratified K-Fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
print(f'Cross-validated accuracy: {cv_scores.mean()}')
