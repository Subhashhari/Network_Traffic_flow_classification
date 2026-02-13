import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. DATA COLLECTION [cite: 252, 259]
# The paper uses the VPN and non-VPN dataset from CIC
data, meta = arff.loadarff("C:\\Users\\Subhash\\Downloads\\Scenario A1-ARFF\\Scenario A1-ARFF\\TimeBasedFeatures-Dataset-15s-VPN.arff")
df = pd.DataFrame(data)

# 2. PRE-PROCESSING (DATA CLEANING) 
# Decode nominal attributes from bytes to strings
for col in df.select_dtypes([object]):
    df[col] = df[col].str.decode('utf-8')

# Replace missing values with the median 
# Note: In this dataset, -1 often represents a missing value
df.replace(-1, np.nan, inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)

# Data Encoding: Convert nominal attributes to numeric [cite: 273]
# Non-VPN and VPN are converted to 0 and 1
df['class1'] = df['class1'].map({'Non-VPN': 0, 'VPN': 1})

X = df.drop('class1', axis=1)
y = df['class1']

# DATA NORMALIZATION (Min-Max Normalization) 
# Scales features between [0, 1] using Equation 1 [cite: 265, 268]
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

# 3. FEATURE CORRELATION [cite: 254]
# The paper examines correlation here before passing to models
# correlation_matrix = X.corr() 

# 4. MACHINE LEARNING CLASSIFIERS [cite: 276]
# Split for training and testing [cite: 139, 569]
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]} samples")

# Random Forest: Combination of multiple decision trees [cite: 299]
# Paper settings: criterion='gini', tuned for high precision/recall [cite: 25, 294]
rf_model = RandomForestClassifier(
    n_estimators=100, 
    criterion='gini',
    max_depth=None, 
    min_samples_split=2, 
    min_samples_leaf=1,
    random_state=42
)

rf_model.fit(X_train, y_train)

# 5. COMPARISON OF RESULTS [cite: 256, 551]
y_pred = rf_model.predict(X_test)
print(f"Strict Replication Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred))