import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              AdaBoostClassifier, BaggingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, accuracy_score, 
                            precision_recall_fscore_support, roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("=" * 80)
print("PERFORMANCE EVALUATION OF SECURED NETWORK TRAFFIC CLASSIFICATION")
print("Using Machine Learning Approach - Replication Study")
print("=" * 80)

# 1. DATA COLLECTION [cite: page 1, section 3.1.1]
print("\n[STEP 1] Loading VPN/non-VPN Dataset from Canadian Institute of Cybersecurity...")
try:
    data, meta = arff.loadarff("TimeBasedFeatures-Dataset-15s-VPN.arff")
    df = pd.DataFrame(data)
    print(f"✓ Dataset loaded successfully: {df.shape[0]} samples, {df.shape[1]} features")
except FileNotFoundError:
    print("⚠ Dataset file not found. Please download from: https://www.unb.ca/cic/datasets/vpn.html")
    print("Using synthetic data for demonstration...")
    # Create synthetic data for demonstration
    np.random.seed(42)
    n_samples = 50000
    n_features = 20
    df = pd.DataFrame(np.random.randn(n_samples, n_features), 
                     columns=[f'feature_{i}' for i in range(n_features)])
    df['class1'] = np.random.choice(['Non-VPN', 'VPN'], n_samples)

# 2. PRE-PROCESSING [cite: page 7, section 3.1.1]
print("\n[STEP 2] Data Pre-processing...")

# Decode bytes to strings for nominal attributes
for col in df.select_dtypes([object]):
    try:
        df[col] = df[col].str.decode('utf-8')
    except AttributeError:
        pass  # Already string

# Replace missing values with median [cite: page 7, Data Normalization]
df.replace(-1, np.nan, inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)
print(f"✓ Missing values handled")

# Remove duplicates [cite: page 7, Data Encoding]
initial_rows = len(df)
df.drop_duplicates(inplace=True)
print(f"✓ Removed {initial_rows - len(df)} duplicate rows")

# Encode class labels: Non-VPN=0, VPN=1 [cite: page 7, Data Encoding]
label_encoder = LabelEncoder()
df['class1'] = label_encoder.fit_transform(df['class1'])
print(f"✓ Class distribution: {dict(df['class1'].value_counts())}")

# Separate features and target
X = df.drop('class1', axis=1)
y = df['class1']

# 3. DATA NORMALIZATION [cite: page 7, Equation 1]
print("\n[STEP 3] Min-Max Normalization (scaling to [0,1])...")
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
print(f"✓ {X.shape[1]} features normalized using Equation 1")

# 4. TRAIN-TEST SPLIT [cite: page 10, section 5]
print("\n[STEP 4] Splitting dataset (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Testing set: {X_test.shape[0]} samples")

# 5. DEFINE ALL CLASSIFIERS [cite: page 2, Table 1 & page 7, section 4]
print("\n[STEP 5] Initializing Machine Learning Classifiers...")

classifiers = {
    # ENSEMBLE MODELS [cite: page 10, section 4.7]
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        criterion='gini',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    ),
    
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ),
    
    'AdaBoost': AdaBoostClassifier(
        n_estimators=50,
        learning_rate=1.0,
        random_state=42
    ),
    
    'Bagging Decision Tree': BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=10,
        random_state=42,
        n_jobs=-1
    ),
    
    # SINGLE MODELS [cite: page 7, section 4]
    'Decision Tree': DecisionTreeClassifier(
        criterion='gini',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    ),
    
    'k-Nearest Neighbors': KNeighborsClassifier(
        n_neighbors=5,
        leaf_size=30,
        metric='minkowski',
        n_jobs=-1
    ),
    
    'Logistic Regression': LogisticRegression(
        random_state=42,
        max_iter=200,
        n_jobs=-1
    ),
    
    'Naive Bayes': GaussianNB(),
    
    'Multilayer Perceptron': MLPClassifier(
        hidden_layer_sizes=(60,),
        activation='relu',
        alpha=0.0001,
        batch_size='auto',
        max_iter=200,
        random_state=42
    ),
    
    'SVM': SVC(
        kernel='rbf',
        probability=True,
        random_state=42
    )
}

print(f"✓ Initialized {len(classifiers)} classifiers (Ensemble + Single models)")

# 6. FUNCTION TO PLOT LEARNING CURVES [cite: Figures 2-17]
def plot_learning_curve(estimator, title, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5)):
    """
    Generate learning curve plots showing training and cross-validation scores
    [cite: Figures 2-17 show these curves for all classifiers]
    """
    plt.figure(figsize=(10, 6))
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes,
        scoring='accuracy', random_state=42
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
    
    plt.xlabel('Training examples', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt

# 7. TRAIN AND EVALUATE ALL CLASSIFIERS [cite: page 10-12, section 5]
print("\n[STEP 6] Training and Evaluating Classifiers...")
print("=" * 80)

results = {}
roc_data = {}

for name, clf in classifiers.items():
    print(f"\n{'='*40}")
    print(f"Training: {name}")
    print(f"{'='*40}")
    
    # Train the model
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Calculate metrics [cite: page 10, Equations 11-14]
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=[0, 1]
    )
    
    # Store results
    results[name] = {
        'Accuracy': accuracy * 100,
        'Precision_NonVPN': precision[0] * 100,
        'Precision_VPN': precision[1] * 100,
        'Recall_NonVPN': recall[0] * 100,
        'Recall_VPN': recall[1] * 100,
        'F1_NonVPN': f1[0] * 100,
        'F1_VPN': f1[1] * 100
    }
    
    # Calculate ROC curve data [cite: Figure 19]
    if hasattr(clf, 'predict_proba'):
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
    
    print(f"✓ Accuracy: {accuracy*100:.2f}%")
    print(f"  VPN - Precision: {precision[1]*100:.2f}%, Recall: {recall[1]*100:.2f}%, F1: {f1[1]*100:.2f}%")
    print(f"  Non-VPN - Precision: {precision[0]*100:.2f}%, Recall: {recall[0]*100:.2f}%, F1: {f1[0]*100:.2f}%")
    
    # Generate learning curves [cite: Figures 2-17]
    print(f"  Generating learning curves...")
    plt_obj = plot_learning_curve(clf, f'{name} Learning Curve', X_train, y_train)
    plt_obj.savefig(f'/mnt/user-data/outputs/{name.replace(" ", "_")}_learning_curve.png', 
                    dpi=300, bbox_inches='tight')
    plt.close()

# 8. RESULTS COMPARISON TABLE [cite: Table 2, page 15]
print("\n" + "=" * 80)
print("COMPARATIVE SUMMARY OF EXPERIMENTAL RESULTS")
print("=" * 80)

results_df = pd.DataFrame(results).T
results_df = results_df.round(2)
print("\n" + results_df.to_string())

# Save results table
results_df.to_csv('/mnt/user-data/outputs/comparative_results.csv')

# 9. PLOT ROC CURVES FOR ALL CLASSIFIERS [cite: Figure 19, page 15]
print("\n[STEP 7] Generating ROC Curve Comparison...")
plt.figure(figsize=(12, 8))

colors = plt.cm.tab10(np.linspace(0, 1, len(roc_data)))
for (name, data), color in zip(roc_data.items(), colors):
    plt.plot(data['fpr'], data['tpr'], color=color, lw=2,
             label=f'{name} (area = {data["auc"]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Threshold (area = 0.0)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity (False Positive Rate)', fontsize=12)
plt.ylabel('Sensitivity (True Positive Rate)', fontsize=12)
plt.title('ROC Curves - All Classifiers Comparison', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/roc_curves_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 10. BAR CHART COMPARISON [cite: Figure 18, page 14]
print("\n[STEP 8] Generating Performance Comparison Charts...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Accuracy comparison
accuracies = [results[name]['Accuracy'] for name in classifiers.keys()]
colors_bar = ['#2ecc71' if acc > 90 else '#3498db' if acc > 80 else '#e74c3c' 
              for acc in accuracies]

ax1.barh(list(classifiers.keys()), accuracies, color=colors_bar, edgecolor='black')
ax1.set_xlabel('Accuracy (%)', fontsize=12)
ax1.set_title('Classifier Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_xlim([0, 100])
ax1.grid(axis='x', alpha=0.3)

for i, v in enumerate(accuracies):
    ax1.text(v + 1, i, f'{v:.2f}%', va='center', fontsize=9)

# F1-Score comparison
f1_vpn = [results[name]['F1_VPN'] for name in classifiers.keys()]
f1_nonvpn = [results[name]['F1_NonVPN'] for name in classifiers.keys()]

x = np.arange(len(classifiers))
width = 0.35

ax2.barh(x - width/2, f1_vpn, width, label='VPN', color='#e74c3c', edgecolor='black')
ax2.barh(x + width/2, f1_nonvpn, width, label='Non-VPN', color='#3498db', edgecolor='black')

ax2.set_xlabel('F1-Score (%)', fontsize=12)
ax2.set_title('F1-Score: VPN vs Non-VPN', fontsize=14, fontweight='bold')
ax2.set_yticks(x)
ax2.set_yticklabels(list(classifiers.keys()))
ax2.set_xlim([0, 100])
ax2.legend()
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/performance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 11. IDENTIFY BEST PERFORMING MODEL [cite: page 12, section 5.2]
print("\n" + "=" * 80)
print("BEST PERFORMING MODEL")
print("=" * 80)

best_model = max(results.items(), key=lambda x: x[1]['Accuracy'])
print(f"\n🏆 Best Classifier: {best_model[0]}")
print(f"   Accuracy: {best_model[1]['Accuracy']:.2f}%")
print(f"   VPN Precision: {best_model[1]['Precision_VPN']:.2f}%")
print(f"   VPN Recall: {best_model[1]['Recall_VPN']:.2f}%")
print(f"   VPN F1-Score: {best_model[1]['F1_VPN']:.2f}%")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nGenerated files:")
print(f"  • {len(classifiers)} learning curve plots")
print(f"  • ROC curves comparison")
print(f"  • Performance comparison charts")
print(f"  • Comparative results CSV")
print(f"\n[cite: Paper achieved 93.80% with Random Forest]")
print(f"Your replication achieved: {results.get('Random Forest', {}).get('Accuracy', 0):.2f}%")

print("\n" + "=" * 80)
print("RECOMMENDATIONS FOR IMPROVEMENT")
print("=" * 80)
print("""
1. HYPERPARAMETER TUNING [cite: page 2, line 16-17]
   - Use GridSearchCV or RandomizedSearchCV to optimize parameters
   - Random Forest: n_estimators, max_depth, min_samples_split
   - Gradient Boosting: learning_rate, n_estimators, max_depth
   - SVM: C, gamma, kernel type
   
2. FEATURE ENGINEERING [cite: page 7, Feature Correlation]
   - Perform feature selection using correlation analysis
   - Remove highly correlated features (correlation > 0.95)
   - Use feature importance from Random Forest to select top features
   - Apply PCA for dimensionality reduction

3. CROSS-VALIDATION [cite: page 10, learning curves use CV=5]
   - Use stratified K-fold cross-validation (K=5 or 10)
   - Calculate mean and std of metrics across folds
   - Helps detect overfitting and provides more robust estimates

4. CLASS IMBALANCE HANDLING
   - Check if dataset is imbalanced
   - Apply SMOTE (Synthetic Minority Over-sampling Technique)
   - Use class_weight='balanced' parameter in classifiers
   
5. ENSEMBLE METHODS [cite: page 10, section 4.7]
   - Voting Classifier: Combine predictions from multiple models
   - Stacking: Use meta-learner on top of base classifiers
   - Paper shows ensemble models (RF, GB) outperform single models

6. DEEP LEARNING [cite: page 4-5, section 2]
   - Try CNN or LSTM for sequence-based features
   - Papers mention deep learning achieves better results
   - Implement attention mechanisms for feature importance

7. MODEL INTERPRETABILITY
   - SHAP values for feature importance
   - LIME for local interpretability
   - Helps understand what features drive VPN vs non-VPN classification

8. TIMEOUT PARAMETER ANALYSIS [cite: page 1, abstract]
   - Paper mentions performance varies with different timeout values
   - Test with 5s, 10s, 15s, 30s, 60s, 120s timeout datasets
   - Analyze which timeout gives best accuracy

9. COMPUTATIONAL EFFICIENCY
   - Reduce SVM training time by using LinearSVC for large datasets
   - Use SGDClassifier for very large datasets
   - Implement parallel processing for ensemble methods

10. VALIDATION ON DIFFERENT DATASETS
    - Test on Scenario A2, B1, B2 datasets from CIC
    - Cross-dataset validation to check generalization
    - Test on encrypted traffic from different VPN providers
""")