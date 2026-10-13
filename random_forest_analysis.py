"""
Random Forest Classification — Figure G
Reinforced Hippocampus-Accumbens pathway marks a shift towards cognitive decline in aging
Deneuville et al., Cell Reports 2026

Dependencies: pandas, numpy, matplotlib, seaborn, scikit-learn, scipy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from scipy.stats import mannwhitneyu

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_FILE    = "data D1 _ .csv"
DELIMITER    = ","
RANDOM_STATE = 42
TEST_SIZE    = 0.30
N_ESTIMATORS = 100
ADULT_LABEL  = '0-1'
AGED_LABEL   = '1-1'

# =============================================================================
# 1. LOAD DATA
# =============================================================================

data = pd.read_csv(DATA_FILE, sep=DELIMITER, header=1, engine='python')
data = data[data['Animal'].notna()]
data = data[data['Animal'].astype(str).str.upper().ne('REMOVE')]
data['Age'] = data['Age'].str.strip()

print(f"Dataset: {data.shape[0]} animals, {data.shape[1]} variables")
print(f"Groups: {data['Age'].value_counts().to_dict()}")

# Convert to numeric
for col in data.columns:
    if col not in ['Animal', 'Age']:
        data[col] = pd.to_numeric(data[col], errors='coerce')

data['Age_Binary'] = (data['Age'] == AGED_LABEL).astype(int)

# =============================================================================
# 2. FEATURE SELECTION
# =============================================================================

# Exclude identifiers and variables with >20% missing data
cols_to_drop = ['Animal', 'Age', 'Age_Binary',
                'O-maze (%)', 'Social interaction (mouse/empty)']
cols_to_drop = [c for c in cols_to_drop if c in data.columns]

X = data.drop(columns=cols_to_drop)
y = data['Age_Binary']

# =============================================================================
# 3. TRAIN / TEST SPLIT AND IMPUTATION
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

imputer     = SimpleImputer(strategy='mean')
X_train_imp = imputer.fit_transform(X_train)
X_test_imp  = imputer.transform(X_test)

print(f"Train: {len(y_train)} animals | Test: {len(y_test)} animals")

# =============================================================================
# 4. RANDOM FOREST CLASSIFIER
# =============================================================================

clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
clf.fit(X_train_imp, y_train)

y_pred = clf.predict(X_test_imp)
acc    = accuracy_score(y_test, y_pred)
cm     = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Adult', 'Aged'])

print(f"\nAccuracy: {acc*100:.2f}%")
print("Confusion matrix:\n", cm)
print("Classification report:\n", report)

# 5-fold stratified cross-validation
cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(clf, imputer.fit_transform(X), y, cv=cv, scoring='accuracy')
print(f"5-fold CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# =============================================================================
# 5. FEATURE IMPORTANCES
# =============================================================================

features_df = pd.DataFrame({
    'Feature':    X.columns,
    'Importance': clf.feature_importances_
}).sort_values('Importance', ascending=False).reset_index(drop=True)

print("\nFeature importances:")
print(features_df.round(4).to_string(index=False))

colors     = ["#7AB6E0", "#5B7DBA"]
cmap       = LinearSegmentedColormap.from_list("blue_gradient", colors, N=len(features_df))
bar_colors = [cmap(v) for v in np.linspace(0, 1, len(features_df))]

plt.figure(figsize=(9, 7))
sns.barplot(x='Importance', y='Feature', data=features_df,
            hue='Feature', palette=dict(zip(features_df['Feature'], bar_colors)),
            edgecolor='white', legend=False)
plt.title("Feature Importances — Random Forest", fontsize=14, fontweight='bold')
plt.xlabel("Importance (mean decrease in Gini impurity)", fontsize=12)
plt.ylabel("Feature", fontsize=12)
sns.despine()
plt.tight_layout()
plt.savefig('RF_feature_importances_figure_G.png', dpi=300)
plt.show()

# =============================================================================
# 6. MANN-WHITNEY U TEST ON TOP FEATURES
# =============================================================================

adult_data  = data[data['Age'] == ADULT_LABEL]
aged_data   = data[data['Age'] == AGED_LABEL]
top_features = features_df['Feature'].head(3).tolist()

print("\nMann-Whitney U test — top 3 features:")
for feat in top_features:
    stat, p = mannwhitneyu(adult_data[feat].dropna(),
                           aged_data[feat].dropna(),
                           alternative='two-sided')
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  {feat}: U={stat:.1f}, p={p:.4f}  {sig}")