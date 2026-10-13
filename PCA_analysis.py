"""
Principal Component Analysis — Figure H
Reinforced Hippocampus-Accumbens pathway marks a shift towards cognitive decline in aging
Deneuville et al., Cell Reports 2026

Dependencies: pandas, numpy, matplotlib, scikit-learn, scipy, statsmodels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy import stats
from statsmodels.multivariate.manova import MANOVA

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_FILE      = "data D1 _ .csv"
DELIMITER      = ","
OUTPUT_CSV     = "pca_scores_with_age.csv"

GROUP_NAMES = {'0-1': 'Adult', '1-1': 'Aged'}
PALETTE     = {'Adult': '#7B5EA7', 'Aged': '#1B4F72'}

# =============================================================================
# 1. LOAD DATA
# =============================================================================

data = pd.read_csv(DATA_FILE, sep=DELIMITER, header=1, engine='python')
data = data[data['Animal'].notna()]
data = data[data['Animal'].astype(str).str.upper().ne('REMOVE')]
data['Age'] = data['Age'].str.strip()

print(f"Dataset: {data.shape[0]} animals, {data.shape[1]} variables")
print(f"Groups: {data['Age'].value_counts().to_dict()}")

# =============================================================================
# 2. FEATURE PREPARATION
# =============================================================================

# Exclude identifiers and variables with >20% missing data
cols_to_drop = ['Animal', 'Age', 'O-maze (%)', 'Social interaction (mouse/empty)']
cols_to_drop = [c for c in cols_to_drop if c in data.columns]

identifiers = data['Animal'].values
age_labels  = data['Age'].map(GROUP_NAMES).values

features      = data.drop(columns=cols_to_drop)
feature_names = list(features.columns)
features      = features.apply(pd.to_numeric, errors='coerce')

# Mean imputation for variables with <5% missing values
imputer             = SimpleImputer(strategy='mean')
features_imputed    = imputer.fit_transform(features)

# Z-score standardisation
scaler              = StandardScaler()
features_standardised = scaler.fit_transform(features_imputed)

# =============================================================================
# 3. PRINCIPAL COMPONENT ANALYSIS
# =============================================================================

n_components = min(10, features_standardised.shape[1])
pca          = PCA(n_components=n_components)
pca_scores   = pca.fit_transform(features_standardised)

summary = pd.DataFrame({
    'Std. deviation':          np.std(pca_scores, axis=0),
    'Proportion of variance':  pca.explained_variance_ratio_,
    'Cumulative proportion':   np.cumsum(pca.explained_variance_ratio_)
}, index=[f'PC{i+1}' for i in range(n_components)])



pca_df           = pd.DataFrame(pca_scores, columns=[f'PC{i+1}' for i in range(n_components)])
pca_df['Age']    = age_labels
pca_df['Animal'] = identifiers
pca_df.to_csv(OUTPUT_CSV, index=False)

# =============================================================================
# 4. BIPLOT
# =============================================================================

LABEL_OFFSETS = {
    'Sucrose preference':            ( 0.05,  0.18),
    'CPP score compartment':         (-0.25, -0.18),
    'CPP score exploration':         (-0.25, -0.32),
    'Connectivity ratio':            ( 0.05, -0.18),
    'PPR':                           (-0.20,  0.12),
    'oEPSC amplitude':               ( 0.18,  0.00),
    'oIPSC amplitude':               ( 0.18, -0.14),
    'sEPSC frequency':               ( 0.18,  0.00),
    'sIPSC frequency':               ( 0.18,  0.12),
    'Synaptic fatigue steady state': ( 0.05,  0.18),
    'EPSc and IPSC/total':           ( 0.18, -0.14),
}

fig, ax = plt.subplots(figsize=(9, 8))

for group, color in PALETTE.items():
    mask = age_labels == group
    ax.scatter(pca_scores[mask, 0], pca_scores[mask, 1],
               c=color, label=group, s=70, alpha=0.85, zorder=3)

loadings     = pca.components_.T
scale_factor = 3.0

for i, feature in enumerate(feature_names):
    lx, ly = loadings[i, 0] * scale_factor, loadings[i, 1] * scale_factor
    ax.annotate('', xy=(lx, ly), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='grey', lw=1.2))
    dx, dy = LABEL_OFFSETS.get(feature, (0.12, 0.10))
    ax.text(lx + dx, ly + dy, feature,
            ha='center', va='center', fontsize=8, color='dimgrey')

var1 = pca.explained_variance_ratio_[0] * 100
var2 = pca.explained_variance_ratio_[1] * 100
ax.set_xlabel(f'PC1 ({var1:.1f}% variance)', fontsize=11)
ax.set_ylabel(f'PC2 ({var2:.1f}% variance)', fontsize=11)
ax.set_title('PCA Biplot — D1R-MSN (Adult vs Aged)', fontsize=12)
ax.axhline(0, color='lightgrey', lw=0.8, zorder=1)
ax.axvline(0, color='lightgrey', lw=0.8, zorder=1)
ax.legend(title='Group', frameon=True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('PCA_biplot_figure_H.png', dpi=300)
plt.show()

# =============================================================================
# 5. STATISTICS
# =============================================================================

# MANOVA on first three principal components
try:
    maov = MANOVA.from_formula('PC1 + PC2 + PC3 ~ Age', data=pca_df)
    print("\nMANOVA Results:")
    print(maov.mv_test())
except Exception as e:
    print(f"MANOVA error: {e}")

# One-way ANOVA per component
print("\nANOVA per principal component:")
groups_unique = np.unique(age_labels)
for pc_col in [f'PC{i+1}' for i in range(min(5, n_components))]:
    groups_data = [pca_df[pca_df['Age'] == g][pc_col].dropna() for g in groups_unique]
    f_stat, p_val = stats.f_oneway(*groups_data)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    print(f"  {pc_col}: F={f_stat:.3f}, p={p_val:.4f}  {sig}")