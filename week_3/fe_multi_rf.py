import pandas as pd
import numpy as np
import io
import re
from scipy.io import arff
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "C:\\Users\\Subhash\\Downloads\\Scenario B-ARFF\\Scenario B-ARFF\\TimeBasedFeatures-Dataset-60s-AllinOne.arff"

CLASSES = ['BROWSING', 'CHAT', 'STREAMING', 'MAIL', 'VOIP', 'P2P', 'FT']

# Best params from hyperparameter search
BEST_PARAMS = dict(
    n_estimators=500,
    min_samples_split=5,
    min_samples_leaf=1,
    max_features=0.5,
    max_depth=None,
    class_weight=None,
    random_state=42,
    n_jobs=-1,
)


# ─────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────
def load_arff_safe(path):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    cleaned = []
    for line in lines:
        stripped = line.rstrip().rstrip(',').rstrip()
        if stripped:
            cleaned.append(stripped)
    clean_text = '\n'.join(cleaned) + '\n'
    data, meta = arff.loadarff(io.StringIO(clean_text))
    df = pd.DataFrame(data)
    for col in df.select_dtypes([object]):
        df[col] = df[col].str.decode('utf-8')
    return df


# ─────────────────────────────────────────────────────────────────
# Load & clean
# ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = load_arff_safe(DATA_PATH)
df.replace(-1, np.nan, inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)
df = df[df['class1'].isin(CLASSES)].reset_index(drop=True)

print(f"Shape: {df.shape}")
print(f"\nClass distribution:")
print(df['class1'].value_counts().to_string())

X_raw = df.drop('class1', axis=1)
y     = df['class1']


# ─────────────────────────────────────────────────────────────────
# Pipeline helper
# ─────────────────────────────────────────────────────────────────
def run_pipeline(X, y, label):
    scaler   = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(**BEST_PARAMS)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred) * 100

    print(f"\n{'='*62}")
    print(f"  {label}")
    print(f"{'='*62}")
    print(f"  Features : {X.shape[1]}")
    print(f"  Accuracy : {acc:.2f}%")
    print(classification_report(y_test, y_pred, target_names=CLASSES, zero_division=0))

    cv_scores = cross_val_score(
        RandomForestClassifier(**BEST_PARAMS),
        X_scaled, y,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring='f1_macro',
    )
    print(f"  5-fold CV macro-F1 : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Refit on full data for importances
    rf_full = RandomForestClassifier(**BEST_PARAMS)
    rf_full.fit(X_scaled, y)
    importances = pd.Series(rf_full.feature_importances_, index=X.columns)

    return acc, y_test, y_pred, importances


# ─────────────────────────────────────────────────────────────────
# BASELINE
# ─────────────────────────────────────────────────────────────────
base_acc, base_yt, base_yp, base_imp = run_pipeline(X_raw, y, "BASELINE — 23 raw features")


# ─────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# This schema has std_fiat and std_biat (unlike the VPN 15s dataset),
# so we can compute full CV features for both directions.
#
# Per-class reasoning for each new feature:
#
#   fiat_cv / biat_cv
#     VOIP uses a fixed codec interval (e.g. 20ms) → CV ≈ 0
#     BROWSING is bursty → high CV
#     STREAMING is steady → low-moderate CV
#     CHAT has sparse keepalives → very high CV
#
#   bytes_per_packet
#     STREAMING / FT → large payloads, high value
#     VOIP / CHAT    → small fixed-size packets, low value
#     BROWSING       → mixed, moderate
#
#   flowiat_cv
#     VOIP → near 0 (metronomic)
#     BROWSING → high (think-time gaps between page loads)
#
#   fiat_range / biat_range / flowiat_range
#     Spread proxy. VOIP: near 0. BROWSING: very wide.
#
#   fwd_bwd_mean_ratio
#     FT and STREAMING are download-heavy → server sends far more
#     than client → mean_biat << mean_fiat → ratio >> 1
#     VOIP is symmetric → ratio ≈ 1
#
#   active_idle_ratio
#     STREAMING stays continuously active → high ratio
#     MAIL is bursty with long idle gaps → low ratio
#     BROWSING has moderate idle periods between page loads
#
#   active_cv / idle_cv
#     VOIP: active periods are all equal length → CV ≈ 0
#     BROWSING: highly variable active bursts → high CV
#
#   duration_x_pktrate
#     Interaction term capturing total flow volume proxy.
#     FT and STREAMING will score high; CHAT and MAIL low.
# ─────────────────────────────────────────────────────────────────
def engineer_features(X_in):
    X   = X_in.copy()
    eps = 1e-9

    # Step 1: Add log features alongside raw (both kept).
    # RF will choose whichever scale produces better splits per feature.
    log_cols = {}
    for col in X_in.columns:
        # clip negatives to 0 before log-transforming — negative IAT values
        # from clock jitter are noise, safely floored at 0
        log_cols[f'log1p_{col}'] = np.log1p(X_in[col].clip(lower=0))
    X = pd.concat([X, pd.DataFrame(log_cols, index=X.index)], axis=1)

    def lc(name): return f'log1p_{name}'

    # Step 2: CV features (computed on log versions)
    X['fiat_cv']    = X[lc('std_fiat')]    / (X[lc('mean_fiat')]    + eps)
    X['biat_cv']    = X[lc('std_biat')]    / (X[lc('mean_biat')]    + eps)
    X['flowiat_cv'] = X[lc('std_flowiat')] / (X[lc('mean_flowiat')] + eps)
    X['active_cv']  = X[lc('std_active')]  / (X[lc('mean_active')]  + eps)
    X['idle_cv']    = X[lc('std_idle')]    / (X[lc('mean_idle')]    + eps)

    # Step 3: Bytes per packet
    X['bytes_per_packet'] = X[lc('flowBytesPerSecond')] / (X[lc('flowPktsPerSecond')] + eps)

    # Step 4: Range features
    X['fiat_range']    = X[lc('max_fiat')]    - X[lc('min_fiat')]
    X['biat_range']    = X[lc('max_biat')]    - X[lc('min_biat')]
    X['flowiat_range'] = X[lc('max_flowiat')] - X[lc('min_flowiat')]
    X['active_range']  = X[lc('max_active')]  - X[lc('min_active')]
    X['idle_range']    = X[lc('max_idle')]    - X[lc('min_idle')]

    # Step 5: Directional asymmetry
    X['fwd_bwd_mean_ratio'] = X[lc('mean_fiat')] / (X[lc('mean_biat')] + eps)

    # Step 6: Active / idle balance
    X['active_idle_ratio']  = X[lc('mean_active')] / (X[lc('mean_idle')]  + eps)
    X['active_to_duration'] = X[lc('mean_active')] / (X[lc('duration')]   + eps)

    # Step 7: Interaction term
    X['duration_x_pktrate'] = X[lc('duration')] * X[lc('flowPktsPerSecond')]

    n_log = len(log_cols)
    n_new = X.shape[1] - len(X_in.columns) - n_log
    print(f"  Raw features kept        : {len(X_in.columns)}")
    print(f"  Log features added       : {n_log}")
    print(f"  Engineered features added: {n_new}")
    print(f"  Total                    : {X.shape[1]} features")
    return X


print("\n── Building engineered feature set ──")
X_engineered = engineer_features(X_raw)

fe_acc, fe_yt, fe_yp, fe_imp = run_pipeline(X_engineered, y, "FEATURE ENGINEERED — log + 15 new features")


# ─────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────
C_BG    = '#0F0F0F'
C_PANEL = '#1A1A1A'
C_TEXT  = '#E0E0E0'
C_BLUE  = '#4A9EFF'
C_GREY  = '#666666'

plt.rcParams.update({
    'figure.facecolor': C_BG,
    'axes.facecolor'  : C_PANEL,
    'axes.labelcolor' : C_TEXT,
    'xtick.color'     : C_TEXT,
    'ytick.color'     : C_TEXT,
    'text.color'      : C_TEXT,
    'font.family'     : 'monospace',
})

present = [c for c in CLASSES if c in fe_yt.values]

# ── Figure 1: Side-by-side confusion matrices ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Baseline vs Feature Engineered — Row-Normalised Confusion Matrix', fontsize=13, color=C_TEXT)
fig.patch.set_facecolor(C_BG)

for ax, y_pred, title in [
    (axes[0], base_yp, f'Baseline  ({base_acc:.1f}%)'),
    (axes[1], fe_yp,   f'FE Model  ({fe_acc:.1f}%)'),
]:
    cm = confusion_matrix(fe_yt, y_pred, labels=present)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(
        cm_norm, annot=True, fmt='.2f',
        xticklabels=present, yticklabels=present,
        cmap='Blues', linewidths=0.5, linecolor='#2A2A2A',
        ax=ax, cbar_kws={'shrink': 0.8}, vmin=0, vmax=1,
    )
    ax.set_title(title, fontsize=11, color=C_TEXT, pad=8)
    ax.set_xlabel('Predicted', fontsize=9)
    ax.set_ylabel('Actual', fontsize=9)
    ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig('fig_cm_comparison.png', dpi=150, bbox_inches='tight', facecolor=C_BG)
print("\nSaved: fig_cm_comparison.png")


# ── Figure 2: Feature importances — FE model, top 25 ─────────────
top25 = fe_imp.nlargest(25).sort_values(ascending=True)
raw_log_cols = [f'log1p_{c}' for c in X_raw.columns]
new_feat_cols = [f for f in top25.index if f not in raw_log_cols]

fig2, ax = plt.subplots(figsize=(10, 8))
fig2.patch.set_facecolor(C_BG)
ax.set_facecolor(C_PANEL)

colors = [C_BLUE if f in new_feat_cols else C_GREY for f in top25.index]
bars   = ax.barh(top25.index, top25.values, color=colors, edgecolor='none', height=0.7)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=C_BLUE,  label='Engineered feature'),
    Patch(facecolor=C_GREY,  label='Raw (log-transformed)'),
]
ax.legend(handles=legend_elements, fontsize=8, framealpha=0.3,
          loc='lower right', labelcolor=C_TEXT)

ax.set_title('Top 25 Feature Importances — FE Model (7-class)', fontsize=12, color=C_TEXT, pad=10)
ax.set_xlabel('Mean Decrease in Impurity', fontsize=9)
ax.tick_params(labelsize=8)
ax.grid(axis='x', color='#2A2A2A', linewidth=0.5)
ax.set_axisbelow(True)

for bar, val in zip(bars, top25.values):
    ax.text(val + 0.0005, bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}', va='center', fontsize=7, color=C_TEXT)

plt.tight_layout()
plt.savefig('fig_feature_importance_fe.png', dpi=150, bbox_inches='tight', facecolor=C_BG)
print("Saved: fig_feature_importance_fe.png")


# ── Figure 3: Per-class F1 comparison bar chart ───────────────────
from sklearn.metrics import f1_score

base_f1s = f1_score(base_yt, base_yp, labels=present, average=None, zero_division=0)
fe_f1s   = f1_score(fe_yt,   fe_yp,   labels=present, average=None, zero_division=0)

x     = np.arange(len(present))
width = 0.35

fig3, ax = plt.subplots(figsize=(12, 5))
fig3.patch.set_facecolor(C_BG)
ax.set_facecolor(C_PANEL)

bars1 = ax.bar(x - width/2, base_f1s, width, label=f'Baseline ({base_acc:.1f}%)',
               color='#FF6B6B', alpha=0.8, edgecolor='none')
bars2 = ax.bar(x + width/2, fe_f1s,   width, label=f'FE Model ({fe_acc:.1f}%)',
               color=C_BLUE,   alpha=0.8, edgecolor='none')

ax.set_xticks(x)
ax.set_xticklabels(present, fontsize=10)
ax.set_ylabel('F1 Score', fontsize=9)
ax.set_ylim(0, 1.1)
ax.set_title('Per-class F1 — Baseline vs Feature Engineered', fontsize=12, color=C_TEXT, pad=10)
ax.legend(fontsize=9, framealpha=0.3, labelcolor=C_TEXT)
ax.grid(axis='y', color='#2A2A2A', linewidth=0.5)
ax.set_axisbelow(True)

# Delta labels above each FE bar
for bar, b_f1, fe_f1 in zip(bars2, base_f1s, fe_f1s):
    delta = fe_f1 - b_f1
    color = '#90EE90' if delta >= 0 else '#FF6B6B'
    ax.text(bar.get_x() + bar.get_width() / 2, fe_f1 + 0.02,
            f'{delta:+.2f}', ha='center', fontsize=8, color=color)

plt.tight_layout()
plt.savefig('fig_f1_comparison.png', dpi=150, bbox_inches='tight', facecolor=C_BG)
print("Saved: fig_f1_comparison.png")

print("\nDone.")