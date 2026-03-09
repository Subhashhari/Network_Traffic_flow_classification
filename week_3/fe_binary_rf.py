import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# Schema (exact, from ARFF header):
#   duration,
#   total_fiat, total_biat,
#   min_fiat, min_biat, max_fiat, max_biat, mean_fiat, mean_biat,
#   flowPktsPerSecond, flowBytesPerSecond,
#   min_flowiat, max_flowiat, mean_flowiat, std_flowiat,
#   min_active, mean_active, max_active, std_active,
#   min_idle,   mean_idle,   max_idle,   std_idle,
#   class1
# ─────────────────────────────────────────────────────────────────

DATA_PATH = "C:\\Users\\Subhash\\Downloads\\Scenario A1-ARFF\\Scenario A1-ARFF\\TimeBasedFeatures-Dataset-15s-VPN.arff"

# ── Load ──────────────────────────────────────────────────────────
data, meta = arff.loadarff(DATA_PATH)
df = pd.DataFrame(data)

for col in df.select_dtypes([object]):
    df[col] = df[col].str.decode('utf-8')

df.replace(-1, np.nan, inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)
df['class1'] = df['class1'].map({'Non-VPN': 0, 'VPN': 1})

X_raw = df.drop('class1', axis=1)
y = df['class1']

print(f"Dataset shape     : {X_raw.shape}")
print(f"Class distribution:\n{y.value_counts()}\n")

# Verify exact column names match schema
EXPECTED = [
    'duration',
    'total_fiat', 'total_biat',
    'min_fiat', 'min_biat', 'max_fiat', 'max_biat', 'mean_fiat', 'mean_biat',
    'flowPktsPerSecond', 'flowBytesPerSecond',
    'min_flowiat', 'max_flowiat', 'mean_flowiat', 'std_flowiat',
    'min_active', 'mean_active', 'max_active', 'std_active',
    'min_idle', 'mean_idle', 'max_idle', 'std_idle',
]
missing = [c for c in EXPECTED if c not in X_raw.columns]
if missing:
    print(f"WARNING — expected columns not found: {missing}")
else:
    print("Schema check passed — all 23 feature columns found.\n")


# ─────────────────────────────────────────────────────────────────
# PIPELINE HELPER
# ─────────────────────────────────────────────────────────────────
def run_pipeline(X, y, label):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    rf = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Feature count  : {X.shape[1]}")
    print(f"  Hold-out acc   : {acc:.2f}%")
    print(classification_report(y_test, y_pred, target_names=['Non-VPN', 'VPN']))

    cv = cross_val_score(
        rf, X_scaled, y,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring='f1'
    )
    print(f"  5-fold CV F1   : {cv.mean():.4f} ± {cv.std():.4f}")

    # Feature importances
    rf_full = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=42, n_jobs=-1)
    rf_full.fit(X_scaled, y)
    importances = pd.Series(rf_full.feature_importances_, index=X.columns)
    return importances


# ─────────────────────────────────────────────────────────────────
# BASELINE
# ─────────────────────────────────────────────────────────────────
baseline_imp = run_pipeline(X_raw, y, "BASELINE — 23 raw features")


# ─────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  (schema-aware, exact column names)
# ─────────────────────────────────────────────────────────────────
def engineer_features(X_in):
    X = X_in.copy()
    eps = 1e-9

    # ── Step 1: Log-transform all 23 raw features (replace originals) ──
    # Every feature here is non-negative and right-skewed in practice.
    # We replace — not add alongside — to avoid raw/log duplicate noise
    # seen in the feature importance output.
    log_cols = X.columns.tolist()
    for col in log_cols:
        if (X[col] >= 0).all():
            X[col] = np.log1p(X[col])   # in-place replacement
    # rename to make it explicit in importance plots
    X.columns = [f'log1p_{c}' for c in X.columns]
    print(f"  Log-transformed all {len(log_cols)} raw features (replaced, not duplicated)")

    # Helper: shorthand for the log-transformed column names
    def lc(name): return f'log1p_{name}'

    # ── Step 2: Timing ratio features ────────────────────────────────
    # fwd/bwd IAT balance — VPN tunnels tend to create
    # a more symmetric timing profile than asymmetric web browsing
    X['fwd_bwd_iat_ratio']   = X[lc('total_fiat')] / (X[lc('total_biat')] + eps)
    X['fiat_symmetry']       = X[lc('total_fiat')] / (X[lc('total_fiat')] + X[lc('total_biat')] + eps)

    # ── Step 3: Rate-derived features ────────────────────────────────
    # This is the only way to approximate packet size — the schema has
    # no raw byte or packet counts, only rates. VPN adds fixed per-packet
    # overhead (tunnel header), pushing this ratio up.
    X['bytes_per_packet']    = X[lc('flowBytesPerSecond')] / (X[lc('flowPktsPerSecond')] + eps)

    # ── Step 4: Range features (proxy for missing std_fiat / std_biat) ──
    # std_fiat and std_biat are not in the schema. Range (max - min) is a
    # cruder but available spread measure. VPN traffic tends to have narrower
    # IAT spread (more regular tunneled packets).
    X['fiat_range']          = X[lc('max_fiat')]    - X[lc('min_fiat')]
    X['biat_range']          = X[lc('max_biat')]    - X[lc('min_biat')]
    X['flowiat_range']       = X[lc('max_flowiat')] - X[lc('min_flowiat')]
    X['active_range']        = X[lc('max_active')]  - X[lc('min_active')]
    X['idle_range']          = X[lc('max_idle')]    - X[lc('min_idle')]

    # ── Step 5: Coefficient of Variation for flow IAT ─────────────────
    # std_flowiat IS available (unlike fiat/biat). CV = std/mean.
    # Lower CV → more regular inter-packet gaps → more VPN-like.
    X['flowiat_cv']          = X[lc('std_flowiat')] / (X[lc('mean_flowiat')] + eps)

    # Same for active and idle periods
    X['active_cv']           = X[lc('std_active')]  / (X[lc('mean_active')] + eps)
    X['idle_cv']             = X[lc('std_idle')]    / (X[lc('mean_idle')]   + eps)

    # ── Step 6: Active/Idle balance ───────────────────────────────────
    # VPN sessions tend to stay active for longer fractions of flow duration.
    X['active_idle_ratio']   = X[lc('mean_active')] / (X[lc('mean_idle')]   + eps)
    X['active_to_duration']  = X[lc('mean_active')] / (X[lc('duration')]    + eps)
    X['idle_to_duration']    = X[lc('mean_idle')]   / (X[lc('duration')]    + eps)

    print(f"  Engineered features added: {X.shape[1] - len(log_cols)}")
    print(f"  Total feature count      : {X.shape[1]}")

    return X


print("\n── Building engineered feature set ──")
X_engineered = engineer_features(X_raw)
fe_imp = run_pipeline(X_engineered, y, "FEATURE ENGINEERED — log-replaced + 11 new features")


# ─────────────────────────────────────────────────────────────────
# SIDE-BY-SIDE IMPORTANCE: which new features broke into top 20?
# ─────────────────────────────────────────────────────────────────
print("\n── Top 20 Features (FE model) ──")
top20_fe = fe_imp.nlargest(20)
new_features = [f for f in top20_fe.index if f not in [f'log1p_{c}' for c in X_raw.columns]]

for feat, imp in top20_fe.items():
    bar  = '█' * int(imp * 400)
    tag  = '  ← NEW' if feat in new_features else ''
    print(f"  {feat:<40} {imp:.4f}  {bar}{tag}")

print(f"\nNew engineered features in top 20: {len(new_features)}")
if new_features:
    print("  " + "\n  ".join(new_features))
