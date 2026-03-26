"""
ML Model Training & Evaluation Pipeline
Trains a RandomForest classifier on loan eligibility data
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import warnings

warnings.filterwarnings('ignore')


def safe_ratio(numerator, denominator):
    if denominator is None or denominator == 0:
        return 0.0
    return numerator / denominator


def main():
    print("=" * 70)
    print("LOAN ELIGIBILITY PREDICTION MODEL - TRAINING PIPELINE")
    print("=" * 70)

    # Load training data
    print("\n[1] Loading training data...")
    train_df = pd.read_csv('train.csv')
    print(f"    ✓ Loaded {train_df.shape[0]} records with {train_df.shape[1]} features")

    # Validate target column
    target_col = None
    for col in train_df.columns:
        if 'status' in col.lower():
            target_col = col
            break
    if target_col is None:
        raise ValueError("Error: train.csv must contain a 'loan_status' or similar target column")

    # Target encoding
    print("\n[2] Preparing target variable...")
    y = train_df[target_col].astype(str).str.strip()
    if y.dtype == 'object':
        y = y.map({'Approved': 1, 'Rejected': 0, 'Y': 1, 'N': 0, 'Yes': 1, 'No': 0, '1': 1, '0': 0})
    
    if y.isna().sum() > 0:
        print(f"    Warning: {y.isna().sum()} null targets after mapping; dropping...")
        valid_idx = y.notna()
        train_df = train_df[valid_idx]
        y = y[valid_idx]

    print(f"    ✓ Class distribution:")
    print(f"      - Approved (1): {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)")
    print(f"      - Rejected (0): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)")

    # Features
    X = train_df.drop(columns=[target_col])
    for c in ['loan_id', 'Loan_ID', 'id', 'ID']:
        if c in X.columns:
            X = X.drop(columns=[c])

    # Exclude credit score from training to align with app inputs.
    for c in [' cibil_score', 'cibil_score', 'CIBIL Score', 'CIBIL_SCORE']:
        if c in X.columns:
            X = X.drop(columns=[c])

    # Feature engineering to improve risk sensitivity
    if ' loan_amount' in X.columns and ' income_annum' in X.columns:
        X['loan_income_ratio'] = X.apply(
            lambda row: safe_ratio(row[' loan_amount'], row[' income_annum']),
            axis=1,
        )

    asset_cols = [
        ' residential_assets_value',
        ' commercial_assets_value',
        ' luxury_assets_value',
        ' bank_asset_value',
    ]
    if all(col in X.columns for col in asset_cols):
        X['total_assets'] = X[asset_cols].sum(axis=1)
        X['asset_coverage_ratio'] = X.apply(
            lambda row: safe_ratio(row['total_assets'], row[' loan_amount']) if ' loan_amount' in X.columns else 0.0,
            axis=1,
        )

    # Identify numeric and categorical columns
    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    print(f"\n[3] Feature breakdown:")
    print(f"    - Numeric features ({len(num_cols)}): {num_cols[:5]}{'...' if len(num_cols) > 5 else ''}")
    print(f"    - Categorical features ({len(cat_cols)}): {cat_cols}")

    # Preprocessing pipelines
    print("\n[4] Building preprocessing & model pipeline...")

    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
    ])

    try:
        onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown='ignore', sparse=False)

    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', onehot),
    ])

    preprocess = ColumnTransformer([
        ('num', numeric_pipe, num_cols),
        ('cat', categorical_pipe, cat_cols),
    ])

    # Random Forest Model
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=15,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced_subsample',
    )

    clf = Pipeline([
        ('prep', preprocess),
        ('model', model),
    ])

    # Train/validation split
    print("\n[5] Splitting data (80% train, 20% validation)...")
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    ✓ Training set: {X_train.shape[0]} samples")
    print(f"    ✓ Validation set: {X_valid.shape[0]} samples")

    # Train model
    print("\n[6] Training RandomForest model...")
    clf.fit(X_train, y_train)
    print("    ✓ Model training complete")

    # Evaluate
    print("\n[7] Evaluating model performance...")
    y_pred = clf.predict(X_valid)
    y_pred_proba = clf.predict_proba(X_valid)[:, 1]

    acc = accuracy_score(y_valid, y_pred)
    auc = roc_auc_score(y_valid, y_pred_proba)

    print(f"\n{'=' * 70}")
    print(f"MODEL ACCURACY: {acc:.4f} ({acc * 100:.2f}%)")
    print(f"ROC-AUC Score:  {auc:.4f}")
    print(f"{'=' * 70}")

    print("\nClassification Report:")
    print(classification_report(y_valid, y_pred, digits=4, target_names=['Rejected', 'Approved']))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_valid, y_pred)
    print(f"                 Predicted")
    print(f"                 Rejected  Approved")
    print(f"Actual Rejected   {cm[0, 0]:5d}    {cm[0, 1]:5d}")
    print(f"       Approved   {cm[1, 0]:5d}    {cm[1, 1]:5d}")

    # Feature importance
    print("\n[8] Feature Importance (Top 10):")
    try:
        feature_names = (
            numeric_pipe.named_steps['imputer'].get_feature_names_out(num_cols).tolist() +
            categorical_pipe.named_steps['onehot'].get_feature_names_out(cat_cols).tolist()
        )
        importances = clf.named_steps['model'].feature_importances_
        top_indices = np.argsort(importances)[::-1][:10]
        for rank, idx in enumerate(top_indices, 1):
            print(f"    {rank:2d}. {feature_names[idx][:40]:40s} : {importances[idx]:.4f}")
    except Exception as e:
        print(f"    (Feature names unavailable: {str(e)[:50]}...)")
        importances = clf.named_steps['model'].feature_importances_
        print(f"    Mean feature importance: {np.mean(importances):.4f}")

    # Save model
    print("\n[9] Saving trained model...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    print("    ✓ Model saved to 'model.pkl'")

    # Save preprocessing info
    with open('model_info.pkl', 'wb') as f:
        pickle.dump({
            'num_cols': num_cols,
            'cat_cols': cat_cols,
            'feature_columns': X.columns.tolist(),
            'accuracy': acc,
            'auc': auc,
        }, f)
    print("    ✓ Model metadata saved to 'model_info.pkl'")

    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
