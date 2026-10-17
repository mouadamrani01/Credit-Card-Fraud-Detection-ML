"""Reproduce the thesis comparison on the public ULB credit-card fraud dataset.

Usage: python -m src.train --data data/creditcard.csv --output results
The full SMOTE+RandomForest run can require substantial memory and CPU.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score,
                             classification_report, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

FEATURES = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']


def validate_dataset(data: pd.DataFrame):
    missing = sorted(set(FEATURES + ['Class']) - set(data.columns))
    if missing:
        raise ValueError(f'Dataset is missing columns: {missing}')
    if data[FEATURES + ['Class']].isna().any().any():
        raise ValueError('Dataset contains missing values')
    labels = set(data['Class'].unique())
    if labels != {0, 1}:
        raise ValueError(f'Expected both classes 0 and 1; received {labels}')


def split_and_prepare(data: pd.DataFrame):
    """Fit scaler and SMOTE ONLY to the training split, never test data."""
    validate_dataset(data)
    X = data[FEATURES]
    y = data['Class'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
    X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])
    if int(y_train.sum()) < 6:
        raise ValueError('Training set needs at least six fraud examples for SMOTE')
    X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
    return X_train, X_test, y_train, y_test


def fit_and_evaluate(data: pd.DataFrame, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = split_and_prepare(data)
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1),
    }
    rows = []
    for name, model in models.items():
        print(f'Training {name}...', flush=True)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        report = classification_report(y_test, pred, output_dict=True, zero_division=0)
        row = {'model': name, 'accuracy': report['accuracy'],
               'precision': report['1']['precision'], 'recall': report['1']['recall'],
               'f1': report['1']['f1-score'],
               'roc_auc': roc_auc_score(y_test, proba),
               'pr_auc': average_precision_score(y_test, proba)}
        rows.append(row)
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(confusion_matrix(y_test, pred),
                               display_labels=['Legitimate', 'Fraud']).plot(ax=ax, colorbar=False)
        ax.set_title(name)
        fig.tight_layout()
        fig.savefig(output / f'{name.lower().replace(" ", "_")}_confusion.png', dpi=140)
        plt.close(fig)
    pd.DataFrame(rows).to_csv(output / 'model_comparison.csv', index=False)
    (output / 'metrics.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(pd.DataFrame(rows).round(4).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/creditcard.csv'))
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f'{args.data} not found. Download the public dataset (see README).')
    fit_and_evaluate(pd.read_csv(args.data), args.output)


if __name__ == '__main__':
    main()
