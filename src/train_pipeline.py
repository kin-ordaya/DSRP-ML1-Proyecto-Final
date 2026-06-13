"""
Train pipeline for Telco Customer Churn prediction.

Usage:
    python src/train_pipeline.py --preprocess
    python src/train_pipeline.py --model logistic
    python src/train_pipeline.py --model tree
    python src/train_pipeline.py --all
"""

import argparse
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score
)


DATA_PATH = 'data/raw/Telco-Customer-Churn.csv'
ARTIFACTS_DIR = 'artifacts'
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, 'preprocessor.pkl')
X_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, 'X_train.pkl')
X_TEST_PATH = os.path.join(ARTIFACTS_DIR, 'X_test.pkl')
Y_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, 'y_train.pkl')
Y_TEST_PATH = os.path.join(ARTIFACTS_DIR, 'y_test.pkl')
LOGISTIC_PATH = os.path.join(ARTIFACTS_DIR, 'logistic_model.pkl')
TREE_PATH = os.path.join(ARTIFACTS_DIR, 'tree_model.pkl')


class DataLoader:
    def load_data(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        print(f'Dataset loaded: {df.shape}')
        df.drop(columns=['customerID'], inplace=True)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
        return df

    def split_features_target(self, df: pd.DataFrame):
        X = df.drop(columns=['Churn'])
        y = df['Churn'].map({'Yes': 1, 'No': 0})
        return X, y


class Preprocessor:
    def __init__(self):
        self.numerical_cols = None
        self.categorical_cols = None

    def build(self, X: pd.DataFrame) -> ColumnTransformer:
        self.numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['str']).columns.tolist()
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ]
        )
        return preprocessor

    def save(self, preprocessor, X_train, X_test, y_train, y_test):
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        joblib.dump(preprocessor, PREPROCESSOR_PATH)
        joblib.dump(X_train, X_TRAIN_PATH)
        joblib.dump(X_test, X_TEST_PATH)
        joblib.dump(y_train, Y_TRAIN_PATH)
        joblib.dump(y_test, Y_TEST_PATH)
        print(f'Preprocessor saved: {PREPROCESSOR_PATH}')
        print(f'Train/Test splits saved in {ARTIFACTS_DIR}/')

    def load_preprocessor(self) -> ColumnTransformer:
        return joblib.load(PREPROCESSOR_PATH)

    def load_splits(self):
        return (
            joblib.load(X_TRAIN_PATH),
            joblib.load(X_TEST_PATH),
            joblib.load(Y_TRAIN_PATH),
            joblib.load(Y_TEST_PATH)
        )


class ModelTrainer:
    def __init__(self, model_type: str):
        self.model_type = model_type
        if model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'tree':
            self.model = DecisionTreeClassifier(max_depth=5, random_state=42)
        else:
            raise ValueError(f'Unknown model: {model_type}')

    def train(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        print(f'{self.model_type} model trained.')
        return self.model

    def save(self):
        path = LOGISTIC_PATH if self.model_type == 'logistic' else TREE_PATH
        joblib.dump(self.model, path)
        print(f'Model saved: {path}')


class Evaluator:
    def evaluate(self, model, X_test: np.ndarray, y_test: pd.Series):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'AUC': roc_auc_score(y_test, y_proba)
        }
        print('\n=== Evaluation Metrics ===')
        for name, value in metrics.items():
            print(f'{name:<12} {value:.4f}')
        print(f'\n{classification_report(y_test, y_pred)}')
        return metrics


def run_preprocessing():
    print('=== Step 1: Load Data ===')
    loader = DataLoader()
    df = loader.load_data(DATA_PATH)
    X, y = loader.split_features_target(df)

    print(f'\n=== Step 2: Split (80/20) ===')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'Train: {X_train.shape}, Test: {X_test.shape}')

    print(f'\n=== Step 3: Build Preprocessor ===')
    preprocessor = Preprocessor()
    column_transformer = preprocessor.build(X)
    column_transformer.fit(X_train)
    print(f'Numerical cols: {preprocessor.numerical_cols}')
    print(f'Categorical cols: {len(preprocessor.categorical_cols)} columns')

    preprocessor.save(column_transformer, X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test


def run_training(model_type: str):
    print(f'\n=== Train {model_type} model ===')
    preprocessor = Preprocessor()
    column_transformer = preprocessor.load_preprocessor()
    X_train, X_test, y_train, y_test = preprocessor.load_splits()

    X_train_processed = column_transformer.transform(X_train)
    X_test_processed = column_transformer.transform(X_test)
    print(f'X_train processed: {X_train_processed.shape}')
    print(f'X_test processed: {X_test_processed.shape}')

    trainer = ModelTrainer(model_type)
    model = trainer.train(X_train_processed, y_train)
    trainer.save()

    evaluator = Evaluator()
    evaluator.evaluate(model, X_test_processed, y_test)


def run_all():
    print('========================================')
    print('  Full Pipeline: Preprocessing + ML')
    print('========================================')
    X_train, X_test, y_train, y_test = run_preprocessing()
    for model_type in ['logistic', 'tree']:
        run_training(model_type)


def main():
    parser = argparse.ArgumentParser(description='Telco Churn - ML Pipeline')
    parser.add_argument('--preprocess', action='store_true', help='Run preprocessing')
    parser.add_argument('--model', type=str, choices=['logistic', 'tree'], help='Train model')
    parser.add_argument('--all', action='store_true', help='Run full pipeline')
    args = parser.parse_args()

    if args.preprocess:
        run_preprocessing()
    elif args.model:
        run_training(args.model)
    elif args.all:
        run_all()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
