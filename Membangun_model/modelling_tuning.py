import pandas as pd
import argparse
import os
import mlflow
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def load_data(input_dir):
    train_path = os.path.join(input_dir, 'train.csv')
    test_path = os.path.join(input_dir, 'test.csv')
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    target_col = 'Heart Disease Status'
    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_test = test.drop(columns=[target_col])
    y_test = test[target_col]
    
    return X_train, y_train, X_test, y_test

def run_tuning(input_dir):
    # Setup DagsHub
    dagshub.init(repo_owner='rafli21xrplc', repo_name='asah-ml-flow', mlflow=True)
    mlflow.set_experiment("Heart Disease Tuning")
    
    X_train, y_train, X_test, y_test = load_data(input_dir)
    
    # Define Parameter Grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }
    
    print("Starting Grid Search...")
    
    # Kita tidak pakai GridSearchCV bawaan sklearn untuk logging otomatis yang detail,
    # tapi kita buat loop manual atau pakai trik callback agar setiap iterasi tercatat.
    # Untuk simplicitas & best practice MLflow, kita iterasi manual:
    
    for n_est in param_grid['n_estimators']:
        for depth in param_grid['max_depth']:
            for split in param_grid['min_samples_split']:
                
                with mlflow.start_run(run_name=f"RF_n{n_est}_d{depth}"):
                    # Train
                    clf = RandomForestClassifier(
                        n_estimators=n_est, 
                        max_depth=depth, 
                        min_samples_split=split,
                        random_state=42
                    )
                    clf.fit(X_train, y_train)
                    
                    # Evaluate
                    preds = clf.predict(X_test)
                    acc = accuracy_score(y_test, preds)
                    
                    # Log Params
                    mlflow.log_param("n_estimators", n_est)
                    mlflow.log_param("max_depth", depth)
                    mlflow.log_param("min_samples_split", split)
                    
                    # Log Metric
                    mlflow.log_metric("accuracy", acc)
                    
                    print(f"Logged: n_est={n_est}, depth={depth}, split={split} -> Acc={acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='processed_data', help='Input directory')
    args = parser.parse_args()
    
    run_tuning(args.input)