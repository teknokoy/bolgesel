# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 16:36:23 2025

@author: dayı
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import warnings
import os
import lime
import lime.lime_tabular
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold, train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, ARDRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
    BaggingRegressor, ExtraTreesRegressor, StackingRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.inspection import permutation_importance
import shap

models_list = [
    "LinearRegression", "Ridge", "Lasso", "ElasticNet", "BayesianRidge", "ARDRegression",
    "SVR", "DecisionTree", "RandomForest", "GradientBoosting", "AdaBoost",
    "Bagging", "ExtraTrees", "XGBoost", "LightGBM", "CatBoost", "KNN", "MLP", "Stacking"
]

def get_model(trial, model_name, n_samples=None, stacking_cv=5):
    if model_name == "LinearRegression":
        return LinearRegression()
    elif model_name == "Ridge":
        return Ridge(alpha=trial.suggest_float("alpha", 1e-3, 100, log=True))
    elif model_name == "Lasso":
        return Lasso(alpha=trial.suggest_float("alpha", 1e-3, 100, log=True))
    elif model_name == "ElasticNet":
        return ElasticNet(
            alpha=trial.suggest_float("alpha", 1e-3, 100, log=True),
            l1_ratio=trial.suggest_float("l1_ratio", 0.0, 1.0)
        )
    elif model_name == "BayesianRidge":
        return BayesianRidge()
    elif model_name == "ARDRegression":
        return ARDRegression()
    elif model_name == "SVR":
        return SVR(
            C=trial.suggest_float("C", 0.1, 100, log=True),
            epsilon=trial.suggest_float("epsilon", 0.01, 1.0),
            kernel=trial.suggest_categorical("kernel", ["linear", "rbf"])
        )
    elif model_name == "DecisionTree":
        return DecisionTreeRegressor(max_depth=trial.suggest_int("max_depth", 3, 20))
    elif model_name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 200),
            max_depth=trial.suggest_int("max_depth", 3, 20),
            random_state=42
        )
    elif model_name == "GradientBoosting":
        return GradientBoostingRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 200),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            random_state=42
        )
    elif model_name == "AdaBoost":
        return AdaBoostRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 200),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
            random_state=42
        )
    elif model_name == "Bagging":
        return BaggingRegressor(n_estimators=trial.suggest_int("n_estimators", 10, 100))
    elif model_name == "ExtraTrees":
        return ExtraTreesRegressor(n_estimators=trial.suggest_int("n_estimators", 50, 200))
    elif model_name == "XGBoost":
        return XGBRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 200),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            objective="reg:squarederror", verbosity=0, random_state=42
        )
    elif model_name == "LightGBM":
        return LGBMRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 200),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            random_state=42
        )
    elif model_name == "CatBoost":
        return CatBoostRegressor(
            iterations=trial.suggest_int("iterations", 50, 200),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            depth=trial.suggest_int("depth", 3, 10),
            verbose=0,
            random_state=42
        )
    elif model_name == "KNN":
        max_n_neighbors = max(2, n_samples) if n_samples is not None else 5
        n_neighbors = trial.suggest_int("n_neighbors", 2, min(20, max_n_neighbors))
        return KNeighborsRegressor(n_neighbors=n_neighbors)
    elif model_name == "MLP":
        return MLPRegressor(
            hidden_layer_sizes=trial.suggest_categorical("hidden_layer_sizes", [(64,), (128,), (128,64)]),
            alpha=trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            learning_rate_init=trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
            max_iter=500, random_state=42
        )
    elif model_name == "Stacking":
        stacking_cv = min(stacking_cv, n_samples) if n_samples is not None else 2
        if stacking_cv < 2:
            stacking_cv = 2
        base_models = [
            ("rf", RandomForestRegressor(n_estimators=50, random_state=42)),
            ("xgb", XGBRegressor(n_estimators=50, random_state=42)),
            ("ridge", Ridge(alpha=1.0))
        ]
        final_estimator = Ridge()
        return StackingRegressor(estimators=base_models, final_estimator=final_estimator, cv=stacking_cv)

def calculate_lime_importance(model, X_train, feature_names):
    """Calculate LIME feature importance."""
    try:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=feature_names,
            mode="regression"
        )
        # Get importance for first instance as representative
        exp = explainer.explain_instance(X_train[0], model.predict)
        importance_dict = {}
        for feat, imp in exp.as_list():
            importance_dict[feat] = abs(imp)
        return pd.Series(importance_dict)
    except Exception as e:
        print(f"LIME hesaplanamadı: {e}")
        return pd.Series(dtype=float)

def calculate_all_feature_importance(model, X, X_train, name_prefix):
    """Calculate and save all feature importance methods."""
    importance_results = {}
    
    # Permutation Importance
    try:
        result = permutation_importance(model, X, model.predict(X), n_repeats=5)
        importance_results['permutation'] = pd.Series(
            result.importances_mean, 
            index=X.columns
        ).sort_values(ascending=False)
    except Exception as e:
        print(f"Permutation importance hesaplanamadı: {e}")
        importance_results['permutation'] = pd.Series(dtype=float)

    # SHAP
    try:
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            shap_importance = pd.Series(
                np.abs(shap_values).mean(axis=0),
                index=X.columns
            ).sort_values(ascending=False)
            importance_results['shap'] = shap_importance
            
            # Save SHAP plot
            shap.summary_plot(shap_values, X, show=False)
            plt.title(f"SHAP Importance: {name_prefix}")
            plt.tight_layout()
            plt.savefig(f"shap_{name_prefix}.png")
            plt.close()
    except Exception as e:
        print(f"SHAP hesaplanamadı: {e}")
        importance_results['shap'] = pd.Series(dtype=float)

    # LIME
    importance_results['lime'] = calculate_lime_importance(
        model, 
        X_train,
        X.columns
    )

    return importance_results

def plot_and_save(y_true, y_pred, name_prefix):
    # Convert lists to NumPy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Scatter plot
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red')
    plt.xlabel("Gerçek Rekolte")
    plt.ylabel("Tahmin")
    plt.title(f"Gerçek vs Tahmin: {name_prefix}")
    plt.grid(True)
    plt.savefig(f"scatter_{name_prefix}.png")
    plt.close()

    # Residual plot
    residuals = y_true - y_pred
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Tahmin")
    plt.ylabel("Residual")
    plt.title(f"Residuals: {name_prefix}")
    plt.grid(True)
    plt.savefig(f"residual_{name_prefix}.png")
    plt.close()

def plot_learning_curve(estimator, X, y, name_prefix):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=3, n_jobs=-1,
        train_sizes=np.linspace(0.2, 1.0, 5),
        scoring='r2'
    )
    plt.figure(figsize=(7,5))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Train')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='CV')
    plt.xlabel("Eğitim Boyutu")
    plt.ylabel("R2")
    plt.title(f"Learning Curve: {name_prefix}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"learning_curve_{name_prefix}.png")
    plt.close()

def process_csv_file(csv_path, bolge_df):
    """Process a single CSV file and return results."""
    print(f"\nProcessing {os.path.basename(csv_path)}...")
    
    # Read data
    df = pd.read_csv(csv_path)
    
    # Get unique areas
    region_names = bolge_df.iloc[:, 0]
    area_names = bolge_df.iloc[:, 1]
    unique_areas = area_names.unique()
    
    # Initialize results storage
    area_results = {area: [] for area in unique_areas}
    area_importances = {area: {} for area in unique_areas}
    
    # Process each area
    for area in unique_areas:
        print(f"\n=== {area} için analiz başlıyor ===")
        area_regions = region_names[area_names == area].tolist()
        area_df = df[df['region_name'].isin(area_regions)].copy()
        
        # Drop unnecessary columns
        drop_cols = [
            'Buday', 'ekim', 'HERBST Toplam', 'FUNGUST Toplam', 'dap', 're',
            '20-20-0', '0.33', '20-32-0', 'date', 
            'weather_wind_speed', 'weather_wind_direction', 'weather_pressure_kpa',
            'weather_evaporation_mm', 'TVI', 'DVI', 'RVI', 'MSAVI', 'NDRE'
        ]
        
        n_samples = len(area_df)
        if n_samples < 2:
            print(f"⚠️ {area} için yeterli veri yok ({n_samples} satır), atlanıyor!")
            continue
            
        y = area_df['rekolte_x'].values.astype(np.float32)
        X = area_df.drop(columns=drop_cols + ['rekolte_x', 'region_name'], errors='ignore')
        X.columns = X.columns.str.strip()
        
        # Remove any columns with all NaN values
        X = X.dropna(axis=1, how='all')
        # Fill remaining NaN values with column means
        X = X.fillna(X.mean())
        
        n_samples = len(X)
        recommended_min = min_required_data(len(X.columns))
        
        # Process models
        for model_name in models_list:
            print(f"\u2699\ufe0f Model: {model_name}")
            
            try:
                # For LinearRegression and other models without hyperparameters,
                # skip optimization and use default parameters
                if model_name in ["LinearRegression", "BayesianRidge", "ARDRegression"]:
                    if model_name == "LinearRegression":
                        model = LinearRegression()
                    elif model_name == "BayesianRidge":
                        model = BayesianRidge()
                    else:
                        model = ARDRegression()
                    
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    model.fit(X_scaled, y)
                    best_params = {}
                    
                else:
                    # Optimize model
                    study = optuna.create_study(direction="maximize")
                    study.optimize(lambda trial: objective(trial, X, y, model_name, n_splits=min(5, n_samples)), n_trials=15)
                    
                    # Check if any trials completed successfully
                    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                    if not completed_trials:
                        print(f"⚠️ {model_name} için başarılı deneme yok, atlanıyor!")
                        continue
                    
                    # Get best model
                    best_trial = study.best_trial
                    best_params = best_trial.params
                    
                    # Train final model
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    if model_name == "Stacking":
                        model = get_model(optuna.trial.FixedTrial(best_params), model_name, n_samples=len(X), stacking_cv=min(5, n_samples))
                    else:
                        model = get_model(optuna.trial.FixedTrial(best_params), model_name, n_samples=len(X))
                    model.fit(X_scaled, y)
                
                # Calculate metrics
                y_pred = model.predict(X_scaled)
                metrics = {
                    "R2": r2_score(y, y_pred),
                    "MAE": mean_absolute_error(y, y_pred),
                    "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
                    "MAPE": mean_absolute_percentage_error(y, y_pred)
                }
                
                # Calculate feature importance
                importance_results = calculate_all_feature_importance(
                    model,
                    X,
                    X_scaled,
                    f"{area}_{model_name}"
                )
                
                area_importances[area][model_name] = importance_results
                
                # Save plots
                plot_and_save(y, y_pred, f"{area}_{model_name}")
                plot_learning_curve(model, X_scaled, y, f"{area}_{model_name}")
                
                # Save results
                area_results[area].append({
                    "Alan": area,
                    "Model": model_name,
                    "R2": metrics["R2"],
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "MAPE": metrics["MAPE"],
                    "Best_Params": str(best_params),
                    "Veri_Sayisi": len(X),
                    "Min_Veri_Oneri": recommended_min
                })
                
            except Exception as e:
                print(f"⚠️ {model_name} modeli için hata: {str(e)}")
                area_results[area].append({
                    "Alan": area,
                    "Model": model_name,
                    "R2": np.nan,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "MAPE": np.nan,
                    "Best_Params": "Error",
                    "Veri_Sayisi": len(X),
                    "Min_Veri_Oneri": recommended_min,
                    "Error": str(e)
                })
                continue
    
    # Save results for this CSV file
    output_filename = f"results_{os.path.splitext(os.path.basename(csv_path))[0]}.xlsx"
    with pd.ExcelWriter(output_filename) as writer:
        # Save overall results
        all_results = []
        for area in unique_areas:
            all_results.extend(area_results[area])
        pd.DataFrame(all_results).to_excel(writer, sheet_name="Overall_Results", index=False)
        
        # Save feature importance results for each area and model
        for area in unique_areas:
            if area in area_importances:
                for model_name, importance_dict in area_importances[area].items():
                    for method, importance in importance_dict.items():
                        if not importance.empty:
                            sheet_name = f"{area}_{model_name}_{method}"[:31]  # Excel sheet name length limit
                            importance.to_excel(writer, sheet_name=sheet_name)

def min_required_data(n_features):
    return max(30, n_features * 5)

def objective(trial, X, y, model_name, n_splits):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        try:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            if model_name == "Stacking":
                stacking_cv = min(n_splits, len(X_train))
                model = get_model(trial, model_name, n_samples=len(X_train), stacking_cv=stacking_cv)
            else:
                model = get_model(trial, model_name, n_samples=len(X_train))
                
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_val_scaled)
            score = r2_score(y_val, preds)
            if np.isnan(score) or np.isinf(score):
                scores.append(-999)
            else:
                scores.append(score)
        except Exception as e:
            print(f"Trial error: {str(e)}")
            scores.append(-999)
            
    return np.mean([s for s in scores if s != -999]) if any(s != -999 for s in scores) else -999

def main():
    # Read bolge.xlsx
    bolge_df = pd.read_excel("/kaggle/input/sentinel-spili/bolge.xlsx")
    
    # Process all CSV files in the Kaggle input directory
    input_dir = "/kaggle/input/sentinel-spili"
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        csv_path = os.path.join(input_dir, csv_file)
        process_csv_file(csv_path, bolge_df)
        print(f"\nCompleted processing {csv_file}")

if __name__ == "__main__":
    main() 