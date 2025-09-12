import os
import sys
import time
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, log_loss
from scipy.stats import binned_statistic_2d
import json
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.samplers import TPESampler

import onnxmltools
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
import tempfile
import xgboost as xgb

# Configuration Constants
DATA_DIR = 'data/'
OUTPUT_DIR = 'model_output'
MODEL_OBJ = 'binary:logistic'
MODEL_VERSION = 'v5'
SKIP_HYPERPARAMETER_TUNING = False
SKIP_ALL_TRAINING = SKIP_HYPERPARAMETER_TUNING
SAVE_MODEL = True
DRAW_STRIKE_ZONE_HEATMAP = False

# Optuna configuration
OPTUNA_N_TRIALS = 200
OPTUNA_TIMEOUT = 3600  # 1 hour timeout

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    MODEL_PATH = os.path.join(OUTPUT_DIR, f'model_{MODEL_VERSION}.json')

    print("=" * 60)
    print("STRIKE PROBABILITY MODEL TRAINING")
    print("=" * 60)

    raw_data = load_data()
    data = process_data(raw_data)

    feature_columns = [
        'b_hits_encoded', 'p_throws_encoded',
        'pfx_x', 'pfx_z',
        'x_norm', 'z_norm',
        'same_handedness',
        'balls', 'strikes', 'outs_when_up'
    ]

    X = data[feature_columns]
    y = data['is_strike']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    print()

    if not SKIP_ALL_TRAINING:
        print("PHASE 1: Train baseline model")
        print("-" * 40)
        baseline_model = train_baseline_model(X_train, y_train, X_val, y_val)
        baseline_score = evaluate_model_quick(baseline_model, X_val, y_val, "Baseline Validation")

        if not SKIP_HYPERPARAMETER_TUNING:
            print("-" * 40)
            tuned_model, best_params, tuning_score = tune_hyperparameters_optuna(
                X_train, y_train, X_val, y_val
            )

            improvement = baseline_score - tuning_score  # Lower is better for log loss
            print("\nOptuna Tuning Results:")
            print(f"Baseline Log Loss: {baseline_score:.4f}")
            print(f"Tuned Log Loss: {tuning_score:.4f}")
            print(f"Improvement: {improvement:.4f}")

            final_model = tuned_model
            print("✓ Using tuned model")
        else:
            final_model = baseline_model
            print("✓ Skipping hyperparameter tuning as per configuration")
    else:
        final_model = xgb.XGBClassifier()
        final_model.load_model(MODEL_PATH)
        print("✓ Loaded pre-trained model from disk")

    print("\nPHASE 3: Final model evaluation...")
    print("-" * 40)
    final_score = evaluate_model_comprehensive(final_model, X_test, y_test, "Final Test")

    print("\nPHASE 4: Model insights...")
    print("-" * 40)

    if SAVE_MODEL and (not SKIP_ALL_TRAINING):
        print("\nPHASE 5: Saving model and inference data...")
        print("-" * 40)

        save_model_and_metadata(final_model, feature_columns, MODEL_PATH)

    if not SKIP_ALL_TRAINING:
        # Summary
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Baseline Log Loss: {baseline_score:.4f}")
        print(f"Final Test Log Loss: {final_score:.4f}")
        print(f"Model saved: {'Yes' if SAVE_MODEL else 'No'}")
        if SAVE_MODEL:
            print(f"Output directory: {OUTPUT_DIR}")
        print("=" * 60)

    if DRAW_STRIKE_ZONE_HEATMAP:
        draw_strikezone_heat_map(data, final_model)

def load_data():
    print("Loading data...")
    data_frames = []

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory {DATA_DIR} not found. Please ensure your data files are in a {DATA_DIR} subdirectory.")

    data_files = [f for f in os.listdir(DATA_DIR) if f.startswith('filtered_pitches_') and f.endswith('.csv')]

    if not data_files:
        raise FileNotFoundError(f"No files matching pattern 'filtered_pitches_*.csv' found in {DATA_DIR} directory.")

    print(f"Found {len(data_files)} data files:")
    for file in data_files:
        print(f"  - {file}")
        df = pd.read_csv(os.path.join(DATA_DIR, file))
        data_frames.append(df)

    data = pd.concat(data_frames, ignore_index=True)
    data = data.rename(columns={'stand': 'b_hits', 'type': 'call'})

    required_columns = [
        'b_hits', 'p_throws',
        'balls', 'strikes', 'outs_when_up',
        'pfx_x', 'pfx_z',
        'plate_x', 'plate_z',
        'sz_top', 'sz_bot',
        'call'
    ]

    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"Available columns: {list(data.columns)}")
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = data[required_columns]
    print(f"Loaded {len(data)} rows of data.")
    return data

def process_data(data):
    print("Processing data...")

    data = data.copy()
    data = data.dropna()

    le_batter = LabelEncoder()
    le_pitcher = LabelEncoder()

    data = data.assign(
        b_hits_encoded=le_batter.fit_transform(data['b_hits']),  # L/R -> 0/1
        p_throws_encoded=le_pitcher.fit_transform(data['p_throws']),  # L/R -> 0/1

        same_handedness=(data['b_hits'] == data['p_throws']).astype(int),

        x_norm=np.where(data['b_hits'] == 'L', -data['plate_x'], data['plate_x']),
        z_norm=(data['plate_z'] - data['sz_bot']) / (data['sz_top'] - data['sz_bot']),

        is_strike=(data['call'] == 'S').astype(int),
    )

    print("Finished processing data.\n")
    return data

def train_baseline_model(X_train, y_train, X_val, y_val):
    print("Training baseline XGBoost model...")

    baseline_params = {
        'objective': MODEL_OBJ,
        'eval_metric': 'logloss',
        'max_depth': 4,
        'learning_rate': 0.4,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_estimators': 100,
        'verbosity': 0,
        'early_stopping_rounds': 100
    }

    model = xgb.XGBClassifier(**baseline_params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )

    return model

def evaluate_model_quick(model, X, y, dataset_name="Dataset"):
    y_pred_proba = model.predict_proba(X)
    logloss = log_loss(y, y_pred_proba)
    print(f"✓ {dataset_name} Log Loss: {logloss:.4f}")
    return logloss

def evaluate_model_comprehensive(model, X_test, y_test, dataset_name="Test"):
    print(f"\nComprehensive evaluation on {dataset_name} set...")

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])  # Keep AUC for reference

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Log Loss: {logloss:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ball', 'Strike']))

    return logloss

def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function for hyperparameter optimization."""

    # Define hyperparameter search space
    params = {
        'objective': MODEL_OBJ,
        'eval_metric': 'logloss',
        'verbosity': 0,
        'random_state': 42,

        # Hyperparameters to optimize
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
        'gamma': trial.suggest_float('gamma', 1e-8, 10.0, log=True),
        'early_stopping_rounds': 50
    }

    # Create and train model
    model = xgb.XGBClassifier(**params)

    # Use early stopping to prevent overfitting
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Evaluate on validation set - return negative log loss since Optuna maximizes
    y_pred_proba = model.predict_proba(X_val)
    logloss = log_loss(y_val, y_pred_proba)

    return -logloss  # Negative because Optuna maximizes, but we want to minimize log loss

def tune_hyperparameters_optuna(X_train, y_train, X_val, y_val):
    """Tune hyperparameters using Optuna optimization."""
    print("Starting Optuna hyperparameter optimization...")
    start_time = time.time()

    # Create study
    study = optuna.create_study(
        direction='maximize',  # We want to maximize negative log loss (minimize log loss)
        sampler=TPESampler(seed=42),  # Tree-structured Parzen Estimator
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=20,  # Don't prune for first 20 trials
            n_warmup_steps=30,    # Don't prune for first 30 steps of each trial
            interval_steps=10     # Prune every 10 steps
        )
    )

    # Optimize
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=OPTUNA_N_TRIALS,
        timeout=OPTUNA_TIMEOUT,
        show_progress_bar=True
    )

    elapsed_time = time.time() - start_time

    print(f"✓ Optuna optimization complete ({elapsed_time:.1f}s)")
    print(f"✓ Number of trials: {len(study.trials)}")
    print(f"✓ Best trial Log Loss: {-study.best_value:.4f}")  # Convert back to positive

    print("\nBest parameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")

    # Train final model with best parameters
    print("\nTraining final model with best parameters...")
    best_params = study.best_params.copy()
    best_params.update({
        'objective': MODEL_OBJ,
        'eval_metric': 'logloss',
        'verbosity': 0,
        'random_state': 42,
        'early_stopping_rounds': 50
    })

    best_model = xgb.XGBClassifier(**best_params)
    best_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )

    # Final validation score
    y_val_pred_proba = best_model.predict_proba(X_val)
    val_logloss = log_loss(y_val, y_val_pred_proba)

    print(f"✓ Final validation Log Loss: {val_logloss:.4f}")

    # Save optimization results
    study_path = os.path.join(OUTPUT_DIR, f'optuna_study_{MODEL_VERSION}.pkl')
    try:
        import joblib
        joblib.dump(study, study_path)
        print(f"✓ Optuna study saved to {study_path}")
    except ImportError:
        print("✓ Optuna study not saved (joblib not available)")

    # Print feature importance from best model
    print("\nFeature importance (top 5):")
    feature_names = [
        'b_hits_encoded', 'p_throws_encoded',
        'pfx_x', 'pfx_z', 'x_norm', 'z_norm',
        'same_handedness', 'balls', 'strikes', 'outs_when_up'
    ]

    importances = best_model.feature_importances_
    feature_importance = list(zip(feature_names, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    for name, importance in feature_importance[:5]:
        print(f"  {name}: {importance:.4f}")

    return best_model, study.best_params, val_logloss

def save_model_and_metadata(final_model, feature_columns, model_path):
    final_model.save_model(model_path)
    print(f"✓ Model saved to {model_path}")

    feature_mapping = {f'f{i}': name for i, name in enumerate(feature_columns)}
    mapping_path = os.path.join(OUTPUT_DIR, f'feature_mapping_{MODEL_VERSION}.json')
    with open(mapping_path, 'w') as f:
        json.dump(feature_mapping, f, indent=2)
    print(f"✓ Feature mapping saved to {mapping_path}")

    try:
        booster = final_model.get_booster()

        original_feature_names = booster.feature_names
        new_feature_names = [f'f{i}' for i in range(len(original_feature_names))]

        booster.feature_names = new_feature_names

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_file:
            booster.save_model(tmp_file.name)
            temp_model = xgb.XGBClassifier()
            temp_model.load_model(tmp_file.name)

            initial_type = [('float_input', FloatTensorType([None, len(feature_columns)]))]

            onnx_model = onnxmltools.convert_xgboost(
                temp_model, 
                initial_types=initial_type,
                target_opset=12
            )

            onnx_path = model_path.replace('.json', '.onnx')
            with open(onnx_path, 'wb') as f:
                f.write(onnx_model.SerializeToString())
            print(f"✓ ONNX model saved to {onnx_path}")

            os.unlink(tmp_file.name)

        booster.feature_names = original_feature_names

    except ImportError as e:
        print(f"✓ ONNX conversion skipped (missing dependencies: {e})")
        print("  Install with: pip install onnxmltools onnxruntime")
    except Exception as e:
        print(f"✓ ONNX conversion with onnxmltools failed: {e}")

def draw_strikezone_heat_map(data, model):
    features = data[[
        'b_hits_encoded', 'p_throws_encoded',
        'pfx_x', 'pfx_z',
        'x_norm', 'z_norm',
        'same_handedness', 'balls', 'strikes', 'outs_when_up'
    ]]
    probs = model.predict_proba(features)[:, 1]

    heatmap_data = data[['plate_x', 'plate_z']].copy()
    heatmap_data['strike_prob'] = probs

    sz_top = 3.5
    sz_bot = 1.5
    sz_left = -0.7083
    sz_right = 0.7083

    xedges = np.arange(-2, 2.01, 0.05)
    yedges = np.arange(0, 5.01, 0.05)

    stat, xedge, yedge, binnumber = binned_statistic_2d(
        heatmap_data['plate_x'], heatmap_data['plate_z'],
        heatmap_data['strike_prob'],
        statistic='mean',
        bins=[xedges, yedges]
    )

    fig, ax = plt.subplots(figsize=(6, 8))

    X, Y = np.meshgrid(xedges, yedges)
    pcm = ax.pcolormesh(X, Y, stat.T, cmap='coolwarm', vmin=0, vmax=1, shading='auto')

    strike_zone = plt.Rectangle((sz_left, sz_bot), sz_right - sz_left, sz_top - sz_bot,
                                edgecolor='black', facecolor='none', lw=2)
    ax.add_patch(strike_zone)

    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 5)
    ax.set_xlabel('Horizontal Position (ft)')
    ax.set_ylabel('Vertical Position (ft)')
    ax.set_title('Strike Zone Heat Map')

    ax.set_aspect('equal')

    ax.set_xticks(np.arange(-2, 2.1, 0.5))
    ax.set_yticks(np.arange(0, 5.1, 0.5))

    fig.colorbar(pcm, ax=ax, label="Strike Probability")

    plt.show()

if __name__ == "__main__":
    main()
