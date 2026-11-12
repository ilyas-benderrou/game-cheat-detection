"""
Anti-cheat detection pipeline (reproducible version of the original notebook).

Run from the project root:
    python src/anticheat_pipeline.py

Steps
-----
1. Load the Kaggle dataset "Online Gaming Behavior and Cheating Metrics".
2. Clean it (missing `Items` -> "No_Item").
3. Feature engineering: Aim_Score, Combat_Score, Activity_Score.
4. Exploratory figures + a "separability" sanity check.
5. Stratified 80/20 train/test split.
6. Train 5 models:
     - supervised   : Logistic Regression, Random Forest
     - unsupervised : Isolation Forest, One-Class SVM, MLP autoencoder
       (the unsupervised models only see *normal* players during training)
7. Metrics: accuracy, precision, recall, F1, ROC-AUC + confusion matrices.
8. Feature importance (Random Forest).
9. Extra checks: 5-fold cross-validation of the supervised models and an
   ablation without the "aim" features, to show how much the result depends
   on a few extremely separating columns.

Every number written in the README comes from the files this script writes
into `outputs/`.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # no window needed: we only save PNG files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import OneClassSVM

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
PROJECT_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_DIR / "figures"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
MODELS_DIR = PROJECT_DIR / "models"

TARGET = "Cheat_Likelihood"
ID_COLUMNS = ["Player_ID", "Game_Session_ID"]
# Raw "aim/combat" columns and the engineered features built from them.
AIM_FEATURES = ["KD_Ratio", "Accuracy", "Headshot_Rate", "Aim_Score", "Combat_Score"]


# --------------------------------------------------------------------------- #
# 1-3. Data loading, cleaning, feature engineering
# --------------------------------------------------------------------------- #
def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    """Read the CSV, fill missing Items and add the three engineered features."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Download 'Online Gaming Behavior and Cheating Metrics' from Kaggle and "
            "place online_gaming_behavior.csv in the data/ folder (see README)."
        )
    df = pd.read_csv(csv_path)
    # Note: pandas reads the literal string "None" as a missing value, so the
    # players who bought no item end up as NaN. We give them an explicit label.
    n_missing = int(df["Items"].isna().sum())
    df["Items"] = df["Items"].fillna("No_Item")

    df["Aim_Score"] = df["Accuracy"] * df["Headshot_Rate"]
    df["Combat_Score"] = df["KD_Ratio"] * df["Accuracy"]
    df["Activity_Score"] = df["APM"] / df["Session_Duration"]

    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns "
          f"({n_missing} missing 'Items' replaced by 'No_Item').")
    return df


# --------------------------------------------------------------------------- #
# 4. Exploratory figures and sanity check
# --------------------------------------------------------------------------- #
def eda_figures(df: pd.DataFrame) -> None:
    """Class balance, correlation matrix and per-class histograms."""
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Normal (0)", "Suspect (1)"], counts.values, color=["#4C72B0", "#DD8452"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v} ({v / counts.sum():.1%})", ha="center", va="bottom")
    ax.set_title("Class distribution")
    ax.set_ylabel("Number of sessions")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_class_distribution.png", dpi=150)
    plt.close(fig)

    numeric_cols = ["Session_Duration", "APM", "KD_Ratio", "Accuracy", "Headshot_Rate",
                    "Aim_Score", "Combat_Score", "Activity_Score"]
    corr = df[numeric_cols + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax)
    ax.set_title("Correlation matrix")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_correlation_matrix.png", dpi=150)
    plt.close(fig)

    cols = ["APM", "KD_Ratio", "Accuracy", "Headshot_Rate", "Aim_Score", "Combat_Score"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), cols):
        ax.hist(df.loc[df[TARGET] == 0, col], bins=40, alpha=0.7, label="Normal", density=True)
        ax.hist(df.loc[df[TARGET] == 1, col], bins=40, alpha=0.7, label="Suspect", density=True)
        ax.set_title(col)
        ax.legend()
    fig.suptitle("Feature distributions by class (normalised histograms)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_feature_distributions.png", dpi=150)
    plt.close(fig)


def separability_check(df: pd.DataFrame) -> pd.DataFrame:
    """For each numeric feature, compare the value ranges of both classes.

    If max(normal) < min(suspect), a single threshold separates the two
    classes perfectly. This explains the "too good" supervised scores.
    """
    rows = []
    for col in ["Session_Duration", "APM", "KD_Ratio", "Accuracy", "Headshot_Rate",
                "Aim_Score", "Combat_Score", "Activity_Score"]:
        normal = df.loc[df[TARGET] == 0, col]
        cheat = df.loc[df[TARGET] == 1, col]
        rows.append({
            "Feature": col,
            "Normal_min": normal.min(), "Normal_max": normal.max(),
            "Suspect_min": cheat.min(), "Suspect_max": cheat.max(),
            "Perfectly_separable_by_one_threshold": bool(normal.max() < cheat.min()),
        })
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUTS_DIR / "separability_check.csv", index=False)
    print("\nSeparability check (one threshold enough?):")
    print(table.to_string(index=False))
    return table


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Standardise numeric columns and one-hot encode categorical columns."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
        sparse_threshold=0.0,  # always return a dense numpy array
    )


def metrics_row(name: str, kind: str, y_true, y_pred, y_score) -> dict:
    """All metrics for one model. `y_score` = higher means more suspicious."""
    return {
        "Model": name,
        "Type": kind,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_score),
        "TN_FP_FN_TP": "/".join(str(v) for v in confusion_matrix(y_true, y_pred).ravel()),
    }


# --------------------------------------------------------------------------- #
# 5-8. Modelling
# --------------------------------------------------------------------------- #
def run_models(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, Pipeline, pd.Series]:
    X = df.drop(columns=[TARGET] + ID_COLUMNS)
    y = df[TARGET]
    categorical = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}, "
          f"suspect ratio train={y_train.mean():.4f}, test={y_test.mean():.4f}")

    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rows = []

    # ---- Model 1: Logistic Regression (supervised baseline) ----------------
    log_model = Pipeline([
        ("preprocess", make_preprocessor(numeric, categorical)),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced",
                                     random_state=RANDOM_STATE)),
    ]).fit(X_train, y_train)
    score = log_model.predict_proba(X_test)[:, 1]
    pred = log_model.predict(X_test)
    predictions["Logistic Regression"] = (pred, score)
    rows.append(metrics_row("Logistic Regression", "Supervised", y_test, pred, score))
    joblib.dump(log_model, MODELS_DIR / "logistic_regression.joblib")

    # ---- Model 2: Random Forest (supervised) --------------------------------
    rf_model = Pipeline([
        ("preprocess", make_preprocessor(numeric, categorical)),
        ("model", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                         random_state=RANDOM_STATE, n_jobs=-1)),
    ]).fit(X_train, y_train)
    score = rf_model.predict_proba(X_test)[:, 1]
    pred = rf_model.predict(X_test)
    predictions["Random Forest"] = (pred, score)
    rows.append(metrics_row("Random Forest", "Supervised", y_test, pred, score))
    joblib.dump(rf_model, MODELS_DIR / "random_forest.joblib")

    # ---- Unsupervised models: trained on NORMAL players only ----------------
    # They learn what "normal" looks like and flag what is far from it.
    # The labels are only used afterwards, to evaluate them.
    X_train_normal = X_train[y_train == 0]
    prep_unsup = make_preprocessor(numeric, categorical)
    Xn = prep_unsup.fit_transform(X_train_normal)
    Xt = prep_unsup.transform(X_test)
    joblib.dump(prep_unsup, MODELS_DIR / "preprocess_unsupervised.joblib")
    fraud_ratio = float(y_train.mean())  # expected share of suspects

    # ---- Model 3: Isolation Forest -----------------------------------------
    iso = IsolationForest(n_estimators=300, contamination=fraud_ratio,
                          random_state=RANDOM_STATE, n_jobs=-1).fit(Xn)
    pred = (iso.predict(Xt) == -1).astype(int)        # -1 = anomaly = suspect
    score = -iso.score_samples(Xt)                    # higher = more anomalous
    predictions["Isolation Forest"] = (pred, score)
    rows.append(metrics_row("Isolation Forest", "Unsupervised", y_test, pred, score))
    joblib.dump(iso, MODELS_DIR / "isolation_forest.joblib")

    # ---- Model 4: One-Class SVM --------------------------------------------
    ocsvm = OneClassSVM(nu=fraud_ratio, kernel="rbf", gamma="scale").fit(Xn)
    pred = (ocsvm.predict(Xt) == -1).astype(int)
    score = -ocsvm.decision_function(Xt)
    predictions["One-Class SVM"] = (pred, score)
    rows.append(metrics_row("One-Class SVM", "Unsupervised", y_test, pred, score))
    joblib.dump(ocsvm, MODELS_DIR / "one_class_svm.joblib")

    # ---- Model 5: MLP autoencoder ------------------------------------------
    # Input = output: the network must rebuild a normal player through a
    # bottleneck of 8 neurons. A cheater is rebuilt badly -> large error.
    ae = MLPRegressor(hidden_layer_sizes=(16, 8, 16), activation="relu", solver="adam",
                      max_iter=500, early_stopping=True, validation_fraction=0.1,
                      random_state=RANDOM_STATE).fit(Xn, Xn)
    train_err = np.mean((Xn - ae.predict(Xn)) ** 2, axis=1)
    test_err = np.mean((Xt - ae.predict(Xt)) ** 2, axis=1)
    threshold = np.quantile(train_err, 1 - fraud_ratio)
    pred = (test_err > threshold).astype(int)
    predictions["Autoencoder (MLP)"] = (pred, test_err)
    rows.append(metrics_row("Autoencoder (MLP)", "Unsupervised", y_test, pred, test_err))
    joblib.dump(ae, MODELS_DIR / "autoencoder_mlp.joblib")
    plot_reconstruction_errors(test_err, y_test, threshold)

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUTS_DIR / "model_comparison_results.csv", index=False)
    print("\nTest-set results:")
    print(results.round(4).to_string(index=False))

    plot_confusion_matrices(predictions, y_test)
    plot_metric_comparison(results)
    plot_roc_curves(predictions, y_test)
    feature_importance(rf_model)
    return results, {"X": X, "y": y, "numeric": numeric, "categorical": categorical}, rf_model, y_test


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_reconstruction_errors(errors, y_test, threshold) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.logspace(np.log10(errors.min()), np.log10(errors.max()), 50)
    ax.hist(errors[y_test.values == 0], bins=bins, alpha=0.7, label="Normal")
    ax.hist(errors[y_test.values == 1], bins=bins, alpha=0.7, label="Suspect")
    ax.axvline(threshold, color="k", linestyle="--", label="Threshold")
    ax.set_xscale("log")
    ax.set_xlabel("Reconstruction error (log scale)")
    ax.set_ylabel("Number of test sessions")
    ax.set_title("Autoencoder reconstruction error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "07_autoencoder_reconstruction_errors.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrices(predictions: dict, y_test) -> None:
    fig, axes = plt.subplots(1, len(predictions), figsize=(4 * len(predictions), 3.8))
    for ax, (name, (pred, _)) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_test, pred)
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
        ax.set_xticks([0, 1], ["Normal", "Suspect"])
        ax.set_yticks([0, 1], ["Normal", "Suspect"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_confusion_matrices.png", dpi=150)
    plt.close(fig)


def plot_metric_comparison(results: pd.DataFrame) -> None:
    metrics = ["Precision", "Recall", "F1", "ROC_AUC"]
    x = np.arange(len(results))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, results[m], width, label=m.replace("_", "-"))
    ax.set_xticks(x, results["Model"])
    ax.set_ylim(0, 1.2)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Score (test set)")
    ax.set_title("Model comparison")
    ax.legend(ncol=4, loc="upper center")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_model_comparison.png", dpi=150)
    plt.close(fig)


def plot_roc_curves(predictions: dict, y_test) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, (_, score) in predictions.items():
        fpr, tpr, _ = roc_curve(y_test, score)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, score):.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test set)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "06_roc_curves.png", dpi=150)
    plt.close(fig)


def feature_importance(rf_model: Pipeline) -> None:
    names = rf_model.named_steps["preprocess"].get_feature_names_out()
    imp = pd.DataFrame({
        "Feature": [n.split("__", 1)[1] for n in names],
        "Importance": rf_model.named_steps["model"].feature_importances_,
    }).sort_values("Importance", ascending=False)
    imp.to_csv(OUTPUTS_DIR / "random_forest_feature_importance.csv", index=False)
    top = imp.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["Feature"], top["Importance"], color="#4C72B0")
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title("Random Forest - top 15 features")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "08_random_forest_feature_importance.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 9. Robustness checks
# --------------------------------------------------------------------------- #
def cross_validation(data: dict) -> pd.DataFrame:
    """5-fold stratified CV on the full dataset for the supervised models."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced",
                                                  random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                random_state=RANDOM_STATE, n_jobs=-1),
    }
    rows = []
    for name, clf in models.items():
        pipe = Pipeline([("preprocess", make_preprocessor(data["numeric"], data["categorical"])),
                         ("model", clf)])
        res = cross_validate(pipe, data["X"], data["y"], cv=cv,
                             scoring=["precision", "recall", "f1", "roc_auc"])
        rows.append({"Model": name,
                     **{f"{m}_mean": res[f"test_{m}"].mean() for m in
                        ["precision", "recall", "f1", "roc_auc"]},
                     "f1_std": res["test_f1"].std()})
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUTS_DIR / "cross_validation_supervised.csv", index=False)
    print("\n5-fold cross-validation (supervised):")
    print(table.round(4).to_string(index=False))
    return table


def ablation_without_aim_features(df: pd.DataFrame) -> pd.DataFrame:
    """Re-train both supervised models WITHOUT KD/Accuracy/Headshot features.

    This simulates a harder (more realistic) situation where the cheater does
    not have absurd statistics, and shows how much the perfect score depends
    on those few columns.
    """
    X = df.drop(columns=[TARGET] + ID_COLUMNS + AIM_FEATURES)
    y = df[TARGET]
    categorical = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    rows = []
    for name, clf in {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced",
                                                  random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                min_samples_leaf=5,
                                                random_state=RANDOM_STATE, n_jobs=-1),
    }.items():
        pipe = Pipeline([("preprocess", make_preprocessor(numeric, categorical)),
                         ("model", clf)]).fit(X_train, y_train)
        rows.append(metrics_row(name + " (no aim features)", "Supervised", y_test,
                                pipe.predict(X_test), pipe.predict_proba(X_test)[:, 1]))
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUTS_DIR / "ablation_without_aim_features.csv", index=False)
    print(f"\nAblation - features kept: {list(X.columns)}")
    print(table.round(4).to_string(index=False))
    return table


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Anti-cheat ML pipeline")
    parser.add_argument("--data", type=Path,
                        default=PROJECT_DIR / "data" / "online_gaming_behavior.csv",
                        help="Path to online_gaming_behavior.csv")
    args = parser.parse_args()

    for folder in (FIGURES_DIR, OUTPUTS_DIR, MODELS_DIR):
        folder.mkdir(exist_ok=True)
    np.random.seed(RANDOM_STATE)

    df = load_and_prepare(args.data)
    eda_figures(df)
    separability_check(df)
    results, data, _, _ = run_models(df)
    cross_validation(data)
    ablation_without_aim_features(df)

    summary = {"n_rows": int(len(df)), "n_suspect": int(df[TARGET].sum()),
               "suspect_ratio": float(df[TARGET].mean())}
    (OUTPUTS_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nDone. Figures in figures/, tables in outputs/, models in models/.")


if __name__ == "__main__":
    main()
