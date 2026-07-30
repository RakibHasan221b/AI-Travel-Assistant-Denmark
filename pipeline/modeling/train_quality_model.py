"""Phase 9 — quality-score model bake-off.

Builds a composite "AI Confidence Score" (not a claim of true restaurant
quality — see docs/architecture.md design decision #2) from real signal for
the 172 places with aggregated sentiment (from Wikivoyage-linked text), then
trains Random Forest, XGBoost, and a small neural net to reproduce that
composite from structured OSM features alone — so every place gets a score,
not just the ones with direct review text. All three (plus a linear
baseline) are compared on the same held-out MSE/MAE, tuned with Optuna,
tracked in MLflow.

Dataset is small (172 labeled places) - this is a real constraint of the
pilot's data volume, not hidden. Tuning budgets are kept modest accordingly.
"""

import logging
import os

import mlflow
import numpy as np
import optuna
import pandas as pd
import psycopg
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

load_dotenv()
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_quality_model")

RANDOM_STATE = 42
N_TRIALS = 25

# "Success rate": % of predictions within this many points of the true
# score (0-100 scale) — the plain-English number for anyone who doesn't
# want to interpret R²/RMSE. Deliberately not MAPE: these are engineered
# composite scores, not naturally ratio data, and MAPE blows up near zero.
SUCCESS_TOLERANCE = 10


def load_places(conn) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT place_id, category, neighborhood, opening_hours, subcategory, osm_tags
            FROM places;
            """
        )
        columns = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=columns)


def load_sentiment_summary(conn) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT place_id, AVG(avg_score) AS mean_score, SUM(num_mentions) AS total_mentions
            FROM aggregated_sentiment GROUP BY place_id;
            """
        )
        columns = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=columns)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["has_opening_hours"] = df["opening_hours"].notna().astype(int)
    df["has_subcategory"] = df["subcategory"].notna().astype(int)
    df["tag_count"] = df["osm_tags"].apply(lambda t: len(t) if t else 0)
    df["has_phone"] = df["osm_tags"].apply(lambda t: int(bool(t and t.get("phone"))))
    df["has_website"] = df["osm_tags"].apply(lambda t: int(bool(t and t.get("website"))))
    df["neighborhood_filled"] = df["neighborhood"].fillna("unassigned")

    cat_dummies = pd.get_dummies(df["category"], prefix="cat")
    nbhd_dummies = pd.get_dummies(df["neighborhood_filled"], prefix="nbhd")
    numeric = df[["has_opening_hours", "has_subcategory", "tag_count", "has_phone", "has_website"]]
    return pd.concat([numeric, cat_dummies, nbhd_dummies], axis=1)


def build_target(sentiment_df: pd.DataFrame, features_labeled: pd.DataFrame) -> pd.Series:
    sentiment_component = (sentiment_df["mean_score"] - 1) / 4 * 100
    volume_component = np.minimum(np.log1p(sentiment_df["total_mentions"]) / np.log1p(10), 1) * 100
    completeness_component = (
        features_labeled["has_opening_hours"]
        + features_labeled["has_subcategory"]
        + features_labeled["has_phone"]
        + features_labeled["has_website"]
    ) / 4 * 100
    return 0.6 * sentiment_component + 0.25 * volume_component + 0.15 * completeness_component


def tune_rf(X_train, y_train) -> dict:
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "random_state": RANDOM_STATE,
        }
        model = RandomForestRegressor(**params)
        kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for tr_idx, val_idx in kf.split(X_train):
            model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            pred = model.predict(X_train.iloc[val_idx])
            scores.append(mean_squared_error(y_train.iloc[val_idx], pred))
        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params


def tune_xgb(X_train, y_train) -> dict:
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "random_state": RANDOM_STATE,
        }
        model = XGBRegressor(**params)
        kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for tr_idx, val_idx in kf.split(X_train):
            model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            pred = model.predict(X_train.iloc[val_idx])
            scores.append(mean_squared_error(y_train.iloc[val_idx], pred))
        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params


def compute_metrics(y_test, pred) -> dict:
    mse = mean_squared_error(y_test, pred)
    success_rate = float(np.mean(np.abs(np.asarray(y_test) - np.asarray(pred)) <= SUCCESS_TOLERANCE))
    return {
        "mse": mse,
        "rmse": mse**0.5,
        "mae": mean_absolute_error(y_test, pred),
        "r2": r2_score(y_test, pred),
        "success_rate_within_10pts": success_rate,
    }


def evaluate(name, model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    metrics = compute_metrics(y_test, pred)
    log.info(
        f"  {name}: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, "
        f"R2={metrics['r2']:.2f}, success_rate(±{SUCCESS_TOLERANCE}pts)={metrics['success_rate_within_10pts']:.0%}"
    )
    return metrics


def main():
    db_url = os.environ["DATABASE_URL"]
    mlflow.set_experiment("quality_score_bakeoff")

    with psycopg.connect(db_url, connect_timeout=15) as conn:
        places = load_places(conn)
        sentiment = load_sentiment_summary(conn)

    log.info(f"{len(places)} total places, {len(sentiment)} with sentiment data")

    all_features = build_features(places)
    labeled_mask = places["place_id"].isin(sentiment["place_id"])
    features_labeled = all_features[labeled_mask].reset_index(drop=True)
    sentiment_ordered = sentiment.set_index("place_id").loc[places[labeled_mask]["place_id"]].reset_index()

    y = build_target(sentiment_ordered, features_labeled)
    X = features_labeled

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    log.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    results = {}
    with mlflow.start_run(run_name="bakeoff"):
        mlflow.log_param("n_labeled_places", len(X))
        mlflow.log_param("n_features", X.shape[1])

        # Baseline
        with mlflow.start_run(run_name="linear_regression", nested=True):
            lr = LinearRegression().fit(X_train, y_train)
            results["linear_regression"] = evaluate("Linear Regression", lr, X_test, y_test)
            mlflow.log_metrics(results["linear_regression"])

        # Random Forest, Optuna-tuned
        log.info("Tuning Random Forest...")
        rf_params = tune_rf(X_train, y_train)
        with mlflow.start_run(run_name="random_forest", nested=True):
            rf = RandomForestRegressor(**rf_params, random_state=RANDOM_STATE).fit(X_train, y_train)
            results["random_forest"] = evaluate("Random Forest", rf, X_test, y_test)
            mlflow.log_params(rf_params)
            mlflow.log_metrics(results["random_forest"])

        # XGBoost, Optuna-tuned
        log.info("Tuning XGBoost...")
        xgb_params = tune_xgb(X_train, y_train)
        with mlflow.start_run(run_name="xgboost", nested=True):
            xgb = XGBRegressor(**xgb_params, random_state=RANDOM_STATE).fit(X_train, y_train)
            results["xgboost"] = evaluate("XGBoost", xgb, X_test, y_test)
            mlflow.log_params(xgb_params)
            mlflow.log_metrics(results["xgboost"])

        # Small NN - deliberately modest, this is the "know when not to reach
        # for deep learning" baseline, not expected to win on 172 samples
        with mlflow.start_run(run_name="small_nn", nested=True):
            scaler = StandardScaler().fit(X_train)
            nn = MLPRegressor(
                hidden_layer_sizes=(16, 8), max_iter=2000, random_state=RANDOM_STATE, early_stopping=True
            ).fit(scaler.transform(X_train), y_train)
            pred = nn.predict(scaler.transform(X_test))
            results["small_nn"] = compute_metrics(y_test, pred)
            log.info(
                f"  Small NN: RMSE={results['small_nn']['rmse']:.2f}, MAE={results['small_nn']['mae']:.2f}, "
                f"R2={results['small_nn']['r2']:.2f}, "
                f"success_rate(±{SUCCESS_TOLERANCE}pts)={results['small_nn']['success_rate_within_10pts']:.0%}"
            )
            mlflow.log_metrics(results["small_nn"])

        winner_name = min(results, key=lambda k: results[k]["mse"])
        log.info(
            f"Winner: {winner_name} (RMSE={results[winner_name]['rmse']:.2f}, "
            f"R2={results[winner_name]['r2']:.2f}, "
            f"success_rate={results[winner_name]['success_rate_within_10pts']:.0%})"
        )
        mlflow.log_param("winner", winner_name)

    # Retrain winner on the full labeled set for production predictions
    if winner_name == "random_forest":
        final_model = RandomForestRegressor(**rf_params, random_state=RANDOM_STATE).fit(X, y)
    elif winner_name == "xgboost":
        final_model = XGBRegressor(**xgb_params, random_state=RANDOM_STATE).fit(X, y)
    elif winner_name == "linear_regression":
        final_model = LinearRegression().fit(X, y)
    else:
        scaler = StandardScaler().fit(X)
        final_model = MLPRegressor(
            hidden_layer_sizes=(16, 8), max_iter=2000, random_state=RANDOM_STATE, early_stopping=True
        ).fit(scaler.transform(X), y)

    log.info(f"Predicting quality scores for all {len(places)} places with {winner_name}...")
    if winner_name == "small_nn":
        all_predictions = final_model.predict(scaler.transform(all_features))
    else:
        all_predictions = final_model.predict(all_features)
    all_predictions = np.clip(all_predictions, 0, 100)

    with psycopg.connect(db_url, connect_timeout=15) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM ml_predictions WHERE target = 'quality_score';")
            for place_id, pred in zip(places["place_id"], all_predictions):
                is_direct = bool(labeled_mask[places["place_id"] == place_id].values[0])
                cur.execute(
                    """
                    INSERT INTO ml_predictions (place_id, target, predicted_value, model_name,
                                                 model_version, used_llm_fallback)
                    VALUES (%s, 'quality_score', %s, %s, 'v1', %s);
                    """,
                    (place_id, float(pred), winner_name, not is_direct),
                )
        conn.commit()

    log.info(f"Done. Stored quality_score predictions for {len(places)} places.")


if __name__ == "__main__":
    main()
