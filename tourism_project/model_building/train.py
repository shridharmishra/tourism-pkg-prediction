import os

import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "ProdTaken"

NUMERIC_FEATURES = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome",
]
CATEGORICAL_FEATURES = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]

PARAM_GRID = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.05, 0.1],
}

MODEL_DIR = "tourism_project/deployment"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")


def load_splits():
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
    ytest = pd.read_csv("ytest.csv").squeeze("columns")
    return Xtrain, Xtest, ytrain, ytest


def build_pipeline():
    preprocessor = make_column_transformer(
        (StandardScaler(), NUMERIC_FEATURES),
        (OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    )
    return make_pipeline(
        preprocessor,
        xgb.XGBClassifier(eval_metric="logloss", random_state=42),
    )


def configure_mlflow():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("tourism-wellness-package")
        mlflow.get_tracking_uri()
    except Exception as exc:
        print(f"Could not reach MLflow server at {tracking_uri} ({exc}); logging to a local sqlite store instead.")
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("tourism-wellness-package")


def main():
    Xtrain, Xtest, ytrain, ytest = load_splits()
    configure_mlflow()

    with mlflow.start_run():
        grid_search = GridSearchCV(
            build_pipeline(),
            PARAM_GRID,
            cv=3,
            scoring="f1",
            n_jobs=-1,
        )
        grid_search.fit(Xtrain, ytrain)

        best_model = grid_search.best_estimator_
        mlflow.log_params(grid_search.best_params_)

        train_report = classification_report(ytrain, best_model.predict(Xtrain), output_dict=True)
        test_pred = best_model.predict(Xtest)
        test_report = classification_report(ytest, test_pred, output_dict=True)

        mlflow.log_metrics({
            "train_accuracy": train_report["accuracy"],
            "train_f1": train_report["1"]["f1-score"],
            "test_accuracy": test_report["accuracy"],
            "test_precision": test_report["1"]["precision"],
            "test_recall": test_report["1"]["recall"],
            "test_f1": test_report["1"]["f1-score"],
        })

        print("Best params:", grid_search.best_params_)
        print("\nTest classification report:")
        print(classification_report(ytest, test_pred))

        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)
        mlflow.log_artifact(MODEL_PATH)
        print(f"\nBest model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
