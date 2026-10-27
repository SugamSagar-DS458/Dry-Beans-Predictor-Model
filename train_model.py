"""Train and save the SVM artifact used by the Streamlit predictor."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


INPUT_FIELDS = (
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "ConvexArea",
    "Extent",
)


def train(data_path: str | Path, model_path: str | Path) -> dict:
    """Fit the notebook's tuned SVM and persist all data needed by the app."""
    data_path = Path(data_path)
    model_path = Path(model_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_excel(data_path)
    if "Class" not in df.columns:
        raise ValueError("The dataset must contain a 'Class' target column.")

    numeric_columns = list(df.select_dtypes(include=[np.number]).columns)
    if not numeric_columns:
        raise ValueError("The dataset must contain numeric feature columns.")
    missing_features = set(INPUT_FIELDS).difference(numeric_columns)
    if missing_features:
        raise ValueError(
            "Dataset is missing required numeric input columns: "
            + ", ".join(sorted(missing_features))
        )

    features_df = df[numeric_columns].copy()
    lower = features_df.quantile(0.01)
    upper = features_df.quantile(0.99)
    features_df = features_df.clip(lower=lower, upper=upper, axis=1)

    skewness = features_df.skew()
    skewed = list(skewness[skewness.abs() > 0.75].index)
    if skewed:
        features_df.loc[:, skewed] = np.log1p(features_df[skewed])

    scaler = StandardScaler()
    X = pd.DataFrame(
        scaler.fit_transform(features_df),
        columns=numeric_columns,
        index=df.index,
    )
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["Class"])
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = CalibratedClassifierCV(
        estimator=SVC(C=10, gamma=0.1, kernel="rbf", random_state=42),
        method="sigmoid",
        cv=5,
        ensemble=False,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    class_stats = {}
    class_share = {}
    presets = {}
    ranges = {}
    for field in INPUT_FIELDS:
        ranges[field] = (float(df[field].min()), float(df[field].max()))

    for bean_class in encoder.classes_:
        class_rows = df.loc[df["Class"] == bean_class]
        class_stats[bean_class] = {
            "Area": float(class_rows["Area"].median()),
            "AspectRation": float(class_rows["AspectRation"].median()),
            "roundness": float(class_rows["roundness"].median()),
        }
        class_share[bean_class] = round(len(class_rows) / len(df) * 100, 1)
        presets[bean_class] = {
            field: float(class_rows[field].median()) for field in INPUT_FIELDS
        }

    artifact = {
        "model": model,
        "scaler": scaler,
        "features": numeric_columns,
        "lower": lower,
        "upper": upper,
        "skewed": skewed,
        "classes": list(encoder.classes_),
        "ranges": ranges,
        "presets": presets,
        "class_stats": class_stats,
        "class_share": class_share,
        "metrics": {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "f1_macro": float(f1_score(y_test, predictions, average="macro")),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
        },
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    return artifact


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    result = train(
        project_dir / "Dry_Beans_Dataset.xlsx",
        project_dir / "model.joblib",
    )
    print(f"Saved model to {project_dir / 'model.joblib'}")
    print(f"Holdout accuracy: {result['metrics']['accuracy']:.2%}")
