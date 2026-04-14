
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def generate_data(n_students=500, seed=42):
    """Generate synthetic Nigerian secondary school student data."""
    np.random.seed(seed)

    student_ids = [f"S{str(i).zfill(3)}" for i in range(1, n_students + 1)]

    class_levels = np.random.choice(
        ["JSS1", "JSS2", "JSS3", "SSS1", "SSS2", "SSS3"],
        size=n_students,
        p=[0.20, 0.18, 0.17, 0.18, 0.15, 0.12]
    )

    attendance_rate = np.clip(
        np.random.normal(72, 18, n_students), 40, 100
    ).round(1)

    assignment_rate = np.clip(
        np.random.normal(68, 20, n_students), 30, 100
    ).round(1)

    performance_factor = (
        (attendance_rate - 72) / 18 * 0.4 +
        (assignment_rate - 68) / 20 * 0.3
    )

    def influenced_scores(mean, std, factor, size):
        base = np.random.normal(mean, std, size)
        return np.clip(base + factor * 8, 20, 100).round(1)

    maths_score    = influenced_scores(52, 18, performance_factor, n_students)
    english_score  = influenced_scores(55, 15, performance_factor, n_students)
    science_score  = influenced_scores(50, 17, performance_factor, n_students)
    social_studies = influenced_scores(58, 14, performance_factor, n_students)

    prev_term_avg = influenced_scores(55, 17, performance_factor, n_students)
    missing_mask  = np.random.random(n_students) < 0.15
    prev_term_avg = prev_term_avg.astype(float)
    prev_term_avg[missing_mask] = np.nan

    term_average = (maths_score + english_score + science_score + social_studies) / 4

    condition_1 = term_average < 50
    condition_2 = (attendance_rate < 60) & (term_average < 60)
    condition_3 = (assignment_rate < 50) & (term_average < 55)
    at_risk = (condition_1 | condition_2 | condition_3).astype(int)

    return pd.DataFrame({
        "student_id"      : student_ids,
        "class_level"     : class_levels,
        "maths_score"     : maths_score,
        "english_score"   : english_score,
        "science_score"   : science_score,
        "social_studies"  : social_studies,
        "attendance_rate" : attendance_rate,
        "assignment_rate" : assignment_rate,
        "prev_term_avg"   : prev_term_avg,
        "term_average"    : term_average.round(1),
        "at_risk"         : at_risk
    })


def train_and_save():
    """Full training pipeline. Returns trained components."""

    print("Generating data...")
    df = generate_data()

    # Preprocessing
    level_order = {"JSS1":0,"JSS2":1,"JSS3":2,"SSS1":3,"SSS2":4,"SSS3":5}
    df["class_level"] = df["class_level"].map(level_order)

    df_model = df.drop(columns=["student_id", "term_average"])
    X = df_model.drop(columns=["at_risk"])
    y = df_model["at_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test  = pd.DataFrame(imputer.transform(X_test),      columns=X_test.columns)

    scaler  = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)

    print("Training model...")
    model = RandomForestClassifier(
        n_estimators     = 300,
        max_depth        = 20,
        min_samples_leaf = 2,
        min_samples_split= 10,
        random_state     = 42,
        class_weight     = "balanced"
    )
    model.fit(X_train_scaled, y_train)

    # Save all three components
    os.makedirs("model", exist_ok=True)
    joblib.dump(model,   "model/best_rf_model.pkl")
    joblib.dump(scaler,  "model/scaler.pkl")
    joblib.dump(imputer, "model/imputer.pkl")

    print("Model, scaler, and imputer saved to model/ directory.")
    return model, scaler, imputer


if __name__ == "__main__":
    train_and_save()
