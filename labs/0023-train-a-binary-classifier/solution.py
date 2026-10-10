import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score

def train(X_train, y_train, X_val, y_val):
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    X_val = np.asarray(X_val, dtype=float)
    y_val = np.asarray(y_val, dtype=int)

    models = [
        LogisticRegression(C=0.1, max_iter=5000),
        LogisticRegression(C=1.0, max_iter=5000),
        LogisticRegression(C=10.0, max_iter=5000),
        SVC(C=1.0, kernel='rbf', gamma='scale'),
        SVC(C=3.0, kernel='rbf', gamma='scale'),
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight='balanced'
        ),
        ExtraTreesClassifier(
            n_estimators=300,
            random_state=42,
            class_weight='balanced'
        )
    ]

    best_model = None
    best_score = -1.0

    for model in models:
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        score = accuracy_score(y_val, predictions)

        if score > best_score:
            best_score = score
            best_model = model

    X_combined = np.vstack((X_train, X_val))
    y_combined = np.concatenate((y_train, y_val))

    best_model.fit(X_combined, y_combined)

    def predict(X):
        X = np.asarray(X, dtype=float)
        return best_model.predict(X).astype(int)

    return predict