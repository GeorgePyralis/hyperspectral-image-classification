import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

def cv_score_pipeline(pipeline, X, y, n_splits=5, name="model", random_state=42):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    for fold, (tr, va) in enumerate(cv.split(X, y), 1):           #tr, va: training, validation indices
        pipeline.fit(X[tr], y[tr])
        pred = pipeline.predict(X[va])
        acc = accuracy_score(y[va], pred)
        scores.append(acc)
        print(f"  {name} | Fold {fold}: acc={acc:.4f}")
    return float(np.mean(scores)), float(np.std(scores))          #mean and standard deviation between the 5 accuracies of the 5 fold