from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

def get_fast_models(n_pca_spectral, num_classes, random_state, cv):
    models_fast = []

    lgb_pca = Pipeline([
        ("scaler", StandardScaler()),                                                   #each row (feature): mean 0, standard deviation 1
        ("pca", PCA(n_components=n_pca_spectral, random_state=random_state)),
        ("model", LGBMClassifier(
            n_estimators=800, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            random_state=random_state, objective="multiclass", num_class=num_classes,
            verbose=-1, n_jobs=-1
        ))
    ])
    models_fast.append(("LightGBM_PCA", lgb_pca, cv))

    xgb_pca = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_pca_spectral, random_state=random_state)),
        ("model", XGBClassifier(
            n_estimators=600, max_depth=10, learning_rate=0.08, subsample=0.9,
            colsample_bytree=0.9, random_state=random_state, objective="multi:softprob",
            num_class=num_classes, eval_metric="mlogloss", tree_method="hist", n_jobs=-1
        ))
    ])
    models_fast.append(("XGBoost_PCA", xgb_pca, cv))

    cb_pca = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_pca_spectral, random_state=random_state)),
        ("model", CatBoostClassifier(
            loss_function="MultiClass", eval_metric="Accuracy", iterations=1200, depth=8,
            learning_rate=0.06, random_seed=random_state, verbose=False
        ))
    ])
    models_fast.append(("CatBoost_PCA", cb_pca, cv))

    return models_fast

def get_neural_net(n_pca_spectral, random_state):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_pca_spectral, random_state=random_state)),
        ("model", MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),      #3 hidden layers
            activation="relu", 
            solver="adam", 
            alpha=1e-4,                             #L2 regularization coefficient
            batch_size=256, 
            learning_rate_init=1e-3,                #learning rate constant
            max_iter=80,                            # epochs
            early_stopping=True,
            n_iter_no_change=10,                    #if no improvement after 10 epochs, stop
            random_state=random_state, 
            verbose=False
        ))
    ])

def get_spatial_lgbm(num_classes, random_state):
    return LGBMClassifier(
        n_estimators=1200, learning_rate=0.03, subsample=0.9, colsample_bytree=0.9,
        random_state=random_state, objective="multiclass", num_class=num_classes,
        verbose=-1, n_jobs=-1
    )