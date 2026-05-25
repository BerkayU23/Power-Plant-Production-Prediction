# CCPP System Forecasting Pipeline
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Folds5x2_pp.csv")
df = df.drop([9568, 19137, 28706, 38275]).reset_index(drop=True)

for i in df.columns:
    df[i] = df[i].astype(float)

def remove_outliers_iqr(df_in, threshold=1.25):
    df_cleaned = df_in.copy()
    numeric_cols = df_cleaned.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]

    return df_cleaned.copy()

df_cleaned = remove_outliers_iqr(df)

from sklearn.model_selection import train_test_split, KFold, cross_validate, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

X = df_cleaned.drop("PE", axis=1)
y = df_cleaned["PE"]

# Veri Bölme ve Ölçeklendirme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=23)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_cols = scaler.get_feature_names_out()

X_train = pd.DataFrame(X_train_scaled, columns=X_cols)
X_test = pd.DataFrame(X_test_scaled, columns=X_cols)

# Çapraz Geçerlilik (Cross-Validation)
kf = KFold(n_splits=5, shuffle=True, random_state=23)

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score


models = {
    "Linear": LinearRegression(),
    "Lasso": Lasso(random_state=23),
    "Ridge": Ridge(random_state=23),
    "Random Forest": RandomForestRegressor(random_state=23),
    "Gradient Boost": GradientBoostingRegressor(random_state=23),
    "XGBoost": XGBRegressor(random_state=23),
    "LightGBM": LGBMRegressor(random_state=23, verbose=-1)
}

results = []
scoring = ["r2", "neg_root_mean_squared_error"]

print("Çapraz Geçerlilik (Cross-Validation) İşlemi\n")

for name, model in models.items():
    # Veri sızıntısını önlemek için tüm X ve y yerine sadece eğitim seti (X_train, y_train) kullanıldı
    cv_results = cross_validate(model, X_train, y_train, cv=kf, scoring=scoring, n_jobs=-1)
    mean_cv_r2 = np.mean(cv_results["test_r2"])
    mean_cv_rmse = np.mean(-cv_results["test_neg_root_mean_squared_error"])

    results.append([name, mean_cv_r2, mean_cv_rmse])

results_df = pd.DataFrame(results, columns=["Name", "CV Mean R2", "CV Mean RMSE"]).sort_values(by="CV Mean R2", ascending=False)
print(results_df.to_string(index=False))
print("\n" + "_" * 50 + "\n")

# Hyperparameter Tuning
top_models = {
    "XGBoost": XGBRegressor(random_state=23),
    "LightGBM": LGBMRegressor(random_state=23, verbose=-1)
}

param_distributions = {
    "XGBoost": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0]
    },
    "LightGBM": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [20, 31, 40],
        "min_child_samples": [10, 20, 30]
    }
}

print("RandomizedSearchCV Optimizasyonu\n")

best_models = {}
for name, model in top_models.items():
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions[name],
        n_iter=10,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        random_state=23
    )
    random_search.fit(X_train, y_train)
    best_models[name] = random_search.best_estimator_
    print(f"Optimize Edilmiş {name} Test R2 Skoru: {r2_score(y_test, best_models[name].predict(X_test)):.4f}")

# En İyi Model ve Sonuçları
model_xgb = XGBRegressor(n_estimators=200, subsample=0.8, max_depth=5, learning_rate=0.1, colsample_bytree=0.8, random_state=23)
model_xgb.fit(X_train, y_train)
y_pred = model_xgb.predict(X_test)

print(f"\nFinal Model RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"Final Model R2 Score: {r2_score(y_test, y_pred):.4f}")


import pickle

with open("CCPP_Systems.pkl", "wb") as f:
    pickle.dump({
        "model": model_xgb,
        "scaler": scaler
    }, f)

pd.DataFrame(X_test_scaled, columns=["AT", "V", "AP", "RH"]).to_csv("ccpptestdata.csv", index=False)