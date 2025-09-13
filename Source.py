# Project: Prediction of Agriculture Crop Production in India


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import joblib

# ---------------------------------------------------------
data = pd.read_csv("crop_production.csv")
print("Dataset loaded successfully!")
print(data.head())

data = data.dropna()

cat_cols = ["State_Name", "Season", "Crop"]  
le = LabelEncoder()
for col in cat_cols:
    data[col] = le.fit_transform(data[col])


X = data.drop("Production", axis=1)  
y = data["Production"]         


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Data preprocessing completed!")

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("\nLinear Regression Performance:")
print("R2 Score:", r2_score(y_test, y_pred_lr))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))

rf = RandomForestRegressor(random_state=42)

params = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}

grid_rf = GridSearchCV(rf, param_grid=params, cv=3, n_jobs=-1, scoring="r2")
grid_rf.fit(X_train, y_train)

best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)

print("\nRandom Forest Performance:")
print("Best Params:", grid_rf.best_params_)
print("R2 Score:", r2_score(y_test, y_pred_rf))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

print("\nXGBoost Performance:")
print("R2 Score:", r2_score(y_test, y_pred_xgb))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_xgb)))

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_xgb, alpha=0.5, color="blue")
plt.xlabel("Actual Production")
plt.ylabel("Predicted Production")
plt.title("Actual vs Predicted Crop Production (XGBoost)")
plt.grid(True)
plt.show()


xgb.plot_importance(xgb_model)
plt.show()
joblib.dump(xgb_model, "final_crop_prediction_model.pkl")
print("\nFinal XGBoost model saved as 'final_crop_prediction_model.pkl'")
