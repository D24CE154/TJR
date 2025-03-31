import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

def load_and_clean_dataset(file_path):
    if not os.path.exists(file_path):
        print("Error: File not found! Please enter a valid path.")
        return None

    df = pd.read_csv(file_path)
    df.replace('', np.nan, inplace=True)
    if df.isnull().all().any():
        print("Error: Dataset contains only missing values.")
        return None
    return df

def extract_category(record_str):
    if pd.isna(record_str):
        return np.nan
    match = re.search(r'_(\d+ml)', str(record_str))
    return match.group(1) if match else np.nan

def cap_outliers(df, columns):
    Q1 = df[columns].quantile(0.25)
    Q3 = df[columns].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[columns] = df[columns].clip(lower=lower_bound, upper=upper_bound, axis=1)
    return df

dataset_path = "dataset.csv"
df = load_and_clean_dataset(dataset_path)
if df is None:
    exit()

df['Category'] = df['Records'].apply(extract_category)
df.drop(columns=['Records'], inplace=True)

spectral_columns = [col for col in df.columns if col.isdigit()]
sensor_columns = ['Ph', 'Nitro (mg/10 g)', 'Posh Nitro (mg/10 g)', 'Pota Nitro (mg/10 g)']

if spectral_columns and sensor_columns:
    df = cap_outliers(df, spectral_columns + sensor_columns)

scaler_spec = MinMaxScaler()
X_spec = scaler_spec.fit_transform(df[spectral_columns])

sensor_key_map = {
    'ph': 'Ph',
    'n': 'Nitro (mg/10 g)',
    'p': 'Posh Nitro (mg/10 g)',
    'k': 'Pota Nitro (mg/10 g)'
}

def category_wise_xgboost_modeling(dataframe, X_data, sensor_key_map):
    results, models, test_data, pred_data = {}, {}, {}, {}
    categories = dataframe['Category'].dropna().unique()

    for cat in categories:
        category_mask = dataframe['Category'] == cat
        X_category = X_data[category_mask]
        y_category = dataframe.loc[category_mask, list(sensor_key_map.values())]

        if len(X_category) < 10:
            print(f"Skipping category {cat}: Not enough data. Available samples: {len(X_category)}")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X_category, y_category, test_size=0.2, random_state=100
        )

        xgb = XGBRegressor(random_state=100)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 7],
            'learning_rate': [0.01, 0.05],
        }

        grid_search = GridSearchCV(xgb, param_grid, scoring='r2', cv=3, n_jobs=-1)
        multi_xgb = MultiOutputRegressor(grid_search)
        multi_xgb.fit(X_train, y_train)

        models[cat] = multi_xgb
        y_pred = multi_xgb.predict(X_test)
        y_pred = np.clip(y_pred, 0, None)

        r2_scores = {key: r2_score(y_test[sensor], y_pred[:, i]) for i, (key, sensor) in enumerate(sensor_key_map.items())}
        results[cat] = r2_scores
        test_data[cat] = y_test
        pred_data[cat] = y_pred

    return results, models, test_data, pred_data

results_xgb, models_xgb, test_data_xgb, pred_data_xgb = category_wise_xgboost_modeling(df, X_spec, sensor_key_map)

performance_matrix = pd.DataFrame(results_xgb).T
performance_matrix['Average R2'] = performance_matrix.mean(axis=1)
print("Performance Matrix:")
print(performance_matrix)

best_category = performance_matrix.sort_values(by='Average R2', ascending=False).head(1)
best_cat_name = best_category.index[0] if not best_category.empty else None
print(f"Best Category Selected: {best_cat_name}")

def process_user_csv(file_path):
    df_user = load_and_clean_dataset(file_path)
    if df_user is None:
        return None, None

    missing_columns = set(spectral_columns) - set(df_user.columns)
    for col in missing_columns:
        df_user[col] = 0  # Fill missing spectral columns with zero

    df_user = df_user[spectral_columns]  # Ensure correct column order
    X_user = scaler_spec.transform(df_user)

    return X_user, df_user

def predict_soil_parameters_from_csv(file_path):
    processed_input, df_user = process_user_csv(file_path)
    if processed_input is None or df_user is None:
        print("Error: Could not process input CSV. Please check missing columns.")
        return

    if best_cat_name not in models_xgb:
        print("Error: No trained model available for prediction.")
        return

    model = models_xgb[best_cat_name]
    predictions = model.predict(processed_input)
    predictions = np.clip(predictions, 0, None)

    pred_df = pd.DataFrame(predictions, columns=sensor_key_map.values())
    result_df = pd.concat([df_user, pred_df], axis=1)

    print("\nPredicted Soil Parameters:")

    print(result_df)

    # Calculate overall R² score
    actual_columns = list(sensor_key_map.values())
    if all(col in df_user.columns for col in actual_columns):
        actual_values = df_user[actual_columns]
        r2_scores = [r2_score(actual_values[col], predictions[:, i]) for i, col in enumerate(actual_columns)]
        overall_r2 = np.mean(r2_scores)

        print(f"\nOverall Prediction Accuracy (R² Score): {overall_r2:.4f}")

    return 'r2_scores',r2_scores, 'overall_r2',overall_r2

user_csv_path = "dataset.csv"
# Run prediction and print accuracy
predict_soil_parameters_from_csv(user_csv_path)

