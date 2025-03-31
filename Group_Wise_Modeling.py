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

def plot_r2_scores(performance_matrix):
    performance_matrix.drop(columns=['Average R2']).plot(kind='bar', figsize=(10, 6), title='R² Scores for Each Category')
    plt.xlabel('Category')
    plt.ylabel('R² Score')
    plt.legend(title='Parameter')
    plt.show()

plot_r2_scores(performance_matrix)

def plot_actual_vs_predicted(y_test, y_pred):
    for i, (key, sensor) in enumerate(sensor_key_map.items()):
        plt.figure(figsize=(6, 4))
        plt.scatter(y_test[sensor], y_pred[:, i], alpha=0.5)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"Actual vs Predicted for {sensor}")
        plt.show()

def plot_spectrometer_vs_sensor(df, best_cat_name, spectral_columns, sensor_key_map, model, scaler, num_samples=5):
    df_best = df[df['Category'] == best_cat_name]
    if df_best.empty:
        print("Error: No data available for the best category.")
        return

    df_samples = df_best.sample(min(num_samples, len(df_best)), random_state=42)
    X_samples = scaler.transform(df_samples[spectral_columns])
    Y_actual = df_samples[list(sensor_key_map.values())].values
    Y_predicted = model.predict(X_samples)
    Y_predicted = np.clip(Y_predicted, 0, None)

    wavelengths = np.array([float(col) for col in spectral_columns])
    sensor_names = list(sensor_key_map.values())

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    for sensor_idx, sensor_name in enumerate(sensor_names):
        ax = axes[sensor_idx]
        for i in range(len(df_samples)):
            ax.plot(wavelengths, X_samples[i], linestyle='-', alpha=0.7, label=f"Sample {i+1}" if sensor_idx == 0 else "")

        for i in range(len(df_samples)):
            ax.axhline(Y_actual[i, sensor_idx], linestyle='-', color='green', alpha=0.8, label=f"Actual {sensor_name}" if i == 0 else "")
            ax.axhline(Y_predicted[i, sensor_idx], linestyle='--', color='red', alpha=0.8, label=f"Predicted {sensor_name}" if i == 0 else "")

        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Spectral Intensity")
        ax.set_title(f"Spectrometer Readings vs {sensor_name}")
        ax.legend()

    plt.tight_layout()
    plt.show()

def plot_actual_vs_predicted(y_test, y_pred, sensor_key_map):
    plt.figure(figsize=(12, 8))
    for i, sensor in enumerate(sensor_key_map.values()):
        plt.subplot(2, 2, i + 1)
        plt.scatter(y_test[sensor], y_pred[:, i], alpha=0.7)
        plt.plot([y_test[sensor].min(), y_test[sensor].max()], [y_test[sensor].min(), y_test[sensor].max()], 'r--')
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"Actual vs Predicted {sensor}")
    plt.tight_layout()
    plt.show()

if best_cat_name and best_cat_name in models_xgb:
    plot_spectrometer_vs_sensor(df, best_cat_name, spectral_columns, sensor_key_map, models_xgb[best_cat_name], scaler_spec)
    plot_actual_vs_predicted(test_data_xgb[best_cat_name], pred_data_xgb[best_cat_name], sensor_key_map)


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

    actual_columns = list(sensor_key_map.values())

    # ✅ Forcefully check & handle missing columns
    missing_cols = [col for col in actual_columns if col not in df_user.columns]

    if missing_cols:
        print(f"\n⚠️ Warning: Missing actual values for {missing_cols}. Accuracy cannot be fully calculated.")
    else:
        actual_values = df_user[actual_columns]
        r2_scores = {col: r2_score(actual_values[col], pred_df[col]) for col in actual_columns}
        overall_r2 = np.mean(list(r2_scores.values()))

        print("\nR² Scores for Each Parameter:")
        for key, value in r2_scores.items():
            print(f"{key}: {value:.4f}")

        print(f"\nOverall Prediction Accuracy (R² Score): {overall_r2:.4f}")

        # Add accuracy to the DataFrame
        accuracy_df = pd.DataFrame([r2_scores], index=['R² Score'])
        result_df = pd.concat([result_df, accuracy_df])

        plot_actual_vs_predicted(actual_values, predictions)

    return result_df


# Call the function
user_csv_path = "dataset.csv"
result_df = predict_soil_parameters_from_csv(user_csv_path)

# Print final DataFrame with Accuracy
print("\nFinal Output (Predictions + Accuracy):")
print(result_df)
