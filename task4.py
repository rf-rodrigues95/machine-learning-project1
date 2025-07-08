from operator import index

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

df = pd.read_csv('X_train.csv')

trajectory_size = 258
indices_to_drop = []

for i in range(0, int((df.shape[0]) / trajectory_size)):
    start = (i * (trajectory_size - 1))
    end = (start + 1) + 255
    if end > df.shape[0]:
        break
    for j in range(start + 1, end):
        if np.isclose(df.at[j, 't'], 0.0, rtol=1e-09, atol=1e-09):
            indices_to_drop.extend(range(start, end + 1))
            break

df = df.drop(df.index[indices_to_drop])

df.to_csv('processed_dataset.cv', index=True)

df = pd.read_csv('processed_dataset.csv')
df = df.sample(frac=0.2)

def compute_accelerations(df):
    ax = []
    ay = []
    G = 1.0  
    masses = [1, 1, 1]
    for index, row in df.iterrows():
        x1, y1 = row['x_1'], row['y_1']
        x2, y2 = row['x_2'], row['y_2']
        x3, y3 = row['x_3'], row['y_3']

        d12 = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        d13 = np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
        d23 = np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)

        ax1, ay1 = 0, 0
        ax2, ay2 = 0, 0
        ax3, ay3 = 0, 0

        if d12 > 0 and d13 > 0:
            ax1 = G * (masses[1] * (x2 - x1) / d12**3 + masses[2] * (x3 - x1) / d13**3)
            ay1 = G * (masses[1] * (y2 - y1) / d12**3 + masses[2] * (y3 - y1) / d13**3)

        if d12 > 0 and d23 > 0:
            ax2 = G * (masses[0] * (x1 - x2) / d12**3 + masses[2] * (x3 - x2) / d23**3)
            ay2 = G * (masses[0] * (y1 - y2) / d12**3 + masses[2] * (y3 - y2) / d23**3)

        if d13 > 0 and d23 > 0:
            ax3 = G * (masses[0] * (x1 - x3) / d13**3 + masses[1] * (x2 - x3) / d23**3)
            ay3 = G * (masses[0] * (y1 - y3) / d13**3 + masses[1] * (y2 - y3) / d23**3)

        ax.append([ax1, ax2, ax3])
        ay.append([ay1, ay2, ay3])

    ax = np.array(ax)
    ay = np.array(ay)

    df['ax_1'], df['ax_2'], df['ax_3'] = ax[:, 0], ax[:, 1], ax[:, 2]
    df['ay_1'], df['ay_2'], df['ay_3'] = ay[:, 0], ay[:, 1], ay[:, 2]

df['d_12'] = np.sqrt((df['x_1'] - df['x_2'])**2 + (df['y_1'] - df['y_2'])**2)
df['d_13'] = np.sqrt((df['x_1'] - df['x_3'])**2 + (df['y_1'] - df['y_3'])**2)
df['d_23'] = np.sqrt((df['x_2'] - df['x_3'])**2 + (df['y_2'] - df['y_3'])**2)

compute_accelerations(df)

df.to_csv('ttttt.csv', index=True)

selected_features = ['t', 'd_12', 'd_13', 'd_23', 'ax_1', 'ay_1', 'ax_2', 'ay_2', 'ax_3', 'ay_3']
targets = ['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']

X_train, X_val, y_train, y_val = train_test_split(df[selected_features], df[targets], test_size=0.2, random_state=42)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)
y_train = scaler_y.fit_transform(y_train)
y_val = scaler_y.transform(y_val)

test_data = pd.read_csv('X_test.csv')

test_data.rename(columns={
    'x0_1': 'x_1', 'y0_1': 'y_1',
    'x0_2': 'x_2', 'y0_2': 'y_2',
    'x0_3': 'x_3', 'y0_3': 'y_3'
}, inplace=True)

test_data['d_12'] = np.sqrt((test_data['x_1'] - test_data['x_2'])**2 + (test_data['y_1'] - test_data['y_2'])**2)
test_data['d_13'] = np.sqrt((test_data['x_1'] - test_data['x_3'])**2 + (test_data['y_1'] - test_data['y_3'])**2)
test_data['d_23'] = np.sqrt((test_data['x_2'] - test_data['x_3'])**2 + (test_data['y_2'] - test_data['y_3'])**2)

compute_accelerations(test_data)

X_test = test_data[selected_features]
X_test = scaler_X.transform(X_test)

def plot_y_yhat(y_val, y_pred):
    labels = ['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']
    MAX = 5000
    if len(y_val) > MAX:
        idx = np.random.choice(len(y_val), MAX, replace=False)
    else:
        idx = np.arange(len(y_val))
    plt.figure(figsize=(10, 10))
    for i in range(6):
        x0 = np.min(y_val[idx, i])
        x1 = np.max(y_val[idx, i])
        plt.subplot(3, 2, i + 1)
        plt.scatter(y_val[idx, i], y_pred[idx, i], alpha=0.6)
        plt.xlabel('True ' + labels[i])
        plt.ylabel('Predicted ' + labels[i])
        plt.plot([x0, x1], [x0, x1], color='red', linestyle='--')
        plt.axis('square')
    plt.suptitle('True vs Predicted Values')
    plt.show()


def knnSub():
    best_k = 6
    model = make_pipeline(KNeighborsRegressor(n_neighbors=best_k))
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)

    y_pred_val_orig = scaler_y.inverse_transform(y_pred_val)
    y_val_orig = scaler_y.inverse_transform(y_val)

    plot_y_yhat(y_val, y_pred_val)

    rmse_val = root_mean_squared_error(y_val_orig, y_pred_val_orig)  # Calculate RMSE using root_mean_squared_error
    print(f"Validation RMSE: {rmse_val:.4f}")
    
    y_test_pred = model.predict(X_test)
    y_test_pred = scaler_y.inverse_transform(y_test_pred)

    submission_df = pd.DataFrame(y_test_pred, columns=['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3'])
    submission_df.index.name = 'Id'
    submission_df.to_csv('knn_submission.csv', index=True)

knnSub()
