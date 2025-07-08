import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import root_mean_squared_error

df = pd.read_csv('X_train.csv')

trajectory_size = 258
x0_1_list, y0_1_list, x0_2_list, y0_2_list, x0_3_list, y0_3_list = [], [], [], [], [], []
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
print("Cleaned dataset shape:", df.shape)


for i in range(0, df.shape[0], trajectory_size):
    trajectory_df = df.iloc[i:i + trajectory_size]
    if (trajectory_df['t'] == 0).any():
        initial_row = trajectory_df[trajectory_df['t'] == 0].iloc[0]
        x0_1_list.extend([initial_row['x_1']] * len(trajectory_df))
        y0_1_list.extend([initial_row['y_1']] * len(trajectory_df))
        x0_2_list.extend([initial_row['x_2']] * len(trajectory_df))
        y0_2_list.extend([initial_row['y_2']] * len(trajectory_df))
        x0_3_list.extend([initial_row['x_3']] * len(trajectory_df))
        y0_3_list.extend([initial_row['y_3']] * len(trajectory_df))
    else:
        continue

df = df.iloc[:len(x0_1_list)]

df = df.copy(deep=True)

df['x0_1'] = x0_1_list
df['y0_1'] = y0_1_list
df['x0_2'] = x0_2_list
df['y0_2'] = y0_2_list
df['x0_3'] = x0_3_list
df['y0_3'] = y0_3_list

df = df.sample(frac=0.2)

selected_features = ['t', 'x0_1', 'y0_1', 'x0_2', 'y0_2', 'x0_3', 'y0_3']
targets = ['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']

X = df[selected_features]
y = df[targets]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)


def plot_y_yhat(y_val,y_pred, plot_title):
    labels = ['x_1','y_1','x_2','y_2','x_3','y_3']
    MAX = 500
    if len(y_val) > MAX:
        idx = np.random.choice(len(y_val),MAX, replace=False)
    else:
        idx = np.arange(len(y_val))
    plt.figure(figsize=(10,10))
    for i in range(6):
        x0 = np.min(y_val[idx,i])
        x1 = np.max(y_val[idx,i])
        plt.subplot(3,2,i+1)
        plt.scatter(y_val[idx,i],y_pred[idx,i])
        plt.xlabel('True '+labels[i])
        plt.ylabel('Predicted '+labels[i])
        plt.plot([x0,x1],[x0,x1],color='red')
        plt.axis('square')
    plt.savefig(plot_title + '.pdf')
    plt.show()


def linear_regression():
    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"RMSE: {rmse:.4f}")

    plot_y_yhat(y_test.values, y_pred, 'baseline_model')

linear_regression()


def polynomial_regression():
    degree = 6
    model = make_pipeline(StandardScaler(), PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"RMSE: {rmse:.4f}")

    plot_y_yhat(y_test.values, y_pred, 'poly_model')

polynomial_regression()


def validate_poly_regression(X_train, y_train, X_val, y_val, regressor=None, degrees=range(1,10), max_features=None):
    if regressor is None:
        regressor = RidgeCV(alphas=[1.0])
    best_degree = None
    for degree in degrees:
        model = make_pipeline(PolynomialFeatures(degree), regressor)
        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)
        rmse_val = root_mean_squared_error(y_val, y_pred_val)
        print(f"Validation RMSE for degree {degree}: {rmse_val:.4f}")
        if best_degree is None or rmse_val < best_rmse:
            best_rmse = rmse_val
            best_degree = degree
            best_model = model
    print(f"Best degree: {best_degree}")

validate_poly_regression(X_train, y_train, X_test, y_test)