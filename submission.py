
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error, root_mean_squared_error
import time

df = pd.read_csv('X_train.csv')
test_data = pd.read_csv('X_test.csv')

trajectory_size = 258
indices_to_drop = []

for i in range(0, int((df.shape[0]) / trajectory_size)):
    start = (i * (trajectory_size-1))
    end = (start + 1 ) + 255
    if end > df.shape[0]:
        break
    for j in range(start +1, end):
        if np.isclose(df.at[j, 't'], 0.0, rtol=1e-09, atol=1e-09):
            indices_to_drop.extend(range(start, end+1))
            break

df = df.drop(df.index[indices_to_drop])

#features = ['v_x_1', 'v_y_1', 'v_x_2', 'v_y_2', 'v_x_3', 'v_y_3', 't']

#reduced_features = ['v_x_1', 'v_y_1', 'v_x_3', 't']

reduced_features = ['v_y_1', 'v_x_3', 'v_y_3', 't']

targets = ['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']

test_data.rename(columns={
    'x0_1': 'x_1', 'y0_1': 'y_1',
    'x0_2': 'x_2', 'y0_2': 'y_2',
    'x0_3': 'x_3', 'y0_3': 'y_3'
}, inplace=True)

#test_data['v_x_1'] = 0
test_data['v_y_1'] = 0
#test_data['v_x_2'] = 0
#test_data['v_y_2'] = 0
test_data['v_x_3'] = 0
test_data['v_y_3'] = 0


X_train, X_val, y_train, y_val = train_test_split(df[reduced_features], df[targets], test_size=0.2)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)
y_train = scaler_y.fit_transform(y_train)
y_val = scaler_y.transform(y_val)

#X_train = df[features]
#y_train = df[targets]

#features = ['t']
#X_test = test_data[features] #submission
#y_test = test_data[targets] #submission


X_test = scaler_X.transform(test_data[reduced_features])

def rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)

def plot_y_yhat(y_val,y_pred):
    labels = ['x_1','y_1','x_2','y_2','x_3','y_3']
    MAX = 5000
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
    #plt.savefig(plot_title + '.pdf')
    plt.show()

def linear_regression():
    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print(f"Linear Regression RMSE: {rmse(y_val, y_pred):.4f}")

    plot_y_yhat(y_val, y_pred)

    y_test_pred = model.predict(X_test)
    y_test_pred = scaler_y.inverse_transform(y_test_pred)

    submission_df = pd.DataFrame(y_test_pred, columns=['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3'])
    submission_df.index.name = 'Id'
    submission_df.to_csv('baseline-model.csv', index=True)

    #plot_y_yhat(y_test, y_pred)

#linear_regression()

def polynomial_regression():
    degree = 6
    model = make_pipeline(PolynomialFeatures(degree), RidgeCV(alphas=[1.0]))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    print(f"Polynomial Regression RMSE: {rmse(y_val, y_pred):.4f}")

    y_test_pred = model.predict(X_test)
    y_test_pred = scaler_y.inverse_transform(y_test_pred)

    plot_y_yhat(y_val, y_test_pred)

    submission_df = pd.DataFrame(y_test_pred, columns=['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3'])
    submission_df.index.name = 'Id'
    submission_df.to_csv('polynomial_submission.csv', index=True)

polynomial_regression()

def validate_knn_regression(X_train, y_train, X_val, y_val, k=range(1, 15)):
    results = []
    times = []

    for neighbors in k:
        start_time = time.time()
        
        model = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=neighbors))
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        results.append((neighbors, rmse))

        end_time = time.time()
        times.append((neighbors, end_time - start_time))
        
        print(f"k = {neighbors}: RMSE = {rmse:.4f}, Time taken = {end_time - start_time:.4f} seconds")
    
    return results, times


def knn_regression(X_train, y_train, X_test, y_test, k=5):
    model = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=k))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Final Test RMSE with k={k}: {rmse:.4f}")

    ids = np.arange(0, len(y_pred))
    y_pred_df = pd.DataFrame({
        'Id': ids,
        'x_1': y_pred[:, 0],
        'y_1': y_pred[:, 1],
        'x_2': y_pred[:, 2],
        'y_2': y_pred[:, 3],
        'x_3': y_pred[:, 4],
        'y_3': y_pred[:, 5]
    })
    
    y_pred_df.to_csv('knn_submission.csv', index=False)

    plot_y_yhat(y_test, y_pred)

#knn_regression(X_train, y_train, X_test, y_test, k = 9) 
