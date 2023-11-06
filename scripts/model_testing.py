from sklearn.metrics import mean_squared_error as mse, r2_score #Метрики от scikit-learn
import pandas as pd
import pickle
import json

X_val = pd.read_csv('data/X_val.csv', index_col=0)
y_val = pd.read_csv('data/y_val.csv', index_col=0)
model = pickle.load(open('models/model.pkl', 'rb'))

y_predict = model.predict(X_val)
r2 = r2_score(y_val, y_predict)

# print('Ошибка на тестовых данных')
# print('MSE: %.1f' % mse(y_val, y_predict))
# print('RMSE: %.1f' % mse(y_val, y_predict, squared=False))
# print('R2 : %.4f' %  r2_score(y_val, y_predict))

with open('evaluate/score.json', 'w') as f:
    json.dump({"score": r2}, f)