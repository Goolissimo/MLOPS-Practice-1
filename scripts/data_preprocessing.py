import pandas as pd
from sklearn.pipeline import Pipeline # Pipeline.Не добавить, не убавить
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler # Импортируем нормализацию и One-Hot Encoding от sklearn
from sklearn.compose import ColumnTransformer # т.н. преобразователь колонок
import yaml
import numpy as np

data = pd.read_csv('data/clear_data.csv', index_col=0)
# Разделим признаки на числовые и категориальные
num_columns = list(data.select_dtypes('number'))
num_columns.remove('Price')#Уберем целевую переменную из признаков
cat_columns = list(data.select_dtypes('object'))

params = yaml.safe_load(open("params.yaml"))["split"]
p_split_ratio = params["split_ratio"]

# Разбиваем тренировочные данные на тренировочную и валидационную выборку
X_Train, X_val, y_Train, y_val = train_test_split(data.drop('Price', axis = 1), data['Price'].ravel(), test_size=p_split_ratio, random_state=44)

# Pipeline для числовых данных (нормализация)
numerical_pipe = Pipeline([
    ('scaler', MinMaxScaler())
])
# Pipeline для категориальных данных (One-Hot кодирование)
categorical_pipe = Pipeline([
    ('encoder', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False))
])

preprocessors = ColumnTransformer(transformers=[
    ('num', numerical_pipe, num_columns),
    ('cat', categorical_pipe, cat_columns)
])
preprocessors.fit(X_Train)

X_train = preprocessors.transform(X_Train) # Преобразуем  тренировочные данные
X_val = preprocessors.transform(X_val) # Преобразуем валидационные данные

with open('data/X_train.npy', 'wb') as f:
    np.save(f, X_train)
with open('data/X_val.npy', 'wb') as f:
    np.save(f, X_val)
with open('data/y_Train.npy', 'wb') as f:
    np.save(f, y_Train)
with open('data/y_val.npy', 'wb') as f:
    np.save(f, y_val)
# X_train.to_csv('data/X_train.csv')
# X_val.to_csv('data/X_val.csv')
# y_Train.to_csv('data/y_Train.csv')
# y_val.to_csv('data/y_val.csv')