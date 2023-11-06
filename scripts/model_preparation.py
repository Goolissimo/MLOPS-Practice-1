import pandas as pd
import scipy.stats as stats # статистические функции библиотеки scipy
from sklearn.model_selection import ShuffleSplit, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor # Регрессия К-Ближайших соседей от scikit-learn
import pickle

X_train = pd.read_csv('data/X_train.csv', index_col=0)
y_Train = pd.read_csv('data/y_Train.csv', index_col=0)

# словарь гиперпараметров в виде
# обозначение гиперпараметров : из какого распределения сэмплируем
parameters = {'n_neighbors':stats.randint(1,50), # задаем распределение как равномерное от 1 до 50
              'weights':['uniform', 'distance']}

# количество итераций
n_iter_search = 100

kNN_search = RandomizedSearchCV(estimator=KNeighborsRegressor(),  # оптимизируем нашу модель
                                verbose=3,  # чтобы он всё подробно расписал. если не интересно - пишем 0
                          param_distributions=parameters, # что оптимизируем - берем из словарика
                          cv=ShuffleSplit(n_splits=5, random_state=42), # указываем тип кросс-валидации
                          n_iter=n_iter_search # количество итераций
                          )
kNN_search.fit(X_train, y_Train) #обучаем поисковик
kNNbest = kNN_search.best_estimator_ #забираем лучшую
kNNbest.fit(X_train, y_Train) #обучаем лучший KNN на наших данных

pickle.dump(kNNbest, open('models/model.pkl', 'wb'))