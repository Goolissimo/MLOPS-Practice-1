import pandas as pd
import re

### data_creation
# Загружаем тренировочные данные
data = pd.read_csv('data/data.csv', index_col=0)

# Удалим пропущенные значения
data.dropna(subset=['Mileage', 'Power', 'Mileage', 'Seats'], inplace=True)

# Удалим столбец потому что большая часть столбца это пропущенные значения
data.drop(['New_Price'], axis=1, inplace=True)

#Засунем столбец Power в отдельный датафрейм, удалим нули, конвертируем и посчитаем среднее
df = pd.DataFrame(data['Power'])
df = df[df.Power != 'null bhp']
df['Power'] = df['Power'].str.replace('[^\d\.]', '', regex=True).astype('float64')

#Заменяем null bhp на наше среднее
data['Power'] = data['Power'].str.replace('null bhp', '113')

#Убираем не числовые символы и конвертируем столбцы в числовой формат
data['Mileage'] = data['Mileage'].str.replace('[^\d\.]', '', regex=True).astype('float64')
data['Engine'] = data['Engine'].str.replace('[^\d\.]', '', regex=True).astype('float64')
data['Power'] = data['Power'].str.replace('[^\d\.]', '', regex=True).astype('float64')

# Удаление выбросов
data.drop(data[data['Kilometers_Driven'] > 100000].index, axis=0, inplace=True)
data.drop(data[data['Engine'] > 5000].index, axis=0, inplace=True)
data.drop(data[data['Power'] > 400].index, axis=0, inplace=True)

# Преобразование столбца чтобы оставить только марку машины и удаляем изначальный столбец
data['Brand'] = data['Name'].apply(lambda row: row.split()[0])
data.drop(['Name'], axis=1, inplace=True)

# Save files
data.to_csv('data/clear_data.csv')