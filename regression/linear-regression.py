import pandas as pd
from sqlalchemy import create_engine
from LinearRegressionFromScratch import LinearRegressionFromScratch
import joblib

connection_string = 'mysql+pymysql://root:SQLAleksa12!@localhost:3306/'
engine = create_engine(connection_string + 'books_cleaned')

data = pd.read_sql_table('books', con=engine)
data.drop(columns=['index'], inplace=True)

x = data.drop(columns=['naslov', 'opis', 'cena', 'autor', 'kategorija'])
y = data['cena']

x[['width', 'height']] = x['format'].str.split('x', expand=True).astype(float)
x['area'] = x['width'] * x['height']

mean_prices_by_publisher = data.groupby('izdavac')['cena'].mean().reset_index()
mean_prices_by_publisher.rename(columns={'cena': 'srednja_cena_po_izdavacu'}, inplace=True)

x = x.merge(mean_prices_by_publisher, on='izdavac', how='left')

publishers_data = x[['izdavac', 'srednja_cena_po_izdavacu']]

publishers_data = publishers_data.drop_duplicates()
publishers_data['izdavac'] = publishers_data['izdavac'].str.lower()

x.drop(columns=['izdavac', 'format', 'width', 'height'], inplace=True)

x = pd.get_dummies(x, columns=['povez'], dtype=int)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

x_train = x_train.to_numpy().reshape(-1,x_train.shape[1])
x_test = x_test.to_numpy().reshape(-1,x_test.shape[1])

y_train = y_train.to_numpy().reshape(-1,1)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x_train, y_train)

reg = LinearRegressionFromScratch()  
reg.fit(x_train, y_train)


joblib.dump(model, 'linear_regression_model.pkl')
joblib.dump(reg, 'custom_linear_regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(publishers_data, 'publishers_data.pkl')


print("Models saved successfully.")