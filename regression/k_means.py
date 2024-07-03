import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sqlalchemy import create_engine

# Povezivanje sa bazom podataka sa ociscenim podacima
connection_string = 'mysql+pymysql://root:SQLAleksa12!@localhost:3306/'
engine = create_engine(connection_string + 'books_cleaned')

# Citanje podataka iz tabele
data = pd.read_sql_table('books', con=engine)
data.drop(columns=['index'], inplace=True)

# Uklanjanje kolona koje necemo koristiti kao feature u linearnoj regresiji
x = data.drop(columns=['naslov', 'opis', 'cena', 'autor', 'kategorija', 'povez'])
y = data['cena']

# Racunanje povrsine knjiga na osnovu formata
x[['width', 'height']] = x['format'].str.split('x', expand=True).astype(float)
x['area'] = x['width'] * x['height']

# Racunanje srednje cene knjige po izdavacu
mean_prices_by_publisher = data.groupby('izdavac')['cena'].mean().reset_index()
mean_prices_by_publisher.rename(columns={'cena': 'srednja_cena_po_izdavacu'}, inplace=True)

x = x.merge(mean_prices_by_publisher, on='izdavac', how='left')
x.drop(columns=['format', 'width', 'height', 'izdavac'], inplace=True)

# Vizuelizacija podataka. Iscrtavanje scatter grafika ulaza koje smo uzeli 
plt.figure(figsize=(12, 10))

plt.subplot(231)
plt.scatter(x['godina_izdanja'], x['broj_strana'])
plt.xlabel('Godina izdanja')
plt.ylabel('Broj strana')

plt.subplot(232)
plt.scatter(x['broj_strana'], x['area'])
plt.xlabel('Broj strana')
plt.ylabel('Area')

plt.subplot(233)
plt.scatter(x['broj_strana'], x['srednja_cena_po_izdavacu'])
plt.xlabel('Broj strana')
plt.ylabel('Srednja cena po izdavaču')

plt.subplot(234)
plt.scatter(x['area'], x['srednja_cena_po_izdavacu'])
plt.xlabel('Area')
plt.ylabel('Srednja cena po izdavaču')

plt.subplot(235)
plt.scatter(x['godina_izdanja'], x['srednja_cena_po_izdavacu'])
plt.xlabel('Godina izdanja')
plt.ylabel('Srednja cena po izdavaču')

plt.subplot(236)
plt.scatter(x['godina_izdanja'], x['area'])
plt.xlabel('Godina izdanja')
plt.ylabel('Srednja cena po izdavaču')

plt.tight_layout()
plt.show()

x = x.to_numpy().reshape(-1,x.shape[1])

# Skaliranje podataka koriscenjem MinMaxScalera
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x = scaler.fit_transform(x)

# Racunanje broja gresaka za razlicite vrednosti K
inertia = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(x)
    inertia.append(kmeans.inertia_)

# Iscrtavanje inercije
plt.plot(range(1,10), inertia)
plt.title("Smanjenje greške sa povećanjem broja klastera")
plt.xlabel("Broj klaster")
plt.ylabel("Inercija")
plt.show()


