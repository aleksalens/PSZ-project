import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib

# Povezivanje sa bazom podataka sa ociscenim podacima
connection_string = 'mysql+pymysql://root:password@localhost:3306/' # Your SQL password instead of "password"
engine = create_engine(connection_string + 'books_cleaned')

# Citanje podataka iz tabele
data = pd.read_sql_table('books', con=engine)
data.drop(columns=['index'], inplace=True)

# Uklanjanje kolona koje necemo koristiti kao feature u logistickoj regresiji
x = data.drop(columns=['naslov', 'opis', 'cena', 'autor', 'kategorija'])
y = data['cena']

# Racunanje povrsine knjiga na osnovu formata
x[['width', 'height']] = x['format'].str.split('x', expand=True).astype(float)
x['area'] = x['width'] * x['height']

# Racunanje srednje cene knjige po izdavacu
mean_prices_by_publisher = data.groupby('izdavac')['cena'].mean().reset_index()
mean_prices_by_publisher.rename(columns={'cena': 'srednja_cena_po_izdavacu'}, inplace=True)

x = x.merge(mean_prices_by_publisher, on='izdavac', how='left')

# Svi izdavaci i odgovarajuce srednje cene knjige
publishers_data = x[['izdavac', 'srednja_cena_po_izdavacu']]

publishers_data = publishers_data.drop_duplicates()
publishers_data['izdavac'] = publishers_data['izdavac'].str.lower()

x.drop(columns=['izdavac', 'format', 'width', 'height'], inplace=True)

# One Hot Encoding kolone 'povez' posto ima vrednosti  'Broš' i 'Tvrd'
x = pd.get_dummies(x, columns=['povez'], dtype=int)

# Definisanje 4 klase na osnovu cene knjige - Veoma jeftina, Jeftina, Skupa i Veoma skupa
def categorize_price(price):
    if price <= 750:
        return 0
    elif price <= 1500:
        return 1
    elif price <= 3000:
        return 2
    elif price <= 5000:
        return 3
    else:
        return 4

# Kreiranje kategorickog izlaza y
y_categorical = y.apply(categorize_price)

# Podela podataka na trening i test skup
x_train, x_test, y_train, y_test = train_test_split(x, y_categorical, test_size=0.2)

x_train = x_train.to_numpy().reshape(-1,x_train.shape[1])
x_test = x_test.to_numpy().reshape(-1,x_test.shape[1])

y_train = y_train.to_numpy().reshape(-1,1)
y_test = y_test.to_numpy().reshape(-1,1)

# Ispravljanje niza
y_train = y_train.ravel()
y_test = y_test.ravel()

# Skaliranje podatak koriscenjem MinMaxScalera
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# One-vs-Rest Logisticka regresija
ovr_model = LogisticRegression(multi_class='ovr')
ovr_model.fit(x_train_scaled, y_train)

# Multinomijalna Logisticka  regresija
multinomial_model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
multinomial_model.fit(x_train_scaled, y_train)

# Predikcije obe regresije
ovr_predictions = ovr_model.predict(x_test_scaled)
multinomial_predictions = multinomial_model.predict(x_test_scaled)

print("One-vs-Rest (OvR) Logistic Regression:")
print(ovr_predictions)

print("\nMultinomial Logistic Regression:")
print(multinomial_predictions)

# Racunanje tacnosti za obe regresije
ovr_accuracy = accuracy_score(y_test, ovr_predictions)
multinomial_accuracy = accuracy_score(y_test, multinomial_predictions)

# Racunanje F1 skora za obe regresije
ovr_f1_score = f1_score(y_test, ovr_predictions, average='weighted')
multinomial_f1_score = f1_score(y_test, multinomial_predictions, average='weighted')

print("One-vs-Rest (OvR) Logistic Regression:")
print(f"Accuracy: {ovr_accuracy:.2f}")
print(f"F1 Score (weighted): {ovr_f1_score:.2f}")

print("\nMultinomial Logistic Regression:")
print(f"Accuracy: {multinomial_accuracy:.2f}")
print(f"F1 Score (weighted): {multinomial_f1_score:.2f}")

# Cuvanje modela potrebnih za rad aplikacije
joblib.dump(ovr_model, 'one_vs_rest_model.pkl')
joblib.dump(multinomial_model, 'multinomial_model.pkl')

print("Models saved successfully.")

# Iscrtavanje konfuzione matrice za oba tipa regresije
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ovr_cm = ConfusionMatrixDisplay.from_predictions(y_test, ovr_predictions)
ovr_cm.ax_.set_title('One-vs-Rest (OvR) Logistic Regression')

multinomial_cm = ConfusionMatrixDisplay.from_predictions(y_test, multinomial_predictions)
multinomial_cm.ax_.set_title('Multinomial Logistic Regression')

plt.show()