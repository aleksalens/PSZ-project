import numpy as np

class LinearRegressionFromScratch:
    def __init__(self, learning_rate=0.23, n_iterations=20000, threshold=1e-5):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.threshold = threshold
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Inicijalizacija tezina i bias-a
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
        
        # Inicijalizacija promenljive koja cuva cost funkciju u prethodnoj iteraciji
        prev_cost = float('inf') 
        
        # Metoda Gradijentnog spusta
        for iteration in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Racunanje gradijenata
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Azuriranje tezina
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Racunanje cost funkcije u trenutnoj iteraciji (Mean Squared Error)
            current_cost = np.mean((y_pred - y) ** 2)
            
            # Provera uslova za zaustavljanje algoritma
            if abs(current_cost - prev_cost) < self.threshold:
                print(f"Stopped at iteration {iteration+1} because change in cost {abs(current_cost - prev_cost)} < threshold {self.threshold}")
                break
            
            prev_cost = current_cost

    # Funkcija za predikciju
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
