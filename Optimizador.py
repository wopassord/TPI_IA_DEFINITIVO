import numpy as np
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

class VerificacionCruzadaKNN:
    def __init__(self, archivo_db="DB/parametros_DB.csv", max_k=20, n_splits=5):
        """
        Inicializa la clase de verificación cruzada.

        :param archivo_db: Ruta del archivo CSV con los parámetros de la base de datos.
        :param max_k: Valor máximo de K a evaluar.
        :param n_splits: Número de divisiones para la validación cruzada.
        """
        self.archivo_db = archivo_db
        self.max_k = max_k
        self.n_splits = n_splits

    def cargar_parametros(self):
        """
        Carga los parámetros y etiquetas desde el archivo CSV.

        :return: Matriz de características (X) y etiquetas (y).
        """
        X = []
        y = []
        with open(self.archivo_db, mode="r") as file:
            reader = csv.reader(file)
            next(reader)  # Saltar la fila de encabezado
            for row in reader:
                X.append([float(value) for value in row[:-1]])  # Características
                y.append(row[-1])  # Etiqueta
        return np.array(X), np.array(y)

    def verificar_k(self):
        """
        Realiza verificación cruzada para encontrar el valor óptimo de K.

        :return: Diccionario con la precisión promedio para cada valor de K.
        """
        X, y = self.cargar_parametros()

        # Configuración de la validación cruzada
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        resultados = {}

        print("Realizando verificación cruzada...")
        for k in range(1, self.max_k + 1):
            precisiones = []

            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # Modelo KNN
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)

                # Calcular precisión
                precision = accuracy_score(y_test, y_pred)
                precisiones.append(precision)

            # Promediar las precisiones para este valor de K
            resultados[k] = np.mean(precisiones)
            print(f"K = {k}, Precisión promedio = {resultados[k]:.4f}")

        return resultados

    def encontrar_mejor_k(self):
        """
        Encuentra el valor de K con la mayor precisión promedio.

        :return: Valor de K óptimo y su precisión correspondiente.
        """
        resultados = self.verificar_k()
        mejor_k = max(resultados, key=resultados.get)
        mejor_precision = resultados[mejor_k]

        print(f"\nEl valor óptimo de K es {mejor_k} con una precisión de {mejor_precision:.4f}.")
        return mejor_k, mejor_precision

if __name__ == "__main__":
    verificador = VerificacionCruzadaKNN()
    verificador.encontrar_mejor_k()
