import os
import csv
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ClasificadorAudios:
    def __init__(self, archivo_db="DB/parametros_DB.csv", archivo_candidato="DB/parametros_candidato.csv", n_componentes=3, k=5):
        """
        Inicializa el clasificador con las configuraciones necesarias.

        :param archivo_db: Ruta del archivo CSV con los parámetros de la base de datos.
        :param archivo_candidato: Ruta del archivo CSV con los parámetros del audio candidato.
        :param n_componentes: Número de componentes para PCA.
        :param k: Número de vecinos para el algoritmo KNN.
        """
        self.archivo_db = archivo_db
        self.archivo_candidato = archivo_candidato
        self.n_componentes = n_componentes
        self.k = k

    def cargar_parametros(self, archivo):
        """
        Carga los parámetros y etiquetas de un archivo CSV.

        :param archivo: Ruta del archivo CSV.
        :return: Matriz de parámetros (X) y etiquetas (y).
        """
        X = []
        y = []
        with open(archivo, mode="r") as file:
            reader = csv.reader(file)
            next(reader)  # Saltar la fila de encabezado
            for row in reader:
                # La última columna es la etiqueta (solo en la base de datos)
                X.append([float(value) for value in row[:-1]])
                if len(row) > len(X[0]):  # Solo en el archivo DB
                    y.append(row[-1])
        return np.array(X), np.array(y) if y else None

    def clasificar_candidato(self):
        """
        Clasifica el audio candidato usando KNN y visualiza la comparación con la base de datos.
        """
        # Cargar parámetros de la base de datos
        X_db, y_db = self.cargar_parametros(self.archivo_db)

        # Cargar parámetros del audio candidato
        X_candidato, _ = self.cargar_parametros(self.archivo_candidato)

        if X_candidato is None or X_db is None:
            print("Error: No se pudieron cargar los parámetros necesarios.")
            return

        # Validar y ajustar dimensiones del audio candidato
        if X_candidato.shape[1] < X_db.shape[1]:  # Si el candidato tiene menos características
            diferencia = X_db.shape[1] - X_candidato.shape[1]
            X_candidato = np.hstack([X_candidato, np.zeros((X_candidato.shape[0], diferencia))])
            print(f"Ajustadas dimensiones del audio candidato: {X_candidato.shape}")

        elif X_candidato.shape[1] > X_db.shape[1]:  # Si el candidato tiene más características
            X_candidato = X_candidato[:, :X_db.shape[1]]
            print(f"Recortadas dimensiones del audio candidato: {X_candidato.shape}")

        # Reducir dimensionalidad con PCA
        pca = PCA(n_components=self.n_componentes)
        X_db_pca = pca.fit_transform(X_db)
        X_candidato_pca = pca.transform(X_candidato)

        # Clasificar con KNN
        knn = KNeighborsClassifier(n_neighbors=self.k)
        knn.fit(X_db_pca, y_db)
        etiqueta_candidato = knn.predict(X_candidato_pca)

        # Visualización en 3D
        self.visualizar_3D(X_db_pca, y_db, X_candidato_pca, etiqueta_candidato[0])

        return etiqueta_candidato[0]

    def visualizar_3D(self, X_db_pca, y_db, X_candidato_pca, etiqueta_candidato):
        """
        Genera un gráfico en 3D para comparar la base de datos y el candidato.

        :param X_db_pca: Parámetros de la base de datos en 3D.
        :param y_db: Etiquetas de la base de datos.
        :param X_candidato_pca: Parámetros del audio candidato en 3D.
        :param etiqueta_candidato: Etiqueta predicha para el audio candidato.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Definir colores específicos para cada verdura
        color_map = {
            "zanahoria": "orange",
            "camote": "violet",  # Violeta
            "berenjena": "darkviolet",  # Violeta oscuro
            "papa": "saddlebrown"  # Marrón similar al color de la papa
        }

        # Graficar puntos de la base de datos agrupados por etiqueta
        for etiqueta, color in color_map.items():
            indices = (y_db == etiqueta)
            if np.any(indices):  # Graficar solo si hay datos con esta etiqueta
                ax.scatter(X_db_pca[indices, 0], X_db_pca[indices, 1], X_db_pca[indices, 2],
                        c=color, label=etiqueta, s=50, alpha=0.7)

        # Graficar el audio candidato
        ax.scatter(X_candidato_pca[0, 0], X_candidato_pca[0, 1], X_candidato_pca[0, 2],
                c="black", label="Candidato (X)", s=100, marker="X")

        # Configuración del gráfico
        ax.set_title("Clasificación de audio candidato")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")

        # Ajustar leyenda para que sea más clara y solo muestre grupos
        ax.legend(loc="upper left")
        plt.show()


if __name__ == "__main__":
    clasificador = ClasificadorAudios()

    # Clasificar el audio candidato
    etiqueta_predicha = clasificador.clasificar_candidato()
    print(f"El audio candidato fue clasificado como: {etiqueta_predicha}")
