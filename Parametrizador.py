import os
import librosa
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler
from joblib import dump  # Importado para guardar el escalador

class ParametrizadorAudios:
    def __init__(self, ruta_db="DB", carpeta_procesados="Processed", archivo_salida="parametros_DB.csv", segmentos=10, sr=48000):
        """
        Inicializa el parametrizador con las configuraciones necesarias.

        :param ruta_db: Ruta principal de la base de datos.
        :param carpeta_procesados: Carpeta donde están los audios procesados.
        :param archivo_salida: Nombre del archivo CSV donde se guardarán los parámetros.
        :param segmentos: Cantidad de segmentos en los que dividir los audios.
        :param sr: Frecuencia de muestreo de los audios.
        """
        self.ruta_procesados = os.path.join(ruta_db, carpeta_procesados)
        self.archivo_salida = os.path.join(ruta_db, archivo_salida)
        self.segmentos = segmentos
        self.sr = sr

    def extraer_mfccs(self, audio):
        """
        Extrae los MFCCs de un audio y calcula el promedio por segmento.

        :param audio: Señal de audio.
        :return: Lista con los promedios de los MFCCs para cada segmento.
        """
        try:
            n_fft = min(len(audio), 2048)  # Ajustar dinámicamente el tamaño de n_fft
            if n_fft < 512:
                raise ValueError("El segmento es demasiado corto para extraer características significativas.")

            mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=13, n_fft=n_fft)
            return np.mean(mfccs.T, axis=0)  # Promedio de los MFCCs
        except ValueError as e:
            print(f"Advertencia: {e}")
            return np.array([])  # Devolver un array vacío en lugar de None

    def parametrizar_audio(self, ruta_audio):
        """
        Parametriza un audio dividiéndolo en segmentos y extrayendo MFCCs.

        :param ruta_audio: Ruta del archivo de audio procesado.
        :return: Lista con las características del audio.
        """
        try:
            # Cargar el audio
            audio, sr_original = librosa.load(ruta_audio, sr=self.sr)

            # Verificar si el audio es demasiado corto
            if len(audio) < self.segmentos:
                raise ValueError(f"El audio {ruta_audio} es demasiado corto para dividirse en {self.segmentos} segmentos.")

            # Dividir el audio en segmentos
            segmentos = np.array_split(audio, self.segmentos)

            # Extraer características de cada segmento
            caracteristicas = []
            for segmento in segmentos:
                mfccs_segmento = self.extraer_mfccs(segmento)
                if mfccs_segmento.size > 0:  # Evaluar correctamente si hay datos en el array
                    caracteristicas.extend(mfccs_segmento)

            return caracteristicas
        except ValueError as e:
            print(f"Advertencia: {e}")
            return None

    def generar_csv_parametros(self):
        """
        Procesa todos los audios en la carpeta de "Processed" y guarda sus características en un archivo CSV.
        """
        datos = []
        etiquetas = []

        for archivo in os.listdir(self.ruta_procesados):
            if archivo.endswith(".wav"):
                ruta_audio = os.path.join(self.ruta_procesados, archivo)
                print(f"Procesando {archivo} para extracción de parámetros...")

                caracteristicas = self.parametrizar_audio(ruta_audio)
                if caracteristicas:
                    # Obtener la etiqueta de la verdura a partir del nombre del archivo
                    etiqueta = "desconocido"
                    if "zanahoria" in archivo.lower():
                        etiqueta = "zanahoria"
                    elif "camote" in archivo.lower():
                        etiqueta = "camote"
                    elif "berenjena" in archivo.lower():
                        etiqueta = "berenjena"
                    elif "papa" in archivo.lower():
                        etiqueta = "papa"

                    datos.append(caracteristicas)
                    etiquetas.append(etiqueta)

        if datos:
            # Escalar los datos
            scaler = StandardScaler()
            datos_escalados = scaler.fit_transform(datos)

            # Guardar el escalador entrenado
            scaler_path = os.path.join("DB", "scaler.pkl")
            dump(scaler, scaler_path)
            print(f"Escalador guardado exitosamente en {scaler_path}")

            # Guardar en el CSV
            with open(self.archivo_salida, mode="w", newline="") as file:
                writer = csv.writer(file)
                # Escribir encabezados
                columnas = [f"MFCC_{i+1}" for i in range(len(datos_escalados[0]))] + ["Etiqueta"]
                writer.writerow(columnas)

                # Escribir datos y etiquetas
                for i in range(len(datos_escalados)):
                    writer.writerow(list(datos_escalados[i]) + [etiquetas[i]])

            print(f"Parámetros guardados exitosamente en {self.archivo_salida}")
        else:
            print("No se encontraron audios procesados para parametrizar.")

if __name__ == "__main__":
    parametrizador = ParametrizadorAudios()
    parametrizador.generar_csv_parametros()
