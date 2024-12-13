import os
import sounddevice as sd
import librosa
import numpy as np
import soundfile as sf
from sklearn.preprocessing import StandardScaler
from Parametrizador import ParametrizadorAudios  # Importa el parametrizador para reutilizarlo
from Preprocesador import PreprocesadorAudios  # Importa el preprocesador

class GrabadoraAudios:
    def __init__(self, ruta_candidato="Candidato", archivo_candidato="audio_candidato.wav", archivo_procesado="processed_audio_candidato.wav", sr=48000, segmentos=10):
        """
        Inicializa la grabadora con las configuraciones necesarias.

        :param ruta_candidato: Carpeta donde se guardarán los audios grabados.
        :param archivo_candidato: Nombre del archivo para el audio crudo.
        :param archivo_procesado: Nombre del archivo para el audio procesado.
        :param sr: Frecuencia de muestreo de los audios.
        :param segmentos: Cantidad de segmentos para parametrización.
        """
        self.ruta_candidato = ruta_candidato
        self.archivo_candidato = os.path.join(ruta_candidato, archivo_candidato)
        self.archivo_procesado = os.path.join(ruta_candidato, archivo_procesado)
        self.sr = sr
        self.segmentos = segmentos

        # Crear la carpeta de Candidato si no existe
        os.makedirs(self.ruta_candidato, exist_ok=True)

    def grabar_audio(self, duracion=5):
        """
        Graba un audio de duración especificada y lo guarda como archivo WAV.

        :param duracion: Duración de la grabación en segundos.
        """
        print(f"Grabando audio por {duracion} segundos...")
        audio = sd.rec(int(duracion * self.sr), samplerate=self.sr, channels=1, dtype="float32")
        sd.wait()  # Espera a que termine la grabación
        sf.write(self.archivo_candidato, audio, self.sr)
        print(f"Audio crudo guardado en {self.archivo_candidato}.")

    def procesar_audio_candidato(self):
        """
        Preprocesa el audio grabado utilizando el preprocesador.
        """
        preprocesador = PreprocesadorAudios(ruta_db=".", carpeta_crudos="Candidato", carpeta_procesados="Candidato", sr=self.sr)
        audio_procesado = preprocesador.procesar_audio(self.archivo_candidato)

        if audio_procesado is not None:
            sf.write(self.archivo_procesado, audio_procesado, self.sr)
            print(f"Audio procesado guardado en {self.archivo_procesado}.")
            return True
        else:
            print("Error al procesar el audio candidato.")
            return False

    def extraer_parametros_candidato(self, archivo_scaler="DB/scaler.pkl", archivo_salida="DB/parametros_candidato.csv"):
        """
        Extrae y escala los parámetros del audio procesado.

        :param archivo_scaler: Archivo que contiene el modelo de escalado.
        :param archivo_salida: Nombre del archivo CSV donde se guardarán los parámetros del candidato.
        """
        # Inicializar el parametrizador
        parametrizador = ParametrizadorAudios(ruta_db=".", carpeta_procesados="Candidato", sr=self.sr, segmentos=self.segmentos)

        # Extraer características del audio procesado
        caracteristicas = parametrizador.parametrizar_audio(self.archivo_procesado)
        if caracteristicas is None:
            print("Error al extraer parámetros del audio candidato.")
            return False

        # Cargar el escalador previamente entrenado
        try:
            from joblib import load
            scaler = load(archivo_scaler)
            caracteristicas_escaladas = scaler.transform([caracteristicas])

            # Guardar los parámetros escalados en un archivo CSV
            with open(archivo_salida, mode="w", newline="") as file:
                import csv
                writer = csv.writer(file)
                columnas = [f"MFCC_{i+1}" for i in range(len(caracteristicas_escaladas[0]))]
                writer.writerow(columnas)
                writer.writerow(caracteristicas_escaladas[0])

            print(f"Parámetros escalados guardados en {archivo_salida}.")
            return True
        except Exception as e:
            print(f"Error al escalar o guardar los parámetros: {e}")
            return False


if __name__ == "__main__":
    grabadora = GrabadoraAudios()

    # Grabación del audio
    grabadora.grabar_audio(duracion=5)

    # Preprocesamiento del audio grabado
    if grabadora.procesar_audio_candidato():
        # Extracción y escalado de parámetros
        grabadora.extraer_parametros_candidato()
