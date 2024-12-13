import os
import librosa
import numpy as np
import soundfile as sf

class AmpliadorBaseDatos:
    def __init__(self, ruta_db="DB", carpeta_crudos="Crudos", carpeta_ampliados="Amplified", sr=48000):
        """
        Inicializa el ampliador de base de datos con las configuraciones necesarias.

        :param ruta_db: Ruta principal de la base de datos.
        :param carpeta_crudos: Carpeta que contiene los audios crudos.
        :param carpeta_ampliados: Carpeta donde se guardarán los audios ampliados.
        :param sr: Frecuencia de muestreo de los audios.
        """
        self.ruta_crudos = os.path.join(ruta_db, carpeta_crudos)
        self.ruta_ampliados = os.path.join(ruta_db, carpeta_ampliados)
        self.sr = sr

        # Crear carpeta de ampliados si no existe
        os.makedirs(self.ruta_ampliados, exist_ok=True)

    def cambiar_tono(self, audio, n_steps):
        """
        Cambia el tono del audio.

        :param audio: Señal de audio.
        :param n_steps: Número de semitonos para cambiar el tono (+/-).
        :return: Señal de audio con el tono cambiado.
        """
        return librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)

    def cambiar_velocidad(self, audio, factor):
        """
        Cambia la velocidad del audio. La duración del audio cambia proporcionalmente.

        :param audio: Señal de audio.
        :param factor: Factor de cambio de velocidad (>1 más rápido, <1 más lento).
        :return: Señal de audio con velocidad modificada.
        """
        try:
            # Convertir al dominio de frecuencia
            stft = librosa.stft(audio)
            stft_modificado = librosa.effects.time_stretch(librosa.istft(stft), factor)

            # Reconstruir la señal de audio
            return librosa.istft(librosa.stft(stft_modificado))
        except Exception as e:
            print(f"Error al cambiar la velocidad: {e}")
            return audio
        
    def agregar_ruido(self, audio, ruido_factor=0.005):
        """
        Agrega ruido blanco al audio.

        :param audio: Señal de audio.
        :param ruido_factor: Amplitud del ruido agregado.
        :return: Señal de audio con ruido añadido.
        """
        ruido = np.random.randn(len(audio))
        return audio + ruido_factor * ruido

    def ampliar_audio(self, ruta_audio):
        """
        Aplica transformaciones de data augmentation a un audio.

        :param ruta_audio: Ruta del archivo de audio.
        :return: Lista de audios ampliados.
        """
        audio, sr_original = librosa.load(ruta_audio, sr=self.sr)

        # Aplicar transformaciones
        audios_ampliados = [
            ("tono_mas_alto", self.cambiar_tono(audio, n_steps=2)),
            ("tono_mas_bajo", self.cambiar_tono(audio, n_steps=-2)),
            ("mas_rapido", self.cambiar_velocidad(audio, factor=1.25)),
            ("mas_lento", self.cambiar_velocidad(audio, factor=0.75)),
            ("con_ruido", self.agregar_ruido(audio)),
        ]

        return audios_ampliados

    def ampliar_base_datos(self):
        """
        Amplía la base de datos cruda generando nuevas versiones de los audios.
        """
        for archivo in os.listdir(self.ruta_crudos):
            if archivo.endswith(".wav"):
                ruta_audio = os.path.join(self.ruta_crudos, archivo)
                print(f"Ampliando {archivo}...")

                # Obtener las versiones ampliadas del audio
                audios_ampliados = self.ampliar_audio(ruta_audio)

                # Guardar cada versión con un nombre único
                for nombre_variacion, audio_ampliado in audios_ampliados:
                    nombre_salida = archivo.replace(".wav", f"_{nombre_variacion}.wav")
                    ruta_salida = os.path.join(self.ruta_ampliados, nombre_salida)
                    sf.write(ruta_salida, audio_ampliado, self.sr)
                    print(f"Guardado: {ruta_salida}")

if __name__ == "__main__":
    ampliador = AmpliadorBaseDatos()
    ampliador.ampliar_base_datos()
