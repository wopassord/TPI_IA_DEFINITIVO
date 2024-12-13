from Preprocesador import PreprocesadorAudios
from Parametrizador import ParametrizadorAudios
from Grabadora import GrabadoraAudios
from Clasificador import ClasificadorAudios

class ProyectoAudios:
    def __init__(self):
        """
        Inicializa el proyecto de reconocimiento de audios con las configuraciones necesarias.
        """
        self.ruta_db = "DB"
        self.ruta_candidato = "Candidato"
        self.segmentos = 10
        self.sr = 48000

        # Inicializar las clases de componentes
        self.preprocesador = PreprocesadorAudios(
            ruta_db=self.ruta_db,
            carpeta_crudos="Crudos",
            carpeta_procesados="Processed",
            sr=self.sr
        )
        self.parametrizador = ParametrizadorAudios(
            ruta_db=self.ruta_db,
            carpeta_procesados="Processed",
            segmentos=self.segmentos,
            sr=self.sr
        )
        self.grabadora = GrabadoraAudios(
            ruta_candidato=self.ruta_candidato,
            sr=self.sr,
            segmentos=self.segmentos
        )
        self.clasificador = ClasificadorAudios()

    def menu_principal(self):
        """
        Muestra el menú principal del proyecto y permite al usuario interactuar con las funcionalidades.
        """
        while True:
            print("\n===== MENÚ PRINCIPAL =====")
            print("1. Procesar base de datos de audios")
            print("2. Extraer parámetros y generar CSV")
            print("3. Grabar y procesar un nuevo audio candidato")
            print("4. Clasificar audio candidato")
            print("5. Salir")

            try:
                opcion = int(input("Selecciona una opción: "))
                if opcion == 1:
                    self.procesar_base_datos()
                elif opcion == 2:
                    self.extraer_parametros()
                elif opcion == 3:
                    self.grabar_y_procesar_audio_candidato()
                elif opcion == 4:
                    self.clasificar_audio_candidato()
                elif opcion == 5:
                    print("Saliendo del programa. ¡Hasta luego!")
                    break
                else:
                    print("Opción no válida. Intenta nuevamente.")
            except ValueError:
                print("Entrada no válida. Por favor, ingresa un número.")

    def procesar_base_datos(self):
        """
        Procesa los audios crudos de la base de datos.
        """
        print("\nProcesando audios crudos de la base de datos...")
        self.preprocesador.procesar_base_datos()
        print("Procesamiento de base de datos finalizado.")

    def extraer_parametros(self):
        """
        Extrae parámetros de los audios procesados y genera el archivo CSV.
        """
        print("\nExtrayendo parámetros y generando CSV...")
        self.parametrizador.generar_csv_parametros()
        print("Extracción de parámetros finalizada.")

    def grabar_y_procesar_audio_candidato(self):
        """
        Graba y procesa un nuevo audio candidato.
        """
        print("\nGrabando un nuevo audio candidato...")
        self.grabadora.grabar_audio()
        if self.grabadora.procesar_audio_candidato():
            self.grabadora.extraer_parametros_candidato()
            print("Audio candidato procesado y parametrizado exitosamente.")

    def clasificar_audio_candidato(self):
        """
        Clasifica el audio candidato contra la base de datos utilizando KNN.
        """
        print("\nClasificando audio candidato...")
        etiqueta_predicha = self.clasificador.clasificar_candidato()
        print(f"El audio candidato fue clasificado como: {etiqueta_predicha}")

        # Validar si el audio candidato fue clasificado correctamente
        guardar_audio = input("¿El audio candidato fue correctamente clasificado? (s/n): ").strip().lower()
        if guardar_audio == "s":
            print("Audio candidato añadido a la base de datos cruda.")
        else:
            print("Audio candidato descartado.")

if __name__ == "__main__":
    proyecto = ProyectoAudios()
    proyecto.menu_principal()
