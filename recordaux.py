import os
import shutil
from Grabadora import GrabadoraAudios
from Clasificador import ClasificadorAudios

class GrabadorClasificador:
    def __init__(self, ruta_db="DB", ruta_candidato="Candidato"):
        """
        Inicializa el flujo de grabación, clasificación y almacenamiento.

        :param ruta_db: Ruta principal de la base de datos.
        :param ruta_candidato: Ruta para guardar el audio candidato.
        """
        self.ruta_crudos = os.path.join(ruta_db, "Crudos")
        self.ruta_candidato = ruta_candidato
        self.grabadora = GrabadoraAudios(ruta_candidato=ruta_candidato)
        self.clasificador = ClasificadorAudios()

        # Crear la carpeta de crudos si no existe
        os.makedirs(self.ruta_crudos, exist_ok=True)

    def seleccionar_verdura(self):
        """
        Permite al usuario seleccionar una verdura.
        """
        opciones = ["zanahoria", "camote", "berenjena", "papa"]
        print("\nSelecciona la verdura que vas a grabar:")
        for i, verdura in enumerate(opciones, start=1):
            print(f"{i}. {verdura}")

        while True:
            try:
                seleccion = int(input("Selecciona una opción (1-4): "))
                if 1 <= seleccion <= 4:
                    return opciones[seleccion - 1]
                else:
                    print("Por favor, selecciona un número entre 1 y 4.")
            except ValueError:
                print("Entrada no válida. Ingresa un número.")

    def grabar_y_clasificar_audio(self):
        """
        Graba un audio, lo procesa, lo clasifica, y decide si guardarlo en la base de datos.
        """
        verdura_seleccionada = self.seleccionar_verdura()
        print(f"\nGrabando un audio para la verdura seleccionada: {verdura_seleccionada}")

        # Grabar el audio
        self.grabadora.grabar_audio(duracion=3)

        # Procesar el audio grabado
        if not self.grabadora.procesar_audio_candidato():
            print("Error durante el procesamiento del audio candidato. Abortando.")
            return

        # Parametrizar y clasificar el audio
        if not self.grabadora.extraer_parametros_candidato():
            print("Error durante la parametrización del audio candidato. Abortando.")
            return

        etiqueta_predicha = self.clasificador.clasificar_candidato()
        print(f"\nEl audio candidato fue clasificado como: {etiqueta_predicha}")

        # Decidir si guardar el audio
        if etiqueta_predicha == verdura_seleccionada:
            print("¡Clasificación correcta! Guardando el audio en la base de datos cruda...")
            nombre_archivo = f"{verdura_seleccionada}_{len(os.listdir(self.ruta_crudos)) + 1}.wav"
            ruta_destino = os.path.join(self.ruta_crudos, nombre_archivo)
            shutil.copy(self.grabadora.archivo_candidato, ruta_destino)
            print(f"Audio guardado como: {ruta_destino}")
        else:
            print("Clasificación incorrecta. El audio no será guardado.")

    def menu_principal(self):
        """
        Muestra el menú principal y permite al usuario grabar y clasificar múltiples audios.
        """
        while True:
            print("\n===== MENÚ PRINCIPAL =====")
            print("1. Grabar y clasificar un nuevo audio")
            print("2. Salir")

            try:
                opcion = int(input("Selecciona una opción: "))
                if opcion == 1:
                    self.grabar_y_clasificar_audio()
                elif opcion == 2:
                    print("Saliendo del programa. ¡Hasta luego!")
                    break
                else:
                    print("Opción no válida. Intenta nuevamente.")
            except ValueError:
                print("Entrada no válida. Por favor, ingresa un número.")

if __name__ == "__main__":
    flujo = GrabadorClasificador()
    flujo.menu_principal()