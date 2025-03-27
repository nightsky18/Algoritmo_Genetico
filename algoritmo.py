import numpy as np
import random
import cv2
import os

TAMANO_POBLACION = 50  
GENERACIONES = 500  

class AlgoritmoGenetico:
    def __init__(self, imagen_objetivo, tamano_poblacion=TAMANO_POBLACION, generaciones=GENERACIONES, tasa_mutacion=0.1, modo_color=True):
        if not os.path.exists(imagen_objetivo):
            raise FileNotFoundError(f"No se encontró la imagen en la ruta: {imagen_objetivo}")

        self.imagen_objetivo = cv2.imread(imagen_objetivo, cv2.IMREAD_COLOR if modo_color else cv2.IMREAD_GRAYSCALE)
        if self.imagen_objetivo is None:
            raise ValueError(f"No se pudo cargar la imagen: {imagen_objetivo}")

        self.imagen_objetivo = cv2.resize(self.imagen_objetivo, (100, 100)).astype(np.uint8)
        self.modo_color = modo_color
        self.tamano_poblacion = tamano_poblacion
        self.generaciones = generaciones
        self.tasa_mutacion = tasa_mutacion
        self.altura, self.ancho = self.imagen_objetivo.shape[:2]

        # Crear carpeta para guardar imágenes
        self.nombre_base = os.path.splitext(os.path.basename(imagen_objetivo))[0]
        self.directorio_guardado = os.path.join("images", self.nombre_base)
        os.makedirs(self.directorio_guardado, exist_ok=True)

        self.poblacion = self.generar_poblacion()
        self.mejor_imagen = None
        self.mejor_fitness = -1
        self.iteraciones_sin_mejora = 0

        # Guardar la imagen de la primera generación
        self.guardar_mejor_imagen(0, "mejor_generacion_0.png")

    def generar_poblacion(self):
        """Genera la población inicial con mayor variabilidad."""
        poblacion = []
        for _ in range(self.tamano_poblacion):
            ruido = np.random.randint(-100, 100, self.imagen_objetivo.shape, dtype=np.int16)
            img_variada = np.clip(self.imagen_objetivo.astype(np.int16) + ruido, 0, 255).astype(np.uint8)
            poblacion.append(img_variada)
        return poblacion

    def calcular_fitness(self, individuo):
        """Evalúa qué tan similar es la imagen al objetivo."""
        diferencia = np.mean(np.abs(self.imagen_objetivo.astype(np.int16) - individuo.astype(np.int16)))
        return np.exp(-diferencia / 1000)  # Ajuste para que la mejora sea más significativa

    def seleccionar_padres(self):
        """Método de selección por torneo."""
        seleccionados = random.sample(self.poblacion, k=10)
        seleccionados.sort(key=self.calcular_fitness, reverse=True)
        return seleccionados[0], seleccionados[1]

    def cruzar(self, padre1, padre2):
        """Crossover con mayor variabilidad."""
        mascara = np.random.rand(self.altura, self.ancho, 3 if self.modo_color else 1) > 0.5
        hijo = np.where(mascara, padre1, padre2).astype(np.uint8)
        return hijo

    def mutar(self, individuo, generacion):
        """Mutación adaptativa con conversión de tipo segura."""
        individuo = individuo.astype(np.int16)
        
        factor_mutacion = 50 if self.iteraciones_sin_mejora > 20 else 20  
        num_pixeles = 50  

        x, y = np.random.randint(0, self.altura, size=num_pixeles), np.random.randint(0, self.ancho, size=num_pixeles)
        if self.modo_color:
            individuo[x, y] += np.random.randint(-factor_mutacion, factor_mutacion, size=(num_pixeles, 3))
        else:
            individuo[x, y] += np.random.randint(-factor_mutacion, factor_mutacion, size=num_pixeles)

        return np.clip(individuo, 0, 255).astype(np.uint8)

    def evolucionar_paso_a_paso(self, generacion):
        """Ejecuta una iteración del algoritmo y aplica elitismo."""
        nueva_poblacion = []

        for _ in range(self.tamano_poblacion - 1):
            padre1, padre2 = self.seleccionar_padres()
            hijo = self.cruzar(padre1, padre2)
            hijo = self.mutar(hijo, generacion)
            nueva_poblacion.append(hijo)

        # Elitismo
        mejor_actual = max(self.poblacion, key=self.calcular_fitness)
        nueva_poblacion.append(mejor_actual)
        self.poblacion = nueva_poblacion

        # Evaluar fitness
        mejor_actual = max(self.poblacion, key=self.calcular_fitness)
        fitness_actual = self.calcular_fitness(mejor_actual)

        if fitness_actual > self.mejor_fitness:
            self.mejor_fitness = fitness_actual
            self.mejor_imagen = mejor_actual
            self.iteraciones_sin_mejora = 0
        else:
            self.iteraciones_sin_mejora += 1

        print(f"Generación {generacion + 1}/{self.generaciones} - Fitness: {fitness_actual:.6f}")

        # Guardar la mejor imagen en momentos clave
        if generacion == 0:
            self.guardar_mejor_imagen(generacion, "mejor_generacion_0.png")
        elif generacion == self.generaciones // 2:
            self.guardar_mejor_imagen(generacion, "mejor_generacion_mitad.png")
        elif generacion == self.generaciones - 1:
            self.guardar_mejor_imagen(generacion, "mejor_generacion_final.png")

        return self.poblacion, self.mejor_imagen

    def guardar_mejor_imagen(self, generacion, nombre_archivo):
        """Guarda la mejor imagen de una generación específica."""
        if self.mejor_imagen is not None:
            ruta_guardado = os.path.join(self.directorio_guardado, nombre_archivo)
            cv2.imwrite(ruta_guardado, self.mejor_imagen)
            print(f"Imagen guardada: {ruta_guardado}")


if __name__ == "__main__":
    imagen_objetivo = "imagen_referencia.png"  # Cambia esto con el nombre correcto de tu imagen
    algoritmo = AlgoritmoGenetico(imagen_objetivo, generaciones=500)

    for generacion in range(500):
        algoritmo.evolucionar_paso_a_paso(generacion)

    print("Evolución completa. Las imágenes clave han sido guardadas.")
