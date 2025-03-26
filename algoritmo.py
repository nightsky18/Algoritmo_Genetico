import numpy as np
import random
import cv2
import os

class AlgoritmoGenetico:
    def __init__(self, imagen_objetivo, tamano_poblacion=50, generaciones=300, tasa_mutacion=0.05, modo_color=True):
        if not os.path.exists(imagen_objetivo):
            raise FileNotFoundError(f"Error: No se encontró la imagen en la ruta: {imagen_objetivo}")

        if modo_color:
            imagen = cv2.imread(imagen_objetivo, cv2.IMREAD_COLOR)
        else:
            imagen = cv2.imread(imagen_objetivo, cv2.IMREAD_GRAYSCALE)

        if imagen is None:
            raise ValueError(f"Error: No se pudo cargar la imagen: {imagen_objetivo}")

        self.imagen_objetivo = cv2.resize(imagen, (100, 100)).astype(np.uint8)
        self.modo_color = modo_color
        self.tamano_poblacion = tamano_poblacion
        self.generaciones = generaciones
        self.tasa_mutacion = tasa_mutacion
        self.altura, self.ancho = self.imagen_objetivo.shape[:2]

        # Crear carpeta para guardar imágenes
        self.nombre_base = os.path.splitext(os.path.basename(imagen_objetivo))[0]  # Nombre del archivo sin extensión
        self.directorio_guardado = os.path.join("images", self.nombre_base)
        os.makedirs(self.directorio_guardado, exist_ok=True)

        self.poblacion = self.generar_poblacion()
        self.mejor_imagen = None
        self.mejor_fitness = -1
        self.iteraciones_sin_mejora = 0

    def generar_poblacion(self):
        """Genera la población inicial con pequeñas variaciones de la imagen original."""
        poblacion = []
        for _ in range(self.tamano_poblacion):
            img_variada = self.imagen_objetivo.astype(np.int16) + np.random.randint(-50, 50, self.imagen_objetivo.shape, dtype=np.int16)
            img_variada = np.clip(img_variada, 0, 255).astype(np.uint8)
            poblacion.append(img_variada)
        return poblacion

    def calcular_fitness(self, individuo):
        diferencia = np.mean(np.abs(self.imagen_objetivo.astype(np.int16) - individuo.astype(np.int16)))
        return 1 / (1 + diferencia)

    def seleccionar_padres(self):
        fitness_scores = [(self.calcular_fitness(ind), ind) for ind in self.poblacion]
        fitness_scores.sort(reverse=True, key=lambda x: x[0])
        return fitness_scores[:2]

    def cruzar(self, padre1, padre2):
        """Crea un hijo combinando píxeles de los padres con una máscara de cruce."""
        mascara = np.random.rand(self.altura, self.ancho, 1 if self.modo_color else 3) > 0.5
        hijo = np.where(mascara, padre1, padre2).astype(np.uint8)
        return hijo

    def mutar(self, individuo, generacion):
        """Introduce mutaciones asegurando que los valores estén dentro del rango uint8."""
        factor_mutacion = max(10, 100 - (generacion * 2))
        cantidad_mutaciones = min(factor_mutacion, self.altura * self.ancho // 10)

        individuo = individuo.astype(np.int16)

        for _ in range(cantidad_mutaciones):
            x, y = random.randint(0, self.altura - 1), random.randint(0, self.ancho - 1)
            if self.modo_color:
                individuo[x, y] += np.random.randint(-80, 80, size=(3,))
            else:
                individuo[x, y] += random.randint(-80, 80)

        individuo = np.clip(individuo, 0, 255).astype(np.uint8)

        # Aplicar mutaciones globales cada 20 generaciones
        if generacion % 20 == 0:
            ruido_global = np.random.randint(-40, 40, self.imagen_objetivo.shape, dtype=np.int16)
            individuo = np.clip(individuo + ruido_global, 0, 255).astype(np.uint8)

        return individuo

    def evolucionar_paso_a_paso(self, generacion):
        padres = self.seleccionar_padres()
        nueva_poblacion = [self.mutar(self.cruzar(padres[0][1], padres[1][1]), generacion) for _ in range(self.tamano_poblacion)]
        self.poblacion = nueva_poblacion
        mejor_actual = max(self.poblacion, key=self.calcular_fitness)
        fitness_actual = self.calcular_fitness(mejor_actual)

        if fitness_actual > self.mejor_fitness:
            self.mejor_fitness = fitness_actual
            self.mejor_imagen = mejor_actual
            self.iteraciones_sin_mejora = 0
        else:
            self.iteraciones_sin_mejora += 1

        # Guardar imágenes en generaciones clave dentro de su carpeta correspondiente
        if generacion in [1, 50, 100, 200, 300]:
            ruta_guardado = os.path.join(self.directorio_guardado, f"generacion_{generacion}.png")
            cv2.imwrite(ruta_guardado, mejor_actual)

        # Imprimir en consola
        diferencia = np.mean(np.abs(self.imagen_objetivo.astype(np.int16) - mejor_actual.astype(np.int16)))
        print(f"Generación {generacion + 1}/{self.generaciones} - Fitness: {fitness_actual:.6f} - Diferencia: {diferencia:.2f}")

        return self.poblacion, self.mejor_imagen

    def obtener_mejor_imagen(self):
        return self.mejor_imagen
