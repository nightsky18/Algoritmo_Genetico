import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from algoritmo import AlgoritmoGenetico
import os

class InterfazGrafica:
    def __init__(self, root):
        self.root = root
        self.root.title("Optimización de Imágenes con Algoritmos Genéticos")

        # Contenedor de imágenes
        self.frame_imagenes = tk.Frame(root)
        self.frame_imagenes.pack()

        # Imagen original
        self.lbl_original = tk.Label(self.frame_imagenes, text="Imagen Original")
        self.lbl_original.grid(row=0, column=0)
        self.canvas_original = tk.Canvas(self.frame_imagenes, width=300, height=300)
        self.canvas_original.grid(row=1, column=0)

        # Imagen optimizada
        self.lbl_optimizada = tk.Label(self.frame_imagenes, text="Imagen Optimizada")
        self.lbl_optimizada.grid(row=0, column=1)
        self.canvas_optimizada = tk.Canvas(self.frame_imagenes, width=300, height=300)
        self.canvas_optimizada.grid(row=1, column=1)

        # Botón para seleccionar imagen
        self.btn_seleccionar = tk.Button(root, text="Seleccionar Imagen", command=self.seleccionar_imagen)
        self.btn_seleccionar.pack(pady=5)

        # Botón para iniciar el algoritmo
        self.btn_iniciar = tk.Button(root, text="Iniciar Optimización", command=self.iniciar_algoritmo, state=tk.DISABLED)
        self.btn_iniciar.pack(pady=5)

        # Barra de progreso
        self.progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(pady=5)

        # Etiqueta de generación
        self.lbl_iteracion = tk.Label(root, text="Generación: 0 - Fitness: 0.000000")
        self.lbl_iteracion.pack()

    def seleccionar_imagen(self):
        """Permite al usuario seleccionar la imagen a optimizar."""
        ruta_imagen = filedialog.askopenfilename(filetypes=[("Imágenes", "*.png;*.jpg;*.jpeg")])
        if ruta_imagen:
            self.ruta_imagen = os.path.abspath(ruta_imagen)
            self.btn_iniciar.config(state=tk.NORMAL)

            # Cargar y mostrar la imagen original
            imagen_pil = Image.open(self.ruta_imagen)
            imagen_pil = imagen_pil.resize((300, 300))  # Ajustamos tamaño
            self.imagen_tk_original = ImageTk.PhotoImage(imagen_pil)
            self.canvas_original.create_image(150, 150, image=self.imagen_tk_original)
            self.canvas_original.image = self.imagen_tk_original

    def iniciar_algoritmo(self):
        """Inicia el algoritmo genético con la imagen seleccionada por el usuario."""
        try:
            self.ag = AlgoritmoGenetico(self.ruta_imagen, tamano_poblacion=50, generaciones=300, tasa_mutacion=0.1)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        self.generacion_actual = 0
        self.evolucionar()

    def evolucionar(self):
        """Evoluciona la imagen y actualiza la interfaz en cada iteración."""
        if self.generacion_actual < self.ag.generaciones:
            _, mejor_imagen = self.ag.evolucionar_paso_a_paso(self.generacion_actual)

            # Mostrar el fitness en la interfaz
            fitness_actual = self.ag.calcular_fitness(mejor_imagen)
            self.lbl_iteracion.config(text=f"Generación: {self.generacion_actual + 1} - Fitness: {fitness_actual:.6f}")

            # Actualizamos la imagen optimizada
            imagen_pil = Image.fromarray(mejor_imagen)
            imagen_pil = imagen_pil.resize((300, 300))
            imagen_tk = ImageTk.PhotoImage(imagen_pil)
            self.canvas_optimizada.create_image(150, 150, image=imagen_tk)
            self.canvas_optimizada.image = imagen_tk

            # Barra de progreso
            self.progress["value"] = (self.generacion_actual / self.ag.generaciones) * 100
            self.root.update_idletasks()

            # Llamamos a la siguiente iteración después de 100ms para visualizar el proceso
            self.generacion_actual += 1
            self.root.after(100, self.evolucionar)

# Ejecutar la interfaz gráfica
if __name__ == "__main__":
    root = tk.Tk()
    app = InterfazGrafica(root)
    root.mainloop()
