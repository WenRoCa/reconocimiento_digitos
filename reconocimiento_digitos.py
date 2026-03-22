"""
Hecho por : Rocha Cantu Nidia Wendoly  Fecha: 22  de Marzo 2026
Clase: Inteligencia artificial y su ética - Tema 4.4 Redes neuronales y aprendizaje profundo - Actividad 22
MIA - Intituto Tecnológico de Nuevo Laredo - Prof. Carlos Arturo Guerrero Crespo
Titulo: Crea tu Primer Reconocedor de Dígitos
Descripción:
Clasifica dígitos escritos a mano usando el dataset MNIST.
"""

# =========================
# LIBRERÍAS
# =========================
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# =========================
# 1. CARGA DE DATOS
# =========================
def cargar_datos():
    """
    Carga y preprocesa el dataset MNIST.
    """
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalizar
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Aplanar
    X_train = X_train.reshape(-1, 784)
    X_test = X_test.reshape(-1, 784)

    return X_train, X_test, y_train, y_test


# =========================
# 2. CREAR MODELO
# =========================
def crear_modelo(capas, learning_rate=0.001):
    """
    Crea una red neuronal según la arquitectura.
    """
    modelo = models.Sequential()

    modelo.add(layers.Dense(capas[0], activation='relu', input_shape=(784,)))

    for neuronas in capas[1:]:
        modelo.add(layers.Dense(neuronas, activation='relu'))

    modelo.add(layers.Dense(10, activation='softmax'))

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return modelo


# =========================
# 3. EXPERIMENTACIÓN
# =========================
def experimentar(X_train, y_train, X_test, y_test):
    """
    Prueba diferentes arquitecturas.
    """
    arquitecturas = [
        {"capas": [64], "nombre": "Pequeña"},
        {"capas": [128, 64], "nombre": "Mediana"},
        {"capas": [256, 128, 64], "nombre": "Grande"}
    ]

    resultados = {}

    for arq in arquitecturas:
        print(f"\n Probando red {arq['nombre']}")

        modelo = crear_modelo(arq["capas"])

        modelo.fit(
            X_train, y_train,
            epochs=5,
            batch_size=128,
            validation_split=0.2,
            verbose=1
        )

        loss, acc = modelo.evaluate(X_test, y_test, verbose=0)
        print(f"Precisión: {acc:.4f}")

        resultados[arq["nombre"]] = (modelo, acc)

    return resultados


# =========================
# 4. ANÁLISIS DE ERRORES
# =========================
def analizar_errores(modelo, X_test, y_test):
    """
    Genera matriz de confusión, analiza errores y muestra ejemplos.
    """
    print("\nAnalizando errores del modelo...")

    y_pred = np.argmax(modelo.predict(X_test), axis=1)

    # Matriz de confusión
    matriz = confusion_matrix(y_test, y_pred)

    print("\nMatriz de confusión:")
    print(matriz)

    # =========================
    # CONFUSIONES MÁS COMUNES
    # =========================
    print("\nConfusiones más frecuentes:")

    for i in range(10):
        for j in range(10):
            if i != j and matriz[i][j] > 5:  # solo muestra errores relevantes
                print(f"Dígito {i} confundido con {j}: {matriz[i][j]} veces")

    # =========================
    # MOSTRAR EJEMPLOS VISUALES
    # =========================
    errores = np.where(y_pred != y_test)[0]

    print(f"\nTotal de errores: {len(errores)}")

    plt.figure(figsize=(10,5))

    for i, idx in enumerate(errores[:10]):
        plt.subplot(2,5,i+1)
        plt.imshow(X_test[idx].reshape(28,28), cmap='gray')
        plt.title(f"Real: {y_test[idx]} \nPred: {y_pred[idx]}")
        plt.axis('off')

    plt.suptitle("Ejemplos de errores del modelo")
    plt.show()

# =========================
# MAIN
# =========================
def main():
    X_train, X_test, y_train, y_test = cargar_datos()

    resultados = experimentar(X_train, y_train, X_test, y_test)

    # Mejor modelo (Grande)
    mejor_modelo = resultados["Grande"][0]

    analizar_errores(mejor_modelo, X_test, y_test)


if __name__ == "__main__":
    main()