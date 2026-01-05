import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# --- CONFIGURAÇÕES ---
PASTA_DATASET = 'meu_dataset_cnn'
BATCH_SIZE = 32
IMG_SIZE = (64, 64)
EPOCHS = 50 # Com 17 mil imagens, 15 épocas é suficiente

print(f"--- FASE 3: TREINANDO ARQUITETURA PRÓPRIA EM {tf.config.list_physical_devices('GPU')} ---")

# 1. Carregar Dataset (Treino e Validação)
print("Carregando imagens...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    PASTA_DATASET,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    PASTA_DATASET,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Descobre os nomes das classes (deve ser 'maduro' e 'verde')
class_names = train_ds.class_names
print(f"Classes encontradas: {class_names}")

# Cache para acelerar o treino
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 2. Data Augmentation (Faz a IA "imaginar" variações)
# Isso evita que ela decore os grãos e ajuda a generalizar
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
  layers.RandomZoom(0.1),
])

# 3. A SUA ARQUITETURA (CNN Customizada)
model = models.Sequential([
    # Camada de entrada + Augmentation
    layers.Input(shape=(64, 64, 3)),
    data_augmentation,
    layers.Rescaling(1./255), # Normaliza pixels (0-1)
    
    # Bloco 1
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    # Bloco 2
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    # Bloco 3
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    # Bloco 4 (Mais profundo para aprender textura)
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    # Classificador
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Ajuda a evitar overfitting
    layers.Dense(len(class_names)) # Saída
])

# 4. Compilar
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

# 5. Treinar
print("\nIniciando treinamento...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# 6. Gerar Gráfico para o TCC
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Acurácia Treino')
plt.plot(epochs_range, val_acc, label='Acurácia Validação')
plt.legend(loc='lower right')
plt.title('Performance de Acurácia')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Erro Treino')
plt.plot(epochs_range, val_loss, label='Erro Validação')
plt.legend(loc='upper right')
plt.title('Performance de Erro (Loss)')

plt.savefig('grafico_tcc.png')
print("\nGráfico salvo como 'grafico_tcc.png'")

# 7. Salvar Modelo
model.save('modelo_final_cafe.keras')
print("Modelo salvo como 'modelo_final_cafe.keras'")