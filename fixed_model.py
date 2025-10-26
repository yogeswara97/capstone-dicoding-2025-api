# train_vgg16.py
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Gunakan pre-trained VGG16 tanpa top classifier
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Bekukan layer awal agar tidak ikut di-train ulang
for layer in base_model.layers:
    layer.trainable = False

# Tambahkan lapisan klasifikasi
x = base_model.output  # Hasil tensor, bukan list
x = Flatten()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation="sigmoid")(x)

# Buat model dengan benar (tanpa [])
model = Model(inputs=base_model.input, outputs=predictions)

# Kompilasi model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Contoh: generator data dummy (ganti dengan path dataset kamu)
train_datagen = ImageDataGenerator(rescale=1.0/255)
train_generator = train_datagen.flow_from_directory(
    "dataset/train",  # ganti ke folder dataset kamu
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary"
)

# Latih model
model.fit(train_generator, epochs=3)

# Simpan model dengan format baru .keras
model.save("model_vgg16_finetuned.keras")

print("✅ Model berhasil disimpan tanpa list output!")
