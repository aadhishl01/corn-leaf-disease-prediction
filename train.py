import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

img_size = 224
batch_size = 32

train_gen = ImageDataGenerator(rescale=1/255)
val_gen = ImageDataGenerator(rescale=1/255)

train_data = train_gen.flow_from_directory(
    "C:/Users/HP/OneDrive/Desktop/ml project/dataset/train",
    target_size=(img_size,img_size),
    batch_size=batch_size,
    class_mode="categorical"
)

val_data = val_gen.flow_from_directory(
    "C:/Users/HP/OneDrive/Desktop/ml project/dataset/val",
    target_size=(img_size,img_size),
    batch_size=batch_size,
    class_mode="categorical"
)

model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)),
    MaxPooling2D(),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128,activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes,activation='softmax')
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_data, validation_data=val_data, epochs=10)

model.save("model/cnn_model.h5")
print("Model saved")
