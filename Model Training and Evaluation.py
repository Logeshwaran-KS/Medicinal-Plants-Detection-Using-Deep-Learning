import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Resizing, Rescaling

# Preprocessing Layer
resize_and_rescale = Sequential([
    Resizing(299, 299),
    Rescaling(1./255)
])

# Importing Xception Model
base_model = Xception(
    weights='imagenet',
    input_shape=(299, 299, 3),
    include_top=False,
    pooling='avg',
    classifier_activation='softmax',
    classes=80
)

# Constructing the Model
base_model.trainable = False

inputs = tf.keras.Input(shape=(299, 299, 3))
x = resize_and_rescale(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(80, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Layer Description of the Model
model.summary()

# Training the Model with 30 Epoch
history = model.fit(
    train_data,
    validation_data=val_data,
    batch_size=32,
    epochs=30
)

# Evaluating the Test Data
model.evaluate(test_data)

# Saving the Model
model.save("Trained_model.keras")

# Plotting Accuracy Graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()