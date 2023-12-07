import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

# Define the shared base model
def build_shared_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    return model

# Define the binary classification head
def build_binary_classification_head(base_model_output):
    x = layers.Dense(64, activation='relu')(base_model_output)
    binary_classification_output = layers.Dense(1, activation='sigmoid', name='binary_classification_output')(x)
    return binary_classification_output

# Define the regression head
def build_regression_head(base_model_output):
    x = layers.Dense(64, activation='relu')(base_model_output)
    regression_output = layers.Dense(1, name='regression_output')(x)
    return regression_output

# Input shape for the image
input_shape = (128, 128, 3)

# Number of classes for binary classification
num_classes_binary = 1

# Shared base model
base_model = build_shared_model(input_shape)

# Binary classification head
binary_classification_output = build_binary_classification_head(base_model.output)

# Regression head
regression_output = build_regression_head(base_model.output)

# Create the multi-task model
model = models.Model(inputs=base_model.input, outputs=[binary_classification_output, regression_output])

# Compile the model
model.compile(optimizer=Adam(),
              loss={'binary_classification_output': 'binary_crossentropy', 'regression_output': 'rmse'},
              metrics={'binary_classification_output': 'accuracy', 'regression_output': 'mse'})

# Display the model summary
model.summary()

# Assuming you have X_train (input images), y_class (classification labels), and y_reg (regression labels)

# Convert classification labels to one-hot encoding
y_class_one_hot = tf.keras.utils.to_categorical(y_class, num_classes=num_classes)

# Train the model using the fit method
epochs = 10
batch_size = 32

history = model.fit(
    X_train,
    {'classification_output': y_class_one_hot, 'regression_output': y_reg},
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2  # Adjust the validation split as needed
)

# Save the trained model if needed
model.save('multi_task_model.h5')

import matplotlib.pyplot as plt

# Plot training and validation loss
plt.figure(figsize=(12, 6))

# Plot Classification Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['classification_output_loss'], label='classification_loss')
plt.plot(history.history['val_classification_output_loss'], label='val_classification_loss')
plt.title('Classification Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Regression Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['regression_output_loss'], label='regression_loss')
plt.plot(history.history['val_regression_output_loss'], label='val_regression_loss')
plt.title('Regression Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()