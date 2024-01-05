import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize pixel values to the range [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0

# Split the training data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)

# Flatten and standardize the data
scaler = StandardScaler()
train_images_flat = scaler.fit_transform(train_images.reshape(-1, 28*28))
val_images_flat = scaler.transform(val_images.reshape(-1, 28*28))
test_images_flat = scaler.transform(test_images.reshape(-1, 28*28))

# Define the deep MLP model
model = keras.Sequential([
    keras.layers.Input(shape=(28*28,)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Implement a learning rate schedule
initial_learning_rate = 1e-3
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=10000, decay_rate=0.9, staircase=True)

# Compile the model with the learning rate schedule and 'adam' optimizer
model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule), 
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early stopping with patience
early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

# ModelCheckpoint callback to save the best model
model_checkpoint = keras.callbacks.ModelCheckpoint("mnist_model.h5", save_best_only=True)

# Train the model
history = model.fit(
    train_images_flat, train_labels,
    epochs=100,
    validation_data=(val_images_flat, val_labels),
    callbacks=[early_stopping, model_checkpoint]
)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_images_flat, test_labels)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
