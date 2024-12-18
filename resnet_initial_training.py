import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input

tf.config.set_visible_devices([], 'GPU')


# Load pre-trained ResNet50 without the top layer
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Add custom regression layers
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Pool features
x = Dense(512, activation='relu')(x)  # Fully connected layer
x = Dense(128, activation='relu')(x)  # Another dense layer
output = Dense(1, activation='linear')(x)  # Linear activation for regression

# Create the model
model = Model(inputs=base_model.input, outputs=output)

#

# Freeze all ResNet50 layers
for layer in base_model.layers:
    layer.trainable = False
    
# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

X_train = np.load("/home/kerem/Desktop/contakbizim/data_matrix/X_train.npy")
y_train = np.load("/home/kerem/Desktop/contakbizim/data_matrix/y_train.npy")
X_val = np.load("/home/kerem/Desktop/contakbizim/data_matrix/X_val.npy")
y_val = np.load("/home/kerem/Desktop/contakbizim/data_matrix/y_val.npy")


history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,  # Start with 10 epochs and adjust based on performance
    batch_size=256,
    #shuffle=True
)
model.save("/home/kerem/Desktop/contakbizim/saved_initial_training/100epoch_256batch.h5")
#model.fit()