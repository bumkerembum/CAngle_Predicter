import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model

tf.config.set_visible_devices([], 'GPU')


# Load initialliy tirained model and data sets
model = load_model('/home/kerem/Desktop/contakbizim/saved_initial_training/100epoch_64batch.h5')

X_train = np.load("/home/kerem/Desktop/contakbizim/data_matrix/X_train.npy")
y_train = np.load("/home/kerem/Desktop/contakbizim/data_matrix/y_train.npy")
X_val = np.load("/home/kerem/Desktop/contakbizim/data_matrix/X_val.npy")
y_val = np.load("/home/kerem/Desktop/contakbizim/data_matrix/y_val.npy")

# Set which layers to train
for layer in model.layers:
    layer.trainable = True
    
    
model.compile(optimizer=Adam(learning_rate=1e-5), 
              loss='mean_squared_error', 
              metrics=['mae'])

history_fine = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,  # Fine-tuning usually requires more epochs
    batch_size=32
)

model.save("/home/kerem/Desktop/contakbizim/unfreeze_training_two/100epoch_64batch/200epoch_32batch.h5")
    
