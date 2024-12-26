import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.models import load_model

tf.config.set_visible_devices([], 'GPU') #Disable thiss option activate GPU computing


# Load initialliy tirained model and data sets
model = load_model("pre_trained model's location")

X_train = np.load("x_training_location")
y_train = np.load("y_training_location")
X_val = np.load("x_validation_locaiton")
y_val = np.load("y_validation_location")

# Set which layers to train
for layer in model.layers:
    layer.trainable = True
    
    
model.compile(optimizer=Adam(learning_rate=1e-5), 
              loss='mean_squared_error', 
              metrics=['mae'])

history_fine = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,   # Adjust based on your preferance
    batch_size=32 # Adjust based on your preferance
)

model.save("location/model.h5")
    
