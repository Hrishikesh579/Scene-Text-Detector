import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Reshape, LSTM, Dense, Bidirectional

def build_crnn_model(input_shape=(32, 128, 1), num_classes=36):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)

    x = Flatten()(x)

    # **Fix:** Compute `feature_dim` correctly
    timesteps = 32 
    feature_dim = tf.shape(x)[-1] // timesteps  # Ensure it's a valid division

    x = Reshape((timesteps, feature_dim))(x)  # ✅ Correct reshape

    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(128))(x)

    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    return model
