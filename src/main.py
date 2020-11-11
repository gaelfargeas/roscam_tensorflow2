import os, random, numpy, sys

# disable tensorblow os debug log
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow

# make use we use tensorflow 2
assert hasattr(tensorflow, "function")


if __name__ == "__main__":

    tensorflow.keras.backend.set_floatx("float64")
    my_model = tensorflow.keras.models.Sequential()

    my_model.add(tensorflow.keras.layers.Dense(512, activation="relu"))
    my_model.add(tensorflow.keras.layers.Dense(256, activation="relu"))
    my_model.add(tensorflow.keras.layers.Dense(128, activation="relu"))
    my_model.add(tensorflow.keras.layers.Dense(10, activation="softmax"))
