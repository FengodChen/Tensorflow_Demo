from tensorflow import keras

def SSD300(input_shape, classes_num):
    model = keras.Sequential()
    # Input Layer
    model.add(keras.layers.Input(shape=input_shape))
    # Block 1
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D(strides=(2, 2)))
    # Block 2
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D(strides=(2, 2)))
    # Block 3
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D(strides=(2, 2)))
    # Block 4
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D(strides=(2, 2)))
    # Block 5
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D((2,2), strides=(2, 2)))
    # FC
    model.add(keras.layers.Conv2D(1024, (3, 3)))
    model.add(keras.layers.Conv2D(1024, (1, 1)))
    # Block 6
    model.add(keras.layers.Conv2D(256, (1, 1), activation='relu'))
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu'))
    # Block 7
    model.add(keras.layers.Conv2D(128, (1, 1), activation='relu'))
    model.add(keras.layers.ZeroPadding2D())
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu'))
    # Block 8
    model.add(keras.layers.Conv2D(128, (1, 1), activation='relu'))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu'))
    # Pool
    model.add(keras.layers.GlobalAvgPool2D())