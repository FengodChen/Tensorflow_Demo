from tensorflow import keras
import ssd_layer

def SSD300(input_shape, classes_num):
    
    # 初始化
    net = {}
    img_size = (input_shape[1], input_shape[0])
    input_tensor = keras.layers.Input(shape=input_shape)
    # Input Layer
    net['input'] = input_tensor
    # Block 1
    net['conv1_1'] = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(net['input'])
    net['conv1_2'] = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(net['conv1_1'])
    net['pool1'] = keras.layers.MaxPool2D(strides=(2, 2))(net['conv1_2'])
    # Block 2
    net['conv2_1'] = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(net['pool1'])
    net['conv2_2'] = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(net['conv2_1'])
    net['pool2'] = keras.layers.MaxPool2D(strides=(2, 2))(net['conv2_2'])
    # Block 3
    net['conv3_1'] = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(net['pool2'])
    net['conv3_2'] = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(net['conv3_1'])
    net['conv3_3'] = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(net['conv3_2'])
    net['pool3'] = keras.layers.MaxPool2D(strides=(2, 2))(net['conv3_3'])
    # Block 4
    net['conv4_1'] = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(net['pool3'])
    net['conv4_2'] = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(net['conv4_1'])
    net['conv4_3'] = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(net['conv4_2'])
    net['pool4'] = keras.layers.MaxPool2D(strides=(2, 2))(net['conv4_3'])
    # Block 5
    net['conv5_1'] = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(net['pool4'])
    net['conv5_2'] = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(net['conv5_1'])
    net['conv5_3'] = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(net['conv5_2'])
    net['pool5'] = keras.layers.MaxPool2D((2,2), strides=(2, 2))(net['conv5_3'])
    # FC 6 带扩张率（dilation_rate）参数的为带洞卷积（AtrousConvolution）
    net['fc6'] = keras.layers.Conv2D(1024, (3, 3), dilation_rate = 2, padding='same')(net['pool5'])
    # FC 7
    net['fc7'] = keras.layers.Conv2D(1024, (1, 1), padding='same')(net['fc6'])
    # Block 6
    net['conv6_1'] = keras.layers.Conv2D(256, (1, 1), activation='relu', padding='same')(net['fc7'])
    net['conv6_2'] = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(net['conv6_1'])
    # Block 7
    net['conv7_1'] = keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same')(net['conv6_2'])
    net['conv7_2'] = keras.layers.ZeroPadding2D()(net['conv7_1'])
    net['conv7_2'] = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(net['conv7_2'])
    # Block 8
    net['conv8_1'] = keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same')(net['conv7_2'])
    net['conv8_2'] = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(net['conv8_1'])
    # Pool
    net['pool6'] = keras.layers.GlobalAvgPool2D()(net['conv8_2'])
    # Prediction from conv4_3
    return net
    #model = keras.Model(inputs = net['input'], outputs = net['pool6'])
