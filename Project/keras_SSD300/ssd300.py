from tensorflow import keras

def SSD300(input_shape, classes_num):
    
    # 初始化
    net = {}
    img_size = (input_shape[1], input_shape[0])
    input_tensor = keras.layers.Input(shape=input_shape)
    # Input Layer
    net['input'] = input_tensor
    # Block 1
    net['conv1_1'] = keras.layers.Conv2D(64, (3, 3), activation='relu')(net['input'])
    net['conv1_2'] = keras.layers.Conv2D(64, (3, 3), activation='relu')(net['conv1_1'])
    net['pool1'] = keras.layers.MaxPool2D(strides=(2, 2))(net['conv1_2'])
    # Block 2
    net['conv2_1'] = keras.layers.Conv2D(128, (3, 3), activation='relu')(net['pool1'])
    net['conv2_2'] = keras.layers.Conv2D(128, (3, 3), activation='relu')(net['conv2_1'])
    net['pool2'] = keras.layers.MaxPool2D(strides=(2, 2))(net['conv2_2'])
    # Block 3
    net['conv3_1'] = keras.layers.Conv2D(256, (3, 3), activation='relu')(net['pool2'])
    net['conv3_2'] = keras.layers.Conv2D(256, (3, 3), activation='relu')(net['conv3_1'])
    net['conv3_3'] = keras.layers.Conv2D(256, (3, 3), activation='relu')(net['conv3_2'])
    net['pool3'] = keras.layers.MaxPool2D(strides=(2, 2))(net['conv3_3'])
    # Block 4
    net['conv4_1'] = keras.layers.Conv2D(512, (3, 3), activation='relu')(net['pool3'])
    net['conv4_2'] = keras.layers.Conv2D(512, (3, 3), activation='relu')(net['conv4_1'])
    net['conv4_3'] = keras.layers.Conv2D(512, (3, 3), activation='relu')(net['conv4_2'])
    net['pool4'] = keras.layers.MaxPool2D(strides=(2, 2))(net['conv4_3'])
    # Block 5
    net['conv5_1'] = keras.layers.Conv2D(512, (3, 3), activation='relu')(net['pool4'])
    net['conv5_2'] = keras.layers.Conv2D(512, (3, 3), activation='relu')(net['conv5_1'])
    net['conv5_3'] = keras.layers.Conv2D(512, (3, 3), activation='relu')(net['conv5_2'])
    net['pool5'] = keras.layers.MaxPool2D((2,2), strides=(2, 2))(net['conv5_3'])
    # FC 6 带扩张率（dilation_rate）参数的为带洞卷积（AtrousConvolution）
    net['fc6'] = keras.layers.Conv2D(1024, (3, 3), dilation_rate = 2)(net['pool5'])
    # FC 7
    net['fc7'] = keras.layers.Conv2D(1024, (1, 1))(net['fc6'])
    # Block 6
    net['conv6_1'] = keras.layers.Conv2D(256, (1, 1), activation='relu')(net['fc7'])
    net['conv6_2'] = keras.layers.Conv2D(512, (3, 3), activation='relu')(net['conv6_1'])
    # Block 7
    net['conv7_1'] = keras.layers.Conv2D(128, (1, 1), activation='relu')(net['conv6_2'])
    net['conv7_2'] = keras.layers.ZeroPadding2D()(net['conv7_1'])
    net['conv7_2'] = keras.layers.Conv2D(256, (3, 3), activation='relu')(net['conv7_2'])
    # Block 8
    net['conv8_1'] = keras.layers.Conv2D(128, (1, 1), activation='relu')(net['conv7_2'])
    net['conv8_2'] = keras.layers.Conv2D(256, (3, 3), activation='relu')(net['conv8_1'])
    # Pool
    net['pool6'] = keras.layers.GlobalAvgPool2D()(net['conv8_2'])
    '''
    net = {}
    # Block 1
    input_tensor = input_tensor = keras.layers.Input(shape=input_shape)
    img_size = (input_shape[1], input_shape[0])
    net['input'] = input_tensor
    net['conv1_1'] = keras.layers.Convolution2D(64, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv1_1')(net['input'])
    net['conv1_2'] = keras.layers.Convolution2D(64, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv1_2')(net['conv1_1'])
    net['pool1'] = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), border_mode='same',
                                name='pool1')(net['conv1_2'])
    # Block 2
    net['conv2_1'] = keras.layers.Convolution2D(128, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv2_1')(net['pool1'])
    net['conv2_2'] = keras.layers.Convolution2D(128, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv2_2')(net['conv2_1'])
    net['pool2'] = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), border_mode='same',
                                name='pool2')(net['conv2_2'])
    # Block 3
    net['conv3_1'] = keras.layers.Convolution2D(256, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv3_1')(net['pool2'])
    net['conv3_2'] = keras.layers.Convolution2D(256, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv3_2')(net['conv3_1'])
    net['conv3_3'] = keras.layers.Convolution2D(256, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv3_3')(net['conv3_2'])
    net['pool3'] = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), border_mode='same',
                                name='pool3')(net['conv3_3'])
    # Block 4
    net['conv4_1'] = keras.layers.Convolution2D(512, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv4_1')(net['pool3'])
    net['conv4_2'] = keras.layers.Convolution2D(512, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv4_2')(net['conv4_1'])
    net['conv4_3'] = keras.layers.Convolution2D(512, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv4_3')(net['conv4_2'])
    net['pool4'] = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), border_mode='same',
                                name='pool4')(net['conv4_3'])
    # Block 5
    net['conv5_1'] = keras.layers.Convolution2D(512, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv5_1')(net['pool4'])
    net['conv5_2'] = keras.layers.Convolution2D(512, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv5_2')(net['conv5_1'])
    net['conv5_3'] = keras.layers.Convolution2D(512, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv5_3')(net['conv5_2'])
    net['pool5'] = keras.layers.MaxPooling2D((3, 3), strides=(1, 1), border_mode='same',
                                name='pool5')(net['conv5_3'])
    # FC6
    net['fc6'] = keras.layers.AtrousConvolution2D(1024, 3, 3, atrous_rate=(6, 6),
                                     activation='relu', border_mode='same',
                                     name='fc6')(net['pool5'])
    # x = Dropout(0.5, name='drop6')(x)
    # FC7
    net['fc7'] = keras.layers.Convolution2D(1024, 1, 1, activation='relu',
                               border_mode='same', name='fc7')(net['fc6'])
    # x = Dropout(0.5, name='drop7')(x)
    # Block 6
    net['conv6_1'] = keras.layers.Convolution2D(256, 1, 1, activation='relu',
                                   border_mode='same',
                                   name='conv6_1')(net['fc7'])
    net['conv6_2'] = keras.layers.Convolution2D(512, 3, 3, subsample=(2, 2),
                                   activation='relu', border_mode='same',
                                   name='conv6_2')(net['conv6_1'])
    # Block 7
    net['conv7_1'] = keras.layers.Convolution2D(128, 1, 1, activation='relu',
                                   border_mode='same',
                                   name='conv7_1')(net['conv6_2'])
    net['conv7_2'] = keras.layers.ZeroPadding2D()(net['conv7_1'])
    net['conv7_2'] = keras.layers.Convolution2D(256, 3, 3, subsample=(2, 2),
                                   activation='relu', border_mode='valid',
                                   name='conv7_2')(net['conv7_2'])
    # Block 8
    net['conv8_1'] = keras.layers.Convolution2D(128, 1, 1, activation='relu',
                                   border_mode='same',
                                   name='conv8_1')(net['conv7_2'])
    net['conv8_2'] = keras.layers.Convolution2D(256, 3, 3, subsample=(2, 2),
                                   activation='relu', border_mode='same',
                                   name='conv8_2')(net['conv8_1'])
    # Last Pool
    net['pool6'] = keras.layers.GlobalAveragePooling2D(name='pool6')(net['conv8_2'])
    '''
    return net
