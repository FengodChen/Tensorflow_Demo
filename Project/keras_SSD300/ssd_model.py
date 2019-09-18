import tensorflow as tf
from tensorflow import keras
import ssd_layer

def SSD300_vgg16(input_shape, classes_num):
    
    # Initialization
    net = {}
    img_size = (input_shape[1], input_shape[0])
    input_tensor = keras.layers.Input(shape=input_shape)
    vgg16_model = keras.applications.VGG16(include_top = False, input_shape = input_shape)
    # Input Layer
    #net['input'] = vgg16_model.get_layer(index = 0)
    net['input'] = input_tensor
    # Block 1
    net['conv1_1'] = vgg16_model.get_layer(index = 1)(net['input'])
    net['conv1_2'] = vgg16_model.get_layer(index = 2)(net['conv1_1'])
    net['pool1'] = vgg16_model.get_layer(index = 3)(net['conv1_2'])
    # Block 2
    net['conv2_1'] = vgg16_model.get_layer(index = 4)(net['pool1'])
    net['conv2_2'] = vgg16_model.get_layer(index = 5)(net['conv2_1'])
    net['pool2'] = vgg16_model.get_layer(index = 6)(net['conv2_2'])
    # Block 3
    net['conv3_1'] = vgg16_model.get_layer(index = 7)(net['pool2'])
    net['conv3_2'] = vgg16_model.get_layer(index = 8)(net['conv3_1'])
    net['conv3_3'] = vgg16_model.get_layer(index = 9)(net['conv3_2'])
    net['pool3'] = vgg16_model.get_layer(index = 10)(net['conv3_3'])
    # Block 4
    net['conv4_1'] = vgg16_model.get_layer(index = 11)(net['pool3'])
    net['conv4_2'] = vgg16_model.get_layer(index = 12)(net['conv4_1'])
    net['conv4_3'] = vgg16_model.get_layer(index = 13)(net['conv4_2'])
    net['pool4'] = vgg16_model.get_layer(index = 14)(net['conv4_3'])
    # Block 5
    net['conv5_1'] = vgg16_model.get_layer(index = 15)(net['pool4'])
    net['conv5_2'] = vgg16_model.get_layer(index = 16)(net['conv5_1'])
    net['conv5_3'] = vgg16_model.get_layer(index = 17)(net['conv5_2'])
    net['pool5'] = vgg16_model.get_layer(index = 18)(net['conv5_3'])
    # FC 6 带扩张率（dilation_rate）参数的为带洞卷积（AtrousConvolution）
    net['fc6'] = keras.layers.Conv2D(1024, (3, 3), dilation_rate = 2, padding='same', name='fc6')(net['pool5'])
    # FC 7
    net['fc7'] = keras.layers.Conv2D(1024, (1, 1), padding='same', name='fc7')(net['fc6'])
    # Block 6
    net['conv6_1'] = keras.layers.Conv2D(256, (1, 1), activation='relu', padding='same', name='conv6_1')(net['fc7'])
    net['conv6_2'] = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv6_2')(net['conv6_1'])
    # Block 7
    net['conv7_1'] = keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same', name='conv7_1')(net['conv6_2'])
    net['conv7_2'] = keras.layers.ZeroPadding2D()(net['conv7_1'])
    net['conv7_2'] = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv7_2')(net['conv7_2'])
    # Block 8
    net['conv8_1'] = keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same', name='conv8_1')(net['conv7_2'])
    net['conv8_2'] = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv8_2')(net['conv8_1'])
    # Last Pool
    net['pool6'] = keras.layers.GlobalAvgPool2D(name='pool6')(net['conv8_2'])

    # Prediction from conv4_3
    priors_num = 3
    name = 'conv4_3_norm_mbox_conf' + '_{}'.format(classes_num)
    # TODO
    net['conv4_3_norm'] = keras.layers.LayerNormalization(name='conv4_3_norm')(net['conv4_3'])
    net['conv4_3_norm_mbox_loc'] = keras.layers.Conv2D(priors_num*4, (3, 3), padding='same', name='conv4_3_norm_mbox_loc')(net['conv4_3_norm'])
    net['conv4_3_norm_mbox_loc_flat'] = keras.layers.Flatten(name='conv4_3_norm_mbox_loc_flat')(net['conv4_3_norm_mbox_loc'])
    net['conv4_3_norm_mbox_conf'] = keras.layers.Conv2D(priors_num*classes_num, (3, 3), padding='same',name=name)(net['conv4_3_norm'])
    net['conv4_3_norm_mbox_conf_flat'] = keras.layers.Flatten(name='conv4_3_norm_mbox_conf_flat')(net['conv4_3_norm_mbox_conf'])
    net['conv4_3_norm_mbox_priorbox'] = ssd_layer.PriorBox(img_size, 30.0, aspect_ratios=[2], variances=[0.1, 0.1, 0.2, 0.2], name='conv4_3_norm_mbox_priorbox')(net['conv4_3_norm'])

    # Prediction from fc7
    priors_num = 6
    name = 'fc7_mbox_conf' + '_{}'.format(classes_num)
    net['fc7_mbox_loc'] = keras.layers.Conv2D(priors_num * 4, (3, 3), padding='same', name='fc7_mbox_loc')(net['fc7'])
    net['fc7_mbox_loc_flat'] = keras.layers.Flatten(name='fc7_mbox_loc_flat')(net['fc7_mbox_loc'])
    net['fc7_mbox_conf'] = keras.layers.Conv2D(priors_num * classes_num, (3, 3), padding='same', name=name)(net['fc7'])
    net['fc7_mbox_conf_flat'] = keras.layers.Flatten(name='fc7_mbox_conf_flat')(net['fc7_mbox_conf'])
    net['fc7_mbox_priorbox'] = ssd_layer.PriorBox(img_size, 60.0, max_size=114.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='fc7_mbox_priorbox')(net['fc7'])

    # Prediction from conv6_2
    priors_num = 6
    name = 'conv6_2_mbox_conf' + '_{}'.format(classes_num)
    net['conv6_2_mbox_loc'] = keras.layers.Conv2D(priors_num * 4, (3, 3), padding='same', name='conv6_2_mbox_loc')(net['conv6_2'])
    net['conv6_2_mbox_loc_flat'] = keras.layers.Flatten(name='conv6_2_mbox_loc_flat')(net['conv6_2_mbox_loc'])
    net['conv6_2_mbox_conf'] = keras.layers.Conv2D(priors_num * classes_num , (3, 3), padding='same', name=name)(net['conv6_2'])
    net['conv6_2_mbox_conf_flat'] = keras.layers.Flatten(name='conv6_2_mbox_conf_flat')(net['conv6_2_mbox_conf'])
    net['conv6_2_mbox_priorbox'] = ssd_layer.PriorBox(img_size, 114.0, max_size=168.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='conv6_2_mbox_priorbox')(net['conv6_2'])

    # Prediction from conv7_2
    priors_num = 6
    name = 'conv7_2_mbox_conf' + '_{}'.format(classes_num)
    net['conv7_2_mbox_loc'] = keras.layers.Conv2D(priors_num * 4, (3, 3), padding='same', name='conv7_2_mbox_loc')(net['conv7_2'])
    net['conv7_2_mbox_loc_flat'] = keras.layers.Flatten(name='conv7_2_mbox_loc_flat')(net['conv7_2_mbox_loc'])
    net['conv7_2_mbox_conf'] = keras.layers.Conv2D(priors_num * classes_num, (3, 3), padding='same', name=name)(net['conv7_2'])
    net['conv7_2_mbox_conf_flat'] = keras.layers.Flatten(name='conv7_2_mbox_conf_flat')(net['conv7_2_mbox_conf'])
    net['conv7_2_mbox_priorbox'] = ssd_layer.PriorBox(img_size, 168.0, max_size=222.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='conv7_2_mbox_priorbox')(net['conv7_2'])

    # Prediction from conv8_2
    priors_num = 6
    name = 'conv8_2_mbox_conf' + '_{}'.format(classes_num)
    net['conv8_2_mbox_loc'] = keras.layers.Conv2D(priors_num * 4, (3, 3), padding='same', name='conv8_2_mbox_loc')(net['conv8_2'])
    net['conv8_2_mbox_loc_flat'] = keras.layers.Flatten(name='conv8_2_mbox_loc_flat')(net['conv8_2_mbox_loc'])
    net['conv8_2_mbox_conf'] = keras.layers.Conv2D(priors_num * classes_num, (3, 3), padding='same', name=name)(net['conv8_2'])
    net['conv8_2_mbox_conf_flat'] = keras.layers.Flatten(name='conv8_2_mbox_conf_flat')(net['conv8_2_mbox_conf'])
    net['conv8_2_mbox_priorbox'] = ssd_layer.PriorBox(img_size, 222.0, max_size=276.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='conv8_2_mbox_priorbox')(net['conv8_2'])

    # Prediction from pool6
    priors_num = 6
    name = 'pool6_mbox_conf_flat' + '_{}'.format(classes_num)
    net['pool6_mbox_loc_flat'] = keras.layers.Dense(priors_num * 4, name='pool6_mbox_loc_flat')(net['pool6'])
    net['pool6_mbox_conf_flat'] = keras.layers.Dense(priors_num * classes_num, name=name)(net['pool6'])
    net['pool6_reshaped'] = keras.layers.Reshape((1, 1, 256), name='pool6_reshaped')(net['pool6'])
    net['pool6_mbox_priorbox'] = ssd_layer.PriorBox(img_size, 276.0, max_size=330.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='pool6_mbox_priorbox')(net['pool6_reshaped'])

    # Gather all predictions
    net['mbox_loc'] = keras.layers.concatenate([net['conv4_3_norm_mbox_loc_flat'],
                             net['fc7_mbox_loc_flat'],
                             net['conv6_2_mbox_loc_flat'],
                             net['conv7_2_mbox_loc_flat'],
                             net['conv8_2_mbox_loc_flat'],
                             net['pool6_mbox_loc_flat']],
                            axis=1, name='mbox_loc')
    net['mbox_conf'] = keras.layers.concatenate([net['conv4_3_norm_mbox_conf_flat'],
                              net['fc7_mbox_conf_flat'],
                              net['conv6_2_mbox_conf_flat'],
                              net['conv7_2_mbox_conf_flat'],
                              net['conv8_2_mbox_conf_flat'],
                              net['pool6_mbox_conf_flat']],
                             axis=1, name='mbox_conf')
    net['mbox_priorbox'] = keras.layers.concatenate([net['conv4_3_norm_mbox_priorbox'],
                                  net['fc7_mbox_priorbox'],
                                  net['conv6_2_mbox_priorbox'],
                                  net['conv7_2_mbox_priorbox'],
                                  net['conv8_2_mbox_priorbox'],
                                  net['pool6_mbox_priorbox']],
                                 axis=1, name='mbox_priorbox')
    num_boxes = keras.backend.int_shape(net['mbox_loc'])[-1] // 4

    net['mbox_loc'] = keras.layers.Reshape((num_boxes, 4),
                              name='mbox_loc_final')(net['mbox_loc'])
    net['mbox_conf'] = keras.layers.Reshape((num_boxes, classes_num),
                               name='mbox_conf_logits')(net['mbox_conf'])
    net['mbox_conf'] = keras.layers.Activation('softmax',
                                  name='mbox_conf_final')(net['mbox_conf'])
    net['predictions'] = keras.layers.concatenate([net['mbox_loc'],
                               net['mbox_conf'],
                               net['mbox_priorbox']],
                               axis=2, name='predictions')
    model = keras.models.Model(net['input'], net['predictions'])
    # <TODO id=190820000>
    # <Debug>
    # Freezen VGG16 weight
    '''
    for vgg in model.layers[:19]:
    #for vgg in model.layers[:11]:
        vgg.trainable = False
    '''
    # </Debug>
    # </TODO>
    return model


def SSD300_manual(input_shape, classes_num):
    
    # Initialization
    net = {}
    img_size = (input_shape[1], input_shape[0])
    input_tensor = keras.layers.Input(shape=input_shape)
    vgg16_model = keras.applications.VGG16(include_top = False, input_shape = input_shape)
    # Input Layer
    #net['input'] = vgg16_model.get_layer(index = 0)
    net['input'] = input_tensor
    # Block 1
    net['conv1_1'] = vgg16_model.get_layer(index = 1)(net['input'])
    net['conv1_2'] = vgg16_model.get_layer(index = 2)(net['conv1_1'])
    net['pool1'] = vgg16_model.get_layer(index = 3)(net['conv1_2'])
    # Block 2
    net['conv2_1'] = vgg16_model.get_layer(index = 4)(net['pool1'])
    net['conv2_2'] = vgg16_model.get_layer(index = 5)(net['conv2_1'])
    net['pool2'] = vgg16_model.get_layer(index = 6)(net['conv2_2'])
    # Block 3
    net['conv3_1'] = vgg16_model.get_layer(index = 7)(net['pool2'])
    net['conv3_2'] = vgg16_model.get_layer(index = 8)(net['conv3_1'])
    net['conv3_3'] = vgg16_model.get_layer(index = 9)(net['conv3_2'])
    net['pool3'] = vgg16_model.get_layer(index = 10)(net['conv3_3'])
    # Block 4
    net['conv4_1'] = vgg16_model.get_layer(index = 11)(net['pool3'])
    net['conv4_2'] = vgg16_model.get_layer(index = 12)(net['conv4_1'])
    net['conv4_3'] = vgg16_model.get_layer(index = 13)(net['conv4_2'])
    net['pool4'] = vgg16_model.get_layer(index = 14)(net['conv4_3'])
    # Block 5
    net['conv5_1'] = vgg16_model.get_layer(index = 15)(net['pool4'])
    net['conv5_2'] = vgg16_model.get_layer(index = 16)(net['conv5_1'])
    net['conv5_3'] = vgg16_model.get_layer(index = 17)(net['conv5_2'])
    net['pool5'] = vgg16_model.get_layer(index = 18)(net['conv5_3'])
    # FC 6 带扩张率（dilation_rate）参数的为带洞卷积（AtrousConvolution）
    net['fc6'] = keras.layers.Conv2D(1024, (3, 3), dilation_rate = 2, padding='same', name='fc6')(net['pool5'])
    # FC 7
    net['fc7'] = keras.layers.Conv2D(1024, (1, 1), padding='same', name='fc7')(net['fc6'])
    # Block 6
    net['conv6_1'] = keras.layers.Conv2D(256, (1, 1), activation='relu', padding='same', name='conv6_1')(net['fc7'])
    net['conv6_2'] = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv6_2')(net['conv6_1'])
    # Block 7
    net['conv7_1'] = keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same', name='conv7_1')(net['conv6_2'])
    net['conv7_2'] = keras.layers.ZeroPadding2D()(net['conv7_1'])
    net['conv7_2'] = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv7_2')(net['conv7_2'])
    # Block 8
    net['conv8_1'] = keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same', name='conv8_1')(net['conv7_2'])
    net['conv8_2'] = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv8_2')(net['conv8_1'])
    # Last Pool
    net['pool6'] = keras.layers.GlobalAvgPool2D(name='pool6')(net['conv8_2'])

    # Prediction from conv4_3
    priors_num = 3
    name = 'conv4_3_norm_mbox_conf' + '_{}'.format(classes_num)
    # TODO
    net['conv4_3_norm'] = keras.layers.LayerNormalization(name='conv4_3_norm')(net['conv4_3'])
    net['conv4_3_norm_mbox_loc'] = keras.layers.Conv2D(priors_num*4, (3, 3), padding='same', name='conv4_3_norm_mbox_loc')(net['conv4_3_norm'])
    net['conv4_3_norm_mbox_loc_flat'] = keras.layers.Flatten(name='conv4_3_norm_mbox_loc_flat')(net['conv4_3_norm_mbox_loc'])
    net['conv4_3_norm_mbox_conf'] = keras.layers.Conv2D(priors_num*classes_num, (3, 3), padding='same',name=name)(net['conv4_3_norm'])
    net['conv4_3_norm_mbox_conf_flat'] = keras.layers.Flatten(name='conv4_3_norm_mbox_conf_flat')(net['conv4_3_norm_mbox_conf'])
    net['conv4_3_norm_mbox_priorbox'] = ssd_layer.PriorBox(img_size, 30.0, aspect_ratios=[2], variances=[0.1, 0.1, 0.2, 0.2], name='conv4_3_norm_mbox_priorbox')(net['conv4_3_norm'])

    # Prediction from fc7
    priors_num = 6
    name = 'fc7_mbox_conf' + '_{}'.format(classes_num)
    net['fc7_mbox_loc'] = keras.layers.Conv2D(priors_num * 4, (3, 3), padding='same', name='fc7_mbox_loc')(net['fc7'])
    net['fc7_mbox_loc_flat'] = keras.layers.Flatten(name='fc7_mbox_loc_flat')(net['fc7_mbox_loc'])
    net['fc7_mbox_conf'] = keras.layers.Conv2D(priors_num * classes_num, (3, 3), padding='same', name=name)(net['fc7'])
    net['fc7_mbox_conf_flat'] = keras.layers.Flatten(name='fc7_mbox_conf_flat')(net['fc7_mbox_conf'])
    net['fc7_mbox_priorbox'] = ssd_layer.PriorBox(img_size, 60.0, max_size=114.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='fc7_mbox_priorbox')(net['fc7'])

    # Prediction from conv6_2
    priors_num = 6
    name = 'conv6_2_mbox_conf' + '_{}'.format(classes_num)
    net['conv6_2_mbox_loc'] = keras.layers.Conv2D(priors_num * 4, (3, 3), padding='same', name='conv6_2_mbox_loc')(net['conv6_2'])
    net['conv6_2_mbox_loc_flat'] = keras.layers.Flatten(name='conv6_2_mbox_loc_flat')(net['conv6_2_mbox_loc'])
    net['conv6_2_mbox_conf'] = keras.layers.Conv2D(priors_num * classes_num , (3, 3), padding='same', name=name)(net['conv6_2'])
    net['conv6_2_mbox_conf_flat'] = keras.layers.Flatten(name='conv6_2_mbox_conf_flat')(net['conv6_2_mbox_conf'])
    net['conv6_2_mbox_priorbox'] = ssd_layer.PriorBox(img_size, 114.0, max_size=168.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='conv6_2_mbox_priorbox')(net['conv6_2'])

    # Prediction from conv7_2
    priors_num = 6
    name = 'conv7_2_mbox_conf' + '_{}'.format(classes_num)
    net['conv7_2_mbox_loc'] = keras.layers.Conv2D(priors_num * 4, (3, 3), padding='same', name='conv7_2_mbox_loc')(net['conv7_2'])
    net['conv7_2_mbox_loc_flat'] = keras.layers.Flatten(name='conv7_2_mbox_loc_flat')(net['conv7_2_mbox_loc'])
    net['conv7_2_mbox_conf'] = keras.layers.Conv2D(priors_num * classes_num, (3, 3), padding='same', name=name)(net['conv7_2'])
    net['conv7_2_mbox_conf_flat'] = keras.layers.Flatten(name='conv7_2_mbox_conf_flat')(net['conv7_2_mbox_conf'])
    net['conv7_2_mbox_priorbox'] = ssd_layer.PriorBox(img_size, 168.0, max_size=222.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='conv7_2_mbox_priorbox')(net['conv7_2'])

    # Prediction from conv8_2
    priors_num = 6
    name = 'conv8_2_mbox_conf' + '_{}'.format(classes_num)
    net['conv8_2_mbox_loc'] = keras.layers.Conv2D(priors_num * 4, (3, 3), padding='same', name='conv8_2_mbox_loc')(net['conv8_2'])
    net['conv8_2_mbox_loc_flat'] = keras.layers.Flatten(name='conv8_2_mbox_loc_flat')(net['conv8_2_mbox_loc'])
    net['conv8_2_mbox_conf'] = keras.layers.Conv2D(priors_num * classes_num, (3, 3), padding='same', name=name)(net['conv8_2'])
    net['conv8_2_mbox_conf_flat'] = keras.layers.Flatten(name='conv8_2_mbox_conf_flat')(net['conv8_2_mbox_conf'])
    net['conv8_2_mbox_priorbox'] = ssd_layer.PriorBox(img_size, 222.0, max_size=276.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='conv8_2_mbox_priorbox')(net['conv8_2'])

    # Prediction from pool6
    priors_num = 6
    name = 'pool6_mbox_conf_flat' + '_{}'.format(classes_num)
    net['pool6_mbox_loc_flat'] = keras.layers.Dense(priors_num * 4, name='pool6_mbox_loc_flat')(net['pool6'])
    net['pool6_mbox_conf_flat'] = keras.layers.Dense(priors_num * classes_num, name=name)(net['pool6'])
    net['pool6_reshaped'] = keras.layers.Reshape((1, 1, 256), name='pool6_reshaped')(net['pool6'])
    net['pool6_mbox_priorbox'] = ssd_layer.PriorBox(img_size, 276.0, max_size=330.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='pool6_mbox_priorbox')(net['pool6_reshaped'])

    # Gather all predictions
    net['mbox_loc'] = keras.layers.concatenate([net['conv4_3_norm_mbox_loc_flat'],
                             net['fc7_mbox_loc_flat'],
                             net['conv6_2_mbox_loc_flat'],
                             net['conv7_2_mbox_loc_flat'],
                             net['conv8_2_mbox_loc_flat'],
                             net['pool6_mbox_loc_flat']],
                            axis=1, name='mbox_loc')
    net['mbox_conf'] = keras.layers.concatenate([net['conv4_3_norm_mbox_conf_flat'],
                              net['fc7_mbox_conf_flat'],
                              net['conv6_2_mbox_conf_flat'],
                              net['conv7_2_mbox_conf_flat'],
                              net['conv8_2_mbox_conf_flat'],
                              net['pool6_mbox_conf_flat']],
                             axis=1, name='mbox_conf')
    net['mbox_priorbox'] = keras.layers.concatenate([net['conv4_3_norm_mbox_priorbox'],
                                  net['fc7_mbox_priorbox'],
                                  net['conv6_2_mbox_priorbox'],
                                  net['conv7_2_mbox_priorbox'],
                                  net['conv8_2_mbox_priorbox'],
                                  net['pool6_mbox_priorbox']],
                                 axis=1, name='mbox_priorbox')
    num_boxes = keras.backend.int_shape(net['mbox_loc'])[-1] // 4

    net['mbox_loc'] = keras.layers.Reshape((num_boxes, 4),
                              name='mbox_loc_final')(net['mbox_loc'])
    net['mbox_conf'] = keras.layers.Reshape((num_boxes, classes_num),
                               name='mbox_conf_logits')(net['mbox_conf'])
    net['mbox_conf'] = keras.layers.Activation('softmax',
                                  name='mbox_conf_final')(net['mbox_conf'])
    net['predictions'] = keras.layers.concatenate([net['mbox_loc'],
                               net['mbox_conf'],
                               net['mbox_priorbox']],
                               axis=2, name='predictions')
    model = keras.models.Model(net['input'], net['predictions'])
    # <TODO id=190820000>
    # <Debug>
    # Freezen VGG16 weight
    '''
    for vgg in model.layers[:19]:
    #for vgg in model.layers[:11]:
        vgg.trainable = False
    '''
    # </Debug>
    # </TODO>
    return model