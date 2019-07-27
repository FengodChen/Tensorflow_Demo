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
    net['conv1_1'] = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(net['input'])
    net['conv1_2'] = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(net['conv1_1'])
    net['pool1'] = keras.layers.MaxPool2D(strides=(2, 2), name='pool1')(net['conv1_2'])
    # Block 2
    net['conv2_1'] = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(net['pool1'])
    net['conv2_2'] = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(net['conv2_1'])
    net['pool2'] = keras.layers.MaxPool2D(strides=(2, 2), name='pool2')(net['conv2_2'])
    # Block 3
    net['conv3_1'] = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(net['pool2'])
    net['conv3_2'] = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(net['conv3_1'])
    net['conv3_3'] = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(net['conv3_2'])
    net['pool3'] = keras.layers.MaxPool2D(strides=(2, 2), name='pool3')(net['conv3_3'])
    # Block 4
    net['conv4_1'] = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(net['pool3'])
    net['conv4_2'] = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(net['conv4_1'])
    net['conv4_3'] = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(net['conv4_2'])
    net['pool4'] = keras.layers.MaxPool2D(strides=(2, 2), name='pool4')(net['conv4_3'])
    # Block 5
    net['conv5_1'] = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(net['pool4'])
    net['conv5_2'] = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(net['conv5_1'])
    net['conv5_3'] = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(net['conv5_2'])
    net['pool5'] = keras.layers.MaxPool2D((2,2), strides=(2, 2), name='pool5')(net['conv5_3'])
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
    net['conv4_3_norm'] = ssd_layer.Normalize(20, name='conv4_3_norm')(net['conv4_3'])
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
    net['pool6_mbox_conf_flat'] = keras.layersDense(priors_num * classes_num, name=name)(net['pool6'])
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
                                 concat_axis=1, name='mbox_priorbox')
    if hasattr(net['mbox_loc'], '_keras_shape'):
        num_boxes = net['mbox_loc']._keras_shape[-1] // 4
    elif hasattr(net['mbox_loc'], 'int_shape'):
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
    return model
    #return net
    #model = keras.Model(inputs = net['input'], outputs = net['pool6'])
    #return model
