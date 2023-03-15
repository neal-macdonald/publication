import numpy as np
import tensorflow as tf
import keras
from keras.layers import MaxPooling2D, Dense, Flatten, Input, Concatenate
from keras.models import Model
from keras.applications.resnet import ResNet50
from keras import backend as K

from conv_settings import Conv
from initializer_settings import NovelMethod, GlorotNormal, RandomUniform


def ordinal_loss(y_true, y_pred):
    weights = K.cast(K.abs(K.argmax(y_true, axis=1) - K.argmax(y_pred, axis=1))/(K.int_shape(y_pred)[1] - 1),
                     dtype='float32')
    return (1.0 + weights) * keras.losses.categorical_crossentropy(y_true, y_pred)


class Zeroizer(tf.keras.constraints.Constraint):
    def __init__(self, scheme, name):
        self.scheme = scheme
        self.name = name

    def __call__(self, w):
        if self.scheme == "NovelMethod":
            try:
                zeros = zero_loc[self.name][0]
                return w * zeros
            except:
                pass
        else:
            return w


def hyperparameters(kernel_size, dilation, weight_init):
    if (kernel_size % 2)-1 != 0 or kernel_size < 3 or dilation < 1:
        print('error')
    if weight_init == 'NovelMethod':
        init = NovelMethod(kernel_size, dilation)
        kernel_size = kernel_size-(dilation-1)+(kernel_size*(dilation-1))
        dilation = 1
    elif weight_init == 'RandomUniform':
        init = RandomUniform
    else:
        init = GlorotNormal
    return kernel_size, dilation, init


def create_autoencoder(shape, kernel_size, dilation, weight_init):
    
    kernel_size, dilation, init = hyperparameters(kernel_size, dilation, weight_init)
    global zero_loc
    zero_loc = {}
    
    input_img = keras.Input(shape=(shape[0], shape[1], shape[2]))
    x = Conv(filters=64, kernel_size=(kernel_size,kernel_size), strides=(1,1),
                 padding='same',dilation_rate=(dilation,dilation),kernel_initializer=init,
                 kernel_constraint=Zeroizer(init,f'conv2D_64e'),name=f'conv2D_64e')(input_img)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.MaxPool2D((2,2))(x)
    
    for filters in [96,128]:
        x = Conv(filters=filters, kernel_size=(kernel_size,kernel_size), strides=(1,1),
                 padding='same',dilation_rate=(dilation,dilation),kernel_initializer=init,
                 kernel_constraint=Zeroizer(init,f'conv2D_{filters}e'),name=f'conv2D_{filters}e')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.MaxPool2D((2,2))(x)
    
    for filters in [128,96,64]:
        x = Conv(filters=filters, kernel_size=(kernel_size,kernel_size),strides=(1,1),
                 padding='same',dilation_rate=(dilation,dilation),kernel_initializer=init,
                 kernel_constraint=Zeroizer(init,f'conv2D_{filters}e'),name=f'conv2D_{filters}d')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
        x = keras.layers.BatchNormalization(momentum=0.5)(x)
        x = keras.layers.LeakyReLU()(x)
    decoded = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
    model = keras.Model(input_img, decoded)

    if weight_init == "NovelMethod" and zero_loc == {}:
        for layer in model.layers:
            if layer.name.startswith("conv"):
                zero_loc[f'{layer.name}'] = layer.get_weights()
    for key, value in zero_loc.items():
        zeros = abs(value[0]) > 0.000001
        zero_loc[key] = np.where(zeros==True,1,0)

    return model


def create_unet(shape, kernel_size, dilation, weight_init):

    kernel_size, dilation, init = hyperparameters(kernel_size, dilation, weight_init)
    global zero_loc
    zero_loc = {}

    input_img = keras.Input(shape=(shape[0], shape[1], shape[2]))
    
    filters = 48
    c1 = Conv(filters=filters, kernel_size=(kernel_size,kernel_size),strides=(1,1),activation='relu',
                 padding='same',dilation_rate=(dilation,dilation),kernel_initializer=init,
                 kernel_constraint=Zeroizer(init,f'conv2D_{filters}_e1'),name=f'conv2D_{filters}_e1')(input_img)
    c1 = keras.layers.BatchNormalization(momentum=0.5)(c1)
    c1 = Conv(filters=filters, kernel_size=(kernel_size,kernel_size),strides=(1,1),activation='relu',
                 padding='same',dilation_rate=(dilation,dilation),kernel_initializer=init,
                 kernel_constraint=Zeroizer(init,f'conv2D_{filters}_e2'),name=f'conv2D_{filters}_e2')(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)

    filters +=6
    c2 = Conv(filters=filters, kernel_size=(kernel_size,kernel_size),strides=(1,1),activation='relu',
                 padding='same',dilation_rate=(dilation,dilation),kernel_initializer=init,
                 kernel_constraint=Zeroizer(init,f'conv2D_{filters}_e1'),name=f'conv2D_{filters}_e1')(p1)
    c2 = keras.layers.BatchNormalization(momentum=0.5)(c2)
    c2 = Conv(filters=filters, kernel_size=(kernel_size,kernel_size),strides=(1,1),activation='relu',
                 padding='same',dilation_rate=(dilation,dilation),kernel_initializer=init,
                 kernel_constraint=Zeroizer(init,f'conv2D_{filters}_e2'),name=f'conv2D_{filters}_e2')(c2)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)

    filters +=6
    c3 = Conv(filters=filters, kernel_size=(kernel_size,kernel_size),strides=(1,1),activation='relu',
                 padding='same',dilation_rate=(dilation,dilation),kernel_initializer=init,
                 kernel_constraint=Zeroizer(init,f'conv2D_{filters}_e1'),name=f'conv2D_{filters}_e1')(p2)
    c3 = keras.layers.BatchNormalization(momentum=0.5)(c3)
    c3 = Conv(filters=filters, kernel_size=(kernel_size,kernel_size),strides=(1,1),activation='relu',
                 padding='same',dilation_rate=(dilation,dilation),kernel_initializer=init,
                 kernel_constraint=Zeroizer(init,f'conv2D_{filters}_e2'),name=f'conv2D_{filters}_e2')(c3)
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)

    filters +=6
    c4 = Conv(filters=filters, kernel_size=(kernel_size,kernel_size),strides=(1,1),activation='relu',
                 padding='same',dilation_rate=(dilation,dilation),kernel_initializer=init,
                 kernel_constraint=Zeroizer(init,f'conv2D_{filters}_e1'),name=f'conv2D_{filters}_e1')(p3)
    c4 = keras.layers.BatchNormalization(momentum=0.5)(c4)
    c4 = Conv(filters=filters, kernel_size=(kernel_size,kernel_size),strides=(1,1),activation='relu',
                 padding='same',dilation_rate=(dilation,dilation),kernel_initializer=init,
                 kernel_constraint=Zeroizer(init,f'conv2D_{filters}_e2'),name=f'conv2D_{filters}_e2')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    filters +=6
    c5 = Conv(filters=filters, kernel_size=(kernel_size,kernel_size),strides=(1,1),activation='relu',
                 padding='same',dilation_rate=(dilation,dilation),kernel_initializer=init,
                 kernel_constraint=Zeroizer(init,f'conv2D_{filters}_e1'),name=f'conv2D_{filters}_e1')(p4)
    c5 = keras.layers.BatchNormalization(momentum=0.5)(c5)
    c5 = Conv(filters=filters, kernel_size=(kernel_size,kernel_size),strides=(1,1),activation='relu',
                 padding='same',dilation_rate=(dilation,dilation),kernel_initializer=init,
                 kernel_constraint=Zeroizer(init,f'conv2D_{filters}_e2'),name=f'conv2D_{filters}_e2')(c5)

    # Expansive path:
    # filters -=6
    u6 = Conv(filters=filters, kernel_size=(kernel_size,kernel_size),strides=(1,1),activation='relu',
                 padding='same',dilation_rate=(dilation,dilation),kernel_initializer=init,
                 kernel_constraint=Zeroizer(init,f'conv2D_{filters}_d1'),name=f'conv2D_{filters}_d1')(c5)
    u6 = keras.layers.UpSampling2D((2, 2))(u6)
    u6 = keras.layers.concatenate([u6, c4])
    c6 = Conv(filters=filters, kernel_size=(kernel_size,kernel_size),strides=(1,1),activation='relu',
                 padding='same',dilation_rate=(dilation,dilation),kernel_initializer=init,
                 kernel_constraint=Zeroizer(init,f'conv2D_{filters}_d2'),name=f'conv2D_{filters}_d2')(u6)
    c6 = keras.layers.BatchNormalization(momentum=0.5)(c6)
    c6 = Conv(filters=filters, kernel_size=(kernel_size,kernel_size),strides=(1,1),activation='relu',
                 padding='same',dilation_rate=(dilation,dilation),kernel_initializer=init,
                 kernel_constraint=Zeroizer(init,f'conv2D_{filters}_d3'),name=f'conv2D_{filters}_d3')(c6)
    
    filters -=6
    u7 = Conv(filters=filters, kernel_size=(kernel_size,kernel_size),strides=(1,1),activation='relu',
                 padding='same',dilation_rate=(dilation,dilation),kernel_initializer=init,
                 kernel_constraint=Zeroizer(init,f'conv2D_{filters}_d1'),name=f'conv2D_{filters}_d1')(c6)
    u7 = keras.layers.UpSampling2D((2, 2))(u7)
    u7 = keras.layers.concatenate([u7, c3])
    c7 = Conv(filters=filters, kernel_size=(kernel_size,kernel_size),strides=(1,1),activation='relu',
                 padding='same',dilation_rate=(dilation,dilation),kernel_initializer=init,
                 kernel_constraint=Zeroizer(init,f'conv2D_{filters}_d2'),name=f'conv2D_{filters}_d2')(u7)
    c7 = keras.layers.BatchNormalization(momentum=0.5)(c7)
    c7 = Conv(filters=filters, kernel_size=(kernel_size,kernel_size),strides=(1,1),activation='relu',
                 padding='same',dilation_rate=(dilation,dilation),kernel_initializer=init,
                 kernel_constraint=Zeroizer(init,f'conv2D_{filters}_d3'),name=f'conv2D_{filters}_d3')(c7)

    filters -=6
    u8 = Conv(filters=filters, kernel_size=(kernel_size,kernel_size),strides=(1,1),activation='relu',
                 padding='same',dilation_rate=(dilation,dilation),kernel_initializer=init,
                 kernel_constraint=Zeroizer(init,f'conv2D_{filters}_1'),name=f'conv2D_{filters}_d1')(c7)
    u8 = keras.layers.UpSampling2D((2, 2))(u7)
    u8 = keras.layers.concatenate([u8, c2])
    c8 = Conv(filters=filters, kernel_size=(kernel_size,kernel_size),strides=(1,1),activation='relu',
                 padding='same',dilation_rate=(dilation,dilation),kernel_initializer=init,
                 kernel_constraint=Zeroizer(init,f'conv2D_{filters}_2'),name=f'conv2D_{filters}_d2')(u8)
    c8 = keras.layers.BatchNormalization(momentum=0.5)(c8)
    c8 = Conv(filters=filters, kernel_size=(kernel_size,kernel_size),strides=(1,1),activation='relu',
                 padding='same',dilation_rate=(dilation,dilation),kernel_initializer=init,
                 kernel_constraint=Zeroizer(init,f'conv2D_{filters}_3'),name=f'conv2D_{filters}_d3')(c8)

    filters -=6
    u9 = Conv(filters=filters, kernel_size=(kernel_size,kernel_size),strides=(1,1),activation='relu',
                 padding='same',dilation_rate=(dilation,dilation),kernel_initializer=init,
                 kernel_constraint=Zeroizer(init,f'conv2D_{filters}_d1'),name=f'conv2D_{filters}_d1')(c8)
    u9 = keras.layers.UpSampling2D((2, 2))(u9)
    u9 = keras.layers.concatenate([u9, c1])
    c9 = Conv(filters=filters, kernel_size=(kernel_size,kernel_size),strides=(1,1),activation='relu',
                 padding='same',dilation_rate=(dilation,dilation),kernel_initializer=init,
                 kernel_constraint=Zeroizer(init,f'conv2D_{filters}_d2'),name=f'conv2D_{filters}_d2')(u9)
    c9 = keras.layers.BatchNormalization(momentum=0.5)(c9)
    c9 = Conv(filters=filters, kernel_size=(kernel_size,kernel_size),strides=(1,1),activation='relu',
                 padding='same',dilation_rate=(dilation,dilation),kernel_initializer=init,
                 kernel_constraint=Zeroizer(init,f'conv2D_{filters}_d3'),name=f'conv2D_{filters}_d3')(c9)

    output = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=input_img, outputs=output)

    if weight_init == "NovelMethod" and zero_loc == {}:
        for layer in model.layers:
            if layer.name.startswith("conv"):
                zero_loc[f'{layer.name}'] = layer.get_weights()
    for key, value in zero_loc.items():
        zeros = abs(value[0]) > 0.000001
        zero_loc[key] = np.where(zeros==True,1,0)

    return model


def create_cnn_model(shape, kernel_size, dilation, weight_init, dense = False, classes = None):

    kernel_size, dilation, init = hyperparameters(kernel_size, dilation, weight_init)
    global zero_loc
    zero_loc = {}

    inputs = Input(shape=(shape[0], shape[1], shape[2]))

    x = Conv(filters=64, kernel_size=(kernel_size,kernel_size), strides=(1,1),
             padding='same',dilation_rate=(dilation,dilation),activation='relu',
             kernel_initializer=init,
             kernel_constraint=Zeroizer(init,'conv2D_1'), name='conv2D_1')(inputs)
    # x = MaxPooling2D((2,2))(x)
    x = Conv(filters=32, kernel_size=(kernel_size,kernel_size), strides=(1,1),
             padding='same',dilation_rate=(dilation,dilation),activation='relu',
             kernel_initializer=init,
             kernel_constraint=Zeroizer(init,'conv2D_2'), name='conv2D_2')(x)
    # x = MaxPooling2D((2,2))(x)
    output = Conv(filters=16, kernel_size=(kernel_size,kernel_size), strides=(1,1),
             padding='same',dilation_rate=(dilation,dilation),activation='relu',
             kernel_initializer=init,
             kernel_constraint=Zeroizer(init,'conv2D_3'), name='conv2D_3')(x)
    # if dense == True:
    #     output = Flatten()(output)
    #     output = Dense(64, activation='relu')(output)
    #     output = Dense(classes)(output)
    
    model = Model(inputs=inputs, outputs=output)

    if weight_init == "NovelMethod" and zero_loc == {}:
        for layer in model.layers:
            if layer.name.startswith("conv"):
                zero_loc[f'{layer.name}'] = layer.get_weights()
        for key, value in zero_loc.items():
            zeros = abs(value[0]) > 0.000001
            zero_loc[key] = np.where(zeros==True,1,0)

    return model


def create_resnet_model(shape, kernel_size, dilation, weight_init):

    kernel_size, dilation, init = hyperparameters(kernel_size, dilation, weight_init)
    global zero_loc
    zero_loc = {}

    input_img = keras.Input(shape=(shape[0], shape[1], shape[2]))
    
    # Entry Block
    x = Conv(filters=32, kernel_size=(kernel_size,kernel_size), strides=(1,1),
             padding='same',dilation_rate=(1,1),activation='relu',
             kernel_initializer=init,
             kernel_constraint=Zeroizer(init,f'conv2D_in_d1'),
             name=f'conv2D_in_d1')(input_img)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    previous_block_activation = x  # Residual
    
    # Residual blocks - downsampling
    for filters in [32, 64, 128]:
        x = keras.layers.Activation("relu")(x)
        x = Conv(filters=filters, kernel_size=(kernel_size,kernel_size), strides=(1,1),
             padding='same',dilation_rate=(dilation,dilation),
             kernel_initializer=init,
             kernel_constraint=Zeroizer(init,f'conv2D_{filters}_d1'),
             name=f'conv2D_{filters}_d1')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = Conv(filters=filters, kernel_size=(kernel_size,kernel_size), strides=(2,2),
             padding='same',dilation_rate=(1,1),
             kernel_initializer=init,
             kernel_constraint=Zeroizer(init,f'conv2D_{filters}_d2'),
             name=f'conv2D_{filters}_d2')(x)
        x = keras.layers.BatchNormalization()(x)
        # x = keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)
        residual = keras.layers.Conv2D(filters, 1, strides=2,
                    padding="same")(previous_block_activation)
        x = keras.layers.add([x, residual])
        previous_block_activation = x
        
    # Residual blocks - upsampling
    for filters in [128, 64, 32]:
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2DTranspose(filters=filters,
             kernel_size=(kernel_size,kernel_size),strides=(1,1),
             padding='same',dilation_rate=(1,1),
             name=f'deconv2D_{filters}_u1')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2DTranspose(filters=filters,
             kernel_size=(kernel_size,kernel_size),strides=(1,1),
             padding='same',dilation_rate=(1,1),
             name=f'deconv2D_{filters}_u2')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.UpSampling2D(2)(x)
        residual = keras.layers.UpSampling2D(2)(previous_block_activation)
        residual = keras.layers.Conv2D(filters, 1, padding="same")(residual)
        x = keras.layers.add([x, residual])
        previous_block_activation = x
    
    outputs = keras.layers.Conv2D(1, 1, activation="sigmoid", padding="same")(x)
    model = keras.Model(input_img, outputs)

    if weight_init == "NovelMethod" and zero_loc == {}:
        for layer in model.layers:
            if layer.name.startswith("conv"):
                zero_loc[f'{layer.name}'] = layer.get_weights()
    for key, value in zero_loc.items():
        zeros = abs(value[0]) > 0.000001
        zero_loc[key] = np.where(zeros==True,1,0)

    return model

# def test1():
#     kernel_size, dilation, init = hyperparameters(kernel_size, dilation, weight_init)
#     global zero_loc
#     zero_loc = {}

#     weights = 'imagenet'
#     inputs = Input(shape=(shape[0], shape[1], shape[2]))

#     base_model = ResNet50(include_top=False, weights=weights, input_shape=(shape[0], shape[1], shape[2]))
#     for layer in base_model.layers:
#         layer.trainable = False

#     x = Conv(filters=16, kernel_size=(kernel_size,kernel_size), strides=(1,1),
#              padding='same',dilation_rate=(dilation,dilation),activation='relu',
#              kernel_initializer=init,
#              kernel_constraint=Zeroizer(init,'conv2D_1'), name='conv2D_1')(inputs)
#     x = MaxPooling2D((2,2))(x)
#     x = Conv(filters=32, kernel_size=(kernel_size,kernel_size), strides=(1,1),
#              padding='same',dilation_rate=(dilation,dilation),activation='relu',
#              kernel_initializer=init,
#              kernel_constraint=Zeroizer(init,'conv2D_2'), name='conv2D_2')(x)
#     x = MaxPooling2D((2,2))(x)
#     x = Conv(filters=64, kernel_size=(kernel_size,kernel_size), strides=(1,1),
#              padding='same',dilation_rate=(dilation,dilation),activation='relu',
#              kernel_initializer=init,
#              kernel_constraint=Zeroizer(init,'conv2D_3'), name='conv2D_3')(x)
#     x = MaxPooling2D((2,2))(x)
#     x = Flatten()(x)

#     base_resnet = base_model(inputs)
#     base_resnet = Flatten()(base_resnet)

#     concated_layers = Concatenate()([x, base_resnet])
#     concated_layers = Dense(524, activation='relu')(concated_layers)
#     concated_layers = Dense(252, activation='relu')(concated_layers)
#     concated_layers = Dense(124, activation='relu')(concated_layers)
#     output = Dense(2, activation='relu')(concated_layers)

#     model = Model(inputs=inputs, outputs=output)

#     if weight_init == "NovelMethod" and zero_loc == {}:
#         for layer in model.layers:
#             if layer.name.startswith("conv"):
#                 zero_loc[f'{layer.name}'] = layer.get_weights()
#         for key, value in zero_loc.items():
#             zeros = abs(value[0]) > 0.000001
#             zero_loc[key] = np.where(zeros==True,1,0)

#     return model


def create_autoencoder2(shape, kernel_size, dilation, weight_init):

    kernel_size, dilation, init = hyperparameters(kernel_size, dilation, weight_init)
    global zero_loc
    zero_loc = {}

    input_img = keras.Input(shape=(shape[0], shape[1], shape[2]))

    x = Conv(filters=256, kernel_size=(kernel_size,kernel_size), strides=(1,1),
             padding='same',dilation_rate=(dilation,dilation),activation='relu',
             kernel_initializer=init,
             kernel_constraint=Zeroizer(init,'conv2D_1'), name='conv2D_1')(input_img)
    x = keras.layers.MaxPool2D((2,2))(x)
    x = Conv(filters=128, kernel_size=(kernel_size,kernel_size), strides=(1,1),
             padding='same',dilation_rate=(dilation,dilation),activation='relu',
             kernel_initializer=init,
             kernel_constraint=Zeroizer(init,'conv2D_2'), name='conv2D_2')(x)
    x = keras.layers.MaxPool2D((2,2))(x)
    x = Conv(filters=64, kernel_size=(kernel_size,kernel_size), strides=(1,1),
             padding='same',dilation_rate=(dilation,dilation),activation='relu',
             kernel_initializer=init,
             kernel_constraint=Zeroizer(init,'conv2D_3'), name='conv2D_3')(x)
    encoded = keras.layers.MaxPool2D((2,2))(x)
    x = Conv(filters=64, kernel_size=(kernel_size,kernel_size),strides=(1,1),
             padding='same',dilation_rate=(dilation,dilation),activation='relu',
             kernel_initializer=init, name='deconv2D_1')(encoded)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = Conv(filters=128, kernel_size=(kernel_size,kernel_size),strides=(1,1),
             padding='same',dilation_rate=(dilation,dilation),activation='relu',
             kernel_initializer=init, name='deconv2D_2')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = Conv(filters=256, kernel_size=(kernel_size,kernel_size),strides=(1,1),
             padding='same',dilation_rate=(dilation,dilation),activation='relu',
             kernel_initializer=init, name='deconv2D_3')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    decoded = Conv(2, kernel_size=(kernel_size,kernel_size),strides=(1,1),
                   padding='same',dilation_rate=(dilation,dilation),
                   kernel_initializer=init,activation='sigmoid', name='deconv2D_4')(x)

    model = keras.Model(input_img, decoded)

    if weight_init == "NovelMethod" and zero_loc == {}:
        for layer in model.layers:
            if layer.name.startswith("conv"):
                zero_loc[f'{layer.name}'] = layer.get_weights()
    for key, value in zero_loc.items():
        zeros = abs(value[0]) > 0.000001
        zero_loc[key] = np.where(zeros==True,1,0)

    return model

def create_c(shape, kernel_size, dilation, weight_init):
    
    kernel_size, dilation, init = hyperparameters(kernel_size, dilation, weight_init)
    global zero_loc
    zero_loc = {}

    input_img = keras.Input(shape=(shape[0], shape[1], shape[2]))
    
    # Entry Block
    x = Conv(filters=32, kernel_size=(kernel_size,kernel_size), strides=(1,1),
             padding='same',dilation_rate=(1,1),activation='relu',
             kernel_initializer=init,
             kernel_constraint=Zeroizer(init,f'conv2D_in_d1'),
             name=f'conv2D_in_d1')(input_img)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)