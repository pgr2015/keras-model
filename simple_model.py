import keras as K
from keras import layers
from keras_preprocessing import image
from keras import Model
from keras.layers import *
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions



def conv_block(input_tensor, filter_number, strides=(1,1)):

    filters1, filters2, filters3 = filter_number

    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3

    x = Conv2D(filters1, (1,1), strides=strides)(input_tensor)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, (3,3), padding='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1,1))(x)
    x = BatchNormalization(axis=bn_axis)(x)
    #x = Activation('relu')(x)

    if K.int_shape(input_tensor)[bn_axis] != filter_number[2]:
        short_cut = Conv2D(filter_number[2], (1,1), strides=strides)(input_tensor)
        short_cut = BatchNormalization(axis=bn_axis)(short_cut)
    else:
        short_cut = input_tensor

    x = layers.add([x, short_cut])
    x = Activation('relu')(x)

    return x

def ResNet_50(input_shape=None, input_tensor=None):

    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=197,
                                      data_format=K.image_data_format(),
                                      require_flatten=True)


    img_input = Input(shape=input_shape)

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3,3))(img_input)
    x = Conv2D(64, (7,7), strides=(2,2))(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3,3), strides=(2,2))(x)

    x = conv_block(x, [64,64,256])
    x = conv_block(x, [64,64,256])
    x = conv_block(x, [64,64,256])

    x = conv_block(x, [128,128,512], strides=(2,2))
    x = conv_block(x, [128,128,512])
    x = conv_block(x, [128,128,512])
    x = conv_block(x, [128,128,512])

    x = conv_block(x, [256, 256, 1024], strides=(2,2))
    x = conv_block(x, [256, 256, 1024])
    x = conv_block(x, [256, 256, 1024])
    x = conv_block(x, [256, 256, 1024])
    x = conv_block(x, [256, 256, 1024])
    x = conv_block(x, [256, 256, 1024])

    x = conv_block(x, [512, 512, 2048], strides=(2,2))
    x = conv_block(x, [512, 512, 2048])
    x = conv_block(x, [512, 512, 2048])

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    x = Flatten()(x)
    x = Dense(1000, activation='softmax', name='fc1000')(x)

    model = Model(img_input, x, name='resnet50')

    return model


if __name__ == '__main__':
    jpg = 'test/cat.jpg'

    img = image.load_img(jpg, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    resnet_50 = ResNet_50()
    resnet_50.load_weights('models/ResNet-50.h5')

    print('Input image shape:', x.shape)

    preds = resnet_50.predict(x)
    print('Predicted:', decode_predictions(preds))
    #resnet_50.save('wrong.h5')







