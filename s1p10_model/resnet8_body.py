from keras.layers import Input, Activation, Conv2D, MaxPooling2D, Flatten
import keras.layers as KL
from keras.models import Model


def convolutional_block(X, num_filters, shape_filters, strides, stage):
    """
    Implementation of convolutional block in Residual network
    
    Input:
        X (tensor): input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        num_filters (list of 3 ints): list of number of filters
        shape_filters (list of 3 ints): list of filters' shape
        strides (list of 3 ints): list of strides
        stage (int): stage of this convolutional block in the whole ResNet
        
    Output:
        tensor of shape (m, n_H, n_W, n_C)
    """
    
    # retrieve filters shape from filters
    n1, n2, n3 = num_filters
    f1, f2, f3 = shape_filters
    
    # retrieve strides from strides
    s1, s2, s3 = strides
    
    # create name
    bn_name_base = 'bn_' + str(stage) + '_'
    conv_name_base = 'conv_' + str(stage) + '_'
    
    # save value of X
    X_shorcut = X
    
    # First component of the main path
    X = KL.normalization.BatchNormalization(name=bn_name_base + 'a')(X)
    X = Activation('relu')(X)
    X = Conv2D(n1, (f1, f1), strides=[s1, s1], padding='same',
               name=conv_name_base + 'a')(X)
    
    # Second component of the main path
    X = KL.normalization.BatchNormalization(name=bn_name_base + 'b')(X)
    X = Activation('relu')(X)
    X = Conv2D(n2, (f2, f2), strides=[s2, s2], padding='same',
               name=conv_name_base + 'b')(X)
    
    # Short-cut
    X_shorcut = Conv2D(n3, (f3, f3), strides=[s3, s3], padding='same', name=conv_name_base + 'c')(X_shorcut)
    
    X = KL.merge.add([X, X_shorcut])
    
    return X


def resnet8_body(input_shape):
    """
    Define encoder architecture as ResNet8
    
    Input:
        input_shape (list of ints): shape of input image [n_H, n_W, n_C]
        
    Output:
        model: a Model instance
    """
    
    # Input
    X_input = Input(shape=input_shape)
    
    # Apply 1st convolution & max pooling on input
    X = Conv2D(32, (5, 5), strides=[2,2], padding='same', name='conv_0')(X_input)
    X = MaxPooling2D(pool_size=(3, 3), strides=[2,2])(X) 
    
    # First convolutional block
    X = convolutional_block(X, [32, 32, 32], [3, 3, 1], [2, 1, 2], stage=1)
    
    # Second convolutional block
    X = convolutional_block(X, [64, 64, 64], [3, 3, 1], [2, 1, 2], stage=2)
    
    # Third convolutional block
    X = convolutional_block(X, [128, 128, 128], [3, 3, 1], [2, 1, 2], stage=3)
    
    # Output layer
    X = Flatten()(X)
    X = Activation('relu')(X)
    
    # Define model
    model = Model(inputs=[X_input], outputs=[X], name="resnet8")    
    return model