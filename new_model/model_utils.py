from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, LSTM
from keras.layers.merge import add
from keras import regularizers
import keras


# -------------- ResNet-8 -------------- # 
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
    X = BatchNormalization(name=bn_name_base + 'a')(X)
    X = Activation('relu')(X)
    X = Conv2D(n1, (f1, f1), strides=[s1, s1], padding='same',
               name=conv_name_base + 'a',
               kernel_initializer='he_normal',
               kernel_regularizer=regularizers.l2(1e-4))(X)
    
    # Second component of the main path
    X = keras.layers.normalization.BatchNormalization(name=bn_name_base + 'b')(X)
    X = Activation('relu')(X)
    X = Conv2D(n2, (f2, f2), strides=[s2, s2], padding='same',
               name=conv_name_base + 'b',
               kernel_initializer='he_normal',
               kernel_regularizer=regularizers.l2(1e-4))(X)
    
    # Short-cut
    X_shorcut = Conv2D(n3, (f3, f3), strides=[s3, s3], padding='same', name=conv_name_base + 'c')(X_shorcut)
    
    X = add([X, X_shorcut])
    
    return X


def resnet8(input_shape):
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
    model = Model(inputs=[X_input], outputs=[X])
    
    return model


# -------------- Classifier -------------- #
def _classifier(input_shape, num_class):
    """
    2 Dense layers & Softmax activation
    """
    X_input = Input(shape=input_shape)
    
    X = Dense(num_class, activation='relu')(X_input)
    
    y = Activation('softmax')(X)
    
    model = Model(inputs=[X_input], outputs=[y])
    
    return model

# -------------- Full Model -------------- #
def hybrid_LSTM_training(img_shape, encoder, LSTM_cell, dim_lstm_hidden, classifier, reshapor, Ty=10):
    """
    Define full hybrid model for training encoder & LSTM_cell
    
    Input:
        img_shape (tuple): shape of images input to encoder 
        encoder (keras.Model): create a features vector for 1 image
        LSTM_cell (keras.layer)
        dim_lstm_hidden (int): number of LSTM hidden units
        classifier (keras.layer): a stack of Dense layers ended with Softmax 
        Ty (int): number of images in a training sample 
    """
    # Input layer
    X_input_list = [Input(shape=img_shape) for i in range(Ty)]
    
    # Encode
    encoded_X_list = [encoder(X_input) for X_input in X_input_list]
    
    # initialize input & cell state
    a_0 = Input(shape=(dim_lstm_hidden, ))  
    c_0 = Input(shape=(dim_lstm_hidden, ))
    
    a = a_0
    c = c_0
    
    outputs = []
    
    # Decode
    for encoded_X in encoded_X_list:
        # perform 1 step of LSTM cell
        X = reshapor(encoded_X)
        a, _, c = LSTM_cell(X, initial_state=[a, c])
        
        # apply regressor to the hidden state of LSTM_cell
        out = classifier(a)
        
        # append out to outputs
        outputs.append(out)
        
    # define model
    model = Model(inputs=X_input_list + [a_0, c_0], outputs=outputs)
    
    return model
