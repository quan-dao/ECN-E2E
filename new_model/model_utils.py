from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, LSTM
from keras.layers.merge import add
from keras import regularizers
import keras


# -------------- ResNet-8 -------------- # 
def convolutional_block(X, num_filters, shape_filters, strides, stage, model_name=None):
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
    if model_name:
        bn_name_base = model_name + '_bn_' 
        conv_name_base = model_name + '_conv_'
    else:
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


def resnet_shorten(input_shape, model_name=None):
    """
    Define a shorten version of resnet8 above to serve the idea of free up the last layer so that encoder can
    learn to extract different features from different image
    
    Input:
        input_shape: shape of input image [n_H, n_W, n_C]
    
    Outpt:
        keras Model instance
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
    
    # Define model
    model = Model(inputs=[X_input], outputs=[X], name=model_name)
    
    return model


# -------------- Classifier -------------- #
def _classifier(input_shape, num_class):
    """
    2 Dense layers & Softmax activation
    """
    X_input = Input(shape=input_shape)
    
    X = Dense(256, activation='relu')(X_input)
    
    X = Dense(num_class, activation='relu')(X)
    
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


def model_shared_private_encoder(image_shape, shared_encoder, sep_encoder_list, LSTM_cell, 
                                 LSTM_dim_hidden_state, Ty):
    """
    Define full model with both shared & private encoder
    
    Input:
        image_shape (tuple): shape of input image
        shared_encoder (keras.Model): shared model used to extract low level feature vector from input image
        sep_encoder_list (list): list of keras.Model storing separate encoder
        LSTM_cell (keras.layers): shared LSTM layer
        LSTM_dim_hidden_state (int): dimension of LSTM_cell's hidden state
        Ty (int): length of spatial history
    
    Output:
        keras Model instance
    """
    # Input layer
    X_input_list = [Input(shape=image_shape) for i in range(Ty)]
    
    # pass each input through shared encoder
    shared_encoded_X = [shared_encoder(X) for X in X_input_list]
    
    # pass each encoded_X through its own convolution block
    separate_encoded_X = [separate_encoder(X) 
                          for separate_encoder, X in zip(sep_encoder_list, shared_encoded_X)]
    
    # initialize input & cell state
    a_0 = Input(shape=(LSTM_dim_hidden_state, ))  
    c_0 = Input(shape=(LSTM_dim_hidden_state, ))
    
    a = a_0
    c = c_0
    
    outputs = []
    
    # Decode
    for encoded_X in separate_encoded_X:
        # flatten & activate encoded_X 
        X = flattener(encoded_X)
        X = activator(X)
        
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


# ---------------------- ResNet50 --------------------------#
def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    ### START CODE HERE ###
    
    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    ### END CODE HERE ###
    
    return X


def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    ### START CODE HERE ###

    # Second component of main path (≈3 lines)
    X = Conv2D(F2, (f, f), strides = (1,1), padding='same', name = conv_name_base + '2b')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(F3, (1, 1), strides = (1,1), padding='valid', name = conv_name_base + '2c')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), name = conv_name_base + '1')(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    ### END CODE HERE ###
    
    return X


def ResNet50(input_shape = (64, 64, 3), classes = 6):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1')(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f = 3, filters = [128,128,512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128,128,512], stage=3, block='b')
    X = identity_block(X, 3, [128,128,512], stage=3, block='c')
    X = identity_block(X, 3, [128,128,512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D(pool_size=(2, 2), name="avg_pool")(X)
    
    ### END CODE HERE ###

    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model
