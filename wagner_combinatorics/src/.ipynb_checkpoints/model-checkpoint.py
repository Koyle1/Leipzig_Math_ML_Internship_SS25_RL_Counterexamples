from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

def get_model(
        first_layer_neurons: int,
        second_layer_neurons: int,
        third_layer_neurons: int,
        input_size: int,
        learning_rate: float,
        verbose: bool=True):
    """Simple 3-layer dense NN with single node sigmoid output."""
    #Model structure: a sequential network with three hidden layers, sigmoid activation in the output.
    #I usually used relu activation in the hidden layers but play around to see what activation function and what optimizer works best.
    #It is important that the loss is binary cross-entropy if alphabet size is 2.

    model = Sequential()
    model.add(Dense(first_layer_neurons,  activation="relu"))
    model.add(Dense(second_layer_neurons, activation="relu"))
    model.add(Dense(third_layer_neurons, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.build((None, input_size))
    model.compile(loss="binary_crossentropy", optimizer=SGD(learning_rate = learning_rate)) #Adam optimizer also works well, with lower learning rate

    if verbose: print(model.summary())   
    
    return model


