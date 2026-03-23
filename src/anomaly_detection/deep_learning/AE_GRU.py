from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, GRU, Reshape, Permute, 
                                     Conv1DTranspose, TimeDistributed, Dense)
from tensorflow.keras.layers import RepeatVector

def build_gru_ae(time_steps, nb_features):
    model = Sequential()
    
    # ENCODER
    model.add(Input(shape=(time_steps, nb_features)))
    model.add(GRU(units=64, return_sequences=True))
    model.add(GRU(units=32, return_sequences=False)) 
    
    # BRIDGE
    model.add(RepeatVector(time_steps))
    
    # DECODER
    model.add(GRU(units=32, return_sequences=True))
    model.add(GRU(units=64, return_sequences=True))
    
    # OUTPUT
    model.add(TimeDistributed(Dense(units=nb_features)))
    
    model.compile(optimizer='adam', loss='mse')
    return model

def build_cnn_gru_ae(time_steps, nb_features):
   
    input_layer = Input(shape=(time_steps, nb_features))

    # CONVOLUTIONAL FEATURE COMPRESSION LAYER
    x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input_layer)
    x = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(x)
    x = Conv1D(filters=8, kernel_size=3, activation='relu', padding='same')(x)


    # TEMPORAL DEPENDENCY MODELING LAYER (ENCODEUR GRU)
    x = GRU(128, return_sequences=True)(x)
    x = GRU(128, return_sequences=True)(x)
    encoder_output = GRU(128, return_sequences=False)(x) 

    # TEMPORAL FEATURE RECONSTRUCTION LAYER (DÉCODEUR GRU)
    x = RepeatVector(time_steps)(encoder_output)
    
    # Inverse GRU (Décodeur)
    x = GRU(128, return_sequences=True)(x)
    x = GRU(128, return_sequences=True)(x)
    gru_decoded = GRU(32, return_sequences=True)(x) 

    # SPATIAL FEATURE RECONSTRUCTION LAYER
    x = Conv1DTranspose(filters=16, kernel_size=3, activation='relu', padding='same')(gru_decoded)
    x = Conv1DTranspose(filters=32, kernel_size=3, activation='relu', padding='same')(x)
    output_layer = Conv1DTranspose(filters=nb_features, kernel_size=3, padding='same')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse')
    return model