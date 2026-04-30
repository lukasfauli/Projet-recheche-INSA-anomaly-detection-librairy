from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, GRU, Reshape, Permute, 
                                     Conv1DTranspose, TimeDistributed, Dense)
from tensorflow.keras.layers import RepeatVector

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input

def build_dnn_ae(time_steps, nb_features, latent_dim):
    """
    latent_dim : la taille du goulot d'étranglement (espace latent)
    """
    input_dim = time_steps * nb_features 
    
    model = Sequential(name=f"DNN_AE_Latent_{latent_dim}")

    model.add(Input(shape=(time_steps, nb_features)))
    model.add(Flatten()) # Devient un vecteur de 75

    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    
    # latent space
    model.add(Dense(latent_dim, activation='relu', name="Latent_Space"))
    
  
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    
    model.add(Dense(input_dim, activation='linear'))
    
    
    model.add(Reshape((time_steps, nb_features)))
    
    model.compile(optimizer='adam', loss='mse')
    return model

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





def build_simple_ae(input_dim=15, latent_dim=3):
    model = Sequential(name="Simple_Autoencoder")
    
    # Entrée : les 15 capteurs en direct
    model.add(Input(shape=(input_dim,)))
    
    # Encodeur
    model.add(Dense(10, activation='relu')) 
    # Espace Latent (3D pour l'interprétabilité)
    model.add(Dense(latent_dim, activation='relu', name="latent_layer"))
    
    # Décodeur
    model.add(Dense(10, activation='relu'))
    # Sortie : on reconstruit les 15 capteurs
    model.add(Dense(input_dim, activation='linear'))
    
    model.compile(optimizer='adam', loss='mse')
    return model

