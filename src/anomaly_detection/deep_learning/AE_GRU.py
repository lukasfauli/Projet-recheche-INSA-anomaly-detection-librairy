from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, RepeatVector, TimeDistributed, Input

def build_autoencoder(time_steps, nb_features):
    model = Sequential()
    
    # ENCODER
    model.add(Input(shape=(time_steps, nb_features)))
    model.add(GRU(units=64, return_sequences=True))
    model.add(GRU(units=32, return_sequences=False)) # Bottleneck
    
    # BRIDGE
    model.add(RepeatVector(time_steps))
    
    # DECODER
    model.add(GRU(units=32, return_sequences=True))
    model.add(GRU(units=64, return_sequences=True))
    
    # OUTPUT
    model.add(TimeDistributed(Dense(units=nb_features)))
    
    model.compile(optimizer='adam', loss='mse')
    return model