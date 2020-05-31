from keras.models import Sequential
from keras.layers import Dense

def get_Model(shape):
    model = Sequential()
    model.add(Dense(200, activation='relu', input_shape = (shape, )))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(150, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model