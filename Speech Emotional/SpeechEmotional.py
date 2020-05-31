from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from LoadData import  load_data
from sklearn.neural_network import MLPClassifier
from BuildModel import get_Model
from livelossplot.keras import PlotLossesCallback
from sklearn.svm import SVC

x_train, x_test, x_valid, y_train, y_test, y_valid = load_data(test_size = 0.25, valid_size = 0.25)
print('Shape of training data', x_train.shape)
print('Shape of training labels', len(y_train))
print('Shape of test data', x_test.shape)
print('Shape of test labels', len(y_test))
print('Shape of validation data', x_valid.shape)
print('Shape of validation labels', len(y_valid))
print(f'\nFeatures extracted:', x_train.shape[1])

#MLP classifier 
model = MLPClassifier(alpha = 0.01, batch_size = 256, epsilon = 1e-08, hidden_layer_sizes=(300,), learning_rate = 'adaptive', max_iter = 500)
model.fit(x_train, y_train)
y_prediction = model.predict(x_test)
accuracy = accuracy_score(y_true = y_test, y_pred = y_prediction)
print("\nAccuracy: {:.2f}%".format(accuracy * 100))
print("\n")
print(classification_report(y_test, y_prediction))

"""#Deep Neural Network
model = get_Model(x_train.shape[1])
model.fit(x_train, y_train,
          batch_size = 10000,
          epochs = 100,
          verbose = 0,
          callbacks = [PlotLossesCallback()],
          validation_data = (x_valid, y_valid))
y_prediction = model.predict(x_test)
accuracy = accuracy_score(y_true = y_test, y_pred = y_prediction)
print("\nAccuracy: {:.2f}%".format(accuracy * 100))
print("\n")
print(classification_report(y_test, y_prediction))"""

"""model = SVC(kernel='linear')
model.fit(x_train, y_train)
y_prediction = model.predict(x_test)
accuracy = accuracy_score(y_true = y_test, y_pred = y_prediction)
print("\nAccuracy: {:.2f}%".format(accuracy * 100))
print("\n")
print(classification_report(y_test, y_prediction))"""