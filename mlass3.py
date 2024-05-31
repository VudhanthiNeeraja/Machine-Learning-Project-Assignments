from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input, Dense
from sklearn.preprocessing import OneHotEncoder

#4 different neural network models using Keras for the regression task


reg_data = fetch_openml(data_id=507)
x_train, x_val, y_train, y_val = train_test_split(reg_data.data, reg_data.target, test_size=0.2, random_state=0)


model1 = Sequential()
model1.add(Input((6,)))
model1.add(Dense(10, activation="sigmoid"))
model2 = Sequential()
model2.add(Input((6,)))
model2.add(Dense(10, activation="relu"))
model3 = Sequential()
model3.add(Input((6,)))
model3.add(Dense(12, activation="sigmoid"))
model3.add(Dense(10, activation="sigmoid"))
model4 = Sequential()
model4.add(Input((6,)))
model4.add(Dense(10, activation="sigmoid"))
model1.add(Dense(1))
model2.add(Dense(1))
model3.add(Dense(1))
model4.add(Dense(1))
model1.compile(optimizer="adam", loss="mse", metrics=["mse"])
model1_trained = model1.fit(x_train, y_train, epochs = 100, validation_data=(x_val, y_val), verbose = 0)
import matplotlib.pyplot as plt
plt.plot(model1_trained.history["mse"], label="Training")
plt.plot(model1_trained.history["val_mse"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.ylim(0,0.10)
plt.legend()
plt.show()


model2.compile(optimizer="adam", loss="mse", metrics=["mse"])
model2_trained = model2.fit(x_train, y_train, epochs = 100, validation_data=(x_val, y_val), verbose = 0)
plt.plot(model2_trained.history["mse"], label="Training")
plt.plot(model2_trained.history["val_mse"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.ylim(0,300000)
plt.legend()
plt.show()


model3.compile(optimizer="adam", loss="mse", metrics=["mse"])
model3_trained = model3.fit(x_train, y_train, epochs = 100, validation_data=(x_val, y_val), verbose = 0)
plt.plot(model3_trained.history["mse"], label="Training")
plt.plot(model3_trained.history["val_mse"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.ylim(0,0.10)
plt.legend()
plt.show()


model4.compile(optimizer="sgd", loss="mse", metrics=["mse"])
model4_trained = model4.fit(x_train, y_train, epochs = 100, validation_data=(x_val, y_val), verbose = 0)
plt.plot(model4_trained.history["mse"], label="Training")
plt.plot(model4_trained.history["val_mse"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.ylim(0,0.10)
plt.legend()
plt.show()

print("Minimum mse for model1, model2, model3, model4 respectively are: ",min(model1_trained.history["val_mse"]), min(model2_trained.history["val_mse"]), min(model3_trained.history["val_mse"]), min(model4_trained.history["val_mse"]))

model1.evaluate(x_val,y_val)
model2.evaluate(x_val,y_val)
model3.evaluate(x_val,y_val)
model4.evaluate(x_val,y_val)

y_pred1 = model1.predict(x_val)
y_pred2 = model2.predict(x_val)
y_pred3 = model3.predict(x_val)
y_pred4 = model4.predict(x_val)

print(y_pred1)
print(y_pred2)
print(y_pred3)
print(y_pred4)

#4 different neural network models using Keras for the regression task.

class_data = fetch_openml(data_id=772)
enc = OneHotEncoder(sparse_output = False)
tmp = [[x] for x in class_data.target]
ohe_target = enc.fit_transform(tmp)
x_train, x_val, y_train, y_val = train_test_split(class_data.data, ohe_target, test_size=0.2, random_state=0)
from keras.models import Sequential
from keras.layers import Input, Dense
model1 = Sequential()
model1.add(Input((3,)))
model1.add(Dense(20, activation="sigmoid"))
model2 = Sequential()
model2.add(Input((3,)))
model2.add(Dense(20, activation="relu"))
model3 = Sequential()
model3.add(Input((3,)))
model3.add(Dense(20, activation="sigmoid"))
model3.add(Dense(10, activation="sigmoid"))
model4 = Sequential()
model4.add(Input((3,)))
model4.add(Dense(20, activation="sigmoid"))
model1.add(Dense(2, activation="softmax"))
model2.add(Dense(2, activation="softmax"))
model3.add(Dense(2, activation="softmax"))
model4.add(Dense(2, activation="softmax"))

model1.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model1_trained = model1.fit(x_train, y_train, epochs = 100, validation_data=(x_val, y_val), verbose = 0)
import matplotlib.pyplot as plt
plt.plot(model1_trained.history["accuracy"], label="Training")
plt.plot(model1_trained.history["val_accuracy"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0.3,0.8)
plt.legend()
plt.show()


model2.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model2_trained = model2.fit(x_train, y_train, epochs = 100, validation_data=(x_val, y_val), verbose = 0)
plt.plot(model2_trained.history["accuracy"], label="Training")
plt.plot(model2_trained.history["val_accuracy"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0.3,0.8)
plt.legend()
plt.show()


model3.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model3_trained = model3.fit(x_train, y_train, epochs = 100, validation_data=(x_val, y_val), verbose = 0)
plt.plot(model3_trained.history["accuracy"], label="Training")
plt.plot(model3_trained.history["val_accuracy"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0.3,0.8)
plt.legend()
plt.show()


model4.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
model4_trained = model4.fit(x_train, y_train, epochs = 100, validation_data=(x_val, y_val), verbose = 0)
plt.plot(model4_trained.history["accuracy"], label="Training")
plt.plot(model4_trained.history["val_accuracy"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0.3,0.8)
plt.legend()
plt.show()

print("Maximum accuracy for model1, model2, model3, model4 respectively are: ",max(model1_trained.history["val_accuracy"]), max(model2_trained.history["val_accuracy"]), max(model3_trained.history["val_accuracy"]), max(model4_trained.history["val_accuracy"]))

model1.evaluate(x_val,y_val)
model2.evaluate(x_val,y_val)
model3.evaluate(x_val,y_val)
model4.evaluate(x_val,y_val)

y_pred1 = model1.predict(x_val)
y_pred2 = model2.predict(x_val)
y_pred3 = model3.predict(x_val)
y_pred4 = model4.predict(x_val)

print(y_pred1)
print(y_pred2)
print(y_pred3)
print(y_pred4)