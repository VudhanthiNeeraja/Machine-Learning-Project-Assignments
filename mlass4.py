import os, glob
files = glob.glob("Monkey Species Data/*/*/*")
for file in files:
    f = open(file,"rb")
    if not b"JFIF" in f.peek(10):
        f.close()
        os.remove(file)
    else:
        f.close()
import matplotlib.pyplot as plt
import numpy as np
from keras import Input, preprocessing
from keras.preprocessing import image_dataset_from_directory
from keras.models import Sequential
from keras.layers import Rescaling, Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.src.layers import GlobalAveragePooling2D

training_set = image_dataset_from_directory("Monkey Species Data/Training Data", label_mode="categorical", image_size=(100,100))
test_data = image_dataset_from_directory("Monkey Species Data/Prediction Data", label_mode="categorical", image_size=(100,100), shuffle=False)

import matplotlib.pyplot as plt

for e in training_set :
    print(e[1][0])
    plt.imshow(e[0][0]/255)
    plt.show()

#Model 1:
m = Sequential()
m.add(Input((100,100,3)))
m.add(Rescaling(1/255))
m.add(Conv2D(32, kernel_size=(3,3), activation="relu"))
m.add(MaxPooling2D(pool_size=(2,2)))
m.add(Conv2D(64, (3,3), activation="relu"))
m.add(MaxPooling2D(pool_size=(2,2)))
m.add(Flatten())
m.add(Dense(128, activation="relu"))
m.add(Dropout(0.2))
m.add(Dense(10, activation="softmax"))
m.compile(loss="categorical_crossentropy", metrics=["accuracy"])
m.summary()
epochs = 20
print("Training:")
for i in range(epochs):
    history = m.fit(training_set, epochs=1)
    print("Epoch:", i+1, "Training Accuracy:", history.history["accuracy"])

score = m.evaluate(test_data)
print("Test accuracy model 1:", score[1])

m.save("my_cnn1.keras")


from keras.models import load_model
old_model = load_model("my_cnn1.keras")


corr = []
for e in test_data:
    corr += list(np.argmax(e[1], axis=1))

p = old_model.predict(test_data)
pr = np.argmax(p, axis=1)

from sklearn.metrics import confusion_matrix

print("Confusion matrix for model 1:",confusion_matrix(corr,pr))


# import matplotlib.pyplot as plt
#
# for e in training_set :
#     print(e[1][0])
#     plt.imshow(e[0][0]/255)
#     plt.show()

#Model 2:

m1 = Sequential()
m1.add(Input((100,100,3)))
m1.add(Rescaling(1/255))
m1.add(Conv2D(64, kernel_size=(3,3), activation="relu"))
m1.add(MaxPooling2D(pool_size=(2,2)))
m1.add(Conv2D(128, (6,6), activation="relu"))
m1.add(MaxPooling2D(pool_size=(2,2)))
m1.add(Flatten())
m1.add(Dense(64, activation="relu"))
m1.add(Dense(128, activation="relu"))
m1.add(Dropout(0.2))
m1.add(Dense(10, activation="softmax"))
m1.compile(loss="categorical_crossentropy", metrics=["accuracy"])
m1.summary()
epochs = 20
print("Training:")
for i in range(epochs):
    history = m1.fit(training_set, epochs=1)
    print("Epoch:", i+1, "Training Accuracy:", history.history["accuracy"])

score = m1.evaluate(test_data)
print("Test accuracy:", score[1])

m1.save("my_cnn2.keras")

from keras.models import load_model
old_model1 = load_model("my_cnn2.keras")

p1 = old_model.predict(test_data)
pr1 = np.argmax(p, axis=1)

print("Confusion matrix for model 2:",confusion_matrix(corr,pr1))

# image_file = "D:/Python Projects/pythonProject3/Monkey Species Data/Prediction Data/Bald Uakari/BU (1).jpg"
# img = preprocessing.image.load_img(image_file,target_size=(100,100))
# img_arr = preprocessing.image.img_to_array(img)
# img_arr.shape
#
# plt.imshow(img_arr/255)
# plt.show()

#
# img_cl = img_arr.reshape(1,100,100,3)
#
# score = old_model.predict(img_cl)
# print(score.round(3))



from keras.applications import EfficientNetV2S
base_model = EfficientNetV2S(include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(32, activation="relu")(x)
x = Dense(64, activation="relu")(x)
output_layer = Dense(10, activation="softmax")(x)

from  keras.models import Model

m = Model(inputs=base_model.input, outputs = output_layer)

for layer in base_model.layers:
    layer.trainable = False

m.compile(loss="categorical_crossentropy", metrics = ["accuracy"])
#print(m.summary())

epochs = 20
print("Training:")
for i in range(epochs):
    history = m.fit(training_set, epochs=1)
    print("Epoch:", i+1, "Training Accuracy:", history.history["accuracy"])

score = m.evaluate(test_data)
print("Test accuracy:", score[1])

m.save("my_cnn3.keras")

from keras.models import load_model
old_model2 = load_model("my_cnn3.keras")

score = old_model2.evaluate(test_data)
print("Test accuracy model3:", score[1])


p2 = old_model2.predict(test_data)
pr2 = np.argmax(p, axis=1)

print("Confusion matrix for model 2:",confusion_matrix(corr,pr2))


#to find the incorrect predictions from model 2 (better model) and model 3 (fine tuned model)
for i in range(len(corr)):
    print(corr[i],end=' ')

print("\n")

for i in range(len(pr)):
    print(pr[i],end=' ')

print("\n")

for i in range(len(pr)):
    print(pr1[i],end=' ')

#task -3:
image_files = ["D:/Python Projects/pythonProject3/Monkey Species Data/Prediction Data/Emperor Tamarin/image (8).jpeg", "D:/Python Projects/pythonProject3/Monkey Species Data/Prediction Data/Emperor Tamarin/images (6).jpeg", "D:/Python Projects/pythonProject3/Monkey Species Data/Prediction Data/Emperor Tamarin/images (8).jpeg", "D:/Python Projects/pythonProject3/Monkey Species Data/Prediction Data/Emperor Tamarin/images (20).jpeg", "D:/Python Projects/pythonProject3/Monkey Species Data/Prediction Data/Emperor Tamarin/images (22).jpeg", "D:/Python Projects/pythonProject3/Monkey Species Data/Prediction Data/Emperor Tamarin/images (33).jpeg", "D:/Python Projects/pythonProject3/Monkey Species Data/Prediction Data/Emperor Tamarin/images (41).jpeg", "D:/Python Projects/pythonProject3/Monkey Species Data/Prediction Data/Golden Monkey/GM (377).jpeg", "D:/Python Projects/pythonProject3/Monkey Species Data/Prediction Data/Golden Monkey/GM (381).jpeg", "D:/Python Projects/pythonProject3/Monkey Species Data/Prediction Data/Golden Monkey/GM (384).jpeg"]
image_files_array = []
image_files_cl = []

for i in range(len(image_files)):
    image_files[i] = preprocessing.image.load_img(image_files[i],target_size=(100,100))
    image_files_array.append(preprocessing.image.img_to_array(image_files[i]))

for i in range(len(image_files_array)):
    image_files_cl.append(image_files_array.reshape(1,100,100,3))

model3_pred_10 = []

for i in range(len(image_files_cl)):
    model3_pred_10.append(old_model.predict(image_files_cl[i]))

print(model3_pred_10)
