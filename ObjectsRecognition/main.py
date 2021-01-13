import cv2 as cv
import os
import re
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches
from tensorflow.keras import datasets, layers, models

testRatio = 0.2
valRatio = 0.2
path='data'
folder=['face', 'nonface']
class_names = ['Face','Nonface']

def label(numpy):
    npList=np.array([])
    for i in range(len(numpy)):
        if numpy[i]=='face':
            npList=np.append(npList,[0])
        else:
            npList=np.append(npList,[1])
    return npList
def file():

    ############################

    images = []  # LIST CONTAINING ALL THE IMAGES
    classNo = []  # LIST CONTAINING ALL THE CORRESPONDING CLASS ID OF IMAGES
    myList = os.listdir(path)
    print("Total Classes Detected:", len(myList))
    noOfClasses = len(myList)
    print("Importing Classes .......")
    for x in folder:
        myPicList = os.listdir(path + "/" + x)
        for y in myPicList:
            curImg = cv.imread(path + "/" + x + "/" + y)
            curImg = cv.resize(curImg, (32, 32))
            images.append(curImg)
            classNo.append(x)

        print(x, end=" ")

    print(" ")

    print("Total Images in Images List = ", len(images))
    print("Total IDS in classNo List= ", len(classNo))
    #######################
    #### CONVERT TO NUMPY ARRAY
    images = np.array(images)
    classNo = np.array(classNo)


    #### SPLITTING THE DATA
    X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
    #X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=valRatio)
    print(len(X_train) )
    print(len(X_test) )
    print(len(y_train) )
    print(len(y_test) )

    ####################
    (training_images, training_labels), (testing_images, testing_labels) = (X_train,label(y_train)), (X_test,label(y_test))
    training_images, testing_images = training_images/255, testing_images/255
    return (training_images, training_labels), (testing_images, testing_labels)


#Building the neural network
def Train_Model():
    model = models.Sequential()#Definim o retea secventiala
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3))) #Adaugam un strat de input conv
    model.add(layers.MaxPooling2D((2,2))) #Simplifica rezultatul si reduce ca si informatia esentiala
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))#output layer doar doua posibiliatati face or nonface

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(training_images, training_labels, epochs=25, validation_data=(testing_images, testing_labels))

    loss, accuracy = model.evaluate(testing_images, testing_labels)
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")

    model.save("image_classifier.model")

#############Train the Model#####################

#(training_images, training_labels), (testing_images, testing_labels)= file()   #Spliting the data
#Train_Model()


def detect_faces(paths):


    for path in paths:
        if os.path.isfile(path):
            img = cv.imread(path)
            detect_face(img)



def detect_face( img ):
    model = models.load_model('image_classifier.model')
    img_copy=img
    img = cv.resize(img, (32, 32))
    img=img/255
    height, width, e = img.shape

    prediction = model.predict(np.array([img]))
    index = np.argmax(prediction)
    print(f'Prediction is {class_names[index]}')
    if class_names[index]:
        fig, ax = plt.subplots(1)
        # Display the image
        ax.imshow(img_copy,label=class_names[index])
        # Create a Rectangle patch
        rect = matplotlib.patches.Rectangle((0, 0), height-1, width-1, linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.show()



detect_faces(['test/vg01.png','test/potato.jpg','test/sc02.png','test/sc01.png','test/hudark3.jpg','test/hudark2.jpg','test/easy_2_1111.jpg','test/deer1.jpg','test/cat1.jpg','test/gorila1.jpg','test/car1.jpg'])













