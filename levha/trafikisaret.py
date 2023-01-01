# -*- coding: utf-8 -*-
"""
Created on Sun May 30 16:42:20 2021

@author: ckmz
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from sklearn.metrics import accuracy_score
from keras.datasets import mnist
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from sklearn.utils import shuffle
from PyQt5.uic import loadUiType
from trafiktasarim import Ui_Dialog
from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt,QTimer,QTime,QAbstractTableModel
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QTableView, QFileDialog,QMessageBox,QMainWindow, QLabel, QGridLayout,QDesktopWidget, QWidget,QTableWidget,QTableWidgetItem,QHeaderView,QGraphicsScene,QGraphicsPixmapItem
from keras.models import model_from_json
from keras.preprocessing import image

import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image

import numpy
#load the trained model to classify sign
from keras.models import load_model

class MainWindow(QWidget,Ui_Dialog):
    dataset_file_path = ""
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)  
        self.setupUi(self)
        self.pushButton.clicked.connect(self.modelolustur)
        self.pushButton_2.clicked.connect(self.resimyukle)
        self.pushButton_3.clicked.connect(self.sorgula)
        
    def resimyukle(self):
        self.dosyam, isim=QtWidgets.QFileDialog.getOpenFileName(None,"Fotograf Seç","","Veri Seti Türü(*.png)")
        print(self.dosyam)
        img=cv2.imread(self.dosyam)
        self.pixmap = QPixmap(self.dosyam)
        self.label_12.setPixmap(self.pixmap)
        
    def sorgula(self):
        classes = { 1:'Hız Sınırı (20km/s)',
                            2:'Hız Sınırı (30km/s)', 
                            3:'Hız Sınırı (50km/s)', 
                            4:'Hız Sınırı (60km/s)', 
                            5:'Hız Sınırı (70km/s)', 
                            6:'Hız Sınırı (80km/s)', 
                            7:'Hız Sınırı Sonu (80km/s)', 
                            8:'Hız Sınırı (100km/s)', 
                            9:'Hız Sınırı (120km/s)', 
                            10:'Sol Şerit Yasak', 
                            11:'3,5 Tonun Üzerindeki Araçlar İçin Sol Şerit Yasak', 
                            12:'Kavşakta Geçiş Hakkı', 
                            13:'Öncelikli Yol', 
                            14:'Yol Ver', 
                            15:'Dur', 
                            16:'Araç Giremez', 
                            17:'3.5 Ton Üzeri Araç İçin Yasak', 
                            18:'Giriş Yasak', 
                            19:'Genel Uyarı', 
                            20:'Solda Tehlikeli Viraj', 
                            21:'Sağda Tehlikeli Viraj', 
                            22:'Çift Viraj', 
                            23:'Engebe Yol', 
                            24:'Kaygan Yol', 
                            25:'Yol Sağda Daralır', 
                            26:'Yol Çalışması', 
                            27:'Trafik Işıkları', 
                            28:'Yaya Geçidi', 
                            29:'Okul Geçidi', 
                            30:'Bisikletli Geçişi', 
                            31:'Buz Ve Kara Karşı Dikkat!',
                            32:'Vahşi Hayvanlar ÇIkabilir', 
                            33:'Tüm Hız Ve Geçiş Sınırlarının Sonu', 
                            34:'Sadece Sağa Dönüş', 
                            35:'Sadece Sola Dönüş', 
                            36:'Düz İlerleyin', 
                            37:'dDz Ya Da Sağa Gidin', 
                            38:'Düz Ya Da sSla Gİdin', 
                            39:'Sağdan Gidin', 
                            40:'Soldan Gidin', 
                            41:'Zorunlu Döner Kavşak', 
                            42:'Sollama Yapmak Yasak', 
                            43:'3,5 Tonun Üzerindeki Araçlar İçin Sollama Yasak' }
        json_file = open('levhamodel.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)

        model.load_weights("levhamodel.h5")
        file_path= str(self.dosyam)
        image = Image.open(file_path)
        image = image.resize((30,30))
        image = numpy.expand_dims(image, axis=0)
        image = numpy.array(image)
        pred = model.predict_classes([image])[0]
        sign = classes[pred+1]
        print(sign)
        self.label_4.setText(str(sign))
        

    def modelolustur(self):
        self.data = []
        self.labels = []
        self.classes = 43
        self.dizi = os.getcwd()

        for i in range(self.classes):
            self.yol = os.path.join(self.dizi,'train',str(i))
            images = os.listdir(self.yol)

            for a in images:
                try:
                    image = Image.open(self.yol + '\\'+ a)
                    image = image.resize((30,30))
                    image = np.array(image)
                    self.data.append(image)
                    self.labels.append(i)
                except:
                    print("Error loading image")
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        print(self.data.shape, self.labels.shape)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.labels, test_size=0.2, random_state=42)
        self.y_train = to_categorical(self.y_train, 43)
        self.y_test = to_categorical(self.y_test, 43)
        print ( "X TRAİN:",self.X_train.shape)
        self.label_2.setText(str(self.X_train.shape))
        
        

        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=self.X_train.shape[1:]))
        model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(43, activation='softmax'))

#Compilation of the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        epochs = 15
        history = model.fit(self.X_train, self.y_train, batch_size=32, epochs=epochs, validation_data=(self.X_test, self.y_test))


        model_json = model.to_json()
        with open("levhamodel.json", "w") as json_file:
            json_file.write(model_json) 
            model.save_weights("levhamodel.h5") 
        print("Saved model to disk")


        plt.figure(0)
        plt.plot(history.history['accuracy'], label='training accuracy')
        plt.plot(history.history['val_accuracy'], label='val accuracy')
        plt.title('Accuracy')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig("./accuracy.png")
        pixmap = QPixmap("./accuracy.png")
        self.label_10.setPixmap(pixmap)
        plt.show()

        plt.figure(1)
        plt.plot(history.history['loss'], label='training loss')
        plt.plot(history.history['val_loss'], label='val loss')
        plt.title('Loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig("./loss.png")
        pixmap = QPixmap("./loss.png")
        self.label_11.setPixmap(pixmap)
        plt.show()
        
        

        y_test = pd.read_csv('Test.csv')

        labels = y_test["ClassId"].values
        imgs = y_test["Path"].values

        data=[]

        for img in imgs:
            image = Image.open(img)
            image = image.resize((30,30))
            data.append(np.array(image))

        X_test=np.array(data)

        pred = model.predict_classes(X_test)


        print(accuracy_score(labels, pred))
        self.label_8.setText(str(accuracy_score(labels, pred)))
















import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from trafikisaret import MainWindow

def main():
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()




