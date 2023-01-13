# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'comvisnew.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QAction, QMainWindow, QSlider, QPushButton, QToolTip, QApplication

import os
import cv2

import sys
import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import PIL
from skimage import data
from skimage.color import rgb2gray


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1562, 727)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setMaximumSize(QtCore.QSize(16777215, 50))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei Light")
        self.frame.setFont(font)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setFamily("Rubik")
        font.setPointSize(15)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout_3.addWidget(self.label)
        self.verticalLayout.addWidget(self.frame)
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame_2)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame_3 = QtWidgets.QFrame(self.frame_2)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame_3)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.frame_3)
        self.label_2.setMaximumSize(QtCore.QSize(500, 500))
        self.label_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_2.setScaledContents(True)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.pushButton = QtWidgets.QPushButton(self.frame_3)
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout_2.addWidget(self.pushButton)
        self.pushButton_2 = QtWidgets.QPushButton(self.frame_3)
        self.pushButton_2.setObjectName("pushButton_2")
        self.verticalLayout_2.addWidget(self.pushButton_2)
        self.horizontalLayout.addWidget(self.frame_3)
        self.frame_4 = QtWidgets.QFrame(self.frame_2)
        self.frame_4.setMaximumSize(QtCore.QSize(250, 16777215))
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.frame_4)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label_3 = QtWidgets.QLabel(self.frame_4)
        self.label_3.setMaximumSize(QtCore.QSize(16777215, 100))
        self.label_3.setFrameShape(QtWidgets.QFrame.Box)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_6.addWidget(self.label_3)
        self.widget = QtWidgets.QWidget(self.frame_4)
        self.widget.setObjectName("widget")
        self.verticalLayout_6.addWidget(self.widget)
        self.listWidget_3 = QtWidgets.QListWidget(self.frame_4)
        self.listWidget_3.setMaximumSize(QtCore.QSize(250, 250))
        self.listWidget_3.setObjectName("listWidget_3")
        self.verticalLayout_6.addWidget(self.listWidget_3)
        self.horizontalLayout.addWidget(self.frame_4)
        self.frame_5 = QtWidgets.QFrame(self.frame_2)
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.frame_5)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.frame_7 = QtWidgets.QFrame(self.frame_5)
        self.frame_7.setMaximumSize(QtCore.QSize(500, 500))
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.frame_7)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_4 = QtWidgets.QLabel(self.frame_7)
        self.label_4.setMaximumSize(QtCore.QSize(400, 400))
        self.label_4.setFrameShape(QtWidgets.QFrame.Box)
        self.label_4.setScaledContents(True)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_3.addWidget(self.label_4)
        self.verticalLayout_4.addWidget(self.frame_7)
        self.frame_8 = QtWidgets.QFrame(self.frame_5)
        self.frame_8.setMaximumSize(QtCore.QSize(16777215, 150))
        self.frame_8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.gridLayout = QtWidgets.QGridLayout(self.frame_8)
        self.gridLayout.setObjectName("gridLayout")
        self.pushButton_3 = QtWidgets.QPushButton(self.frame_8)
        self.pushButton_3.setObjectName("pushButton_3")
        self.gridLayout.addWidget(self.pushButton_3, 0, 0, 1, 1)
        self.pushButton_4 = QtWidgets.QPushButton(self.frame_8)
        self.pushButton_4.setObjectName("pushButton_4")
        self.gridLayout.addWidget(self.pushButton_4, 0, 1, 1, 1)
        self.pushButton_6 = QtWidgets.QPushButton(self.frame_8)
        self.pushButton_6.setObjectName("pushButton_6")
        self.gridLayout.addWidget(self.pushButton_6, 1, 0, 1, 1)
        self.pushButton_5 = QtWidgets.QPushButton(self.frame_8)
        self.pushButton_5.setObjectName("pushButton_5")
        self.gridLayout.addWidget(self.pushButton_5, 1, 1, 1, 1)
        self.pushButton_7 = QtWidgets.QPushButton(self.frame_8)
        self.pushButton_7.setObjectName("pushButton_7")
        self.gridLayout.addWidget(self.pushButton_7, 2, 0, 1, 1)
        self.pushButton_8 = QtWidgets.QPushButton(self.frame_8)
        self.pushButton_8.setObjectName("pushButton_8")
        self.gridLayout.addWidget(self.pushButton_8, 2, 1, 1, 1)
        self.verticalLayout_4.addWidget(self.frame_8)
        self.horizontalLayout.addWidget(self.frame_5)
        self.frame_6 = QtWidgets.QFrame(self.frame_2)
        self.frame_6.setMaximumSize(QtCore.QSize(250, 16777215))
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.frame_6)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.label_5 = QtWidgets.QLabel(self.frame_6)
        self.label_5.setMaximumSize(QtCore.QSize(16777215, 100))
        self.label_5.setFrameShape(QtWidgets.QFrame.Box)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_5.addWidget(self.label_5)
        self.widget_2 = QtWidgets.QWidget(self.frame_6)
        self.widget_2.setObjectName("widget_2")
        self.verticalLayout_5.addWidget(self.widget_2)
        self.listWidget_2 = QtWidgets.QListWidget(self.frame_6)
        self.listWidget_2.setMaximumSize(QtCore.QSize(250, 250))
        self.listWidget_2.setObjectName("listWidget_2")
        self.verticalLayout_5.addWidget(self.listWidget_2)
        self.horizontalLayout.addWidget(self.frame_6)
        self.verticalLayout.addWidget(self.frame_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1562, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.imgName = ''

        #Function Button
        self.pushButton.clicked.connect(self.openIMG)
        self.pushButton_2.clicked.connect(self.propertiesIMG)
        self.pushButton_3.clicked.connect(self.negasiIMG)
        self.pushButton_4.clicked.connect(self.pengembanganIMG)
        self.pushButton_5.clicked.connect(self.flippingHIMG)
        self.pushButton_6.clicked.connect(self.flippingVIMG)
        self.pushButton_7.clicked.connect(self.erosiIMG)
        self.pushButton_8.clicked.connect(self.dilateIMG)



    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Computer Viziong"))
        self.label_2.setText(_translate("MainWindow", "TextLabel"))
        self.pushButton.setText(_translate("MainWindow", "Open Image"))
        self.pushButton_2.setText(_translate("MainWindow", "Image Propertires"))
        self.label_3.setText(_translate("MainWindow", "TextLabel"))
        self.label_4.setText(_translate("MainWindow", "TextLabel"))
        self.pushButton_3.setText(_translate("MainWindow", "Negasi"))
        self.pushButton_4.setText(_translate("MainWindow", "Pengembangan"))
        self.pushButton_6.setText(_translate("MainWindow", "Flipping Vertical"))
        self.pushButton_5.setText(_translate("MainWindow", "Flipping Horizontal"))
        self.pushButton_7.setText(_translate("MainWindow", "Erosi"))
        self.pushButton_8.setText(_translate("MainWindow", "Dilasi"))
        self.label_5.setText(_translate("MainWindow", "TextLabel"))


    def openIMG(self):
        self.listWidget_3.clear()
        self.label_3.clear()
        plt.close()

        self.imgName, imgType = QFileDialog.getOpenFileName(self.centralwidget, "Open Image", "",
                                                              "*.jpg;;All Files(*)")
        jpg = QPixmap(self.imgName)

        pictureBox1 = self.label_2
        pictureBox1.setPixmap(jpg)



    def propertiesIMG(self):

        self.filename = os.path.basename(self.imgName)
        cv_img = cv2.imread(self.imgName, cv2.IMREAD_UNCHANGED)
        size = cv_img.shape
        bit = cv_img.dtype
        print(bit)
        # self.bit_print = print(bit)
        self.label_3.setText('Name File : ' + self.filename +
                             '\nDimension : ' + str(size) +
                             '\nHeight : ' + str(size[1]) +
                             '\nWidth : ' + str(size[0]) +
                             '\nDepth Color : ' + str(size[2]) +
                             '\nBit Depth : {}'.format(str(bit)[4]) + '-bit')


        self.listWidget_3.addItem('RGB per pixel :')

        h, w, c = size
        if h <= 20 and w <= 20 and c <=20:
            for x in range(h):
                for y in range(w):
                    self.listWidget_3.addItem(' {},{}\t: {}'.format(x, y, cv_img[x, y]))
        else:
            for x in range(50):
                for y in range(50):
                    self.listWidget_3.addItem(' {},{}\t: {}'.format(x, y, cv_img[x, y]))
        # self.listView.addItem('({},{},{})'.format(h, w, c))


        # Histogram
        hist1 = cv2.calcHist([cv_img],[0],None,[256],[0,256])
        hist2 = cv2.calcHist([cv_img],[1],None,[256],[0,256])
        hist3 = cv2.calcHist([cv_img],[2],None,[256],[0,256])
        
        fig, (ax0,ax1,ax2) = plt.subplots(3,1,figsize=(8,4))
        fig.tight_layout(pad=3)

        ax0.title.set_text('Red')
        ax0.set_xlim([0,256])
        ax0.plot(hist3,color='red')

        ax1.title.set_text('Green')
        ax1.set_xlim([0,256])
        ax1.plot(hist2,color='green')

        ax2.title.set_text('Blue')
        ax2.set_xlim([0,256])
        ax2.plot(hist1,color='blue')

        plt.show()

    def negasiIMG(self):
        self.listWidget_2.clear()
        self.label_5.clear()
        # ========== Open Image ===========
        CITRA = Image.open(self.imgName)
        # size_citraNegasi = citraNegasi.size

        ukuran_horizontal = CITRA.size[0]
        ukuran_vertikal = CITRA.size[1]


        PIXEL = CITRA.load()

        for x in range(ukuran_horizontal):
            for y in range(ukuran_vertikal):
                R = 255 - PIXEL[x, y][0]
                G = 255 - PIXEL[x, y][1]
                B = 255 - PIXEL[x, y][2]
                PIXEL[x, y] = (R, G, B)

        CITRA.save('gambar_negatif.jpg')

        
        negasiIMG = QPixmap('gambar_negatif.jpg')

        pictureBox1 = self.label_4
        pictureBox1.setPixmap(negasiIMG)


        # ============ NEGASI NEW ============

        CITRA_BINER = Image.open(self.imgName).convert('1')
        PIXEL_BINER = CITRA_BINER.load()

        ukuran_horizontal = CITRA_BINER.size[0]
        ukuran_vertikal = CITRA_BINER.size[1]

        for x in range(ukuran_horizontal):
            for y in range(ukuran_vertikal):
                PIXEL_BINER[x,y] = 255 - PIXEL_BINER[x,y]

        CITRA_BINER.save('gambar_negatif.jpg')

        # ============== Properties ===============

        self.fileNegasi = os.path.basename('gambar_negatif.jpg')
        cv_img = cv2.imread(self.fileNegasi, cv2.IMREAD_UNCHANGED)
        size = cv_img.shape
        bit = cv_img.dtype
        # self.bit_print = print(bit)
        self.label_5.setText('Name File : ' + self.fileNegasi +
                             '\nDimension : ' + str(size) +
                             '\nHeight : ' + str(size[1]) +
                             '\nWidth : ' + str(size[0]) +
                             '\nDepth Color : ' + str(size[2]) +
                             '\nBit Depth : {}'.format(str(bit)[4]) + '-bit')


        self.listWidget_2.addItem('RGB per pixel :')

        h, w, c = size
        if h <= 20 and w <= 20 and c <=20:
            for x in range(h):
                for y in range(w):
                    self.listWidget_2.addItem(' {},{}\t: {}'.format(x, y, cv_img[x, y]))
        else:
            for x in range(50):
                for y in range(50):
                    self.listWidget_2.addItem(' {},{}\t: {}'.format(x, y, cv_img[x, y]))
        # self.listView.addItem('({},{},{})'.format(h, w, c))

        plt.close()

        hist1 = cv2.calcHist([cv_img],[0],None,[256],[0,256])
        hist2 = cv2.calcHist([cv_img],[1],None,[256],[0,256])
        hist3 = cv2.calcHist([cv_img],[2],None,[256],[0,256])

        fig, (ax0,ax1,ax2) = plt.subplots(3,1,figsize=(8,4))
        fig.tight_layout(pad=3)

        ax0.title.set_text('Red')
        ax0.set_xlim([0,256])
        ax0.plot(hist3,color='red')

        ax1.title.set_text('Green')
        ax1.set_xlim([0,256])
        ax1.plot(hist2,color='green')

        ax2.title.set_text('Blue')
        ax2.set_xlim([0,256])
        ax2.plot(hist1,color='blue')

        plt.show()

    def pengembanganIMG(self):
        # fungsi citra biner
        self.listWidget_2.clear()
        self.label_5.clear()

        # konversi gambar RGB ke grayscale
        # https://stackoverflow.com/a/18778280/9157799
        CITRA_GRAYSCALE = Image.open(self.imgName).convert('L')
        PIXEL_GRAYSCALE = CITRA_GRAYSCALE.load()
        nilai_ambang = 200

        ukuran_horizontal = CITRA_GRAYSCALE.size[0]
        ukuran_vertikal = CITRA_GRAYSCALE.size[1]

        for x in range(ukuran_horizontal):
            for y in range(ukuran_vertikal):
                if PIXEL_GRAYSCALE[x, y] < nilai_ambang:
                    PIXEL_GRAYSCALE[x, y] = 0
                else:
                    PIXEL_GRAYSCALE[x, y] = 255

        nama_setelah_disave = 'gambar_biner_' + str(nilai_ambang) + '.jpg'
        CITRA_GRAYSCALE.save(nama_setelah_disave)

        pengembanganIMG = QPixmap(nama_setelah_disave)

        pictureBox1 = self.label_4
        pictureBox1.setPixmap(pengembanganIMG)

        # ============ Properties ===========
        self.filePengembangan = os.path.basename(nama_setelah_disave)
        cv_img = cv2.imread(self.filePengembangan, cv2.IMREAD_UNCHANGED)
        size = cv_img.shape
        bit = cv_img.dtype
        # self.bit_print = print(bit)
        self.label_5.setText('Name File : ' + self.filePengembangan +
                             '\nDimension : ' + str(size) +
                             '\nHeight : ' + str(size[1]) +
                             '\nWidth : ' + str(size[0]) +
                              
                             '\nBit Depth : {}'.format(str(bit)[4]) + '-bit')


        self.listWidget_2.addItem('RGB per pixel :')

        h, w = size
        if h <= 20 and w <= 20:
            for x in range(h):
                for y in range(w):
                    self.listWidget_2.addItem(' {},{}\t: {}'.format(x, y, cv_img[x, y]))
        else:
            for x in range(50):
                for y in range(50):
                    self.listWidget_2.addItem(' {},{}\t: {}'.format(x, y, cv_img[x, y]))
        # self.listView.addItem('({},{},{})'.format(h, w, c))

        plt.close()

        hist1 = cv2.calcHist([cv_img],[0],None,[256],[0,256])

        fig, (ax2) = plt.subplots(1,1,figsize=(8,4))
        fig.tight_layout(pad=3)

        ax2.title.set_text('Black')
        ax2.set_xlim([0,256])
        ax2.plot(hist1,color='black')

        plt.show()

    def flippingHIMG(self):
        self.listWidget_2.clear()
        self.label_5.clear()

        # ============ Flipping Horizontal ============
        CITRA = Image.open(self.imgName)
        PIXEL = CITRA.load()
        ukuran_horizontal = CITRA.size[0]
        ukuran_vertikal = CITRA.size[1]

        CITRA_BARU = Image.new("RGB", (ukuran_horizontal, ukuran_vertikal))
        PIXEL_BARU = CITRA_BARU.load()

        for x in range(ukuran_horizontal):
            for y in range(ukuran_vertikal):
                PIXEL_BARU[x, y] = PIXEL[ukuran_horizontal - 1 - x, y]

        CITRA_BARU.save('flippingHIMG.jpg')

        flippingHIMGJPG = QPixmap('flippingHIMG.jpg')

        pictureBox1 = self.label_4
        pictureBox1.setPixmap(flippingHIMGJPG)

        # =========== PROPERTIES =============
        self.fileFlippingH = os.path.basename('flippingHIMG.jpg')
        cv_img = cv2.imread(self.fileFlippingH, cv2.IMREAD_UNCHANGED)
        size = cv_img.shape
        bit = cv_img.dtype
        # self.bit_print = print(bit)
        self.label_5.setText('Name File : ' + self.fileFlippingH +
                             '\nDimension : ' + str(size) +
                             '\nHeight : ' + str(size[1]) +
                             '\nWidth : ' + str(size[0]) +
                             '\nDepth Color : ' + str(size[2]) +
                             '\nBit Depth : {}'.format(str(bit)[4]) + '-bit')


        self.listWidget_2.addItem('RGB per pixel :')

        h, w, c = size
        if h <= 20 and w <= 20 and c <=20:
            for x in range(h):
                for y in range(w):
                    self.listWidget_2.addItem(' {},{}\t: {}'.format(x, y, cv_img[x, y]))
        else:
            for x in range(50):
                for y in range(50):
                    self.listWidget_2.addItem(' {},{}\t: {}'.format(x, y, cv_img[x, y]))
        # self.listView.addItem('({},{},{})'.format(h, w, c))

        plt.close()

        hist1 = cv2.calcHist([cv_img],[0],None,[256],[0,256])
        hist2 = cv2.calcHist([cv_img],[1],None,[256],[0,256])
        hist3 = cv2.calcHist([cv_img],[2],None,[256],[0,256])

        fig, (ax0,ax1,ax2) = plt.subplots(3,1,figsize=(8,4))
        fig.tight_layout(pad=3)

        ax0.title.set_text('Red')
        ax0.set_xlim([0,256])
        ax0.plot(hist3,color='red')

        ax1.title.set_text('Green')
        ax1.set_xlim([0,256])
        ax1.plot(hist2,color='green')

        ax2.title.set_text('Blue')
        ax2.set_xlim([0,256])
        ax2.plot(hist1,color='blue')

        plt.show()

    def flippingVIMG(self):
        self.listWidget_2.clear()
        self.label_5.clear()
        CITRA = Image.open(self.imgName)
        PIXEL = CITRA.load()
        ukuran_horizontal = CITRA.size[0]
        ukuran_vertikal = CITRA.size[1]

        CITRA_BARU = Image.new("RGB", (ukuran_horizontal, ukuran_vertikal))
        PIXEL_BARU = CITRA_BARU.load()

        for x in range(ukuran_horizontal):
            for y in range(ukuran_vertikal):
                PIXEL_BARU[x, y] = PIXEL[x, ukuran_horizontal - 1 - y]

        CITRA_BARU.save('flippingVIMG.jpg')

        flippingHIMGJPG = QPixmap('flippingVIMG.jpg')

        pictureBox1 = self.label_4
        pictureBox1.setPixmap(flippingHIMGJPG)

        # =========== PROPERTIES =============
        self.fileFlippingV = os.path.basename('flippingVIMG.jpg')
        cv_img = cv2.imread(self.fileFlippingV, cv2.IMREAD_UNCHANGED)
        size = cv_img.shape
        bit = cv_img.dtype
        # self.bit_print = print(bit)
        self.label_5.setText('Name File : ' + self.fileFlippingV +
                             '\nDimension : ' + str(size) +
                             '\nHeight : ' + str(size[1]) +
                             '\nWidth : ' + str(size[0]) +
                             '\nDepth Color : ' + str(size[2]) +
                             '\nBit Depth : {}'.format(str(bit)[4]) + '-bit')


        self.listWidget_2.addItem('RGB per pixel :')

        h, w, c = size
        if h <= 20 and w <= 20 and c <=20:
            for x in range(h):
                for y in range(w):
                    self.listWidget_2.addItem(' {},{}\t: {}'.format(x, y, cv_img[x, y]))
        else:
            for x in range(50):
                for y in range(50):
                    self.listWidget_2.addItem(' {},{}\t: {}'.format(x, y, cv_img[x, y]))
        # self.listView.addItem('({},{},{})'.format(h, w, c))

        plt.close()

        hist1 = cv2.calcHist([cv_img],[0],None,[256],[0,256])
        hist2 = cv2.calcHist([cv_img],[1],None,[256],[0,256])
        hist3 = cv2.calcHist([cv_img],[2],None,[256],[0,256])

        fig, (ax0,ax1,ax2) = plt.subplots(3,1,figsize=(8,4))
        fig.tight_layout(pad=3)

        ax0.title.set_text('Red')
        ax0.set_xlim([0,256])
        ax0.plot(hist3,color='red')

        ax1.title.set_text('Green')
        ax1.set_xlim([0,256])
        ax1.plot(hist2,color='green')

        ax2.title.set_text('Blue')
        ax2.set_xlim([0,256])
        ax2.plot(hist1,color='blue')

        plt.show()

    def dilateIMG(self):
        self.listWidget_2.clear()
        self.label_5.clear()

        self.filename = os.path.basename(self.imgName)
        cv_img = cv2.imread(self.imgName, cv2.IMREAD_UNCHANGED)
        # Balikin Citra
        cv_img = ~cv_img

        # =========== Dilasi IMG ===========

        ret, thresh = cv2.threshold(cv_img, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5,5),np.uint8)
        dilasi = cv2.dilate(thresh,kernel,iterations = 1)
        
        printDilasi = cv2.imwrite('DilasiImage.jpg',dilasi)

        dilasiCitra = QPixmap('DilasiImage.jpg')

        pictureBox1 = self.label_4
        pictureBox1.setPixmap(dilasiCitra)

        # =========== PROPERTIES =============
        self.fileFlippingV = os.path.basename('DilasiImage.jpg')
        cv_img = cv2.imread(self.fileFlippingV, cv2.IMREAD_UNCHANGED)
        size = cv_img.shape
        bit = cv_img.dtype
        # self.bit_print = print(bit)
        self.label_5.setText('Name File : ' + self.fileFlippingV +
                             '\nDimension : ' + str(size) +
                             '\nHeight : ' + str(size[1]) +
                             '\nWidth : ' + str(size[0]) +
                             '\nDepth Color : ' + str(size[2]) +
                             '\nBit Depth : {}'.format(str(bit)[4]) + '-bit')


        self.listWidget_2.addItem('RGB per pixel :')

        h, w, c = size
        if h <= 20 and w <= 20 and c <=20:
            for x in range(h):
                for y in range(w):
                    self.listWidget_2.addItem(' {},{}\t: {}'.format(x, y, cv_img[x, y]))
        else:
            for x in range(50):
                for y in range(50):
                    self.listWidget_2.addItem(' {},{}\t: {}'.format(x, y, cv_img[x, y]))
        # self.listView.addItem('({},{},{})'.format(h, w, c))

        plt.close()

        hist1 = cv2.calcHist([cv_img],[0],None,[256],[0,256])
        hist2 = cv2.calcHist([cv_img],[1],None,[256],[0,256])
        hist3 = cv2.calcHist([cv_img],[2],None,[256],[0,256])

        fig, (ax0,ax1,ax2) = plt.subplots(3,1,figsize=(8,4))
        fig.tight_layout(pad=3)

        ax0.title.set_text('Red')
        ax0.set_xlim([0,256])
        ax0.plot(hist3,color='red')

        ax1.title.set_text('Green')
        ax1.set_xlim([0,256])
        ax1.plot(hist2,color='green')

        ax2.title.set_text('Blue')
        ax2.set_xlim([0,256])
        ax2.plot(hist1,color='blue')

        plt.show()

    def erosiIMG(self):
        self.listWidget_2.clear()
        self.label_5.clear()

        self.filename = os.path.basename(self.imgName)
        cv_img = cv2.imread(self.imgName, cv2.IMREAD_UNCHANGED)
        # Balikin citra
        cv_img = ~cv_img

        ret, thresh = cv2.threshold(cv_img, 200 , 255, cv2.THRESH_BINARY)
        kernel = np.ones((5,5),np.uint8)
        erosi = cv2.erode(thresh,kernel,iterations = 1)

        printErosi = cv2.imwrite('ErosiImage.jpg',erosi)
        

        erosiCitra = QPixmap('ErosiImage.jpg')

        pictureBox1 = self.label_4
        pictureBox1.setPixmap(erosiCitra)


        # =========== PROPERTIES =============
        self.fileFlippingV = os.path.basename('ErosiImage.jpg')
        cv_img = cv2.imread(self.fileFlippingV, cv2.IMREAD_UNCHANGED)
        size = cv_img.shape
        bit = cv_img.dtype
        # self.bit_print = print(bit)
        self.label_5.setText('Name File : ' + self.fileFlippingV +
                             '\nDimension : ' + str(size) +
                             '\nHeight : ' + str(size[1]) +
                             '\nWidth : ' + str(size[0]) +
                             '\nDepth Color : ' + str(size[2]) +
                             '\nBit Depth : {}'.format(str(bit)[4]) + '-bit')


        self.listWidget_2.addItem('RGB per pixel :')

        h, w, c = size
        if h <= 20 and w <= 20 and c <=20:
            for x in range(h):
                for y in range(w):
                    self.listWidget_2.addItem(' {},{}\t: {}'.format(x, y, cv_img[x, y]))
        else:
            for x in range(50):
                for y in range(50):
                    self.listWidget_2.addItem(' {},{}\t: {}'.format(x, y, cv_img[x, y]))
        # self.listView.addItem('({},{},{})'.format(h, w, c))

        plt.close()

        hist1 = cv2.calcHist([cv_img],[0],None,[256],[0,256])
        hist2 = cv2.calcHist([cv_img],[1],None,[256],[0,256])
        hist3 = cv2.calcHist([cv_img],[2],None,[256],[0,256])

        fig, (ax0,ax1,ax2) = plt.subplots(3,1,figsize=(8,4))
        fig.tight_layout(pad=3)

        ax0.title.set_text('Red')
        ax0.set_xlim([0,256])
        ax0.plot(hist3,color='red')

        ax1.title.set_text('Green')
        ax1.set_xlim([0,256])
        ax1.plot(hist2,color='green')

        ax2.title.set_text('Blue')
        ax2.set_xlim([0,256])
        ax2.plot(hist1,color='blue')

        plt.show()

    def konvulsiIMG(self):
        print('Hello')

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())