from PyQt5.uic import loadUi

from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QAction, QMainWindow, QSlider, QPushButton, QToolTip, QApplication

import glob
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

class Ui_MainWindow(QMainWindow):
    
    def __init__(self):
        QMainWindow.__init__(self)
        loadUi('comvis.ui', self)

        # Declaration
        self.image = ''

        # Function Button
        self.pushButton.clicked.connect(self.openImg)
        self.pushButton_2.clicked.connect(self.negasiIMG)
        self.pushButton_3.clicked.connect(self.pengembanganIMG)
        self.pushButton_4.clicked.connect(self.flippingHIMG)
        self.pushButton_5.clicked.connect(self.flippingVIMG)
        self.pushButton_6.clicked.connect(self.flippingHVIMG)   
        self.pushButton_8.clicked.connect(self.negasiIMGErosi)   
        self.pushButton_9.clicked.connect(self.erosiIMG)           
        self.pushButton_10.clicked.connect(self.negasiIMGDilasi) 
        self.pushButton_11.clicked.connect(self.dilasiIMG)   

        self.pushButton_7.clicked.connect(self.sobelXEdge)     
        self.pushButton_16.clicked.connect(self.sobelYEdge)
        self.pushButton_17.clicked.connect(self.laplacianEdge)    
        self.pushButton_13.clicked.connect(self.cannyEdge)    

        self.pushButton_15.clicked.connect(self.circleHough)       
        self.pushButton_14.clicked.connect(self.lineHough)          

    def openImg(self):
        
        files = glob.glob('*.jpg')
        for file in files:
            os.remove(file)


        plt.close()

        self.listWidget.clear()
        self.label_6.clear()

        self.imgName, imgType = QFileDialog.getOpenFileName(self.centralwidget, "Open Image", "",
                                                              "*.jpg;;All Files(*)")
        jpg = QPixmap(self.imgName)

        pictureBox1 = self.label_5
        pictureBox1.setPixmap(jpg)



        self.filename = os.path.basename(self.imgName)
        cv_img = cv2.imread(self.imgName, cv2.IMREAD_UNCHANGED)
        size = cv_img.shape
        bit = cv_img.dtype
        print(bit)
        # self.bit_print = print(bit)
        self.label_6.setText('Name File : ' + self.filename +
                             '\nDimension : ' + str(size) +
                             '\nHeight : ' + str(size[1]) +
                             '\nWidth : ' + str(size[0]) +
                             '\nDepth Color : ' + str(size[2]) +
                             '\nBit Depth : {}'.format(str(bit)[4]) + '-bit')


        self.listWidget.addItem('RGB per pixel :')

        h, w, c = size
        if h <= 20 and w <= 20 and c <=20:
            for x in range(h):
                for y in range(w):
                    self.listWidget.addItem(' {},{}\t: {}'.format(x, y, cv_img[x, y]))
        else:
            for x in range(50):
                for y in range(50):
                    self.listWidget.addItem(' {},{}\t: {}'.format(x, y, cv_img[x, y]))
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

    # =========== Threshold ===========
    def pengembanganIMG(self):
        # fungsi citra biner
        self.listWidget_2.clear()
        self.label_12.clear()

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

        pictureBox1 = self.label_11
        pictureBox1.setPixmap(pengembanganIMG)

        # ============ Properties ===========
        self.filePengembangan = os.path.basename(nama_setelah_disave)
        cv_img = cv2.imread(self.filePengembangan, cv2.IMREAD_UNCHANGED)
        size = cv_img.shape
        bit = cv_img.dtype
        # self.bit_print = print(bit)
        self.label_12.setText('Name File : ' + self.filePengembangan +
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

        fig, (ax2) = plt.subplots(1,1,figsize=(8,6))
        fig.tight_layout(pad=3)

        ax2.title.set_text('Grayscale')
        ax2.set_xlim([0,256])
        ax2.plot(hist1,color='black')

        plt.show()

    # =========== Negasi =============
    def negasiIMG(self):
        self.listWidget_3.clear()
        self.label_9.clear()
        # ========== Open Image ===========
        # ============ NEGASI NEW ============

        CITRA_BINER = Image.open(self.imgName).convert('1')
        PIXEL_BINER = CITRA_BINER.load()

        ukuran_horizontal = CITRA_BINER.size[0]
        ukuran_vertikal = CITRA_BINER.size[1]

        for x in range(ukuran_horizontal):
            for y in range(ukuran_vertikal):
                PIXEL_BINER[x,y] = 255 - PIXEL_BINER[x,y]

        CITRA_BINER.save('gambar_negatif.jpg')

        negasiIMG = QPixmap('gambar_negatif.jpg')

        pictureBox1 = self.label_8
        pictureBox1.setPixmap(negasiIMG)

        # ============== Properties ===============

        self.fileNegasi = os.path.basename('gambar_negatif.jpg')
        cv_img = cv2.imread(self.fileNegasi, cv2.IMREAD_UNCHANGED)
        size = cv_img.shape
        bit = cv_img.dtype
        # self.bit_print = print(bit)
        self.label_9.setText('Name File : ' + self.fileNegasi +
                             '\nDimension : ' + str(size) +
                             '\nHeight : ' + str(size[1]) +
                             '\nWidth : ' + str(size[0]) +
                             '\nBit Depth : {}'.format(str(bit)[4]) + '-bit')


        self.listWidget_3.addItem('RGB per pixel :')

        h, w= size
        if h <= 20 and w <= 20:
            for x in range(h):
                for y in range(w):
                    self.listWidget_3.addItem(' {},{}\t: {}'.format(x, y, cv_img[x, y]))
        else:
            for x in range(50):
                for y in range(50):
                    self.listWidget_3.addItem(' {},{}\t: {}'.format(x, y, cv_img[x, y]))
        # self.listView.addItem('({},{},{})'.format(h, w, c))

        plt.close()

        hist1 = cv2.calcHist([cv_img],[0],None,[256],[0,256])

        fig, (ax2) = plt.subplots(1,1,figsize=(8,6))
        fig.tight_layout(pad=1)

        ax2.title.set_text('Grayscale')
        ax2.set_xlim([0,256])
        ax2.plot(hist1,color='black')

        plt.show()

    # ========= Flipping =============
    def flippingHIMG(self):
        self.listWidget_4.clear()
        self.label_15.clear()

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

        pictureBox1 = self.label_14
        pictureBox1.setPixmap(flippingHIMGJPG)

        # =========== PROPERTIES =============
        self.fileFlippingH = os.path.basename('flippingHIMG.jpg')
        cv_img = cv2.imread(self.fileFlippingH, cv2.IMREAD_UNCHANGED)
        size = cv_img.shape
        bit = cv_img.dtype
        # self.bit_print = print(bit)
        self.label_15.setText('Name File : ' + self.fileFlippingH +
                             '\nDimension : ' + str(size) +
                             '\nHeight : ' + str(size[1]) +
                             '\nWidth : ' + str(size[0]) +
                             '\nDepth Color : ' + str(size[2]) +
                             '\nBit Depth : {}'.format(str(bit)[4]) + '-bit')


        self.listWidget_4.addItem('RGB per pixel :')

        h, w, c = size
        if h <= 20 and w <= 20 and c <=20:
            for x in range(h):
                for y in range(w):
                    self.listWidget_4.addItem(' {},{}\t: {}'.format(x, y, cv_img[x, y]))
        else:
            for x in range(50):
                for y in range(50):
                    self.listWidget_4.addItem(' {},{}\t: {}'.format(x, y, cv_img[x, y]))
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
        self.listWidget_5.clear()
        self.label_18.clear()
        CITRA = Image.open(self.imgName)
        PIXEL = CITRA.load()
        ukuran_horizontal = CITRA.size[0]
        ukuran_vertikal = CITRA.size[1]

        CITRA_BARU = Image.new("RGB", (ukuran_horizontal, ukuran_vertikal))
        PIXEL_BARU = CITRA_BARU.load()

        for x in range(ukuran_horizontal):
            for y in range(ukuran_vertikal):
                PIXEL_BARU[x, y] = PIXEL[x, ukuran_vertikal - 1 - y]

        CITRA_BARU.save('flippingVIMG.jpg')

        flippingHIMGJPG = QPixmap('flippingVIMG.jpg')

        pictureBox1 = self.label_17
        pictureBox1.setPixmap(flippingHIMGJPG)

        # =========== PROPERTIES =============
        self.fileFlippingV = os.path.basename('flippingVIMG.jpg')
        cv_img = cv2.imread(self.fileFlippingV, cv2.IMREAD_UNCHANGED)
        size = cv_img.shape
        bit = cv_img.dtype
        # self.bit_print = print(bit)
        self.label_18.setText('Name File : ' + self.fileFlippingV +
                             '\nDimension : ' + str(size) +
                             '\nHeight : ' + str(size[1]) +
                             '\nWidth : ' + str(size[0]) +
                             '\nDepth Color : ' + str(size[2]) +
                             '\nBit Depth : {}'.format(str(bit)[4]) + '-bit')


        self.listWidget_5.addItem('RGB per pixel :')

        h, w, c = size
        if h <= 20 and w <= 20 and c <=20:
            for x in range(h):
                for y in range(w):
                    self.listWidget_5.addItem(' {},{}\t: {}'.format(x, y, cv_img[x, y]))
        else:
            for x in range(50):
                for y in range(50):
                    self.listWidget_5.addItem(' {},{}\t: {}'.format(x, y, cv_img[x, y]))
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

    def flippingHVIMG(self):
        self.listWidget_6.clear()
        self.label_20.clear()
        CITRA = Image.open(self.imgName)
        PIXEL = CITRA.load()
        ukuran_horizontal = CITRA.size[0]
        ukuran_vertikal = CITRA.size[1]

        # ============ Flipping Horizontal Vertical =============
        CITRA_BARU = Image.new("RGB", (ukuran_horizontal, ukuran_vertikal))
        PIXEL_BARU = CITRA_BARU.load()

        for x in range(ukuran_horizontal):
            for y in range(ukuran_vertikal):
                PIXEL_BARU[x, y] = PIXEL[ukuran_horizontal - 1 - x,ukuran_vertikal - 1 - y]

        CITRA_BARU.save('flippingHVIMG.jpg')

        flippingHIMGJPG = QPixmap('flippingHVIMG.jpg')

        pictureBox1 = self.label_22
        pictureBox1.setPixmap(flippingHIMGJPG)

        # =========== PROPERTIES =============
        self.fileFlippingV = os.path.basename('flippingHVIMG.jpg')
        cv_img = cv2.imread(self.fileFlippingV, cv2.IMREAD_UNCHANGED)
        size = cv_img.shape
        bit = cv_img.dtype
        # self.bit_print = print(bit)
        self.label_20.setText('Name File : ' + self.fileFlippingV +
                             '\nDimension : ' + str(size) +
                             '\nHeight : ' + str(size[1]) +
                             '\nWidth : ' + str(size[0]) +
                             '\nDepth Color : ' + str(size[2]) +
                             '\nBit Depth : {}'.format(str(bit)[4]) + '-bit')


        self.listWidget_6.addItem('RGB per pixel :')

        h, w, c = size
        if h <= 20 and w <= 20 and c <=20:
            for x in range(h):
                for y in range(w):
                    self.listWidget_6.addItem(' {},{}\t: {}'.format(x, y, cv_img[x, y]))
        else:
            for x in range(50):
                for y in range(50):
                    self.listWidget_6.addItem(' {},{}\t: {}'.format(x, y, cv_img[x, y]))
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

    # ============= EROSI ================
    def negasiIMGErosi(self):
        self.listWidget_7.clear()
        self.label_24.clear()
        # ========== Open Image ===========
        # ============ NEGASI NEW ============

        CITRA_BINER = Image.open(self.imgName).convert('1')
        PIXEL_BINER = CITRA_BINER.load()

        ukuran_horizontal = CITRA_BINER.size[0]
        ukuran_vertikal = CITRA_BINER.size[1]

        for x in range(ukuran_horizontal):
            for y in range(ukuran_vertikal):
                PIXEL_BINER[x,y] = 255 - PIXEL_BINER[x,y]

        CITRA_BINER.save('gambar_negatif1.jpg')

        negasiIMG = QPixmap('gambar_negatif1.jpg')

        pictureBox1 = self.label_23
        pictureBox1.setPixmap(negasiIMG)

        # ============== Properties ===============

        self.fileNegasi = os.path.basename('gambar_negatif1.jpg')
        cv_img = cv2.imread(self.fileNegasi, cv2.IMREAD_UNCHANGED)
        size = cv_img.shape
        bit = cv_img.dtype
        # self.bit_print = print(bit)
        self.label_24.setText('Name File : ' + self.fileNegasi +
                             '\nDimension : ' + str(size) +
                             '\nHeight : ' + str(size[1]) +
                             '\nWidth : ' + str(size[0]) +
                             '\nBit Depth : {}'.format(str(bit)[4]) + '-bit')


        self.listWidget_7.addItem('RGB per pixel :')

        h, w= size
        if h <= 20 and w <= 20:
            for x in range(h):
                for y in range(w):
                    self.listWidget_7.addItem(' {},{}\t: {}'.format(x, y, cv_img[x, y]))
        else:
            for x in range(50):
                for y in range(50):
                    self.listWidget_7.addItem(' {},{}\t: {}'.format(x, y, cv_img[x, y]))
        # self.listView.addItem('({},{},{})'.format(h, w, c))

        plt.close()

        hist1 = cv2.calcHist([cv_img],[0],None,[256],[0,256])

        fig, (ax2) = plt.subplots(1,1,figsize=(8,6))
        fig.tight_layout(pad=1)

        ax2.title.set_text('Grayscale')
        ax2.set_xlim([0,256])
        ax2.plot(hist1,color='black')

        plt.show()

    def erosiIMG(self):
        self.listWidget_7.clear()
        self.label_24.clear()

        if os.path.exists("gambar_negatif1.jpg"):
            self.filename = os.path.basename('gambar_negatif1.jpg')
            cv_img = cv2.imread('gambar_negatif1.jpg', cv2.IMREAD_UNCHANGED)
        else:
            self.filename = os.path.basename(self.imgName)
            cv_img = cv2.imread(self.imgName, cv2.IMREAD_UNCHANGED)


        ret, thresh = cv2.threshold(cv_img, 200 , 255, cv2.THRESH_BINARY)
        kernel = np.ones((5,5),np.uint8)
        erosi = cv2.erode(thresh,kernel,iterations = 1)

        printErosi = cv2.imwrite('ErosiImage.jpg',erosi)
        
        erosiCitra = QPixmap('ErosiImage.jpg')

        pictureBox1 = self.label_23
        pictureBox1.setPixmap(erosiCitra)


        # =========== PROPERTIES =============
        self.erosiImgProp = os.path.basename('ErosiImage.jpg')
        cv_img = cv2.imread(self.erosiImgProp, cv2.IMREAD_UNCHANGED)

        if not os.path.exists("gambar_negatif1.jpg"):
            cv_img = cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY)

        size = cv_img.shape
        bit = cv_img.dtype
        # self.bit_print = print(bit)
        self.label_24.setText('Name File : ' + self.erosiImgProp +
                             '\nDimension : ' + str(size) +
                             '\nHeight : ' + str(size[1]) +
                             '\nWidth : ' + str(size[0]) +
                             '\nBit Depth : {}'.format(str(bit)[4]) + '-bit')


        self.listWidget_7.addItem('RGB per pixel :')

        h, w = size
        if h <= 20 and w <= 20:
            for x in range(h):
                for y in range(w):
                    self.listWidget_7.addItem(' {},{}\t: {}'.format(x, y, cv_img[x, y]))
        else:
            for x in range(50):
                for y in range(50):
                    self.listWidget_7.addItem(' {},{}\t: {}'.format(x, y, cv_img[x, y]))
        # self.listView.addItem('({},{},{})'.format(h, w, c))

        plt.close()

        hist1 = cv2.calcHist([cv_img],[0],None,[256],[0,256])

        fig, (ax2) = plt.subplots(1,1,figsize=(8,4))
        fig.tight_layout(pad=1)

        ax2.title.set_text('Grayscale')
        ax2.set_xlim([0,256])
        ax2.plot(hist1,color='black')

        plt.show()

    # ============= DILASI ==============
    def negasiIMGDilasi(self):
        self.listWidget_8.clear()
        self.label_27.clear()
        # ========== Open Image ===========
        # ============ NEGASI NEW ============

        CITRA_BINER = Image.open(self.imgName).convert('1')
        PIXEL_BINER = CITRA_BINER.load()

        ukuran_horizontal = CITRA_BINER.size[0]
        ukuran_vertikal = CITRA_BINER.size[1]

        for x in range(ukuran_horizontal):
            for y in range(ukuran_vertikal):
                PIXEL_BINER[x,y] = 255 - PIXEL_BINER[x,y]

        CITRA_BINER.save('gambar_negatif2.jpg')

        negasiIMG = QPixmap('gambar_negatif2.jpg')

        pictureBox1 = self.label_26
        pictureBox1.setPixmap(negasiIMG)

        # ============== Properties ===============

        self.fileNegasi = os.path.basename('gambar_negatif2.jpg')
        cv_img = cv2.imread(self.fileNegasi, cv2.IMREAD_UNCHANGED)
        size = cv_img.shape
        bit = cv_img.dtype
        # self.bit_print = print(bit)
        self.label_27.setText('Name File : ' + self.fileNegasi +
                             '\nDimension : ' + str(size) +
                             '\nHeight : ' + str(size[1]) +
                             '\nWidth : ' + str(size[0]) +
                             '\nBit Depth : {}'.format(str(bit)[4]) + '-bit')


        self.listWidget_8.addItem('RGB per pixel :')

        h, w= size
        if h <= 20 and w <= 20:
            for x in range(h):
                for y in range(w):
                    self.listWidget_8.addItem(' {},{}\t: {}'.format(x, y, cv_img[x, y]))
        else:
            for x in range(50):
                for y in range(50):
                    self.listWidget_8.addItem(' {},{}\t: {}'.format(x, y, cv_img[x, y]))
        # self.listView.addItem('({},{},{})'.format(h, w, c))

        plt.close()

        hist1 = cv2.calcHist([cv_img],[0],None,[256],[0,256])

        fig, (ax2) = plt.subplots(1,1,figsize=(8,6))
        fig.tight_layout(pad=1)

        ax2.title.set_text('Grayscale')
        ax2.set_xlim([0,256])
        ax2.plot(hist1,color='black')

        plt.show()

    def dilasiIMG(self):
        self.listWidget_8.clear()
        self.label_27.clear()

        if os.path.exists("gambar_negatif2.jpg"):
            self.filename = os.path.basename('gambar_negatif2.jpg')
            cv_img = cv2.imread('gambar_negatif2.jpg', cv2.IMREAD_UNCHANGED)
        else:
            self.filename = os.path.basename(self.imgName)
            cv_img = cv2.imread(self.imgName, cv2.IMREAD_UNCHANGED)

        # =========== Dilasi IMG ==========

        ret, thresh = cv2.threshold(cv_img, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5,5),np.uint8)
        dilasi = cv2.dilate(thresh,kernel,iterations = 1)
        
        printDilasi = cv2.imwrite('DilasiImage.jpg',dilasi)

        dilasiCitra = QPixmap('DilasiImage.jpg')

        pictureBox1 = self.label_26
        pictureBox1.setPixmap(dilasiCitra)


        # =========== PROPERTIES =============
        self.erosiImgProp = os.path.basename('DilasiImage.jpg')
        cv_img = cv2.imread(self.erosiImgProp, cv2.IMREAD_UNCHANGED)

        if not os.path.exists("gambar_negatif2.jpg"):
            cv_img = cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY)

        size = cv_img.shape
        bit = cv_img.dtype
        # self.bit_print = print(bit)
        self.label_27.setText('Name File : ' + self.erosiImgProp +
                                '\nDimension : ' + str(size) +
                                '\nHeight : ' + str(size[1]) +
                                '\nWidth : ' + str(size[0]) +
                                '\nBit Depth : {}'.format(str(bit)[4]) + '-bit')


        self.listWidget_8.addItem('RGB per pixel :')

        h, w = size
        if h <= 20 and w <= 20:
            for x in range(h):
                for y in range(w):
                    self.listWidget_8.addItem(' {},{}\t: {}'.format(x, y, cv_img[x, y]))
        else:
            for x in range(50):
                for y in range(50):
                    self.listWidget_8.addItem(' {},{}\t: {}'.format(x, y, cv_img[x, y]))
        # self.listView.addItem('({},{},{})'.format(h, w, c))

        plt.close()

        hist1 = cv2.calcHist([cv_img],[0],None,[256],[0,256])

        fig, (ax2) = plt.subplots(1,1,figsize=(8,4))
        fig.tight_layout(pad=1)

        ax2.title.set_text('Grayscale')
        ax2.set_xlim([0,256])
        ax2.plot(hist1,color='black')

        plt.show()

    # ============ Edge Detection ===============
    def sobelXEdge(self):
        self.sobelXEDGEFILE = os.path.basename(self.imgName)
        cv_img = cv2.imread(self.imgName,cv2.IMREAD_UNCHANGED)
        cv_img_gray = cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY)

        # Kernel 3x3 Sobel X
        sobel_x = np.array([[-1,0,1],
                            [-2,0,2],
                            [-1,0,1]])

        filtered_image_x = cv2.filter2D(cv_img_gray,-1,sobel_x)

        cv2.imwrite('sobelXEdge.jpg',filtered_image_x)

        sobelXCitra = QPixmap('sobelXEdge.jpg')

        pictureBox1 = self.label_7
        pictureBox1.setPixmap(sobelXCitra)

        plt.close()

        hist1 = cv2.calcHist([filtered_image_x],[0],None,[256],[0,256])

        fig, (ax2) = plt.subplots(1,1,figsize=(8,4))
        fig.tight_layout(pad=1)

        ax2.title.set_text('Grayscale')
        ax2.set_xlim([0,256])
        ax2.plot(hist1,color='black')

        plt.show()

    def sobelYEdge(self):
        cv_img = cv2.imread(self.imgName,cv2.IMREAD_UNCHANGED)
        cv_img_gray = cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY)

        # Kernel 3x3 Sobel X
        sobel_y = np.array([[-1,-2,-1],
                            [0,0,0],
                            [1,2,1]])

        filtered_image_y = cv2.filter2D(cv_img_gray,-1,sobel_y)

        cv2.imwrite('sobelYEdge.jpg',filtered_image_y)

        sobelYCitra = QPixmap('sobelYEdge.jpg')

        pictureBox1 = self.label_7
        pictureBox1.setPixmap(sobelYCitra)

        plt.close()

        hist1 = cv2.calcHist([filtered_image_y],[0],None,[256],[0,256])

        fig, (ax2) = plt.subplots(1,1,figsize=(8,4))
        fig.tight_layout(pad=1)

        ax2.title.set_text('Grayscale')
        ax2.set_xlim([0,256])
        ax2.plot(hist1,color='black')

        plt.show()

    def laplacianEdge(self):
        cv_img = cv2.imread(self.imgName,cv2.IMREAD_UNCHANGED)
        cv_img_gray = cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY)
        cv_img_blur = cv2.GaussianBlur(cv_img_gray,(3,3),0)

        filtered_image = cv2.Laplacian(cv_img_blur, ksize=3, ddepth=cv2.CV_16S)
        filtered_image = cv2.convertScaleAbs(filtered_image)

        cv2.imwrite('laplacianEdge.jpg',filtered_image)

        laplacianEdgeIMG = QPixmap('laplacianEdge.jpg')

        pictureBox1 = self.label_7
        pictureBox1.setPixmap(laplacianEdgeIMG)

        plt.close()

        hist1 = cv2.calcHist([filtered_image],[0],None,[256],[0,256])

        fig, (ax2) = plt.subplots(1,1,figsize=(8,4))
        fig.tight_layout(pad=1)

        ax2.title.set_text('Grayscale')
        ax2.set_xlim([0,256])
        ax2.plot(hist1,color='black')

        plt.show()

    def cannyEdge(self):
        cv_img = cv2.imread(self.imgName,cv2.IMREAD_UNCHANGED)
        cv_img_gray = cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY)


        filtered_image = cv2.Canny(cv_img_gray, threshold1=20, threshold2=200)

        cv2.imwrite('cannyEdge.jpg',filtered_image)

        laplacianEdgeIMG = QPixmap('cannyEdge.jpg')

        pictureBox1 = self.label_7
        pictureBox1.setPixmap(laplacianEdgeIMG)

        plt.close()

        hist1 = cv2.calcHist([filtered_image],[0],None,[256],[0,256])

        fig, (ax2) = plt.subplots(1,1,figsize=(8,4))
        fig.tight_layout(pad=1)

        ax2.title.set_text('Grayscale')
        ax2.set_xlim([0,256])
        ax2.plot(hist1,color='black')

        plt.show()

    # ============== Hough Transformation ============
    def circleHough(self):
        cv_img = cv2.imread(self.imgName,cv2.IMREAD_UNCHANGED)
        grey = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2GRAY)

        blur = cv2.medianBlur(grey, 5)

        circles = cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,1,150, param1=200, param2=10, minRadius=0, maxRadius=70)
        print(circles)
        # Changing the dtype  to int
        cimg = cv_img.copy()
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Draw outer circle
                cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # Draw inner circle
                cv2.circle(cimg, (i[0], i[1]), 2, (0, 255, 0), 3)

        cv2.imwrite('Hough_circles.jpg',cimg)

        hough_circle = QPixmap('Hough_circles.jpg')

        pictureBox1 = self.label_10
        pictureBox1.setPixmap(hough_circle)

    def lineHough(self):
        cv_img = cv2.imread(self.imgName,cv2.IMREAD_UNCHANGED)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2GRAY)

        edges = cv2.Canny(gray, 50, 200,apertureSize=3)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180,120, minLineLength=10, maxLineGap=250)
        print(lines)
        # Changing the dtype  to int
        limg = cv_img.copy()
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(limg, (x1, y1), (x2, y2), (0, 255, 0), 3)


        cv2.imwrite('Hough_line.jpg',limg)

        hough_line = QPixmap('Hough_line.jpg')

        pictureBox1 = self.label_10
        pictureBox1.setPixmap(hough_line)


app = QApplication([])
window = Ui_MainWindow()
window.show()
app.exec_()