from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QScrollArea, QLabel, QDialog, QListWidget, QVBoxLayout, QPushButton, QApplication
import os, cv2
import pandas as pd
from feature_extractor import Histogram_Color_Extractor, Histogram_HSV_Extractor, SIFT_Extractor, ORB_Extractor, GLCM_Extractor, LBP_Extractor, HOG_Extractor, fusion_features, ResNet34_ImageNet_Extractor, InceptionV4_2_steps_data_augmentation_Extractor, ResNet34_2_steps_data_augmentation_Steps_Extractor, ViT_Extractor

from PyQt5.QtCore import QItemSelectionModel
from retrieval import extract_class_id, getkVoisins
from metrics import *
import qdarkstyle
from PyQt5.QtWidgets import QTableView
import timm
import torch
from torchvision import transforms

FEATURES = {
    "Histogram_HSV": Histogram_HSV_Extractor,
    "Histogram_Color": Histogram_Color_Extractor,
    "SIFT": SIFT_Extractor,
    "ORB": ORB_Extractor,
    "GLCM": GLCM_Extractor,
    "LBP": LBP_Extractor,
    "HOG": HOG_Extractor,
    "ResNet34_ImageNet": ResNet34_ImageNet_Extractor,
    "ResNet34_2_steps_data_augmentation": ResNet34_2_steps_data_augmentation_Steps_Extractor,
    "InceptionV4_2_steps_data_augmentation": InceptionV4_2_steps_data_augmentation_Extractor,
    "ViT": ViT_Extractor
}


DISTANCES = [
    "Euclidienne",
    "Chicarre",
    "Bhattacharyya",
    "Flann",
    "Brute force",
    "Intersection",
    "Correlation"
]

DISTANCES_FEATURES = {
    "SIFT": ["Brute force", "Flann"],
    "ORB": ["Brute force", "Flann"],
    "BGR": ["Euclidienne", "Chicarre", "Bhattacharyya", "Intersection", "Correlation"],
    "Histogram_HSV": ["Euclidienne", "Chicarre", "Bhattacharyya", "Intersection", "Correlation"],
    "GLCM": ["Euclidienne", "Chicarre", "Bhattacharyya", "Intersection", "Correlation"],
    "Histogram_Color": ["Euclidienne", "Chicarre", "Bhattacharyya", "Intersection", "Correlation"],
    "LBP": ["Euclidienne", "Chicarre", "Bhattacharyya", "Intersection", "Correlation"],
    "HOG": ["Euclidienne", "Chicarre", "Bhattacharyya", "Intersection", "Correlation"],
    "ResNet34_ImageNet": ["Euclidienne", "Chicarre", "Bhattacharyya", "Intersection", "Correlation"],
    "ResNet34_2_steps_data_augmentation": ["Euclidienne", "Chicarre", "Bhattacharyya", "Intersection", "Correlation"],
    "InceptionV4_2_steps_data_augmentation": ["Euclidienne", "Chicarre", "Bhattacharyya", "Intersection", "Correlation"],
    "ViT": ["Euclidienne", "Chicarre", "Bhattacharyya", "Intersection", "Correlation"],
}



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowTitle("Image Retrieval System")
        MainWindow.setWindowIcon(QtGui.QIcon("icon.png"))
        self.mainWindow = MainWindow

        screen = QtWidgets.QApplication.primaryScreen().size()
        width_ratio = screen.width() / 1920
        height_ratio = screen.height() / 1080 
        self.width_ratio = width_ratio
        self.height_ratio = height_ratio

        MainWindow.setFixedSize(int(961 * width_ratio), int(543 * height_ratio))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabMenu = QtWidgets.QTabWidget(self.centralwidget)
        self.tabMenu.setGeometry(QtCore.QRect(int(10 * width_ratio), int(10 * height_ratio), int(921 * width_ratio), int(511 * height_ratio)))
        self.tabMenu.setObjectName("tabMenu")
        self.tabMenu.currentChanged.connect(self.change_tab)

        ### PAGE 1
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")

        self.buttonLoadImages = QtWidgets.QPushButton(self.tab)
        self.buttonLoadImages.setGeometry(QtCore.QRect(int(400 * width_ratio), int(10 * height_ratio), int(101 * width_ratio), int(41 * height_ratio)))
        self.buttonLoadImages.setObjectName("buttonLoadImages")
        self.buttonLoadImages.clicked.connect(self.load_images)
        self.progressBarImages = QtWidgets.QProgressBar(self.tab)
        self.progressBarImages.setGeometry(QtCore.QRect(int(250 * width_ratio), int(60 * height_ratio), int(401 * width_ratio), int(23 * height_ratio)))
        self.progressBarImages.setObjectName("progressBarImages")
        self.progressBarImages.setProperty("value", 0)
        self.tabMenu.addTab(self.tab, "")
        self.tabMenu.setTabEnabled(0, True)

        ### PAGE 2
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")

        self.scrollAreaDescriptor = QScrollArea(self.tab_2)
        self.scrollAreaDescriptor.setGeometry(QtCore.QRect(int(30 * width_ratio), int(30 * height_ratio), int(871 * width_ratio), int(281 * height_ratio)))
        self.scrollAreaDescriptorContents = QtWidgets.QWidget()
        self.scrollAreaDescriptorContents.setGeometry(QtCore.QRect(int(10 * width_ratio), int(30 * height_ratio), int(851 * width_ratio), int(241 * height_ratio)))
        self.gridDescriptor = QtWidgets.QGridLayout(self.scrollAreaDescriptorContents)
        self.checkBoxDescriptors = []
        checkBox = QtWidgets.QCheckBox(self.scrollAreaDescriptorContents)
        checkBox.setObjectName("Fusion de descripteurs")
        checkBox.setText("Fusion de descripteurs")
        checkBox.clicked.connect(self.checkBoxDescriptors_clicked)
        self.checkBoxDescriptors.append(checkBox)
        for descriptor in FEATURES.keys() :
            checkBox = QtWidgets.QCheckBox(self.scrollAreaDescriptorContents)
            checkBox.setObjectName(descriptor)
            checkBox.setText(descriptor)
            checkBox.clicked.connect(self.checkBoxDescriptors_clicked)
            self.checkBoxDescriptors.append(checkBox)
        nbr_columns_descriptor = 3
        for i, checkBox in enumerate(self.checkBoxDescriptors) :
            self.gridDescriptor.addWidget(checkBox, i // nbr_columns_descriptor, i % nbr_columns_descriptor)

        self.scrollAreaDescriptor.setWidgetResizable(True)
        self.scrollAreaDescriptor.setWidget(self.scrollAreaDescriptorContents)

        self.buttonLoadDescriptors = QtWidgets.QPushButton(self.tab_2)
        self.buttonLoadDescriptors.setGeometry(QtCore.QRect(int(420 * width_ratio), int(330 * height_ratio), int(111 * width_ratio), int(31 * height_ratio)))
        self.buttonLoadDescriptors.setObjectName("buttonLoadDescriptors")
        self.buttonLoadDescriptors.clicked.connect(self.load_descriptors)
        self.progressBarDescriptor = QtWidgets.QProgressBar(self.tab_2)
        self.progressBarDescriptor.setGeometry(QtCore.QRect(int(120 * width_ratio), int(370 * height_ratio), int(711 * width_ratio), int(23 * height_ratio)))
        self.progressBarDescriptor.setProperty("value", 0)
        self.progressBarDescriptor.setObjectName("progressBarDescriptor")
        self.tabMenu.addTab(self.tab_2, "")
        self.tabMenu.setTabEnabled(1, False)

        ### PAGE 3
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")

        self.scrollDistance = QScrollArea(self.tab_3)
        self.scrollDistance.setGeometry(QtCore.QRect(int(20 * width_ratio), int(20 * height_ratio), int(311 * width_ratio), int(131 * height_ratio)))
        self.scrollDistanceContents = QtWidgets.QWidget()
        self.scrollDistanceContents.setGeometry(QtCore.QRect(int(30 * width_ratio), int(40 * height_ratio), int(261 * width_ratio), int(71 * height_ratio)))
        self.listViewDistance = QtWidgets.QListView(self.scrollDistanceContents)
        self.listViewDistance.setModel(QtGui.QStandardItemModel())
        self.listViewDistance.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

        for distance in DISTANCES :
            item = QtGui.QStandardItem(distance)
            self.listViewDistance.model().appendRow(item)
        self.listViewDistance.selectionModel().select(self.listViewDistance.model().index(0, 0), QItemSelectionModel.Select)
        self.listViewDistance.selectionModel().selectionChanged.connect(self.distance_selected_changed)

        self.scrollDistance.setWidgetResizable(True)
        self.scrollDistance.setWidget(self.scrollDistanceContents)

        self.label_number = QtWidgets.QLabel(self.tab_3)
        self.label_number.setGeometry(QtCore.QRect(int(400 * width_ratio), int(100 * height_ratio), int(131 * width_ratio), int(41 * height_ratio)))
        self.label_number.setObjectName("label_number")
        self.spinResults = QtWidgets.QSpinBox(self.tab_3)
        self.spinResults.setGeometry(QtCore.QRect(int(510 * width_ratio), int(110 * height_ratio), int(47 * width_ratio), int(24 * height_ratio)))
        self.spinResults.setMinimum(1)
        self.spinResults.setMaximum(500)
        self.spinResults.setObjectName("spinResults")
        self.buttonSearch = QtWidgets.QPushButton(self.tab_3)
        self.buttonSearch.setGeometry(QtCore.QRect(int(670 * width_ratio), int(50 * height_ratio), int(151 * width_ratio), int(51 * height_ratio)))
        self.buttonSearch.setObjectName("buttonSearch")
        self.buttonSearch.clicked.connect(self.search)
        self.buttonSearch.setEnabled(False)
        self.progressBarSearch = QtWidgets.QProgressBar(self.tab_3)
        self.progressBarSearch.setGeometry(QtCore.QRect(int(10 * width_ratio), int(450 * height_ratio), int(881 * width_ratio), int(23 * height_ratio)))
        self.progressBarSearch.setProperty("value", 0)
        self.progressBarSearch.setObjectName("progressBarSearch")
        self.buttonLoadImageRequest = QtWidgets.QPushButton(self.tab_3)
        self.buttonLoadImageRequest.setGeometry(QtCore.QRect(int(420 * width_ratio), int(50 * height_ratio), int(101 * width_ratio), int(41 * height_ratio)))
        self.buttonLoadImageRequest.setObjectName("buttonLoadImageRequest")
        self.buttonLoadImageRequest.clicked.connect(self.load_image)
        self.label_request = QtWidgets.QLabel(self.tab_3)
        self.label_request.setGeometry(QtCore.QRect(int(400 * width_ratio), int(10 * height_ratio), int(131 * width_ratio), int(41 * height_ratio)))
        self.label_request.setObjectName("label_request")
        self.label_image_request = QtWidgets.QLabel(self.tab_3)
        self.label_image_request.setGeometry(QtCore.QRect(int(270 * width_ratio), int(180 * height_ratio), int(341 * width_ratio), int(251 * height_ratio)))
        self.label_image_request.setObjectName("label_image_request")
        self.tabMenu.addTab(self.tab_3, "")
        self.tabMenu.setTabEnabled(2, False)

        ### PAGE 4
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")

        # First box: scrollRes
        self.scrollRes = QScrollArea(self.tab_4)
        self.scrollRes.setGeometry(QtCore.QRect(int(10 * width_ratio), int(10 * height_ratio), int(200 * width_ratio), int(300 * height_ratio)))
        self.scrollResContents = QtWidgets.QWidget()
        self.scrollResContents.setGeometry(QtCore.QRect(0, 0, int(200 * width_ratio), int(300 * height_ratio)))
        self.listViewRes = QtWidgets.QListView(self.scrollResContents)
        self.listViewRes.setModel(QtGui.QStandardItemModel())
        self.listViewRes.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.listViewRes.setGeometry(QtCore.QRect(0, 0, int(180 * width_ratio), int(360 * height_ratio)))
        self.listViewRes.selectionModel().selectionChanged.connect(self.show_results)

        self.scrollRes.setWidgetResizable(True)
        self.scrollRes.setWidget(self.scrollResContents)

        # Second box: scrollMetric
        self.scrollMetric = QScrollArea(self.tab_4)
        self.scrollMetric.setGeometry(QtCore.QRect(int(220 * width_ratio), int(10 * height_ratio), int(400 * width_ratio), int(300 * height_ratio)))
        self.scrollMetricContents = QtWidgets.QWidget()
        self.scrollMetricContents.setGeometry(QtCore.QRect(0, 0, int(400 * width_ratio), int(300 * height_ratio)))

        self.verticalLayout = QtWidgets.QVBoxLayout(self.scrollMetricContents)

        self.scrollMetric.setWidgetResizable(True)
        self.scrollMetric.setWidget(self.scrollMetricContents)

        # Third box: scrollImageRes
        self.scrollImageRes = QtWidgets.QScrollArea(self.tab_4)
        self.scrollImageRes.setGeometry(QtCore.QRect(int(630 * width_ratio), int(10 * height_ratio), int(720 * width_ratio), int(300 * height_ratio)))
        self.scrollImageResContents = QtWidgets.QWidget()
        self.scrollImageResContents.setGeometry(QtCore.QRect(0, 0, int(720 * width_ratio), int(300 * height_ratio)))
        self.gridImageRes = QtWidgets.QGridLayout(self.scrollImageResContents)
        self.gridImageRes.setContentsMargins(5, 5, 5, 5)  # Set margins
        self.gridImageRes.setSpacing(5)  # Set spacing between images

        self.scrollImageRes.setWidgetResizable(True)
        self.scrollImageRes.setWidget(self.scrollImageResContents)

        # New box for matplotlib plot
        self.scrollPlot = QScrollArea(self.tab_4)
        self.scrollPlot.setGeometry(QtCore.QRect(int(10 * width_ratio), int(320 * height_ratio), int(750 * width_ratio), int(500 * height_ratio)))
        self.scrollPlotContents = QtWidgets.QWidget()
        self.scrollPlotContents.setGeometry(QtCore.QRect(0, 0, int(750 * width_ratio), int(500 * height_ratio)))
        self.verticalLayoutPlot = QtWidgets.QVBoxLayout(self.scrollPlotContents)

        self.scrollPlot.setWidgetResizable(True)
        self.scrollPlot.setWidget(self.scrollPlotContents)

        # Box dataframe

        self.tableView = QTableView(self.tab_4)
        self.tableView.setGeometry(QtCore.QRect(int(700 * width_ratio), int(320 * height_ratio), int(600 * width_ratio), int(500 * height_ratio)))
        self.tableView.setObjectName("tableView")
        self.tableView.resizeColumnsToContents()
        self.tableView.resizeRowsToContents()

        self.tabMenu.addTab(self.tab_4, "")

        self.tabMenu.setTabEnabled(3, False)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabMenu.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.init_variables()
        self.set_size_for_page_1()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Image Retrieval System"))
        self.buttonLoadImages.setText(_translate("MainWindow", "Load Images"))
        self.tabMenu.setTabText(self.tabMenu.indexOf(self.tab), _translate("MainWindow", "Images loader"))
        self.buttonLoadDescriptors.setText(_translate("MainWindow", "Load"))
        self.tabMenu.setTabText(self.tabMenu.indexOf(self.tab_2), _translate("MainWindow", "Descriptor loader"))
        self.label_number.setText(_translate("MainWindow", "Number results :"))
        self.buttonSearch.setText(_translate("MainWindow", "Search"))
        self.buttonLoadImageRequest.setText(_translate("MainWindow", "Load"))
        self.label_request.setText(_translate("MainWindow", "Load request image :"))
        self.tabMenu.setTabText(self.tabMenu.indexOf(self.tab_3), _translate("MainWindow", "Search"))
        self.tabMenu.setTabText(self.tabMenu.indexOf(self.tab_4), _translate("MainWindow", "Results"))

    def init_variables(self):
        self.imagesPath = []
        self.folder_images = None
        self.features = {}
        self.last_descriptors = set()
        self.image_request = None
        self.res = {}
        self.df = None
        self.descriptors_plot = None

    def checkBoxDescriptors_clicked(self):
        if self.last_descriptors :
            new_descriptors = set([checkBox.objectName() for checkBox in self.checkBoxDescriptors if checkBox.isChecked()])
            if new_descriptors != self.last_descriptors :
                self.clean_tab_2()
                self.clean_tab_3()
                self.clean_tab_4()



    def clean_tab_4(self):
        self.clear_vertical_layout()
        self.listViewRes.model().clear()
        if self.gridImageRes.count() > 0:
            for i in reversed(range(self.gridImageRes.count())):
                widget = self.gridImageRes.itemAt(i).widget()
                if widget is not None:
                    widget.deleteLater()
        self.clear_vertical_layout_plot()
        self.tableView.setModel(QtGui.QStandardItemModel())
        self.df = None
        

    def clean_tab_3(self):
        self.progressBarSearch.setValue(0)
        self.res = {}
        self.tabMenu.setTabEnabled(3, False)
        self.spinResults.setValue(1)
        self.buttonSearch.setEnabled(False)
        self.label_image_request.clear()
        self.image_request = None

    def clean_tab_2(self):
        self.progressBarDescriptor.setValue(0)
        self.features = {}
        self.last_descriptors = set()
        self.tabMenu.setTabEnabled(2, False)

    def clean_tab_1(self):
        self.progressBarImages.setValue(0)
        self.imagesPath = []
        self.folder_images = None

    def find_image_in_imagesPath(self, image_path):
        image_name = image_path.split("/")[-1].split("\\")[-1].split(".")[0]
        for path in self.imagesPath :
            image_name_path = path.split("/")[-1].split("\\")[-1].split(".")[0]
            if image_name == image_name_path :
                return path
            
    def add_image_to_list(self, image_path):
        image_path = image_path.replace("\\", "/")
        self.imagesPath.append(image_path)

    def add_image_to_layout(self, liste_images):
        for i, (row, col, image_path) in enumerate(liste_images) :
            image_path = image_path.replace("\\", "/")
            label = QLabel()
            label.setAlignment(QtCore.Qt.AlignCenter)
            img = cv2.imread(image_path)
            b, g, r = cv2.split(img)
            img = cv2.merge([r, g, b])
            height, width, channel = img.shape
            bytesPerLine = 3 * width
            qImg = QtGui.QImage(img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qImg)
            pixmap_show = pixmap.scaled(self.showImage.width(), self.showImage.height(), QtCore.Qt.KeepAspectRatio)
            target_width, target_height = int(130 * self.width_ratio), int(130 * self.height_ratio)
            pixmap = pixmap.scaled(target_width, target_height, QtCore.Qt.KeepAspectRatio)
            label.setPixmap(pixmap)
            def handle_mouse_click(event):
                self.showImage.setPixmap(pixmap_show)
                self.showImageLabel.setText(image_path.split("/")[-1].split(".")[0].replace("_", " "))

            label.mousePressEvent = handle_mouse_click
            self.gridLayout.addWidget(label, row, col)

            if i == 0 :
                self.showImage.setPixmap(pixmap_show)
                self.showImageLabel.setText(image_path.split("/")[-1].split(".")[0].replace("_", " "))


    def freeze_during_load_images(self) :
        self.buttonLoadImages.setEnabled(False)
        self.tabMenu.setTabEnabled(1, False)

    def finalize_progress(self):
        self.progressBarImages.setValue(100)
        self.buttonLoadImages.setEnabled(True)
        self.tabMenu.setTabEnabled(1, True)

    def load_images(self):
        folder = str(QFileDialog.getExistingDirectory(None, "Select dataset folder"))
        if folder:
            self.clean_tab_1()
            self.folder_images = folder
            self.image_loader_thread = ImageLoaderThread(folder, 3)
            self.image_loader_thread.add_image_path.connect(self.add_image_to_list)
            self.image_loader_thread.update_progress.connect(self.progressBarImages.setValue)
            self.image_loader_thread.completed.connect(self.finalize_progress)
            self.image_loader_thread.start()

    def load_image(self):
        image_path = str(QFileDialog.getOpenFileName(None, "Select image request", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif)")[0])
        if image_path:
            img = cv2.imread(image_path)
            b, g, r = cv2.split(img)
            img = cv2.merge([r, g, b])
            height, width, channel = img.shape
            bytesPerLine = 3 * width
            qImg = QtGui.QImage(img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qImg)
            pixmap = pixmap.scaled(self.label_image_request.width(), self.label_image_request.height(), QtCore.Qt.KeepAspectRatio)
            self.label_image_request.setPixmap(pixmap)
            self.label_image_request.setAlignment(QtCore.Qt.AlignCenter)
            self.image_request = image_path
            if self.buttonSearch.isEnabled() == False :
                self.buttonSearch.setEnabled(True)

    def freeze_during_load_descriptors(self) :
        self.buttonLoadDescriptors.setEnabled(False)
        for checkBox in self.checkBoxDescriptors :
            checkBox.setEnabled(False)
        self.tabMenu.setTabEnabled(2, False)
        self.tabMenu.setTabEnabled(3, False)
        self.tabMenu.setTabEnabled(0, False)

    def finalize_progress_descriptors(self):
        self.progressBarDescriptor.setValue(100)
        self.buttonLoadDescriptors.setEnabled(True)
        self.tabMenu.setTabEnabled(2, True)
        self.tabMenu.setTabEnabled(0, True)
        for checkBox in self.checkBoxDescriptors :
            checkBox.setEnabled(True)
        selected_descriptors = [checkBox.objectName() for checkBox in self.checkBoxDescriptors if checkBox.isChecked() and checkBox.objectName() != "Fusion de descripteurs"]
        autoriazed_distances = []
        for descriptor in selected_descriptors :
            autoriazed_distances.append(DISTANCES_FEATURES[descriptor])
        distances = set(autoriazed_distances[0])
        for autoriazed_distance in autoriazed_distances :
            distances = distances.intersection(set(autoriazed_distance))
        if len(distances) == 0 :
            self.tabMenu.setTabEnabled(2, False)
            AlertDialog("No common distance between selected descriptors", parent=self.mainWindow).exec_()
        # desactivate distances not in common
        idx = 0
        for i in range(self.listViewDistance.model().rowCount()) :
            distance = self.listViewDistance.model().item(i).text()
            if distance not in distances :
                self.listViewDistance.model().item(i).setEnabled(False)
            else :
                idx = i
                self.listViewDistance.model().item(i).setEnabled(True)
        self.listViewDistance.selectionModel().select(self.listViewDistance.model().index(idx, 0), QItemSelectionModel.Select)
        
        
    def add_feature_to_features(self, descriptor, feature):
        self.features[descriptor] = feature

    def fusion_descriptors_list(self, features):
        if len(features) > 0 :
            new_features = []
            size = len(features[list(features.keys())[0]])
            for i in range(size) :
                pack = []
                for descriptor in features.keys() :
                    pack.append(features[descriptor][i])
                feature = fusion_features(pack)
                new_features.append((pack[0][0].split("/")[-1].split("\\")[-1], feature))
            self.add_feature_to_features(f"Fusion de descripteurs", new_features)

    def load_descriptors(self):
        selected_descriptors = [checkBox.objectName() for checkBox in self.checkBoxDescriptors if checkBox.isChecked() and checkBox.objectName() != "Fusion de descripteurs"]
        # check if fusion is selected
        fusion_descriptors = self.checkBoxDescriptors[0].isChecked()
        if len(selected_descriptors) == 0 :
            AlertDialog("Please select at least one descriptor", parent=self.mainWindow).exec_()
            return
        folder = str(QFileDialog.getExistingDirectory(None, "Select where features are saved"))
        if folder:
            self.last_descriptors = set(selected_descriptors)
            self.features = {}
            self.progressBarDescriptor.setValue(0)
            self.descriptor_loader_thread = DescriptorLoaderThread(folder, selected_descriptors, self.imagesPath, fusion_descriptors)
            self.descriptor_loader_thread.starting.connect(self.freeze_during_load_descriptors)
            self.descriptor_loader_thread.update_progress.connect(self.progressBarDescriptor.setValue)
            self.descriptor_loader_thread.completed.connect(self.finalize_progress_descriptors)
            self.descriptor_loader_thread.add_feature.connect(self.add_feature_to_features)
            self.descriptor_loader_thread.fusion_features.connect(self.fusion_descriptors_list)
            self.descriptor_loader_thread.start()

    def add_result(self, descriptor, n_voisins):
        self.res[descriptor] = n_voisins

    def add_dataframe(self, df):
        self.df = df

    def plot(self, desciptors):
        self.descriptors_plot = desciptors

    def distance_selected_changed(self):
        self.clean_tab_3()
        self.clean_tab_4()

    def finalize_search(self):
        self.progressBarSearch.setValue(100)
        self.buttonSearch.setEnabled(True)
        self.tabMenu.setTabEnabled(3, True)
        self.tabMenu.setTabEnabled(1, True)
        self.tabMenu.setTabEnabled(0, True)
        self.tabMenu.setCurrentIndex(3)
        self.init_show_results()

    def freeze_during_search(self):
        self.buttonSearch.setEnabled(False)
        self.tabMenu.setTabEnabled(3, False)
        self.tabMenu.setTabEnabled(1, False)
        self.tabMenu.setTabEnabled(0, False)
    
    def search(self):
        self.clean_tab_4()
        if not self.image_request :
            AlertDialog("Please load an image request", parent=self.mainWindow).exec_()
            return
        self.image_request_descriptors = {}
        self.res = {}
        for descriptor in self.features.keys() :
            if descriptor == "Fusion de descripteurs" :
                continue
            else:
                feature_extractor = FEATURES[descriptor]()
            feature = feature_extractor.extract_feature(self.image_request)
            self.image_request_descriptors[descriptor] = feature
        if "Fusion de descripteurs" in self.features.keys() :
            fusion = []
            for descriptor in self.image_request_descriptors.keys() :
                fusion.append((descriptor, self.image_request_descriptors[descriptor]))
            feature = fusion_features(fusion)
            self.image_request_descriptors["Fusion de descripteurs"] = feature

        self.progressBarSearch.setValue(0)
        self.search_thread = SearchThread(self.features, self.image_request_descriptors, self.listViewDistance.selectedIndexes()[0].data(), self.spinResults.value(), self.image_request, self.imagesPath)
        self.search_thread.starting.connect(self.freeze_during_search)
        self.search_thread.update_progress.connect(self.progressBarSearch.setValue)
        self.search_thread.add_result.connect(self.add_result)
        self.search_thread.completed.connect(self.finalize_search)
        self.search_thread.dataframe.connect(self.add_dataframe)
        self.search_thread.plot.connect(self.plot)
        self.search_thread.start()

    def set_size_for_page_4(self):
        self.mainWindow.setFixedSize(int(1480 * self.width_ratio), int(880 * self.height_ratio))
        self.tabMenu.setGeometry(QtCore.QRect(int(10 * self.width_ratio), int(10 * self.height_ratio), int(1440 * self.width_ratio), int(840 * self.height_ratio)))

    def set_size_for_page_1(self):
        self.mainWindow.setFixedSize(int(961 * self.width_ratio), int(150 * self.height_ratio))
        self.tabMenu.setGeometry(QtCore.QRect(int(10 * self.width_ratio), int(10 * self.height_ratio), int(921 * self.width_ratio), int(110 * self.height_ratio)))

    def set_size_for_page_2(self):
        self.mainWindow.setFixedSize(int(961 * self.width_ratio), int(460 * self.height_ratio))
        self.tabMenu.setGeometry(QtCore.QRect(int(10 * self.width_ratio), int(10 * self.height_ratio), int(921 * self.width_ratio), int(420 * self.height_ratio)))

    def reset_size_page(self) :
        self.mainWindow.setFixedSize(int(961 * self.width_ratio), int(543 * self.height_ratio))
        self.tabMenu.setGeometry(QtCore.QRect(int(10 * self.width_ratio), int(10 * self.height_ratio), int(921 * self.width_ratio), int(511 * self.height_ratio)))

    def change_tab(self):
        if self.tabMenu.currentIndex() == 0 :
            self.set_size_for_page_1()
        elif self.tabMenu.currentIndex() == 1 :
            self.set_size_for_page_2()
        elif self.tabMenu.currentIndex() == 2 :
            self.reset_size_page()
        elif self.tabMenu.currentIndex() == 3 :
            self.set_size_for_page_4()

    def init_show_results(self):
        self.set_size_for_page_4()
        self.clear_vertical_layout()
        self.listViewRes.model().clear()
        if self.res and len(self.res) > 0 :
            for descriptor in self.res.keys() :
                self.listViewRes.model().appendRow(QtGui.QStandardItem(descriptor))
            self.listViewRes.selectionModel().select(self.listViewRes.model().index(0, 0), QItemSelectionModel.Select)
        #show dataframe
        if self.df is not None :
            df = self.df[self.df["Feature_Extractor"] == list(self.res.keys())[0]]
            df = df.drop(df.columns[0], axis=1)
            model = PandasModel(df)
            self.tableView.setModel(model)
            self.tableView.resizeColumnsToContents()
            self.tableView.resizeRowsToContents()

    def clear_vertical_layout(self):
        for i in reversed(range(self.verticalLayout.count())) :
            widget = self.verticalLayout.itemAt(i).widget()
            if widget is not None :
                widget.deleteLater()

    def clear_vertical_layout_plot(self):
        for i in reversed(range(self.verticalLayoutPlot.count())) :
            widget = self.verticalLayoutPlot.itemAt(i).widget()
            if widget is not None :
                widget.deleteLater()

    def show_results(self):
        self.clear_vertical_layout()
        nbr_columns = 3
        if self.res and len(self.res) > 0:
            descriptor = self.listViewRes.selectedIndexes()[0].data()
            # Show dataframe
            df = self.df[self.df["Feature_Extractor"] == descriptor]
            df = df.drop(df.columns[0], axis=1)
            model = PandasModel(df)
            self.tableView.setModel(model)
            self.tableView.resizeColumnsToContents()
            self.tableView.resizeRowsToContents()
            # Show plot
            if self.descriptors_plot is not None :
                if descriptor in self.descriptors_plot.keys() :
                    liste_x_y = self.descriptors_plot[descriptor]
                    x_brand_model = liste_x_y[0]
                    y_brand_model = liste_x_y[1]
                    x_brand = liste_x_y[2]
                    y_brand = liste_x_y[3]
                    self.clear_vertical_layout_plot()
                    sc = MplCanvas(self.scrollPlotContents, width=5, height=2, dpi=50)
                    sc.axes.plot(x_brand_model, y_brand_model, label="Brand + Model")
                    sc.axes.set_xlabel('Rappel (%)')
                    sc.axes.set_ylabel('Précision (%)')
                    sc.axes.set_title('Courbe Précision-Rappel (Marque + Modèle)')
                    self.verticalLayoutPlot.addWidget(sc)
                    sc_brand_only = MplCanvas(self.scrollPlotContents, width=5, height=2, dpi=50)
                    sc_brand_only.axes.plot(x_brand, y_brand, label="Brand")
                    sc_brand_only.axes.set_xlabel('Rappel (%)')
                    sc_brand_only.axes.set_ylabel('Précision (%)')
                    sc_brand_only.axes.set_title('Courbe Précision-Rappel (Marque)')
                    self.verticalLayoutPlot.addWidget(sc_brand_only)
            if descriptor in self.res.keys():
                n_voisins = self.res[descriptor]
                if len(n_voisins) > 0:
                    self.verticalLayout.addWidget(QtWidgets.QLabel(f"Descriptor : {descriptor}"))
                    self.verticalLayout.addWidget(QtWidgets.QLabel(f"Distance : {self.listViewDistance.selectedIndexes()[0].data()}"))
                    self.verticalLayout.addWidget(QtWidgets.QLabel(f"Number of results : {len(n_voisins)}"))
                    for i, (image_path_descriptor, _, distance) in enumerate(n_voisins):
                        if i == self.spinResults.value() :
                            break
                        image_path = os.path.join(self.folder_images, self.find_image_in_imagesPath(image_path_descriptor))
                        img = cv2.imread(image_path)
                        b, g, r = cv2.split(img)
                        img = cv2.merge([r, g, b])
                        height, width, channel = img.shape
                        bytesPerLine = 3 * width
                        qImg = QtGui.QImage(img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
                        pixmap = QtGui.QPixmap.fromImage(qImg)
                        pixmap = pixmap.scaled(int(220 * self.width_ratio), int(220 * self.height_ratio), QtCore.Qt.KeepAspectRatio)
                        label = QtWidgets.QLabel()
                        label.setPixmap(pixmap)
                        label.setAlignment(QtCore.Qt.AlignCenter)
                        # Set tooltip with the image name
                        label.setToolTip(image_path.split('/')[-1])
                        self.gridImageRes.addWidget(label, i // nbr_columns, i % nbr_columns)
                        self.verticalLayout.addWidget(QtWidgets.QLabel(f"Image {i + 1} : {image_path.split('/')[-1]}"))
                        self.verticalLayout.addWidget(QtWidgets.QLabel(f"Distance : {distance}"))

class PandasModel(QtCore.QAbstractTableModel):

    def __init__(self, df=pd.DataFrame(), parent=None):
        super(PandasModel, self).__init__(parent)
        self._df = df

    def rowCount(self, parent=None):
        return self._df.shape[0]

    def columnCount(self, parent=None):
        return self._df.shape[1]

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                value = float(self._df.iloc[index.row(), index.column()])
                value = round(value, 3)
                return str(value)
        return None

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return str(self._df.columns[section])
            elif orientation == QtCore.Qt.Vertical:
                return str(self._df.index[section])
        return None 

class SearchThread(QtCore.QThread):
    update_progress = QtCore.pyqtSignal(int)
    add_result = QtCore.pyqtSignal(str, list)
    starting = QtCore.pyqtSignal()
    dataframe = QtCore.pyqtSignal(pd.DataFrame)
    plot = QtCore.pyqtSignal(dict)
    completed = QtCore.pyqtSignal()

    def __init__(self, features, image_request_descriptors, distance_name, k, image_request, imagesPath):
        super().__init__()
        self.features = features
        self.image_request_descriptors = image_request_descriptors
        self.distance_name = distance_name
        self.k = k
        self.image_request = image_request
        self.imagesPath = imagesPath

    def run(self):
        #debugpy.debug_this_thread()
        self.starting.emit()
        df = pd.DataFrame(columns=["Feature_Extractor", "k", "brand_only", "Recall@k", "R-precision", "Average precision@k"])
        request_class = extract_class_id(self.image_request)
        request_class_brand_only = extract_class_id(self.image_request, brand_only=True)
        relevant_items = len([image_path for image_path in self.imagesPath if extract_class_id(image_path) == request_class])
        maximum = max(100, relevant_items)
        plots = {}
        for i, descriptor in enumerate(self.image_request_descriptors.keys()) :
            n_voisins = getkVoisins(self.features[descriptor], self.image_request_descriptors[descriptor], maximum, self.distance_name)
            param = [self.k, 50, 100, relevant_items]
            param = [p for p in param if p <= maximum]
            
            #Calcul des métrique brand+model
            retrieved_classes = [extract_class_id(image_path) for image_path, _, _ in n_voisins]
            for k in param :
                r = recall(k=k, relevant_items=relevant_items, retrieved_classes=retrieved_classes, relevant_class=[request_class])
                r_p = r_precision(k=k, retrieved_classes=retrieved_classes, relevant_class=[request_class])
                ap = average_precision(k=k, retrieved_classes=retrieved_classes, relevant_class=[request_class])
                df = pd.concat([df, pd.DataFrame({"Feature_Extractor": descriptor, "k": k, "brand_only": False, "Recall@k": r, "R-precision": r_p, "Average precision@k": ap}, index=[0])])

            #Calcul la courve PR brand+model
            x_brand_model, y_brand_model = create_plot_pyqt(k=relevant_items, retrieved_classes=retrieved_classes, relevant_class=[request_class], relevant_items=relevant_items)
            

            #Calcul des métrique brand
            retrieved_classes = [extract_class_id(image_path, brand_only=True) for image_path, _, _ in n_voisins]
            for k in param:
                r = recall(k=k, relevant_items=relevant_items, retrieved_classes=retrieved_classes, relevant_class=[request_class_brand_only])
                r_p = r_precision(k=k, retrieved_classes=retrieved_classes, relevant_class=[request_class_brand_only])
                ap = average_precision(k=k, retrieved_classes=retrieved_classes, relevant_class=[request_class_brand_only])
                df = pd.concat([df, pd.DataFrame({"Feature_Extractor": descriptor, "k": k, "brand_only": True, "Recall@k": r, "R-precision": r_p, "Average precision@k": ap}, index=[0])])

            #Calcul la courve PR brand_only
            x_brand_only, y_brand_only = create_plot_pyqt(k=relevant_items, retrieved_classes=retrieved_classes, relevant_class=[request_class_brand_only], relevant_items=relevant_items)
            plots[descriptor] = [x_brand_model, y_brand_model, x_brand_only, y_brand_only]

            self.add_result.emit(descriptor, n_voisins)
            self.update_progress.emit(int((i + 1) / len(self.image_request_descriptors) * 100))
        
        #La dataframe contient toutes les métriques pour l'ensemble des k, brand et brand+model
        self.plot.emit(plots)
        self.dataframe.emit(df)
        self.completed.emit()
                
class DescriptorLoaderThread(QtCore.QThread):
    update_progress = QtCore.pyqtSignal(int)
    completed = QtCore.pyqtSignal()
    starting = QtCore.pyqtSignal()
    add_feature = QtCore.pyqtSignal(str, list)
    fusion_features = QtCore.pyqtSignal(dict)
    def __init__(self, folder, descriptors, imagesPath, fusion):
        super().__init__()
        self.folder = folder
        self.descriptors = descriptors
        self.imagesPath = imagesPath
        self.fusion = fusion

    def run(self):
        self.starting.emit()
        descriptors = {}
        for n, descriptor in enumerate(self.descriptors) :
            feature_extractor = FEATURES[descriptor]()
            descriptor_folder = self.folder + "/" + descriptor
            if not os.path.exists(descriptor_folder) :
                print(f"Folder {descriptor_folder} does not exist -> indexing")
                feature_extractor.index_database(self.imagesPath, self.folder)
            else :
                for image_path in self.imagesPath :
                    image_descriptor_path = self.folder + "/" + descriptor + "/" + image_path.split("/")[-1].split(".")[0] + ".txt"
                    if not os.path.exists(image_descriptor_path) :
                        print(f"File {image_descriptor_path}.txt does not exist -> indexing")
                        feature_extractor.index_element(image_path, self.folder)
            features = feature_extractor.load_features(self.folder)
            self.add_feature.emit(descriptor, features)
            if self.fusion :
                descriptors[descriptor] = features
            if self.fusion :
                self.update_progress.emit(int((n + 1) / (len(self.descriptors) +1 ) * 100))
            else :
                self.update_progress.emit(int((n + 1) / len(self.descriptors) * 100))
        if self.fusion :
            self.fusion_features.emit(descriptors)
        self.completed.emit()

class ImageLoaderThread(QtCore.QThread):
    add_image_path = QtCore.pyqtSignal(str)
    update_progress = QtCore.pyqtSignal(int)
    completed = QtCore.pyqtSignal()

    def __init__(self, folder, num_columns):
        super().__init__()
        self.folder = folder
        self.num_columns = num_columns

    def run(self):
        images = os.listdir(self.folder)
        total_images = len(images)
        for i, image_name in enumerate(images):
            image_path = os.path.join(self.folder, image_name)
            self.add_image_path.emit(image_path)
            if not os.path.exists(image_path) or not os.path.isfile(image_path):
                continue
            if not image_path.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
                continue
            progress = int((i + 1) / total_images * 100)
            self.update_progress.emit(progress)
        self.completed.emit()

class AlertDialog(QDialog):
    def __init__(self, message, parent=None):
        super(AlertDialog, self).__init__(parent)
        self.setWindowTitle("Alert")
        screen = QtWidgets.QApplication.primaryScreen().size()
        width_ratio = screen.width() / 1920
        height_ratio = screen.height() / 1080 
        self.setFixedSize(int(300 * width_ratio), int(100 * height_ratio))
        self.label = QLabel(message, self)
        self.label.setGeometry(int(10 * width_ratio), int(10 * height_ratio), int(280 * width_ratio), int(40 * height_ratio))
        self.confirm_button = QPushButton("OK", self)
        self.confirm_button.setGeometry(int(100 * width_ratio), int(60 * height_ratio), int(100 * width_ratio), int(30 * height_ratio))
        self.confirm_button.clicked.connect(self.accept)

class SelectionDialog(QDialog):
    def __init__(self, options, title, parent=None, multi_selection=False):
        super(SelectionDialog, self).__init__(parent)
        self.setWindowTitle(title)
        self.setWindowIcon(QtGui.QIcon("icon.png"))
        screen = QApplication.primaryScreen().size()
        width_ratio = screen.width() / 1920
        height_ratio = screen.height() / 1080
        self.setFixedSize(int(300 * width_ratio), int(300 * height_ratio))

        self.list_widget = QListWidget(self)
        if multi_selection:
            self.list_widget.setSelectionMode(QListWidget.ExtendedSelection)  # Enable multi-selection
        for option in options:
            self.list_widget.addItem(option)

        self.confirm_button = QPushButton("Confirm", self)
        self.confirm_button.clicked.connect(self.confirm_selection)

        layout = QVBoxLayout(self)
        layout.addWidget(self.list_widget)
        layout.addWidget(self.confirm_button)

        self.selected_options = []  # List to hold multiple selections

    def confirm_selection(self):
        self.selected_options = [item.text() for item in self.list_widget.selectedItems()]  # Collect all selected items
        self.accept()  # Close the dialog and return accept

    def get_selected_options(self):
        return self.selected_options


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)

    #open qss file
    LIST_STYLES = []
    LIST_STYLES.append("PyQt5 Fusion")
    LIST_STYLES.append("QDarkStyle")
    LIST_STYLES.append("PyQt5 Breeze")
    LIST_STYLES.append('PyQt5 Oxygen')
    LIST_STYLES.append('PyQt5 QtCurve')
    LIST_STYLES.append('PyQt5 Windows')
    LIST_STYLES.append('PyQt5 Fusion')
    dialog = SelectionDialog(LIST_STYLES, "Select a style")
    if dialog.exec_() == QDialog.Accepted:
        style = dialog.get_selected_options()[0]
        if "PyQt5" in style.split(" "):
            app.setStyle(style.split(" ")[1])
        elif style == "QDarkStyle":
            app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    else :
        app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
