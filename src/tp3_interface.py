import sys
import os
import cv2
import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QFileDialog, QScrollArea, QLabel, QGroupBox, 
                             QVBoxLayout, QHBoxLayout, QGridLayout, 
                             QPushButton, QCheckBox, QComboBox, QProgressBar)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import existing logic
from feature_extractor import (Histogram_Color_Extractor, Histogram_HSV_Extractor, 
                               SIFT_Extractor, ORB_Extractor, GLCM_Extractor, 
                               LBP_Extractor, HOG_Extractor, fusion_features, ViT_Extractor)
from retrieval import extract_class_id, getkVoisins
from metrics import *

FEATURES = {
    "BGR": Histogram_Color_Extractor,
    "HSV": Histogram_HSV_Extractor,
    "HOG": HOG_Extractor,
    "SIFT": SIFT_Extractor,
    "ORB": ORB_Extractor,
    "Mom.": None, # Placeholder for Moments if needed
    "LBP": LBP_Extractor,
    "GLCM": GLCM_Extractor,
    "Other": None
}

DISTANCES = ["Euclidienne", "Chicarre", "Bhattacharyya", "Flann", "Brute force", "Intersection", "Correlation"]

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class TP3MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MainWindow")
        self.resize(1100, 700)
        
        self.centralwidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralwidget)
        self.main_layout = QVBoxLayout(self.centralwidget)
        
        # --- Top Controls ---
        self.top_controls = QHBoxLayout()
        
        # Request Box
        self.request_group = QGroupBox("Request")
        self.request_vbox = QVBoxLayout()
        self.btn_load = QPushButton("Load")
        self.btn_load.setFixedSize(80, 80)
        self.request_vbox.addWidget(self.btn_load)
        self.request_group.setLayout(self.request_vbox)
        self.top_controls.addWidget(self.request_group)
        
        # Choice of Descriptor Box
        self.descriptor_group = QGroupBox("Choice of descriptor")
        self.descriptor_grid = QGridLayout()
        self.checkboxes = {}
        descriptors = ["BGR", "HSV", "HOG", "SIFT", "ORB", "Mom.", "LBP", "GLCM", "Other"]
        for i, name in enumerate(descriptors):
            cb = QCheckBox(name)
            self.checkboxes[name] = cb
            self.descriptor_grid.addWidget(cb, i // 3, i % 3)
        self.descriptor_group.setLayout(self.descriptor_grid)
        self.top_controls.addWidget(self.descriptor_group)
        
        # Search Box
        self.search_group = QGroupBox("Search")
        self.search_hbox = QHBoxLayout()
        self.btn_load_desc = QPushButton("Load descriptors")
        self.label_distance = QLabel("Distance :")
        self.combo_distance = QComboBox()
        self.combo_distance.addItems(DISTANCES)
        self.btn_search = QPushButton("Search")
        self.search_hbox.addWidget(self.btn_load_desc)
        self.search_hbox.addWidget(self.label_distance)
        self.search_hbox.addWidget(self.combo_distance)
        self.search_hbox.addWidget(self.btn_search)
        self.search_group.setLayout(self.search_hbox)
        self.top_controls.addWidget(self.search_group)
        
        # Recall/Precision Box
        self.rp_group = QGroupBox("Recall/Precision")
        self.rp_vbox = QVBoxLayout()
        self.btn_compute_rp = QPushButton("Compute R/P Curve")
        self.rp_vbox.addWidget(self.btn_compute_rp)
        self.rp_group.setLayout(self.rp_vbox)
        self.top_controls.addWidget(self.rp_group)
        
        self.main_layout.addLayout(self.top_controls)
        
        # --- Titles Bar ---
        self.titles_hbox = QHBoxLayout()
        self.titles_hbox.addWidget(QLabel("Request Image"), 1)
        self.titles_hbox.addWidget(QLabel("Results"), 3)
        self.titles_hbox.addWidget(QLabel("R/P Curve"), 1)
        self.main_layout.addLayout(self.titles_hbox)
        
        # --- Display Area ---
        self.display_hbox = QHBoxLayout()
        
        self.label_image_request = QLabel("Label")
        self.label_image_request.setFrameStyle(QtWidgets.QFrame.Box | QtWidgets.QFrame.Plain)
        self.label_image_request.setFixedSize(250, 350)
        self.display_hbox.addWidget(self.label_image_request)
        
        self.scroll_results = QScrollArea()
        self.scroll_results.setWidgetResizable(True)
        self.results_widget = QtWidgets.QWidget()
        self.results_grid = QGridLayout(self.results_widget)
        self.scroll_results.setWidget(self.results_widget)
        self.display_hbox.addWidget(self.scroll_results, 3)
        
        self.canvas_rp = MplCanvas(self, width=3, height=4, dpi=70)
        self.display_hbox.addWidget(self.canvas_rp)
        
        self.main_layout.addLayout(self.display_hbox)
        
        # --- Bottom Bar ---
        self.bottom_hbox = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.btn_quit = QPushButton("Quit")
        self.btn_quit.clicked.connect(self.close)
        self.bottom_hbox.addWidget(QLabel("ProgressBar"))
        self.bottom_hbox.addWidget(self.progress_bar, 1)
        self.bottom_hbox.addWidget(self.btn_quit)
        self.main_layout.addLayout(self.bottom_hbox)
        
        # Initialization
        self.image_request_path = None
        self.features = {}
        self.imagesPath = []
        self.folder_images = None
        
        # Connections
        self.btn_load.clicked.connect(self.load_image_request)
        self.btn_load_desc.clicked.connect(self.load_descriptors)
        self.btn_search.clicked.connect(self.search)
        self.btn_compute_rp.clicked.connect(self.compute_rp_curve)
        
        # Load dataset images automatically if possible
        self.auto_load_images()

    def auto_load_images(self):
        # Default dataset path
        default_path = os.path.abspath('data/dataset')
        if os.path.exists(default_path):
            self.folder_images = default_path
            self.imagesPath = [os.path.join(default_path, f) for f in os.listdir(default_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"Auto-loaded {len(self.imagesPath)} images.")

    def load_image_request(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '', "Image files (*.jpg *.jpeg *.png)")
        if fname:
            self.image_request_path = fname
            pixmap = QtGui.QPixmap(fname)
            self.label_image_request.setPixmap(pixmap.scaled(self.label_image_request.size(), QtCore.Qt.KeepAspectRatio))

    def load_descriptors(self):
        selected_descs = [name for name, cb in self.checkboxes.items() if cb.isChecked()]
        if not selected_descs:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select at least one descriptor.")
            return
            
        folder = os.path.abspath('extracted_features')
        os.makedirs(folder, exist_ok=True)
        
        self.features = {}
        self.progress_bar.setValue(0)
        
        # Simplified loading logic for demonstration
        for i, desc in enumerate(selected_descs):
            extractor_class = FEATURES.get(desc)
            if extractor_class:
                extractor = extractor_class()
                # Check if aggregate index exists
                index_path = os.path.join(folder, f"{desc}_all.npy")
                if os.path.exists(index_path):
                    data = np.load(index_path, allow_pickle=True).item()
                    self.features[desc] = list(data.items())
                else:
                    print(f"Index for {desc} not found. Please run indexing.")
            self.progress_bar.setValue(int((i + 1) / len(selected_descs) * 100))
        
        QtWidgets.QMessageBox.information(self, "Success", "Descriptors loaded successfully.")

    def search(self):
        if not self.image_request_path:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please load a request image.")
            return
            
        if not self.features:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please load descriptors first.")
            return

        distance = self.combo_distance.currentText()
        results_per_desc = {}
        
        # Clear results grid
        for i in reversed(range(self.results_grid.count())): 
            self.results_grid.itemAt(i).widget().setParent(None)

        # Single descriptor search for simplicity or first selected
        desc = list(self.features.keys())[0]
        extractor = FEATURES[desc]()
        query_feature = extractor.extract_feature(self.image_request_path)
        
        n_voisins = getkVoisins(self.features[desc], query_feature, 20, distance)
        
        for i, (path, _, score) in enumerate(n_voisins):
            full_path = os.path.join(self.folder_images, path) if self.folder_images else path
            img_label = QLabel()
            pixmap = QtGui.QPixmap(full_path)
            if not pixmap.isNull():
                img_label.setPixmap(pixmap.scaled(120, 120, QtCore.Qt.KeepAspectRatio))
                img_label.setToolTip(f"Score: {score:.4f}")
            self.results_grid.addWidget(img_label, i // 4, i % 4)
            
        self.last_results = n_voisins
        self.last_query_path = self.image_request_path

    def compute_rp_curve(self):
        if not hasattr(self, 'last_results') or not self.last_results:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please run a search first.")
            return

        request_class = extract_class_id(self.last_query_path)
        total_relevant = len([p for p in self.imagesPath if extract_class_id(p) == request_class])
        
        retrieved_classes = [extract_class_id(res[0]) for res in self.last_results]
        
        precisions = []
        recalls = []
        relevant_count = 0
        for i, res_class in enumerate(retrieved_classes):
            if res_class == request_class:
                relevant_count += 1
            precisions.append(relevant_count / (i + 1))
            recalls.append(relevant_count / total_relevant if total_relevant > 0 else 0)

        self.canvas_rp.axes.clear()
        self.canvas_rp.axes.plot(recalls, precisions, marker='.', color='#55b6a3')
        self.canvas_rp.axes.set_xlabel('Recall')
        self.canvas_rp.axes.set_ylabel('Precision')
        self.canvas_rp.axes.set_title('Recall-Precision Curve')
        self.canvas_rp.axes.grid(True, linestyle='--', alpha=0.7)
        self.canvas_rp.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TP3MainWindow()
    window.show()
    sys.exit(app.exec_())

