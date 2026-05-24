import cv2
import os
import torch
import timm

import numpy as np

from abc import ABC, abstractmethod
from tqdm import tqdm
from torchvision import transforms
from torchvision import models
from PIL import Image

from skimage.feature import graycomatrix, graycoprops
from skimage.util import img_as_ubyte
from skimage.feature import local_binary_pattern


class FeatureExtractor(ABC):
    """Abstract class for feature extraction"""

    def __init__(self, name=None):
        self.name = name

    def index_database(self, image_files, output_directory, save_features=True):
        """Index the database and save the features in the output directory"""
        output_directory = output_directory + "/" + self.name
        if save_features:
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
        print("Indexing database...")
        for file in tqdm(image_files):
            if save_features:
                path = os.path.join(output_directory, file.split("/")[-1].split(".")[0] + ".txt")
                path = path.replace("\\", "/")
                if os.path.exists(path):
                    continue
            img_name = file.split("/")[-1].split("\\")[-1].split(".")[0]
            feature = self.extract_feature(file)
            path = os.path.join(output_directory, img_name + ".txt")
            path = path.replace("\\", "/")
            if save_features:
                np.savetxt(path, feature)

    def load_features(self, output_directory):
        """Load the features from the output directory"""
        output_directory = os.path.join(output_directory, self.name)
        output_directory = output_directory.replace("\\", "/")
        lfeatures = []
        files = os.listdir(output_directory)
        files = [os.path.join(output_directory, file).replace("\\", "/") for file in files]
        print("Loading features...")
        for file in tqdm(files):
            feature = np.loadtxt(file)
            lfeatures.append((file, feature))
        return lfeatures

    @abstractmethod
    def extract_feature(self, image):
        pass

class Histogram_HSV_Extractor(FeatureExtractor):
    """Extract HSV histogram features from an image"""

    def __init__(self):
        super().__init__("Histogram_HSV")

    def extract_feature(self, file):
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        histH = cv2.calcHist([image],[0],None,[180],[0,180])
        histS = cv2.calcHist([image],[1],None,[256],[0,256])
        histV = cv2.calcHist([image],[2],None,[256],[0,256])
        feature = np.concatenate((histH, np.concatenate((histS,histV),axis=None)),axis=None)
        return feature 

class Histogram_Color_Extractor(FeatureExtractor):
    """Extract RGB histogram features from an image"""

    def __init__(self):
        super().__init__("Histogram_Color")

    def extract_feature(self, file):
        image = cv2.imread(file)
        histB = cv2.calcHist([image],[0],None,[256],[0,256])
        histG = cv2.calcHist([image],[1],None,[256],[0,256])
        histR = cv2.calcHist([image],[2],None,[256],[0,256])
        feature = np.concatenate((histB, np.concatenate((histG,histR),axis=None)),axis=None)
        return feature 

class SIFT_Extractor(FeatureExtractor):
    """Extract SIFT features from an image"""

    def __init__(self):
        super().__init__("SIFT")

    def extract_feature(self, file):
        image = cv2.imread(file)
        sift = cv2.SIFT_create()  
        _ , des = sift.detectAndCompute(image,None)
        return des

class ORB_Extractor(FeatureExtractor):
    """Extract ORB features from an image"""

    def __init__(self):
        super().__init__("ORB")

    def extract_feature(self, file):
        image = cv2.imread(file)
        orb = cv2.ORB_create()
        _ , des = orb.detectAndCompute(image,None)
        return des
    
class GLCM_Extractor(FeatureExtractor):
    """Extract GLCM features from an image"""
    
    def __init__(self):
        super().__init__("GLCM")

    def extract_feature(self, file):
        image = cv2.imread(file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = img_as_ubyte(gray)  # Convert to 8-bit byte format
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcmMatrix = graycomatrix(gray, distances=distances, angles=angles, symmetric=True, normed=True)
        features = []
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        for prop in properties:
            features.append(graycoprops(glcmMatrix, prop).ravel())
        return np.concatenate(features)


class LBP_Extractor(FeatureExtractor):
    """Extract LBP features from an image"""
    
    def __init__(self):
        super().__init__("LBP")

    def extract_feature(self, file):
        image = cv2.imread(file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        radius = 1
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='default')
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        # Normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        return hist

class HOG_Extractor(FeatureExtractor):
    """Extract HOG features from an image"""
        
    def __init__(self):
        super().__init__("HOG")
    
    def extract_feature(self, file):
        cell_size = (25, 25)  # h x w pixels
        block_size = (50, 50)  # h x w cells
        blockStride = (25,25)
        nbins = 9  # number of orientation bins
        winSize = (350,350)
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, winSize)
        hog = cv2.HOGDescriptor(winSize, block_size, blockStride, cell_size, nbins)
        feature = hog.compute(image)
        return feature
        

class Deep_learning_Extractor(FeatureExtractor):
    """Extract deep learning features from an image"""

    def __init__(self, model_name, model, transform):
        super().__init__(model_name)
        self.model = model
        self.transform = transform

    def extract_feature(self, file):
        img = Image.open(file).convert('RGB')
        img = self.transform(img)
        img = img.unsqueeze(0)
        with torch.no_grad():
            feature = self.model(img)
        feature = feature.squeeze()
        feature = feature.numpy()
        return feature

class ResNet34_ImageNet_Extractor(Deep_learning_Extractor):
    """Extract ResNet34 features from an image pre-trained on ImageNet"""

    def __init__(self):
        model = models.resnet34(pretrained=True)
        model.fc = torch.nn.Linear(512, 100)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model.eval()
        t= transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),            
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
        super().__init__("ResNet34_ImageNet", model, t)

class ResNet34_2_steps_data_augmentation_Steps_Extractor(Deep_learning_Extractor):
    """Extract ResNet34 features from an image pre-trained on ImageNet and fine-tuned on the dataset"""

    def __init__(self):
        model_path = "../outputs/resnet34_2_steps_data_augmentation/step2_model.pth"
        model = models.resnet34(pretrained=False)
        model.fc = torch.nn.Linear(512, 100)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model.eval()
        t= transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),            
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        super().__init__("ResNet34_2_steps_data_augmentation", model, t)

class InceptionV4_2_steps_data_augmentation_Extractor(Deep_learning_Extractor):
    """Extract InceptionV4 features from an image pre-trained on ImageNet and fine-tuned on the dataset"""

    def __init__(self):
        t = transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),            
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225] )
                ])
        model = model = timm.create_model('inception_v4', pretrained=True, num_classes=100)
        model_path = "../outputs/inceptionv4_2_steps_data_augmentation/step2_model.pth"
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model.eval()
        super().__init__("InceptionV4_2_steps_data_augmentation", model, t)
        
class ViT_Extractor(Deep_learning_Extractor):
    """Extract Vision Transformer (ViT) features from an image"""

    def __init__(self):
        # Using a standard ViT model from timm
        model_name = 'vit_base_patch16_224'
        model = timm.create_model(model_name, pretrained=True, num_classes=0) # num_classes=0 returns the features before the head
        model.eval()
        
        # Standard preprocessing for ViT
        t = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        super().__init__("ViT", model, t)


    
def fusion_features(features):
    """Concatenate features"""
    concatenated_feature = np.concatenate([feature.ravel() for _, feature in features])
    return concatenated_feature