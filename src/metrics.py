from matplotlib import pyplot as plt
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
matplotlib.use('agg')

"""
This file contains metrics functions that are intended to be used to evaluate the performance of the system.
Note that classes are from 0 to 100
"""
def precision(k, retrieved_classes, relevant_class):
    """
    Precision@k = (# of recommended items @k that are relevant) / (# of recommended items)
    """
    pertinent = 0
    for i in range(k):
        if retrieved_classes[i] in relevant_class:
            pertinent += 1
    return pertinent / len(retrieved_classes)

def recall(k, retrieved_classes, relevant_class, relevant_items):
    """
    Recall@k () = (# of recommended items @k that are relevant) / (total # of relevant items)
    """
    pertinent = 0
    for i in range(k):
        if retrieved_classes[i] in relevant_class:
            pertinent += 1
    return pertinent / relevant_items 

def r_precision(k, retrieved_classes, relevant_class):
    """
    R-precision = (# of recommended items @k that are relevant) / (# of recommended items @k) 
    """
    pertinent = 0
    for i in range(k):
        if retrieved_classes[i] in relevant_class:
            pertinent += 1
    return pertinent / k

def average_precision(k, retrieved_classes, relevant_class):
    """
    Average precision AP = precision@1 + precision@2 + precision@3 + ... + precision@k / k
    """
    avg_precision = 0
    for i in range(1, k+1):
        avg_precision += r_precision(i, retrieved_classes, relevant_class)
    return avg_precision / k

def mean_average_precision(k, retrieved_classes_list, relevant_class_list):
    """
    Mean average precision MAP = average of AP for multiple queries 
    """
    map = 0
    for i in range(len(relevant_class_list)):
        map += average_precision(k, retrieved_classes_list[i], relevant_class_list[i])
    return map / len(relevant_class_list)

class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)





def create_plot_pyqt(k, retrieved_classes, relevant_class, relevant_items):
    x = []
    y = []
    for i in range(1, k):
        x.append(recall(i, retrieved_classes, relevant_class, relevant_items) * 100)
        y.append(r_precision(i, retrieved_classes, relevant_class) * 100)

    return x, y


def plot_precision_recall_curve(k, retrieved_classes, relevant_class, relevant_items):
    """
    Plot the precision-recall curve
    """
    x = []
    y = []
    for i in range(1, k):
        x.append(recall(i, retrieved_classes, relevant_class, relevant_items)*100)
        y.append(r_precision(i, retrieved_classes, relevant_class)*100)

    fig = plt.figure()
    plt.plot(x, y)
    plt.xlabel('Rappel (%)')
    plt.ylabel('Précision (%)')
    plt.title('Courbe Précision-Rappel')
    return fig
    
    