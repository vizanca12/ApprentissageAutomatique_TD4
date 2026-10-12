import matplotlib.pyplot as plt
import os
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from sklearn.neighbors import (KNeighborsClassifier,NeighborhoodComponentsAnalysis)
from sklearn.pipeline import Pipeline
import skimage.feature
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
import joblib


def NN_histogram(img_lbp,N,n_bins):
    X_lbp_histNN_counts = np.zeros([N*N , n_bins ])
    X_lbp_histNN_bins = np.zeros([N*N , n_bins+1 ])
    for j in range(N*N) :
        ii = np.floor(j/N)
        jj = j%N
        h1 = int( ii*(48/N) )
        h2 = int( (ii+1)*(48/N) )
        w1 = int( jj*(48/N) )
        w2 = int( (jj+1)*(48/N) )
        counts, bins = np.histogram(img_lbp[ h1 : h2 ,  w1 : w2].flatten(), bins=n_bins, range=(0,9))
        X_lbp_histNN_counts[j,:] = counts
        X_lbp_histNN_bins[j,:] = bins
    return X_lbp_histNN_counts , X_lbp_histNN_bins


def expression_features(img):
    """Vetor LBP + histograma de um rosto 48x48.

    Unica fonte dos parametros do descritor: detect_expression e o app
    Gradio passam os dois por aqui, entao nao ha como divergirem.
    """
    # LBP parameters
    numpoints_lbp = 25
    radius_lbp = 5
    lbp_method = "uniform"
    # historgram parameters
    N = 4
    n_bins = 10
    img_lbp = skimage.feature.local_binary_pattern(
        img, numpoints_lbp, radius_lbp, method=lbp_method)
    X_test_histNN_counts, X_test_histNN_bins = NN_histogram(img_lbp, N, n_bins)
    return X_test_histNN_counts.flatten()


def detect_expression(img, neigh):
    #prediction
    category = neigh.predict(np.array([expression_features(img)]))
    return category
