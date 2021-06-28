#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 19:30:04 2021

@author: pcgl
"""
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import math
import seaborn as sns
import pandas as pd
from matplotlib.colors import ListedColormap
from timeit import default_timer
import sys

# Chi Cuadrado
from scipy.stats import chisquare

#No mostramos los warnings
import warnings

warnings.filterwarnings("ignore")

#De sklearn
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics.pairwise import rbf_kernel  # Para funciones de base radial
from sklearn.cluster import MiniBatchKMeans
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import learning_curve


# Para trabajar con ficheros
import os


########################################
#	 	VARIABLES GLOBALES
########################################
np.random.seed(777)
MOSTRAR_GRAFICOS_TSNE = False
MOSTRAR_CURVA_DE_APRENDIZAJE = False
path="./datos"


##############################################################################
##############################################################################
##
##
##	 	 	 	 	FUNCIONES Y CLASES AUXILIARES
##
##
##############################################################################
##############################################################################
"""
Estas funciones serán de utilidad en el desarrollo del preprocesado para los 
datos. En particular, todas las funciones auxiliares que aparecen más abajo 
están dedicadas a la generación de características. Además se incluirán 
funciones auxiliares como wait() para pausar la ejecución

"""

def wait():
	"""
	Función para pausar la ejecución
	"""
	input("\n--- Pulsar tecla para continuar ---\n")
	


# Clase que representa la generación de característica de distancia mínima entre cada par de CETOs en cada instancia
class DistanciaMinima:
    
    def __init__(self):
        pass
    
    #Cálculo de las características a añadir
    def fit(self,X,y):
        self.mindist = np.zeros(X.shape[0])       # Aquí guardaremos la distancia máxima entre cada par de CETOs para cada instancia
        
        for k in range(X.shape[0]):             # Para cada instancia
          dist = []
          for i in range(16):                   # Calculamos las distancias entre cada par distinto de CETOs
            for j in range(i+1,16):
              dist.append(norm([(X[k,i] - X[k,j]), (X[k,i+16] - X[k,j+16])]))

          self.mindist[k] = np.min(np.array(dist))  # Añadimos la distancia mínima
        return self

    # Añadimos la propia característica a los datos
    def transform(self,X):
        return np.c_[X, self.mindist]


# Clase que representa la generación de característica de distancia máxima entre cada par de CETOs en cada instancia
class DistanciaMaxima:
    
    def __init__(self):
        pass
    
    #Cálculo de la característica a añadir
    def fit(self,X,y):
        self.maxdist = np.zeros(X.shape[0])        # Aquí guardaremos la distancia máxima entre cada par de CETOs para cada instancia
        
        for k in range(X.shape[0]):                 # Para cada instancia
          dist = np.zeros((X.shape[1], X.shape[1]))
          for i in range(16):                           
            for j in range(i+1,16):
              dist[i,j] = norm([(X[k,i] - X[k,j]), (X[k,i+16] - X[k,j+16])])  # Calculamos las distancias entre cada par distinto de CETOs
          self.maxdist[k] = np.max(dist)                 # Añadimos la distancia máxima
        return self

    # Añadimos la propia característica a los datos
    def transform(self,X):
        return np.c_[X, self.maxdist]


# Clase que representa la generación de característica de la media de las distancias mínimas entre cada CETOs con los demás
class MediaDistanciasMinimas:
    
    def __init__(self):
        pass
    
    #Cálculo de la característica a añadir
    def fit(self,X,y):
        self.mindist = np.zeros(X.shape[0])       # Aquí guardaremos las características a añadir
        
        for k in range(X.shape[0]): # Para cada instancia
          mindists = np.zeros(16) # Aquí almacenaremos las distancias mínimas entre cada CETO y cualquier otro
          for i in range(16):         # Para cada CETO en dicha instancia
            min = float('inf')
            for j in range(16):
              curdist = norm([(X[k,i] - X[k,j]), (X[k,i+16] - X[k,j+16])])
              if curdist < min and not i == j:
                min = curdist                     # Calculamos la distancia mínima entre el CETO fijo y cualquier otro
            mindists[i] = min                     # La almacenamos
          self.mindist[k] = np.mean(mindists)     # Añadimos la media de las distancias mínimas calculadas
        return self

    # Añadimos la propia característica a los datos
    def transform(self,X):
        return np.c_[X, self.mindist]


# Clase que representa la generación de característica de la media de las distancias máximas entre cada CETOs con los demás
class MediaDistanciasMaximas:
    
    def __init__(self):
        pass
    
    #Cálculo de la característica a añadir
    def fit(self,X,y):
        self.maxdist = np.zeros(X.shape[0])       # Aquí guardaremos las características a añadir
        
        for k in range(X.shape[0]):     # Para cada instancia
          maxdists = np.zeros(16) # Aquí almacenaremos las distancias mínimas entre cada CETO y cualquier otro
          for i in range(16):       # Para cada CETO en dicha instancia
            max = 0
            for j in range(16):
              curdist = norm([(X[k,i] - X[k,j]), (X[k,i+16] - X[k,j+16])])
              if curdist > max and not i == j:
                max = curdist                   # Calculamos la distancia máxima entre el CETO fijo y cualquier otro
            maxdists[i] = max                   # La almacenamos
          self.maxdist[k] = np.mean(maxdists)   # Añadimos la media de las distancias máximas calculadas
        return self

    # Añadimos la propia característica a los datos
    def transform(self,X):
        return np.c_[X, self.maxdist]

  
# Clase simple que representa el cálculo de la característica 'Distancia media al centroide de los puntos'
class DistanciaMediaACentroide:
    
    def __init__(self):
        pass
    
    #Cálculo de la característica a añadir
    def fit(self,X,y):
        self.distcent = np.zeros(X.shape[0])       # Aquí guardaremos la nueva característica a añadir

        for k in range(X.shape[0]):               # Para cada instancia

          # Calculamos las coordenadas del centroide
          x_centr = np.mean(X[k,0:16])
          y_centr = np.mean(X[k,16:32])

          dists = np.zeros(16)
          for i in range(16):
            dists[i] = norm([(X[k,i]-x_centr), (X[k,i+16]-y_centr)])  # Nos quedamos con la distancia de cada CETO al centroide

          self.distcent[k] = np.mean(dists)         # Por último, guardamos la media de esas distancias

        return self

    #Método para aplicar la transformación calculada en fit.
    def transform(self,X):
        return np.c_[X, self.distcent]


# Clase simple que representa el cálculo de la característica de la varianza en las coordenadas de los CETOs
class Varianza:
    
    def init(self):
        pass
    
    #Cálculo de la característica a añadir
    def fit(self,X,y):
        self.vars = np.zeros(X.shape[0])       # Aquí guardaremos la nueva característica a añadir

        for k in range(X.shape[0]):              # Para cada instancia

            # Ponemos los datos en forma de pares [x,y]
            pares = []
            for i in range(16):
                pares.append([X[k,i], X[k,i+16]])
            
            self.vars[k] = np.var(pares)      # Calculamos y guardamos la varianza de las coordeanadas
        return self
    
    #Método para aplicar la transformación calculada en fit.
    def transform(self,X):
        return np.c_[X, self.vars]


# Clase para generar una característica haciendo un test estadístico de uniformidad Chi^2 sobre las coordenadas X de cada CETO en una instancia
# La hipótesis nula H0 es que las coordenadas X de los CETOs con una distribución Uniforme sobre las posiciones X

# que representa el p-value del test Chi cuadrado
class Chi2x:
    
    def init(self):
        pass
    
    # Método para encontrar las variables con las que nos vamos a quedar
    def fit(self,X,y):
        self.p = np.zeros(X.shape[0])

        for k in range(X.shape[0]):
          freq = np.zeros(3)
          
          for i in range(16):
            if X[k][i]<566/3:
              freq[0] = freq[0]+1
            elif X[k][i]<2*566/3:
              freq[1] = freq[1]+1
            else:
              freq[2] = freq[2]+1     
          
          chisq, p2 = chisquare(freq) # f_exp por defecto asume que las frecuencias son equivalentes

          self.p[k] = p2

        return self
    
    def transform(self,X):
        return np.c_[X, self.p]

# Clase simple que representa el p-value del test Chi cuadrado con una distribución Uniforme sobre las posiciones Y
class Chi2y:
    
    def init(self):
        pass
    
    # Método para encontrar las variables con las que nos vamos a quedar
    def fit(self,X,y):
        self.p = np.zeros(X.shape[0])

        for k in range(X.shape[0]):
          freq = np.zeros(3)
          
          for i in range(16,32):
            if X[k][i]<566/3:
              freq[0] = freq[0]+1
            elif X[k][i]<2*566/3:
              freq[1] = freq[1]+1
            else:
              freq[2] = freq[2]+1     
          
          chisq, p2 = chisquare(freq) # f_exp por defecto asume que las frecuencias son equivalentes

          self.p[k] = p2

        return self
    
    def transform(self,X):
        return np.c_[X, self.p]



# Clase simple que representa el p-value del test Chi cuadrado con una distribución Uniforme
class Chi2Diagonal:
    
    def init(self):
        pass
    
    # Método para encontrar las variables con las que nos vamos a quedar
    def fit(self,X,y):
        self.p = np.zeros(X.shape[0])

        for k in range(X.shape[0]):      #Para cada instancia
          freq = np.zeros(3)

          for i in range(16):           #Para cada CETO en dicha instancia
            # Le asignamos una clase en función de su posición
            if (X[k][i]<2*566/3 and X[k][i+16]>566/3) and not (X[k][i]>566/3 and X[k][i+16]<2*566/3):
              freq[0] = freq[0]+1
            elif (X[k][i]>566/3 and X[k][i+16]<2*566/3) and not (X[k][i]<2*566/3 and X[k][i+16]>566/3):
              freq[1] = freq[1]+1
            else:
              freq[2] = freq[2]+1     
          
          chisq, p2 = chisquare(freq) # f_exp por defecto asume que las frecuencias son equivalentes

          self.p[k] = p2

        return self
    
    def transform(self,X):
        return np.c_[X, self.p]




# CLASE PARA LA RED DE FUNCIONES DE BASE RADIAL
class RBFNetworkRegressor(BaseEstimator, RegressorMixin):
    """
       Regresor de red de funciones (gaussianas) de base radial.
       Uso de un modelo Ridge para ajustar los pesos del modelo final.
    """

    def __init__(self, k = 16, alpha = 1.0, batch_size = 128,
                 random_state = None, gamma=1):
        """
          Instancia un regresor con los hiperparámetros:
             - k:             número de centros a elegir.
             - alpha:         constante de regularización.
             - batch_size:    tamaño del batch para el clustering no supervisado.
             - random_state:  semilla aleatoria.ç
             - gamma:         Parámetro para controlar la escala de r
        """

        self.k = k
        self.alpha = alpha
        self.batch_size = batch_size
        self.random_state = random_state
        self.centers = None                 # Centroides
        self.r = None               
        self.gamma = gamma          

    def generar_centros_kmeans(self, X):
        """
          Elección de k centros a partir del algoritmo de K-Medias
        """

        init_size = 3 * self.k if 3 * self.batch_size <= self.k else None

        kmeans = MiniBatchKMeans(
            n_clusters = self.k,
            batch_size = self.batch_size,
            init_size = init_size,
            random_state = self.random_state)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_

    def calcular_radio(self):
        """
          Cálculo del radio para aplicar en el kernel
        """

        # Calculamos la media de las distancias mínimas entre los centros
        mindists = np.zeros(self.centers.shape[1]) #Aquí almacenaremos las distancias entre cada par de CETOs
        
        for i in range(self.centers.shape[0]):
          min = float('inf')
          for j in range(self.centers.shape[0]):
            curdist = norm(self.centers[i]-self.centers[j])
            if curdist < min and not i == j:
              min = curdist
          mindists[i] = min

        self.r = self.gamma * mindists    # Nuestro sigma de la función gaussiana
       

    def _transform_rbf(self, X):
        """
          Aplicación del kernel RBF.
        """

        ret = []
        for i in range(self.centers.shape[0]):
          ret.append(rbf_kernel(X, self.centers[i].reshape(-1,1).T , 1 / (2 * self.r[i] ** 2)))
        
        return np.array(ret).T.reshape(X.shape[0], self.centers.shape[0])

    def fit(self, X, y):
        """
          Entrena el modelo.
        """

        self.model = Ridge(alpha = self.alpha,
                           random_state = self.random_state)

        # Obtenemos los k centros y el radio para el kernel 
        self.generar_centros_kmeans(X)
        self.calcular_radio()

        # Transformamos los datos a través de nuestro kernel y centros
        Z = self._transform_rbf(X)

        # Entrenamos
        self.model.fit(Z, y)

        # Guardamos los coeficientes obtenidos
        self.intercept_ = self.model.intercept_
        self.coef_ = self.model.coef_

        return self

    def score(self, X, y = None):
        Z = self._transform_rbf(X)

        return self.model.score(Z, y)

    def predict(self, X):
        Z = self._transform_rbf(X)

        return self.model.predict(Z)

    def decision_function(self, X):
        Z = self._transform_rbf(X)

        return self.model.decision_function(Z)



##############################################################################
##############################################################################
##
##
##	 	 	 	 	 VISUALIZACIÓN DE DATOS
##
##
##############################################################################
##############################################################################
"""
En este módulo estarán las funciones necesarias para visualizar los datos

"""
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
  Fuente:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
  
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing fit and predict methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:CV splitter,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:StratifiedKFold used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:KFold is used.

        Refer :ref:User Guide <cross_validation> for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:joblib.parallel_backend context.
        ``-1`` means using all processors. See :term:Glossary <n_jobs>
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)

    # Transformamos los scores en función del error cuadrático medio  
    train_scores = np.sqrt(-train_scores)
    test_scores = np.sqrt(-test_scores)                    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt



def plt_errors(X_train,y_train,X_test,y_test,model):
	"""
	Genera dos plots:
		1.- Gráfica de residuos
		2.- Gráfica valores predecidos vs valores observados

	Parameters
	----------
	X_train : Dataset de entrenamiento
	y_train : Etiquetas de entrenamiento
	X_test : Dataset de test
	y_test : Etiquetas de test
	model : Modelo predictor

	Returns
	-------
	Gráficas (1) y (2).

	"""	
	for name, X, y, name in [("training", X_train, y_train, 'train'), ("test", X_test, y_test, 'test')]:
	    y_pred = model.predict(X)
	    # Creamos el plot de RESIDUOS
	    fig, ax = plt.subplots()
	      
	    ax.plot(y, y-y_pred, 'bo-', linewidth=0, ms=0.75, label='y-y_pred')
	    ax.plot([min(y), max(y)],[0,0], 'ro-', linewidth=2, ms=0.5, label='error 0')
	    ax.set(title='Gráfica de residuos en '+str(name))
	    ax.set_xlabel('Valores observados')
	    ax.set_ylabel('Residuo')
	    ax.legend(loc='lower right', shadow=False)
	    
	    plt.show()
    
    
	    # Creamos el plot PREDICCIÓN VS REAL
	    fig, ax = plt.subplots()
	      
	    ax.plot(y, y_pred, 'bo-', linewidth=0, ms=0.75, label='Predicción vs Real')
	    ax.plot([max(min(y),min(y_pred)), min(max(y),max(y_pred))],[max(min(y),min(y_pred)), min(max(y),max(y_pred))], 'ro-', linewidth=2, ms=0.5, label='Identidad')
	    ax.set(title='Error de predicción '+str(name))
	    ax.set_xlabel('Valores observados')
	    ax.set_ylabel('Valores predichos')
	    ax.legend(loc='lower right', shadow=False)
	
	    plt.show()





def plt_freq_train_test(y_train, y_test):
	"""
	Description
	----------
	Muestra por pantalla los histogramas en términos de frecuencias relativas
	de los conjuntos train y test.

	"""	
	print("Frecuencias relativas de las potencias totales por rangos de los valores de entrenamiento")
	
	bin = np.linspace(np.min(y_train),np.max(y_train), 100)
	bin_heights, bin_borders, _ = plt.hist(y_train, bins=bin, label='histogram', color='green', weights=np.zeros_like(y_train) + 1. / y_train.shape[0])
	bin_heights = bin_heights / sum(bin_heights)
	plt.legend()
	plt.ylabel("Frecuencias relativas por rango")
	plt.xlabel("Potencia (MW)")
	plt.show()
	
	print("Frecuencias relativas de las potencias totales por rangos de los valores de test")
	
	bin = np.linspace(np.min(y_test),np.max(y_test), 100)
	bin_heights, bin_borders, _ = plt.hist(y_test, bins=bin, label='histogram', color='red', weights=np.zeros_like(y_test) + 1. / y_test.shape[0])
	bin_heights = bin_heights / sum(bin_heights)
	plt.legend()
	plt.ylabel("Frecuencias relativas por rango")
	plt.xlabel("Potencia (MW)")
	plt.show()
	

def plt_todo(X_train,y_train,X_test,y_test):
	# Mostramos la distribución de los cojnuntos train y test
	plt_freq_train_test(y_train, y_test)

	# Graficamos la distribución de los CETOs en la superficie
	for i in range(16):
	  print("---Gráfico nube de puntos del CETO ", i+1, "---")
	  sns.scatterplot(x=X_train[:,i], y=X_train[:,i+16], palette="coolwarm", s=1)
	  plt.show()
	
	for i in range(16):
	  print("---Gráfico nube de puntos del CETO ", i+1, "---")
	  sns.scatterplot(x=X_train[:,i], y=X_train[:,i+16], color='#3A76AF', s=0.05 )
	plt.show()

		
	Xdf = pd.DataFrame(X_train)		 # Lo pasamos a pandas DataFrame

	# Correlaciones
	print("---Matriz de correlación de los atributos iniciales---")
	corrMatrix = Xdf.iloc[:,:32].corr()
	sns.heatmap(corrMatrix, vmin=-1, vmax=1, cmap="RdBu_r")
	plt.show()
	
	print("---Matriz de correlación de todos los atributos---")
	corrMatrix = Xdf.corr()
	sns.heatmap(corrMatrix, vmin=-1, vmax=1, cmap="RdBu_r")
	plt.show()

	# Gráficos t-SNE
	if MOSTRAR_GRAFICOS_TSNE:
		# Comentado debido al gran tiempo de ejecución
		print("-> TSNE sólo con los atributos originales 2D")
		tsne = TSNE()
		tsne_results = tsne.fit_transform(Xdf.iloc[:,:32])
		sns.scatterplot(x=tsne_results[:,0], y=tsne_results[:,1], hue=y_train, palette="coolwarm")
		plt.show()
		
		
		print("-> TSNE sólo con los atributos originales 3D")
		tsne = TSNE(n_components=3)
		tsne_results = tsne.fit_transform(Xdf.iloc[:,:32])
	

##############################################################################
##############################################################################
##
##
##	 	 	 	 	 FUNCIONES LECTURA DE DATOS
##
##
##############################################################################
##############################################################################
"""
Vamos a crear una función para leer los archivos. Leeremos sólo el archivo de
datos de Sydney porque los conjuntos de datos contienen muestras que no se 
extraen de una misma distribución. Por lo tanto nos limitaremos a hacer el 
estudio con un conjunto de datos, el cual tiene un número suficiente de instancias 
como para poder obtener resultados significativos (alrededor de 70000)

"""
def leerArchivos(path):
  """
	Description
	-----------
	Lee los datos de path como un csv en donde la última columna
	es la característica a predecir
  
  """
  data_tasmania = pd.read_csv(path + "/Tasmania_Data.csv")
  data_adelaide = pd.read_csv(path + "/Adelaide_Data.csv")
  data_sydney = pd.read_csv(path + "/Sydney_Data.csv")
  data_perth = pd.read_csv(path + "/Perth_Data.csv")


  return data_tasmania, data_adelaide, data_sydney, data_perth







##############################################################################
##############################################################################
##
##
##	 	 	 	 	 	TRATADO DE DATOS
##
##
##############################################################################
##############################################################################
"""
Funciones para añadir características y preprocesar datos
"""

def comprobacion_datos(data_tasmania, data_adelaide, data_sydney, data_perth):
	testsize=0.20

	X_tasmania = data_tasmania.iloc[:,:32].values
	y_tasmania = data_tasmania.iloc[:,-1].values
	X_adelaide = data_adelaide.iloc[:,:32].values
	y_adelaide = data_adelaide.iloc[:,-1].values
	X_perth = data_perth.iloc[:,:32].values
	y_perth = data_perth.iloc[:,-1].values
	
	
	X = data_sydney.iloc[:,:32].values
	y = data_sydney.iloc[:,-1].values
	
	
	
	X_train_tasmania, X_test_tasmania, y_train_tasmania, y_test_tasmania = train_test_split(X_tasmania,y_tasmania, test_size = testsize)
	X_train_adelaide, X_test_adelaide, y_train_adelaide, y_test_adelaide = train_test_split(X_adelaide,y_adelaide, test_size = testsize)
	X_train_perth, X_test_perth, y_train_perth, y_test_perth = train_test_split(X_perth,y_perth, test_size = testsize)
	
	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = testsize)
	
	
	print("Media de potencia de los datos de Sydney: ", np.mean(y_train))
	print("Media de potencia de los datos de Tasmania: ", np.mean(y_train_tasmania))
	print("Media de potencia de los datos de Adelaide: ", np.mean(y_train_adelaide))
	print("Media de potencia de los datos de Perth: ", np.mean(y_train_perth))
	print("")


def limpiar_valores_atipicos(X_train, y_train):
	"""
	Description
	----------
	 
	Elimina valores de posición fuera del cuadrado [0,566]x[0,566]

	"""
	print("\nEliminando valores atípicos:")
	Xdf = pd.DataFrame(X_train)
	ydf = pd.DataFrame(y_train)
	print("\tNúmero de instancias antes: ", Xdf.shape[0])
	for cols in Xdf.columns.tolist():
		# Eliminamos valores inferiores a 0
	    rows = (Xdf[cols] >= 0)
	    Xdf = Xdf.loc[rows]
	    ydf = ydf.loc[rows]
	
		# Eliminamos valores superiores a 566
	    rows = (Xdf[cols] <= 566)
	    Xdf = Xdf.loc[rows]
	    ydf = ydf.loc[rows]
		
	print("\tNúmero de instancias después: ", Xdf.shape[0])
	X_train = Xdf.values
	y_train = ydf.values.flatten()
	
	print()
	
	return X_train, y_train



def get_pipelines_transformaciones():
	"""
	Description
	-------
	Devuelve los cauces de las transformaciones.

	"""
	featuregeneration = [("DistanciaMinima", DistanciaMinima() ),
                     ("DistanciaMaxima", DistanciaMaxima() ),
                     ("MediaDistMin", MediaDistanciasMinimas() ),
                     ("MediaDistMax", MediaDistanciasMaximas() ),
                     ("DistMediaCentroide", DistanciaMediaACentroide()),
                     ("Chi2x",Chi2x()),
                     ("Chi2y",Chi2y()),
                     ("Chi2Diagonal", Chi2Diagonal())]

	scaler = [("MinMaxScaler", MinMaxScaler())]
	
	pipegeneration = Pipeline(featuregeneration)
	pipescaler = Pipeline(scaler)
	
	return pipegeneration, pipescaler

##############################################################################
##############################################################################
##
##
##	 	 	 	 	 EJECUCIÓN DE TODO EL PROBLEMA
##
##
##############################################################################
##############################################################################
"""
En esta parte vamos a incluir las órdenes que hagan ejecutar toda la práctica

"""
def ejecucion():
	"""
	Description
	-----------
	Ejecuta todo el problema

	"""
	print("Problema WEC. Inicio de ejecución. Leyendo datos...\n")
	# Leemos todos los datos
	data_tasmania, data_adelaide, data_sydney, data_perth = leerArchivos(path)
	
	# Comprobamos que no siguen una misma distribución de potencias
	comprobacion_datos(data_tasmania, data_adelaide, data_sydney, data_perth)

	# Dividimos entre características y etiquetas
	X = data_sydney.iloc[:,:32].values  	# Nos quedamos con las posiciones	
	y = data_sydney.iloc[:,-1].values
	
	# Separamos en train y test
	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20)	
	print("Cantidad total de datos del archivo Sydney_Data.csv: ", X.shape[0])
	print("Cantidad total de datos en entrenamiento: ", X_train.shape[0])
	print("Cantidad total de datos en test: ", X_test.shape[0])
	
	# Cogemos un resumen estadístico inicial de los datos
	print("\nResumen estadístico inicial de los datos de entrenamiento:")
	print(pd.DataFrame(X_train).describe())
	
	# Limpiamos los valores que estén fuera del rango [0,566]
	X_train, y_train = limpiar_valores_atipicos(X_train, y_train)
	
	
	##########
	# 	Transformaciones de datos
	##########
	print("\nTransformamos los datos añadiendo características y cambiando escalas")

	# Reducción de la escala de la variable a predecir: Debido a que la variable
	# a predecir tiene una escala mucho mayor que la de los atributos que usamos 
	# para predecir, lo que haremos será reducir su escala para que los pesos que 
	# se encuentren en los modelos que deban hallarse no tengan que ser de forma
	# intrínseca tan grandes y que algunos modelos les cueste llegar hasta ellos.
	# Además, los parámetros de regularización tendrán un ajuste más complicado
	# en estos casos. Por tanto pasaremos de W (Vatios) a MW (Megavatios)
	y_train = y_train/1e6
	y_test = y_test/1e6
	print("Nuevos valores de la variable a predecir tras el paso a Mega Vatios")
	pd.DataFrame(y_train).describe()
	
	
	# Transformamos los datos 
	pipegeneration, pipescaler = get_pipelines_transformaciones()
	
	# Guardado
	"""
	filepip = path+'pipeline1.txt'
	if os.path.exists(filepip):
		X_train = np.loadtxt(filepip)
	else:
		X_train = pipegeneration.fit_transform(X_train)
		np.savetxt(filepip,X_train)
		
	# Mostramos distribuciones
	for i in range(32, X_train.shape[1]):
	    sns.displot(X_train[:,i], label=i)
	    plt.show()
	pipescaler.fit(X_train)
	X_train = pipescaler.transform(X_train)
	
	# Guardado
	filepip2 = path+'pipeline2.txt'
	if os.path.exists(filepip2):
		X_test = np.loadtxt(filepip2)
	else:
		X_test = pipegeneration.fit_transform(X_test)     # Las características se deben ejecutar sobre el propio conjunto de test
		np.savetxt(filepip2,X_test)
		
	X_test = pipescaler.transform(X_test)             # Pero la normalización se debe de realizar respecto a la hecha en el conjunto de entrenamiento
	"""
	X_train = pipegeneration.fit_transform(X_train)
	# Mostramos distribuciones
	for i in range(32, X_train.shape[1]):
	    sns.displot(X_train[:,i], label=i)
	    plt.show()
	pipescaler.fit(X_train)
	X_train = pipescaler.transform(X_train)
	
	X_test = pipegeneration.fit_transform(X_test)     # Las características se deben ejecutar sobre el propio conjunto de test
	X_test = pipescaler.transform(X_test)             # Pero la normalización se debe de realizar respecto a la hecha en el conjunto de entrenamiento

	
	# Mostramos una descripción de las características añadidas
	Xdf = pd.DataFrame(X_train)	
	print(Xdf.iloc[:,32:].describe())
	
	print("Analizamos si hay valores perdidos:")
	if Xdf.isnull().sum().to_numpy().sum():        #Comprueba si hay ALGÚN valor perdido en todo el dataset
	    print("Número de valores perdidos por cada atributo: ", X_train.isnull().sum().to_numpy())
	else:
	    print("No hay valores perdidos en el dataset")

	

	##########
	# 	Visualizamos los datos
	##########
	plt_todo(X_train, y_train, X_test, y_test)


	
	##########
	# 	Definimos los modelos
	##########
	maxIter = 2000
	randomstate = 777
	LinRegSpace = np.logspace(-4, 1, 6)
	MLPSpace = np.logspace(-4, 1, 6)
	
	# Se ha ejecutado con los siguientes valores:
	# SVRCSpace = np.logspace(-4,-2,3)
	# SVRepsilonSpace = np.logspace(-4,-2,3)
	# Para estos valores, los mejores resultados obtenidos han sido
	# 1e-3 en ambos valores. Por cuestiones duras de eficiencia, se han dejado sólo
	# estos valores de cara a una posterior ejecución
	maxIterSVR = 8000
	SVRCSpace = [1e-3]      
	SVRepsilonSpace = [1e-3]
	
	BoostSpace = [0.5, 0.7, 0.9]
	RFSpace = [50, 100, 200]
	kSpace = np.logspace(4,6,3,base=2).astype('int')
	gammaSpace = np.linspace(1,3,3)
	alpha = np.logspace(-2,0,3)
	
	algs_and_params = [
	                    #Regresión lineal sólo car. originales
	                    [{'Model': [Ridge(max_iter=maxIter, random_state=randomstate)], #Modelos y parámetros para primer pipeline
	                        "Model__alpha": LinRegSpace}],
	                   
	                   #Regresión lineal con todas las características
	                    [{'Model': [Ridge(max_iter=maxIter, random_state=randomstate)], #Modelos y parámetros para primer pipeline
	                        "Model__alpha": LinRegSpace}],
	                    
	                   #Perceptrón multicapa
	                    [{'Model': [MLPRegressor(hidden_layer_sizes=(100,100), activation='relu', solver="sgd", max_iter=200, 
	                                             learning_rate='adaptive', learning_rate_init=1e-2,
	                                             random_state=randomstate, early_stopping=True)],
	                        "Model__alpha": MLPSpace}], #La regularización es L2 de forma implícita
	                    
	                    #SVR, con kernel lineal
	                    [{'Model': [SVR(kernel='poly', max_iter = maxIterSVR)],
	                        "Model__C": SVRCSpace,
	                        "Model__epsilon": SVRepsilonSpace}],
	
	                    #AdaBoost
	                    [{'Model': [AdaBoostRegressor(n_estimators=100, loss='linear', random_state=randomstate)],
	                        "Model__learning_rate": BoostSpace}],
	                   
	                    #Random Forest
	                    [{'Model': [RandomForestRegressor(max_depth=16, max_features='sqrt', n_jobs=-1)],
	                        "Model__n_estimators": RFSpace}],
	
	                     #Funciones de base radial
	                    [{'Model': [RBFNetworkRegressor(batch_size=128)],
	                        "Model__k": kSpace,
	                        "Model__gamma": gammaSpace,
	                        "Model__alpha": alpha}]
	                   ]


	# Ejecutamos Cross-Validation para encontrar el mejor modelo
	print("Ejecutando Cross-Validation y el fit para la selección del mejor modelo...")
	start = default_timer()
	
	pipe = Pipeline([('Model', AdaBoostRegressor())])
	
	
	tiempo = []
	best_models = []
	best_params = []
	for i in range(len(algs_and_params)):
	  start = default_timer()
	  best_model = GridSearchCV(pipe, 
	                algs_and_params[i], 
	                scoring=['neg_mean_squared_error',
	                      'r2'],	
	                refit='neg_mean_squared_error',
	                return_train_score=True,			   
	                cv=5,
	                n_jobs=-1,
	                verbose=3)
	  
	  if i == 0: 	# Para comparar regresiones lineales con y sin 
	    best_model.fit(X_train[:,:32], y_train)
	  else:
	    best_model.fit(X_train[:,32:], y_train)
		
	  tiempo.append(default_timer() - start)
	  best_models.append(best_model)
	  best_params.append(best_model.best_params_['Model'])
	  
	 
	 
	arraytext = ['Regresión lineal originales', 'Regresión lineal', 'Perceptrón Multicapa', 'SVR', 'AdaBoost', 'Random Forest', 'Funciones de base radial']
	minerrortrain = minerrortest = float('inf')
	
	for i in range(len(algs_and_params)):
	
	    print("\n>>> Resultados para ", arraytext[i], " <<<")
	    print("\t-> Tiempo empleado en validación cruzada: {:.3f} minutos".format(tiempo[i] / 60.0))
	    print('\t-> Mejores parámetros: {}.'.format(best_params[i]))
	    print('\t-> RMSE en validación cruzada: {:.4f}'.format(np.sqrt(-best_models[i].best_score_)))
	    if i == 0 or i == 1 :
	      print('\t-> Coeficientes de la regresión lineal:{}'.format(best_models[i].best_estimator_['Model'].coef_))
	
	    if i == 0:
	      y_pred = best_models[i].predict(X_train[:,:32])
	    else:
	      y_pred = best_models[i].predict(X_train[:,32:])
	    print(X_train.shape)
	    print(y_train.shape)
	    print(y_pred)
	    errortrain = np.sqrt(mean_squared_error(y_train, y_pred))
	    if errortrain < minerrortrain:
	        minerrortrain = errortrain
	        minindextrain = i
	    print("\t-> RMSE en entrenamiento: {:.4f}".format(errortrain))
	    print('\t-> R2 en entrenamiento: {:.4f}'.format(r2_score(y_train, y_pred)))
	
	    if i == 0:
	      y_pred = best_models[i].predict(X_test[:,:32])
	    else:
	      y_pred = best_models[i].predict(X_test[:,32:])
		  
	    errortest = np.sqrt(mean_squared_error(y_test, y_pred))
	    if errortest < minerrortest:
	        minerrortest = errortest
	        minindextest = i
	    print("\t-> RMSE en test: {:.4f}".format(np.sqrt(mean_squared_error(y_test, y_pred))))
	    print('\t-> R2 en test: {:.4f}'.format(r2_score(y_test, y_pred)))
		
	
	print("El mejor modelo en entrenamiento (menor RSME) ha sido encontrado utilizando el ", arraytext[minindextrain])

	if minindextrain == minindextest:
	    print("El mejor modelo en test ha sido también el mismo modelo")
	else:
	    print("El mejor modelo en test es sin embargo el modelo obtenido con el ", arraytext[minindextest])
	
	
	
	
	#########
	# VISUALIZACIONES FINALES
	#########
	print("--- Visualizaciones finales ---")
	print("1.- Visualización de errores")
	plt_errors(X_train,y_train,X_test,y_test,best_models[minindextest])

	print("2.- Visualización de errores")
	pesos = [ 1.07379869e-03,  7.87559049e-04,  2.61783723e-04,  7.95175038e-04,
			  3.33727099e-04, -3.28745176e-04, -8.12812497e-04,  1.98531982e-04,
			  1.41510010e-04,  3.45870754e-04,  9.46068129e-05,  3.25965922e-04,
			 -5.07098155e-04, -6.23780502e-04,  2.74622002e-03, -2.48875523e-04,
			  2.05125540e-03,  8.91825775e-04,  1.09566431e-03, -8.20629593e-04,
			 -2.23073843e-05,  1.37623224e-03, -5.97086384e-04, -1.65091458e-03,
			  4.82577195e-04,  3.67573152e-04,  1.03192110e-03,  1.44662765e-03,
			 -7.97422227e-04,  4.61735339e-05, -7.26298979e-04,  1.72964373e-04,
			  1.04299163e-02,  4.01842467e-03,  4.03132596e-02,  2.42084193e-02,
			  9.35376169e-02,  2.15294113e-03,  1.30902095e-02,  5.98624347e-03]
 
	print(np.linspace(1,40,1))
	plt.bar(np.linspace(1,40,40), pesos)
	plt.xlabel("Número del atributo")
	plt.ylabel("Peso asociado al atributo")
	plt.show()
	
	if MOSTRAR_CURVA_DE_APRENDIZAJE:
		print("3.- Curvas de aprendizaje")
		fig, axes = plt.subplots(3, 1, figsize=(10, 15))
		plot_learning_curve(best_models[minindextest], 'Learning Curves', X_train, y_train, axes=axes, cv = 5, n_jobs=-1)
		plt.show()
	
	print("---Fin de la ejecución---\n")


##############################################################################
ejecucion()




