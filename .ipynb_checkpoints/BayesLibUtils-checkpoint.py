"""BayesLibUtils. Proporciona las herramientas utilitarias para la obtención de metricas, transformaciones de datos y calculo de bloques bayesianos."""
# ------------------------------------------------
# Name        : BayesLibUtils.py
# Authors     : Juan Tarbes Vergara
#               Pamela Morales Vergara
# Contact     : j.tarbesvergara@uandresbello.edu
#               p.moralesvergara2@uandresbello.edu
# Licence     : Universidad Andrés Bello (Chile)
# ------------------------------------------------

# %% Librerias
# Libreria para el manejo de archivos
import os

# Libreria BNLEARN es un paquete de Python para aprender la estructura gráfica de redes bayesianas, 
# aprendizaje de parámetros, métodos de inferencia y muestreo. 
import bnlearn as bn

#import numpy as np

# libreria para el manejo de las métricas
from sklearn.metrics import *

# Libreria para la obntención de bloques bayesianos
from astropy.stats import *

# %% Obtención de métricas
def getMetrics (y_true, y_pred, y_class, fold_type, fold_no, flagModel):
    """ Obtención de métricas para los modelos Bayesianos
    
    Descripción
    -----------
    Calcula las métricas de los modelos bayesianos calculados de Python o R y los registra en un archivo de texto
    Las metricas calculadas son:
        1. Accuracy: Puntaje de clasificación de precisión. En la clasificación de etiquetas múltiples, esta función 
        calcula la precisión del subconjunto: el conjunto de etiquetas predichas para una muestra debe coincidir exactamente 
        con el conjunto de etiquetas correspondiente en y_true.
        
        2. Balanced Accuracy: Calculo de precisión equilibrada. La precisión equilibrada en problemas de clasificación binaria 
        y multiclase para hacer frente a conjuntos de datos desequilibrados. Se define como el promedio de recuerdo obtenido 
        en cada clase. El mejor valor es 1 y el peor valor es 0.
        
        3. Precision: Calcula la precisión. La precisión es la relación "tp / (tp + fp)" donde "tp" está el número de 
        verdaderos positivos y "fp" el número de falsos positivos. La precisión es intuitivamente la capacidad del 
        clasificador de no etiquetar como positiva una muestra que es negativa. El mejor valor es 1 y el peor valor es 0.
        
        4. Recall: Calcula la recuperación. La recuperación es la relación "tp / (tp + fn)" donde "tp" es el número de 
        verdaderos positivos y "fn" el número de falsos negativos. El retiro es intuitivamente la capacidad del 
        clasificador de encontrar todas las muestras positivas. El mejor valor es 1 y el peor valor es 0.
        
        5. ROC AUC: Calcula el área bajo la curva de características operativas del receptor (ROC AUC) a partir de las 
        puntuaciones de predicción. 
        
        6. Ratio: Calcula la proporción de la clase
        
    Parámetros
    ----------
    y_true: Matriz de tipo 1d o indicador de etiqueta / matriz dispersa. Etiquetas de verdad fundamental (correctas).
    
    y_pred: Matriz de tipo 1d o indicador de etiqueta / matriz dispersa. Etiquetas previstas / objetivos estimados, 
    como la retorna un clasificador.
    
    y_class: Matriz de tipo 1d que contiene las etiquetas de la clase a calcular.
    
    fold_type: Indicador del tipo de metricas que se está calculando.  Valores posibles:
        a. "TRAIN"
        b. "TEST"
        
    fold_no: Valor númerico que indica el cross-validation del cual se está solicitando las métricas
    flagmodel: Indicador númerico que permite identificar en que carpeta se deberá guardar el resultado de las métricas.
        a. 0 = Identifica que se ejecutó el modelo Python y registra las metricas en "\Experimentos"
        a. 1 = Identifica que se ejecutó el modelo R y registra las metricas en "\ExperimentosR"
    
    Retorna
    -------
    No retorna nada en particular ya que guarda los resultados en un archivo de texto
    
    Ejemplo
    -------
    RESULTADOS DE LAS PRUEBAS:
    ===============================================
    (TEST) Fold 1 Accuracy          : 0.9042553191489362
    (TEST) Fold 1 Balanced Accuracy : 0.7893772893772893
    (TEST) Fold 1 Precision Score   : 0.9629432624113475
    (TEST) Fold 1 Recall Score      : 0.9042553191489362
    (TEST) Fold 1 ROC AUC           : 0.594047619047619
    (TEST) Fold 1 Class Ratio       : 0.10638297872340426
    
    """
    
    if flagModel == 0: 
        folder = 'Experimentos\\'
    if flagModel == 1:
        folder = 'ExperimentosR\\'
    
    # Calculando las métricas
    varAccuracy = accuracy_score(y_true, y_pred)
    varBalancedAccuracy = balanced_accuracy_score(y_true, y_pred)
    varPrecision = precision_score(y_true, y_pred, average='weighted')
    varRecall = recall_score(y_true, y_pred, average='weighted')
    varRocAuc = roc_auc_score(y_pred, y_true, multi_class='ovr')
    varRatio = sum(y_class)/len(y_class)
    
    cnf_matrix = confusion_matrix(y_pred, y_true)
    print(cnf_matrix)
    
    print('RESULTADOS DEL ENTRENAMIENTO:')
    print('===============================================')
    print('('+fold_type+') Fold', str(fold_no), 'Accuracy          :', str(varAccuracy))  
    print('('+fold_type+') Fold', str(fold_no), 'Balanced Accuracy :', str(varBalancedAccuracy))  
    print('('+fold_type+') Fold', str(fold_no), 'Precision Score   :', str(varPrecision))  
    print('('+fold_type+') Fold', str(fold_no), 'Recall Score      :', str(varRecall))  
    print('('+fold_type+') Fold', str(fold_no), 'ROC AUC           :', str(varRocAuc)) 
    print('('+fold_type+') Fold', str(fold_no), 'Class Ratio       :', str(varRatio))
    
    # Guardando las métricas en un archivo
    filename =  folder+'Metricas_'+fold_type+'_'+str(fold_no)+'.txt'
    if os.path.exists(filename): 
        file = open(filename, "a")
    else:
        file = open(filename, "w")

    file.write('RESULTADOS DE LAS PRUEBAS:' + "\n")
    file.write('===============================================' + "\n")
    file.write('('+fold_type+') Fold ' + str(fold_no) + ' Accuracy          : ' + str(varAccuracy) + "\n")  
    file.write('('+fold_type+') Fold ' + str(fold_no) + ' Balanced Accuracy : ' + str(varBalancedAccuracy) + "\n")  
    file.write('('+fold_type+') Fold ' + str(fold_no) + ' Precision Score   : ' + str(varPrecision) + "\n")  
    file.write('('+fold_type+') Fold ' + str(fold_no) + ' Recall Score      : ' + str(varRecall) + "\n")  
    file.write('('+fold_type+') Fold ' + str(fold_no) + ' ROC AUC           : ' + str(varRocAuc) + "\n") 
    file.write('('+fold_type+') Fold ' + str(fold_no) + ' Class Ratio       : ' + str(varRatio) + "\n")
    file.close()

    return True

# %% De modelo Python aprendido a DataFrame Python
def modelToDataFrame(model, test):
    """ De modelo Python aprendido a DataFrame Python
    
    Descripción
    -----------
    Permite, a partir de un modelo aprendido comparar las variables con el dataset y así eliminar las variables que no se
    deben utilizar para realizar calculos posteriores.
        
    Parámetros
    ----------
    model: modelo del tipo "dict"
    
    test: data frame Python
      
    Retorna
    -------
    Retorna un Data Frame Python
       
    """
    
    # Obteniendo la matriz de Source y Target del modelo
    vector = bn.adjmat2vec(model['adjmat'])
    col = []
    
    # Se recorre el la matriz para obtener todas las columnas que esta matriz posee
    for columna in vector: #recorriendo las columnas
        if columna in ('weight'): 
            continue
        else:
            for fila in vector.index: # recorriendo las filas
                col.append(vector[columna][fila])
                       
    # Se lista la lista dejando solo valores únicos        
    vectorUnique = list(set(col))        

    # Buscando la columna que el modelo descarto para eliminarla del dataset
    for j in range(len(test.columns.values)):   
        x = -1
        
        for i in range(len(vectorUnique)):
            if vectorUnique[i] == test.columns.values[j]: 
                x = j
                break
        
        if x < 0:
            print('COLUMNA ELIMINADA DE LA INFERENCIA: ', test.columns.values[j])
            del(test[test.columns.values[j]])
            break
    
    return test    

# %% Bloques Bayesianos
def bayesBlock(data, retornar="width"):
    """ Calculo de bloques bayesianos
    
    Descripción
    -----------
    Calcula los bloques bayesianos a partir de los datos de una matriz
        
    Parámetros
    ----------
    data: matriz 1d con los datos que son usados para el calculo de bloques
    
    retornar: {"width", "bins"}
        a. si "width", devuelve el ancho estimado para cada intervalo.
        b. si "bins", devuelve el número de bins sugerido por la regla.
      
    Retorna
    -------
    Retorna el numero de bordes calculados.
       
    """

    if retornar == "width":
        edges = len(bayesian_blocks(data, fitness='events', p0=0.01))
    else:
        edges = bayesian_blocks(data, fitness='events', p0=0.01)
    
    return edges
