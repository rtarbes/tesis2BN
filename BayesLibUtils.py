# Configuración del ámbito de R sobre Python
import os
import bnlearn as bn
import numpy as np

from sklearn.metrics import *
from astropy.stats import *

def getMetrics (y_true, y_pred, y_class, fold_type, fold_no, flagModel):
    
    if flagModel == 0: 
        folder = 'Experimentos\\'
    if flagModel == 1:
        folder = 'ExperimentosR\\'
    
    #Calculando las metricas
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
    
    #Guardando las metricas en un archivo
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

def modelToDataFrame(model, test):
    #Obteniendo la matriz de Source y Target del modelo
    vector = bn.adjmat2vec(model['adjmat'])
    col = []
    
    #Se recorre el la matriz para obtener todas las columnas que esta matriz posee
    for columna in vector: #recorriendo las columnas
        if columna in ('weight'): 
            continue
        else:
            for fila in vector.index: #recorriendo las filas
                col.append(vector[columna][fila])
                       
    #Se lista la lista dejando solo valores únicos        
    vectorUnique = list(set(col))        

    #buscando la columna que el modelo descarto para eliminarla del dataset
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

   
def freedmanDiaconis(data, retornar="width"):
    """
    Use Freedman Diaconis rule to compute optimal histogram bin width. 
    ``returnas`` can be one of "width" or "bins", indicating whether
    the bin width or number of bins should be returned respectively. 


    Parameters
    ----------
    data: np.ndarray
        One-dimensional array.

    retornar: {"width", "bins"}
        If "width", return the estimated width for each histogram bin. 
        If "bins", return the number of bins suggested by rule.
    """
    data = np.asarray(data, dtype=np.float_)
    IQR  = stats.iqr(data, rng=(25, 75), scale="raw", nan_policy="omit")
    N    = data.size
    bw   = (2 * IQR) / np.power(N, 1/3)

    if retornar=="width":
        result = bw
    else:
        datmin, datmax = data.min(), data.max()
        datrng = datmax - datmin
        result = int((datrng / bw) + 1)
        
    return(result)


def bayesBlock(data, retornar="width"):
    """"
    Parameters
    ----------
    data: np.ndarray
        One-dimensional array.

    retornar: {"width", "bins"}
        If "width", return the estimated width for each histogram bin. 
        If "bins", return the number of bins suggested by rule.
    """

    if retornar == "width":
        edges = len(bayesian_blocks(data, fitness='events', p0=0.01))
    else:
        edges = bayesian_blocks(data, fitness='events', p0=0.01)
    
    return edges
