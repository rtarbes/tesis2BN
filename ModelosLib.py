from imblearn.over_sampling import SMOTE
import BayesLib as bl
import BayesLibR as blR
import BayesLibUtils as blU

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from time import time

def modeloPython (df, clase, numSplits):
    #indica cual va a ser el muestreo estratificado usando la clase "estado"
    #cada fold mantiene la proporcion orignal de clases
    #n_splits = el numero de experimentos a realizar
    skf = StratifiedKFold(n_splits=numSplits, shuffle=True, random_state=1) 
    target = df.loc[:, clase] #todas las filas de la columna "estado"

    # toma inicial de tiempo del proceso completo
    start_time_full = time()  

    fold_no = 1
    for train_index, test_index in skf.split(df, target):
        #toma inicial de tiempo del fold "fold_no"
        start_time = time()    
    
        #---------------------------------------------------------------------------
        #INICIO: SECCION DE ENTRENAMIENTO
        #---------------------------------------------------------------------------
        print("INICIO DE SECCION DE ENTRENAMIENTO, FOLD: ", str(fold_no))
    
        #entrega la porción de datos que serán usados como entrenamiento
        train = df.loc[train_index,:] #todas las columnas de la fila "train_index"
         
        #aprendiendo la estructura y los parametros de la porción de datos entrenados "sobre muestrados"
        modelo = bl.Aprendizaje(train, fold_no, "TRAIN")
    
        #transformando el modelo aprendido en un dataset que pueda ser inferido
        newModel = blU.modelToDataFrame(modelo, train)

        #realizando la inferencia de los datos de entrenamiento
        probTrain = bl.probabilidadConjunta(modelo, newModel, fold_no, "TRAIN")
            
        i = 0 #columna que queremos obtener
        lista_train = [fila[i] for fila in probTrain]

        #Metricas finales TRAIN
        blU.getMetrics(lista_train, train.loc[:, clase], train[clase], 'TRAIN', fold_no, 0)
    
        print("FIN DE SECCION DE ENTRENAMIENTO, FOLD: ", str(fold_no))
        #---------------------------------------------------------------------------
        #FIN: SECCION DE ENTRENAMIENTO
        #---------------------------------------------------------------------------

        #---------------------------------------------------------------------------
        #INICIO: SECCION DE PRUEBAS
        #---------------------------------------------------------------------------
        print("INICIO DE SECCION DE PRUEBAS, FOLD: ", str(fold_no))
    
        #entrega la porción de datos que serán usados como pruebas
        test = df.loc[test_index,:] #todas las columnas de la fila "test_index"
         
        #aprendiendo la estructura y los parametros de la porción de datos de pruebas
        modelo = bl.Aprendizaje(test, fold_no, "TEST")

        #transformando el modelo aprendido en un dataset que pueda ser inferido
        newModel = blU.modelToDataFrame(modelo, test)
    
        #realizando la inferencia de los datos de prueba
        probTest = bl.probabilidadConjunta(modelo, newModel, fold_no, "TEST")
    
        ##i = 0 #columna que queremos obtener
        lista_test = [fila[i] for fila in probTest]
        print(lista_test)
        #Metricas finales TEST
        blU.getMetrics(lista_test, test.loc[:, clase], test[clase], 'TEST', fold_no, 0)

        print("FIN DE SECCION DE PRUEBAS, FOLD: ", str(fold_no))
        #---------------------------------------------------------------------------
        #FIN: SECCION DE PRUEBAS
        #---------------------------------------------------------------------------
     
        # lapso de tiempo calculado para el fold "fold_no"
        elapsed_time = time() - start_time
        print("Tiempo estimado del fold "+str(fold_no)+": %0.10f seconds." % elapsed_time)
    
        #Cambiando de fold
        break
        #fold_no += 1

    # lapso de tiempo calculado del proceso completo
    elapsed_time_full = time() - start_time_full
    print("Tiempo estimado del proceso completo: %0.10f seconds." % elapsed_time_full)
    
    
def modeloR (df, clase, numSplits, discreta, score, balanceado):
    #indica cual va a ser el muestreo estratificado usando la clase "estado"
    #cada fold mantiene la proporcion orignal de clases
    #n_splits = el numero de experimentos a realizar
    skf = StratifiedKFold(n_splits=numSplits, shuffle=True, random_state=1) 
    target = df.loc[:, clase] #todas las filas de la columna "estado"
   
    # toma inicial de tiempo del proceso completo
    start_time_full = time()  

    fold_no = 1
    for train_index, test_index in skf.split(df, target):           
        #toma inicial de tiempo del fold "fold_no"
        start_time = time()    

        #---------------------------------------------------------------------------
        #INICIO: SECCION DE ENTRENAMIENTO
        #---------------------------------------------------------------------------
        print("INICIO DE SECCION DE ENTRENAMIENTO, FOLD: ", str(fold_no))
    
        #entrega la porción de datos que serán usados como entrenamiento
        train = df.loc[train_index, :] #todas las columnas de la fila "train_index"
        
        if balanceado == True:
            print("Balanceando porción de entrenamiento")
            #Balanceando la clase 
            oversample = SMOTE()
            X_trainOversample, y_trainOversample = oversample.fit_resample(train, train.loc[:, 'estado'])
            
            #aprendiendo la estructura y los parametros de la porción de datos entrenados "sobre muestrados"
            modeloAprendido = blR.AprendizajeR(X_trainOversample, fold_no, "TRAIN", discreta, score)

            #realizando la inferencia de los datos de entrenamiento
            probTrain = blR.probabilidadConjuntaR(modeloAprendido, X_trainOversample, fold_no, "TRAIN", discreta)

            i = 0 #columna que queremos obtener
            lista_train = [fila[i] for fila in probTrain]

            #Metricas finales TRAIN
            blU.getMetrics(lista_train, y_trainOversample, y_trainOversample, 'TRAIN', fold_no, 1)
        else:
            #aprendiendo la estructura y los parametros de la porción de datos entrenados "sobre muestrados"
            modeloAprendido = blR.AprendizajeR(train, fold_no, "TRAIN", discreta, score)
 
            #realizando la inferencia de los datos de entrenamiento
            probTrain = blR.probabilidadConjuntaR(modeloAprendido, train, fold_no, "TRAIN", discreta)

            i = 0 #columna que queremos obtener
            lista_train = [fila[i] for fila in probTrain]

            #Metricas finales TRAIN
            blU.getMetrics(lista_train, train.loc[:, clase], train[clase], 'TRAIN', fold_no, 1)
    

        print("FIN DE SECCION DE ENTRENAMIENTO, FOLD: ", str(fold_no))
        #---------------------------------------------------------------------------
        #FIN: SECCION DE ENTRENAMIENTO
        #---------------------------------------------------------------------------
    

        #---------------------------------------------------------------------------
        #INICIO: SECCION DE PRUEBAS
        #---------------------------------------------------------------------------
        print("INICIO DE SECCION DE PRUEBAS, FOLD: ", str(fold_no))
    
        #entrega la porción de datos que serán usados como pruebas
        test = df.loc[test_index,:] #todas las columnas de la fila "test_index"
         
        #aprendiendo la estructura y los parametros de la porción de datos de pruebas
        #modeloTest = blR.AprendizajeR(test, fold_no, "TEST", discreta, score)

        #realizando la inferencia de los datos de prueba utilizando el modelo aprendido
        probTest = blR.probabilidadConjuntaR(modeloAprendido, test, fold_no, "TEST", discreta)
    
        ##i = 0 #columna que queremos obtener
        lista_test = [fila[i] for fila in probTest]
        
        #Metricas finales de los datos de prueba
        blU.getMetrics(lista_test, test.loc[:, clase], test[clase], 'TEST', fold_no, 1)

        print("FIN DE SECCION DE PRUEBAS, FOLD: ", str(fold_no))
        #---------------------------------------------------------------------------
        #FIN: SECCION DE PRUEBAS
        #---------------------------------------------------------------------------
     
        # lapso de tiempo calculado para el fold "fold_no"
        elapsed_time = time() - start_time
        print("Tiempo estimado del fold "+str(fold_no)+": %0.10f seconds." % elapsed_time)
    
        #Cambiando de fold
        #break
        fold_no += 1

    # lapso de tiempo calculado del proceso completo
    elapsed_time_full = time() - start_time_full
    print("Tiempo estimado del proceso completo: %0.10f seconds." % elapsed_time_full)
