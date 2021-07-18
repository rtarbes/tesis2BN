"""ModelosLib. Contiene el calculo del modelo bayesiano"""
# -------------------------------------------------
# Name        : ModelosLib.py
# Authors     : Juan Tarbes Vergara
#               Pamela Morales Vergara
# Contact     : j.tarbesvergara@uandresbello.edu
#               p.moralesvergara2@uandresbello.edu
# Licence     : Universidad Andrés Bello (Chile)
# ------------------------------------------------

# %% Librerias
# Librerías propietarias que permiten la ejecución de los modelos bayesianos de Python y R 
import BayesLib as bl
import BayesLibR as blR
import BayesLibUtils as blU

# Librería para el uso del Cross Validation
from sklearn.model_selection import StratifiedKFold

# Librería para balancear los datos de un dataset
from imblearn.over_sampling import SMOTE

# Librería para el uso de la selección de variables
from sklearn.feature_selection import SelectKBest, f_classif, chi2

# Librería que permite el control del tiempo de ejecución
from time import time

# %% Modelo Bayesiano en Python
def modeloPython (df, clase, numSplits, score, balanceado, seleccionVariables):
    """ Calculo del modelo Bayesiano utilizando Python
    
    Descripción
    -----------
    Calcula el modelo bayesiano utilizando la librería Python de BNLEARN.
    Los pasos algorítmicos del calculo del modelo son:
        a. Obtener el Cross-Validation en base a un número de splits entregado
        2. Obtener la matriz con la clase
        3. Calcular las porciones de entrenamiento y prueba en base a la matriz de la clase y el CV (split)
        4. Por cada split se aprende la estructura de la porción de datos de entrenamiento
        5. se aprenden los parámetros en base modelo aprendido en paso 4 (probabilidad anterior)
        6. Se calcula la probabilidad conjunta (inferencia, probabilidad posterior)
        7. Se obtienen las métricas para la validación del modelo train.
        8. se calcula la probabilidad conjunta (inferencia, probabilidad posterior) de la porcion de datos de prueba usando
        el modelo aprendido en el paso 4.
        9. Se obtienen las métricas para la validación del modelo en test.
        
    Los pasos del 4 al 9 se ejecutan por cada fold
        
    Parámetros
    ----------
    df      : Data Frame con los datos 
    clase   : nombre de la variabla "clase" que se está calculando
    numSlits: Valor númerico que indica la cantidad de split (experimentos) que se desean calcular
    score   : Tipo de score a utilizar para el aprendizaje de la estructura
        a. "bic"  -> Bayesian Information Criterion (Tambien llamado MDL)
        b. "k2"   -> 
        c. "bdeu" -> (DB) Bayesian Dirichlet, (e) for likelihood-equivalence, (u) for uniform joint distibution        
    
    balanceado: {True, False}
        a. True  -> usar datos balanceados
        b. False -> usar datos desbalanceados
        
    seleccionVariables: {0, n}
        a. 0         -> Si el valor es 0 NO hace uso de la característica de selección de variables
        b. mayor a 0 -> Indica el número de variables a seleccionar.

    Retorna
    -------
    NO retorna nada
    
    Ejemplo
    -------
     Este ejemplo realizar el calculo del modelo usando como clase la variable "estado", usando 5 splits con el algoritmo de 
     puntuación de aprendizaje de estructura "bic", sin datos balanceados y sin selección de variables.
     
     modeloPython (df, 'estado', 5, "bic", False, 0)
     
    """
    
    # indica cual va a ser el muestreo estratificado usando el parámetro "clase" 
    # cada fold mantiene la proporcion orignal de clases
    # n_splits = el numero de experimentos a realizar
    skf = StratifiedKFold(n_splits=numSplits, shuffle=True, random_state=1) 
    target = df.loc[:, clase] # todas las filas de la columna "clase"

    # toma inicial de tiempo del proceso completo
    start_time_full = time()  

    fold_no = 1
    for train_index, test_index in skf.split(df, target):
        # toma inicial de tiempo del fold "fold_no"
        start_time = time()    
    
        # ---------------------------------------------------------------------------
        # INICIO: SECCION DE ENTRENAMIENTO
        # ---------------------------------------------------------------------------
        print("INICIO DE SECCION DE ENTRENAMIENTO, FOLD: ", str(fold_no))
    
        # entrega la porción de datos que serán usados como entrenamiento
        train = df.loc[train_index,:] # todas las columnas de la fila "train_index"
         
        if balanceado == True:
            print("Balanceando porción de entrenamiento")
            # Balanceando la clase 
            oversample = SMOTE()
            X_trainOversample, y_trainOversample = oversample.fit_resample(train, train.loc[:, clase])

            if seleccionVariables == 0:
                # Aprendiendo la estructura y los parametros de la porción de datos entrenados "sobre muestrados"
                modeloTrain = bl.Aprendizaje(X_trainOversample, fold_no, "TRAIN", score)
    
                # Transformando el modelo aprendido en un dataset que pueda ser inferido
                newModelTrain = blU.modelToDataFrame(modeloTrain, X_trainOversample)
            else:
                # Aplicando selección de variables univariate
                X_univariate = SelectKBest(f_classif, k=seleccionVariables).fit(X_trainOversample, y_trainOversample)   
                selectK_mask = X_univariate.get_support()
                X_reduced = X_trainOversample[X_trainOversample.columns[selectK_mask]]
                
                # Aprendiendo la estructura y los parametros de la porción de datos entrenados "sobre muestrados"
                modeloTrain = bl.Aprendizaje(X_reduced, fold_no, "TRAIN", score)
    
                # Transformando el modelo aprendido en un dataset que pueda ser inferido
                newModelTrain = blU.modelToDataFrame(modeloTrain, X_reduced)

            # Realizando la inferencia de los datos de entrenamiento
            probTrain = bl.probabilidadConjunta(modeloTrain, newModelTrain, fold_no, "TRAIN", clase)
            
            i = 0 # columna que queremos obtener
            lista_train = [fila[i] for fila in probTrain]

            # Metricas finales TRAIN
            blU.getMetrics(lista_train, y_trainOversample, y_trainOversample, 'TRAIN', fold_no, 0)
        else:
            if seleccionVariables == 0:
               # Aprendiendo la estructura y los parametros de la porción de datos entrenados "sobre muestrados"
               modeloTrain = bl.Aprendizaje(train, fold_no, "TRAIN", score)
    
               # Transformando el modelo aprendido en un dataset que pueda ser inferido
               newModelTrain = blU.modelToDataFrame(modeloTrain, train)
            else:
                # Aplicando selección de variables univariate
                X_univariate = SelectKBest(f_classif, k=seleccionVariables).fit(train, train.loc[:, clase])   
                selectK_mask = X_univariate.get_support()
                X_reduced = train[train.columns[selectK_mask]]
                
                # Aprendiendo la estructura y los parametros de la porción de datos entrenados "sobre muestrados"
                modeloTrain = bl.Aprendizaje(X_reduced, fold_no, "TRAIN", score)
    
                # Transformando el modelo aprendido en un dataset que pueda ser inferido
                newModelTrain = blU.modelToDataFrame(modeloTrain, X_reduced)
                
            # Realizando la inferencia de los datos de entrenamiento
            probTrain = bl.probabilidadConjunta(modeloTrain, newModelTrain, fold_no, "TRAIN", clase)
            
            i = 0 # columna que queremos obtener
            lista_train = [fila[i] for fila in probTrain]

            # Metricas finales TRAIN
            blU.getMetrics(lista_train, train.loc[:, clase], train[clase], 'TRAIN', fold_no, 0)
    
        print("FIN DE SECCION DE ENTRENAMIENTO, FOLD: ", str(fold_no))
        # ---------------------------------------------------------------------------
        # FIN: SECCION DE ENTRENAMIENTO
        # ---------------------------------------------------------------------------

        # ---------------------------------------------------------------------------
        # INICIO: SECCION DE PRUEBAS
        # ---------------------------------------------------------------------------
        print("INICIO DE SECCION DE PRUEBAS, FOLD: ", str(fold_no))
    
        # Entrega la porción de datos que serán usados como pruebas
        test = df.loc[test_index,:] # todas las columnas de la fila "test_index"
         
        if seleccionVariables == 0:
            # Aprendiendo la estructura y los parametros de la porción de datos de pruebas
            # modelo = bl.Aprendizaje(test, fold_no, "TEST", score)

            # Transformando el modelo aprendido en un dataset que pueda ser inferido
            newModelTest = blU.modelToDataFrame(modeloTrain, test)
        else:
            X_reducedTest = test[test.columns[selectK_mask]]
            # Aprendiendo la estructura y los parametros de la porción de datos de pruebas
            # modelo = bl.Aprendizaje(X_reducedTest, fold_no, "TEST", score)

            # Transformando el modelo aprendido en un dataset que pueda ser inferido
            newModelTest = blU.modelToDataFrame(modeloTrain, X_reducedTest)
            
        # Realizando la inferencia de los datos de prueba
        probTest = bl.probabilidadConjunta(modeloTrain, newModelTest, fold_no, "TEST", clase)
    
        # i = 0 
        # columna que queremos obtener
        lista_test = [fila[i] for fila in probTest]
        
        # Metricas finales TEST
        blU.getMetrics(lista_test, test.loc[:, clase], test[clase], 'TEST', fold_no, 0)

        print("FIN DE SECCION DE PRUEBAS, FOLD: ", str(fold_no))
        # ---------------------------------------------------------------------------
        # FIN: SECCION DE PRUEBAS
        # ---------------------------------------------------------------------------
     
        # Lapso de tiempo calculado para el fold "fold_no"
        elapsed_time = time() - start_time
        print("Tiempo estimado del fold "+str(fold_no)+": %0.10f seconds." % elapsed_time)
    
        # Cambiando de fold
        # break
        fold_no += 1

    # Lapso de tiempo calculado del proceso completo
    elapsed_time_full = time() - start_time_full
    print("Tiempo estimado del proceso completo: %0.10f seconds." % elapsed_time_full)
    
# %% Modelo Bayesiano en R
def modeloR (df, clase, numSplits, discreta, score, balanceado, seleccionVariables, listaBlanca, listaNegra):
    """ Calculo del modelo Bayesiano utilizando R
    
    Descripción
    -----------
    Calcula el modelo bayesiano utilizando la librería R de BNLEARN.
    Los pasos algorítmicos del calculo del modelo son:
        a. Obtener el Cross-Validation en base a un número de splits entregado
        2. Obtener la matriz con la clase
        3. Calcular las porciones de entrenamiento y prueba en base a la matriz de la clase y el CV (split)
        4. Por cada split se aprende la estructura de la porción de datos de entrenamiento
        5. se aprenden los parámetros en base modelo aprendido en paso 4 (probabilidad anterior)
        6. Se calcula la probabilidad conjunta (inferencia, probabilidad posterior)
        7. Se obtienen las métricas para la validación del modelo train.
        8. se calcula la probabilidad conjunta (inferencia, probabilidad posterior) de la porcion de datos de prueba usando
        el modelo aprendido en el paso 4.
        9. Se obtienen las métricas para la validación del modelo en test.
        
    Los pasos del 4 al 9 se ejecutan por cada fold
        
    Parámetros
    ----------
    df      : Data Frame con los datos 
    clase   : nombre de la variabla "clase" que se está calculando
    numSlits: Valor númerico que indica la cantidad de split (experimentos) que se desean calcular
    discreta: {True, False}
        a. True  -> Le indica al aprendizaje de estructura y parámetros que todas las variables serán factores (discretas)
        b. False -> Le indica al aprendizaje de estructura y parámetros que las variables serán mixtas (discretas y continuas)
        
    score   : Tipo de score a utilizar para el aprendizaje de la estructura
        a. "aic"       -> Criterio de información de Akaike. Solo para variables discretas, es decir, cuando flag "discreta = True"
        b. "bic"       -> Criterio de información Bayesiano. Solo para variables discretas, es decir, cuando flag "discreta = True"
        c. "loglik"    -> Probabilidad logarítmica. Solo para variables discretas, es decir, cuando flag "discreta = True"
        d. "aic-cg"    -> Solo para variables mixtas, es decir, cuando flag "discreta = False"
        e. "bic-cg"    -> Solo para variables mixtas, es decir, cuando flag "discreta = False"
        f. "loglik-cg" -> Solo para variables mixtas, es decir, cuando flag "discreta = False"
    
    balanceado: {True, False}
        a. True  -> usar datos balanceados
        b. False -> usar datos desbalanceados
        
    seleccionVariables: {0, n}
        a. 0         -> Si el valor es 0 NO hace uso de la característica de selección de variables
        b. mayor a 0 -> Indica el número de variables a seleccionar.
        
    listaBlanca: corresponde a un vector con las variables "desde" y "hasta" que contienen los arcos que deben ser incluídos en 
    el aprendizaje de la estructura.

    listaNegra: corresponde a un vector con las variables "desde" y "hasta" que contienen los arcos que no deben ser incluídos en
    el aprendizaje de la estructura.

    Retorna
    -------
    NO retorna nada
    
    Ejemplo
    -------
     Este ejemplo realizar el calculo del modelo usando como clase la variable "estado", usando 5 splits y datos discretos 
     con el algoritmo de puntuación de aprendizaje de estructura "aic", sin datos balanceados y sin selección de variables.
     
     modeloR (df, 'estado', 5, True, "aic", False, 0)
     
    """

    # indica cual va a ser el muestreo estratificado usando el parámetro "clase"
    # cada fold mantiene la proporcion orignal de clases
    # n_splits = el numero de experimentos a realizar
    skf = StratifiedKFold(n_splits=numSplits, shuffle=True, random_state=1) 
    target = df.loc[:, clase] # todas las filas de la columna "clase"
   
    # toma inicial de tiempo del proceso completo
    start_time_full = time()  

    fold_no = 1
    for train_index, test_index in skf.split(df, target):           
        # toma inicial de tiempo del fold "fold_no"
        start_time = time()    

        # ---------------------------------------------------------------------------
        # INICIO: SECCION DE ENTRENAMIENTO
        # ---------------------------------------------------------------------------
        print("INICIO DE SECCION DE ENTRENAMIENTO, FOLD: ", str(fold_no))
    
        # Entrega la porción de datos que serán usados como entrenamiento
        train = df.loc[train_index, :] #todas las columnas de la fila "train_index"
        
        if balanceado == True:
            print("Balanceando porción de entrenamiento")
            # Balanceando la clase 
            oversample = SMOTE()
            X_trainOversample, y_trainOversample = oversample.fit_resample(train, train.loc[:, clase])
            
            if seleccionVariables == 0:
                # Aprendiendo la estructura y los parametros de la porción de datos entrenados "sobre muestrados"
                modeloAprendido = blR.AprendizajeR(X_trainOversample, fold_no, "TRAIN", discreta, score, clase, listaBlanca, listaNegra)

                # Realizando la inferencia de los datos de entrenamiento
                probTrain = blR.probabilidadConjuntaR(modeloAprendido, X_trainOversample, fold_no, "TRAIN", discreta, clase)
            else:
                # Aplicando selección de variables univariate
                X_univariate = SelectKBest(f_classif, k=seleccionVariables).fit(X_trainOversample, y_trainOversample)   
                selectK_mask = X_univariate.get_support()
                X_reduced = X_trainOversample[X_trainOversample.columns[selectK_mask]]
                
                modeloAprendido = blR.AprendizajeR(X_reduced, fold_no, "TRAIN", discreta, score, clase, listaBlanca, listaNegra)
                
                # Realizando la inferencia de los datos de entrenamiento
                probTrain = blR.probabilidadConjuntaR(modeloAprendido, X_reduced, fold_no, "TRAIN", discreta, clase)

            i = 0 # columna que queremos obtener
            lista_train = [fila[i] for fila in probTrain]

            # Metricas finales TRAIN
            blU.getMetrics(lista_train, y_trainOversample, y_trainOversample, 'TRAIN', fold_no, 1)
        else:
            if seleccionVariables == 0:
                # Aprendiendo la estructura y los parametros de la porción de datos entrenados "sobre muestrados"
                modeloAprendido = blR.AprendizajeR(train, fold_no, "TRAIN", discreta, score, clase, listaBlanca, listaNegra)

                # Realizando la inferencia de los datos de entrenamiento
                probTrain = blR.probabilidadConjuntaR(modeloAprendido, train, fold_no, "TRAIN", discreta, clase)
            else:
                # Aplicando selección de variables univariate
                X_univariate = SelectKBest(f_classif, k=seleccionVariables).fit(train, train.loc[:, clase])   
                selectK_mask = X_univariate.get_support()
                X_reduced = train[train.columns[selectK_mask]]
                 
                # Aprendiendo la estructura y los parametros de la porción de datos entrenados "sobre muestrados"
                modeloAprendido = blR.AprendizajeR(X_reduced, fold_no, "TRAIN", discreta, score, clase, listaBlanca, listaNegra)

                # Realizando la inferencia de los datos de entrenamiento
                probTrain = blR.probabilidadConjuntaR(modeloAprendido, X_reduced, fold_no, "TRAIN", discreta, clase)

            i = 0 
            # columna que queremos obtener
            lista_train = [fila[i] for fila in probTrain]

            # Metricas finales TRAIN
            blU.getMetrics(lista_train, train.loc[:, clase], train[clase], 'TRAIN', fold_no, 1)
    

        print("FIN DE SECCION DE ENTRENAMIENTO, FOLD: ", str(fold_no))
        # ---------------------------------------------------------------------------
        # FIN: SECCION DE ENTRENAMIENTO
        # ---------------------------------------------------------------------------
    

        # ---------------------------------------------------------------------------
        # INICIO: SECCION DE PRUEBAS
        # ---------------------------------------------------------------------------
        print("INICIO DE SECCION DE PRUEBAS, FOLD: ", str(fold_no))
    
        # Entrega la porción de datos que serán usados como pruebas
        test = df.loc[test_index,:] #todas las columnas de la fila "test_index"
         
        if seleccionVariables == 0:
            # Aprendiendo la estructura y los parametros de la porción de datos de pruebas
            # modeloTest = blR.AprendizajeR(test, fold_no, "TEST", discreta, score, clase, listaBlanca, listaNegra)

            # Realizando la inferencia de los datos de prueba
            probTest = blR.probabilidadConjuntaR(modeloAprendido, test, fold_no, "TEST", discreta, clase)
        else:
            X_reducedTest = test[test.columns[selectK_mask]]
            # Aprendiendo la estructura y los parametros de la porción de datos de pruebas
            # modeloTest = blR.AprendizajeR(X_reducedTest, fold_no, "TEST", discreta, score, clase, listaBlanca, listaNegra)

            # Realizando la inferencia de los datos de prueba utilizando el modelo aprendido
            probTest = blR.probabilidadConjuntaR(modeloAprendido, X_reducedTest, fold_no, "TEST", discreta, clase)
    
        # i = 0 
        # columna que queremos obtener
        lista_test = [fila[i] for fila in probTest]
        
        # Metricas finales de los datos de prueba
        blU.getMetrics(lista_test, test.loc[:, clase], test[clase], 'TEST', fold_no, 1)

        print("FIN DE SECCION DE PRUEBAS, FOLD: ", str(fold_no))
        # ---------------------------------------------------------------------------
        # FIN: SECCION DE PRUEBAS
        # ---------------------------------------------------------------------------
     
        # Lapso de tiempo calculado para el fold "fold_no"
        elapsed_time = time() - start_time
        print("Tiempo estimado del fold "+str(fold_no)+": %0.10f seconds." % elapsed_time)
    
        # Cambiando de fold
        #break
        fold_no += 1

    # lapso de tiempo calculado del proceso completo
    elapsed_time_full = time() - start_time_full
    print("Tiempo estimado del proceso completo: %0.10f seconds." % elapsed_time_full)
