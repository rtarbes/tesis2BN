#!/usr/bin/env python
# coding: utf-8

# In[1]:

import bnlearn as bn
import json as js
import sys
import os

from sklearn.metrics import *

def probabilidadConjunta(model, test, fold, tipo):
    #se define un arreglo unidimensional para registrar el resultado de la inferencia
    arreglo = []
    indice  = 0
    
    for fila in test.index: #recorriendo las filas
        valor = "{"
        for columna in test: #recorriendo las columnas
            if columna in ('estado'): 
               continue
            else:
               try:
                  valor = valor + "\""+columna + "\":" +test[columna][fila] + ", "
               except:
                  valor = valor + "\""+columna + "\":" +str(test[columna][fila]) + ", "
                
        valor = valor[:-2] + '}'
        regSalida = "FILA N°: " + str(fila+1) + " -> P(\"Estado\" | " + "[" + valor + "]"
        print(regSalida)
        res = js.loads(valor)
        
        arreglo.append([])
        try:
            q1 = bn.inference.fit(model, variables=['estado'], evidence=res)
           
            #Guardando los resultados de la inferencia
            filename = "Experimentos\ProbConjunta_"+tipo+"_"+str(fold)+".txt"
            if os.path.exists(filename): 
               tf = open(filename, "a")
            else:
               tf = open(filename, "w")
                
            tf.write(regSalida+"\n")
            tf.write(str(q1) + os.linesep)
            tf.close()
    
            #Extrayendo la clase con probabilidad mas alta
            if (q1.get_value(estado=0) > q1.get_value(estado=1)):
                 arreglo[indice].append(0)
                 arreglo[indice].append(q1.get_value(estado=0))
            else:
                 arreglo[indice].append(1)
                 arreglo[indice].append(q1.get_value(estado=1))
        except: 
            arreglo[indice].append(-1)
            arreglo[indice].append(-1)
            e = sys.exc_info()[1]
            print('ERROR AL REALIZAR LA INFERENCIA: ',e) 
        
        indice += 1
        valor = "{"
    
    return arreglo

def Aprendizaje(model, fold, tipo):
    #aprendiendo la estructura y los parametros de la porción de datos entrenados
    modelo = bn.structure_learning.fit(model, methodtype='hc', scoretype='bic', verbose=3)
    
    #Guardando la estructura aprendida
    vector = modelo['adjmat']
    vector = bn.adjmat2vec(vector)
    vector.to_csv("Experimentos\EstructuraCPD_"+tipo+"_"+str(fold)+".csv", sep=";")
    G = bn.plot(modelo, verbose=0)
    
    modelo = bn.parameter_learning.fit(modelo, model, verbose=1)
    # Convertiendo a BayesianModel para guardar los parametros aprendidos
    if isinstance(modelo, dict):
        model1 = modelo['model']

    filename =  "Experimentos\ParametrosCPD_"+tipo+"_"+str(fold)+".txt"
    if os.path.exists(filename): 
        file = open(filename, "a")
    else:
        file = open(filename, "w")

    for cpd in model1.get_cpds():
        file.write("CPD of {variable}:".format(variable=cpd.variable)+"\n")
        file.write(str(cpd) + os.linesep)

    file.close()

    #muestra como queda la red bayesiana con la porción de los datos entrenados y los parametros aprendidos
    G = bn.plot(modelo, verbose=0) 
                
    return modelo

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

def getMetrics (y_true, y_pred, y_class, fold_type, fold_no):
    #Calculando las metricas
    varAccuracy = accuracy_score(y_true, y_pred)
    varBalancedAccuracy = balanced_accuracy_score(y_true, y_pred)
    varPrecision = precision_score(y_true, y_pred, average='weighted')
    varRecall = recall_score(y_true, y_pred, average='weighted')
    varRocAuc = roc_auc_score(y_pred, y_true, multi_class='ovr')
    varRatio = sum(y_class)/len(y_class)
    
    print('RESULTADOS DEL ENTRENAMIENTO:')
    print('===============================================')
    print('('+fold_type+') Fold', str(fold_no), 'Accuracy          :', str(varAccuracy))  
    print('('+fold_type+') Fold', str(fold_no), 'Balanced Accuracy :', str(varBalancedAccuracy))  
    print('('+fold_type+') Fold', str(fold_no), 'Precision Score   :', str(varPrecision))  
    print('('+fold_type+') Fold', str(fold_no), 'Recall Score      :', str(varRecall))  
    print('('+fold_type+') Fold', str(fold_no), 'ROC AUC           :', str(varRocAuc)) 
    print('('+fold_type+') Fold', str(fold_no), 'Class Ratio       :', str(varRatio))
    
    #Guardando las metricas en un archivo
    filename =  'Experimentos\Metricas_'+fold_type+'_'+str(fold_no)+'.txt'
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
   
# In[ ]:




