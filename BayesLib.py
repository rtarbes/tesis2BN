# Configuración del ámbito de R sobre Python
import os
#os.environ['R_HOME'] = r'C:\Users\jtarb\anaconda3\envs\rBNLEARN\Lib\R'
#os.environ['R_PATH'] = r'C:\Users\jtarb\anaconda3\envs\rBNLEARN\Lib\R\bin\x64'

import BayesLibUtils as bnU
import bnlearn as bn
import json as js
import sys
#import numpy as np

#from astropy.stats import *
#from scipy import stats
#from sklearn.metrics import *

#Librería para el graficado de los grafos DAG
#from graphviz import Digraph


def Aprendizaje(model, fold, tipo):
    #####################################################################
    # APRENDIENDO LA ESTRUCTURA DE UNA PORCION DE DATOS DE TRAIN O TEST #
    #####################################################################
    
    modelo = bn.structure_learning.fit(model, methodtype='hc', scoretype='bic', verbose=3) #, bw_list_method = 'enforce', black_list = ['programa', 'estado']
    
    #guardando la estructura estructura aprendida
    nombreArchivo = 'EstructuraCPD_'+tipo+'_'+str(fold)
    nombreArchivoExt = 'Experimentos\\'+nombreArchivo+'.gv'
    f = Digraph(name=nombreArchivo, filename=nombreArchivoExt, format='pdf', engine='dot', encoding="utf-8")

    #Obteniendo la matriz de Source y Target del modelo
    vector = bn.adjmat2vec(modelo['adjmat'])
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
    
    #Asignando los nodos de la estructura aprendida
    f.attr('node', shape='circle')
    for x in range(len(vectorUnique)):
        f.node(vectorUnique[x])

    #Asignando los arcos de la estructura aprendida
    edges = modelo['model'].edges()
    f.attr('node', shape='circle')
    for edge in edges:
        xfrom = edge[0]
        xto   = edge[1]
        f.edge(xfrom, xto)

    #f.view()
    f.save()
    f.render(filename=nombreArchivoExt, view=False, cleanup=1)
    
    #Guardando la estructura aprendida
    vector = modelo['adjmat']
    vector = bn.adjmat2vec(vector)
    vector.to_csv("Experimentos\EstructuraCPD_"+tipo+"_"+str(fold)+".csv", sep=";")
    G = bn.plot(modelo, verbose=0)
    
    ###########################################################################
    # APRENDIENDO LOS PARAMETROS DEL MODELO RECIEN APRENDIDO DE SU ESTRUCTURA #
    ###########################################################################

    modelo = bn.parameter_learning.fit(modelo, model, verbose=1)
    print(modelo)
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
