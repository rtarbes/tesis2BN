"""BayesLib. Proporciona las herramientas necesarias para la calcular los modelos Bayesianos en Python."""
# ------------------------------------------------
# Name        : BayesLib.py
# Authors     : Juan Tarbes Vergara
#               Pamela Morales Vergara
# Contact     : j.tarbesvergara@uandresbello.edu
#               p.moralesvergara2@uandresbello.edu
# Licence     : Universidad Andrés Bello (Chile)
# ------------------------------------------------


# %% Librerias
# Libreria para el manejo de archivos
import os

# Librería propietaria utilitaria
import BayesLibUtils as bnU

# Libreria BNLEARN es un paquete de Python para aprender la estructura gráfica de redes bayesianas, 
# aprendizaje de parámetros, métodos de inferencia y muestreo. 
import bnlearn as bn

# Libreria para el manejo de estructuras JSON (del tipo dict)
import json as js

# Libreria para el manejo de excepciones en Python
import sys

# Librería para graficar los grafos (DAG)
from graphviz import Digraph

def Aprendizaje(model, fold, tipo, score):
    """ Aprendizaje de estructura y parámetros de BNLEARN en Python
    
    Descripción
    -----------
    Encapsula el aprendizaje de estructura y el aprendizaje de parámetros (probabilidad anterior).
    El aprendizaje de la estructura es guardado en un archivo CSV y el DAG aprendido es renderizado y guardado en un 
    archivo PDF en la carpeta \Experimentos con el nombre "EstructuraCPD_'tipo'_fold.gv.pdf"
        
    Parámetros
    ----------
    model: recibe un dataframe con los la información a aprender 
    fold : Valor númerico que indica el cross-validation del cual se está solicitando el aprendizaje
    tipo :  Indicador del tipo de datos del que se desa aprender.  Valores posibles:
        a. "TRAIN"
        b. "TEST"
    score   : Tipo de score a utilizar para el aprendizaje de la estructura
        a. "bic"  -> Bayesian Information Criterion (Tambien llamado MDL)
        b. "k2"   -> 
        c. "bdeu" -> (DB) Bayesian Dirichlet, (e) for likelihood-equivalence, (u) for uniform joint distibution        
         
    Retorna
    -------
    Retorna un modelo aprendido "dict"
       
    """    
    # -------------------------------------------------------------------
    # APRENDIENDO LA ESTRUCTURA DE UNA PORCION DE DATOS DE TRAIN O TEST -
    # -------------------------------------------------------------------
    
    modelo = bn.structure_learning.fit(model, methodtype='hc', scoretype=score, verbose=3)
    # modelo = bn.structure_learning.fit(model, methodtype='hc', scoretype=score, verbose=3, bw_list_method = 'enforce', black_list = ['programa', 'estado'])
    
    # Guardando la estructura estructura aprendida
    nombreArchivo = 'EstructuraCPD_'+tipo+'_'+str(fold)
    nombreArchivoExt = 'Experimentos\\'+nombreArchivo+'.gv'
    f = Digraph(name=nombreArchivo, filename=nombreArchivoExt, format='pdf', engine='dot', encoding="utf-8")

    # Obteniendo la matriz de Source y Target del modelo
    vector = bn.adjmat2vec(modelo['adjmat'])
    col = []
    
    # Se recorre el la matriz para obtener todas las columnas que esta matriz posee
    for columna in vector: # recorriendo las columnas
        if columna in ('weight'): 
            continue
        else:
            for fila in vector.index: # recorriendo las filas
                col.append(vector[columna][fila])
                       
    # Se lista la lista dejando solo valores únicos        
    vectorUnique = list(set(col))  
    
    # Asignando los nodos de la estructura aprendida
    f.attr('node', shape='circle')
    for x in range(len(vectorUnique)):
        f.node(vectorUnique[x])

    # Asignando los arcos de la estructura aprendida
    edges = modelo['model'].edges()
    f.attr('node', shape='circle')
    for edge in edges:
        xfrom = edge[0]
        xto   = edge[1]
        f.edge(xfrom, xto)

    # f.view()
    f.save()
    f.render(filename=nombreArchivoExt, view=False, cleanup=1)
    
    # Guardando la estructura aprendida
    vector = modelo['adjmat']
    vector = bn.adjmat2vec(vector)
    vector.to_csv("Experimentos\EstructuraCPD_"+tipo+"_"+str(fold)+".csv", sep=";")
    G = bn.plot(modelo, verbose=0)
    
    # -------------------------------------------------------------------------
    # APRENDIENDO LOS PARAMETROS DEL MODELO RECIEN APRENDIDO DE SU ESTRUCTURA -
    # -------------------------------------------------------------------------

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

    # Muestra como queda la red bayesiana con la porción de los datos entrenados y los parametros aprendidos
    G = bn.plot(modelo, verbose=0) 
                
    return modelo

def probabilidadConjunta(model, test, fold, tipo, clase):
    """ Probabilidad conjunta (inferencia) de BNLEARN de Python
    
    Descripción
    -----------
    Realiza el calculo de la probabilidad conjunta (inferencia)
        
    Parámetros
    ----------
    model: recibe un modelo DAG con los los parámetros aprendidos (probabilidad anterior)
    test : Data Frame con los datos con los cuales se construira la evidencia para la probabilidad conjunta.
    fold : Valor númerico que indica el cross-validation del cual se está solicitando el aprendizaje
    tipo :  Indicador del tipo de datos del que se desa aprender.  Valores posibles:
        a. "TRAIN"
        b. "TEST"
    clase: nombre de la variabla "clase" que se está calculando
         
    Retorna
    -------
    Retorna una matriz con los los valores de inferencia máximos de cada registro del dataframe "parámetro: test"
       
    """    
    # Se define un arreglo unidimensional para registrar el resultado de la inferencia
    arreglo = []
    indice  = 0
    
    for fila in test.index: # recorriendo las filas
        valor = "{"
        for columna in test: # recorriendo las columnas
            if columna in (clase): 
               continue
            else:
               try:
                  valor = valor + "\""+columna + "\":" +test[columna][fila] + ", "
               except:
                  valor = valor + "\""+columna + "\":" +str(test[columna][fila]) + ", "
                
        valor = valor[:-2] + '}'
        xclase = '\"' + clase + '\"'
        regSalida = "FILA N°: " + str(fila+1) + " -> P(" + xclase + " | " + "[" + valor + "]"
        print(regSalida)
        res = js.loads(valor)
        
        arreglo.append([])
        try:
            q1 = bn.inference.fit(model, variables=[clase], evidence=res)
           
            # Guardando los resultados de la inferencia
            filename = "Experimentos\ProbConjunta_"+tipo+"_"+str(fold)+".txt"
            if os.path.exists(filename): 
               tf = open(filename, "a")
            else:
               tf = open(filename, "w")
                
            tf.write(regSalida+"\n")
            tf.write(str(q1) + os.linesep)
            tf.close()
    
            # Extrayendo la clase con probabilidad mas alta
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
