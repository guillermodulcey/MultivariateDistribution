import math
import numpy
import random

class Probabilidad():
    def __init__(self):
        pass

    def generarDataSet(self, mu:list, sigma:list, k):
        #k es la cantidad de datos
        datos = []
        numVariables = len(mu)
        ro = random.uniform(0,1)
        matrizCo = self.generarMatriz(sigma)
        for i in range(0,k):
            while True:
                z, vector = self.generarVector(numVariables)
                evaluacion = self.pmfBivariate(vector,mu,sigma,ro)
                if z <= evaluacion:
                    datos.append(vector)
                    break
        return datos

    def generarVector(self,longitud):
        vector = []
        for i in range(0,longitud):
            vector.append(random.uniform(0,1))
        z = random.uniform(0,1)
        return z, vector
            

    def pmf(self, x:list, mu:list, matrixSigma):
        k = len(x)

        xmu = numpy.array(x) - numpy.array(mu)
        trans_xmu = xmu.transpose()

        nmatriz = numpy.array(matrixSigma)

        pi = math.pow(2*math.pi,(-k/2))

        ####Revisar####
        determinante = numpy.linalg.det(nmatriz)
        det = math.pow(abs(determinante),(-1/2))
        ###############

        inv = numpy.linalg.inv(nmatriz)

        prodXmuInv = numpy.dot(xmu,inv)
        prodFinal = numpy.dot(prodXmuInv,xmu)
        potencia = (-1/2)*prodFinal
        
        e = math.pow(math.e,potencia)
        return pi*det*e

    def generarMatriz(self, sigma:list):
        random.seed(1)
        matriz = []
        lFilas = len(sigma)
        for i in range(0,lFilas):
            fila = []
            for j in range(0,lFilas):
                ####Revisar####
                p = random.uniform(0,1)
                ###############
                fila.append(p*sigma[i]*sigma[j])
            matriz.append(fila)
        return matriz

    #########Bivariate
    def pmfBivariate(self, x:list, mu:list, sigma:list, ro):
        resta = 1-math.pow(ro,2)
        raiz = math.pow(resta,(1/2))
        coeficiente = 2*math.pi*sigma[0]*sigma[1]

        parte1 = 1/coeficiente
        potencia = -(1/(2*resta))*(self.__cuadradosP(x[0],mu[0],sigma[0])-(2*ro*self.__productoP(x,mu,sigma))+self.__cuadradosP(x[1],mu[1],sigma[1]))
        parte2 = math.exp(potencia)
        return parte1*parte2
    
    def __cuadradosP(self,x,mu,sigma):
        expresion = (x-mu)/sigma
        return math.pow(expresion,2)

    def __productoP(self, x:list, mu:list, sigma:list):
        return (x[0]-mu[0])*(x[1]-mu[1])/(sigma[0]*sigma[1])

p = Probabilidad()
DataSet = p.generarDataSet([0.8,0.2],[0.1,0.1],100)
print(DataSet)
acumulador1 = 0
acumulador2 = 0
for i in range(0,len(DataSet)):
    acumulador1 += DataSet[i][0]
    acumulador2 += DataSet[i][1]
print(str(acumulador1/len(DataSet)))
print(str(acumulador2/len(DataSet)))
