
import numpy as np
import math
from fractions import Fraction
import sympy
import itertools
from itertools import combinations
import random

def subsets_k(collection, k): 
    yield from partition_k(collection, k, k)
    
def partition_k(collection, min, k):
  if len(collection) == 1:
    yield [ collection ]
    return

  first = collection[0]
  for smaller in partition_k(collection[1:], min - 1, k):
    if len(smaller) > k: continue
    # insert `first` in each of the subpartition's subsets
    if len(smaller) >= min:
      for n, subset in enumerate(smaller):
        yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
    # put `first` in its own subset 
    if len(smaller) < k: yield [ [ first ] ] + smaller


def mandini_volume(V):
    
    '''Computes the Volume of the Moduli space of poligons
    
       :params V, list of integers'''
    
    
    
    N = len(V)
       
    num = Fraction(np.power(2, 2*N-7))
    Prefactor = Fraction(num, math.factorial(N-3))*np.power(sympy.pi, N-3)
    
    Result = 0
    
    for I in range(0,N):
        R = list(combinations(V,I))
        factor = 0
        for j in range(0, np.array(R, dtype = 'object').shape[0]):
            Q = 1
            S = np.power(np.max([0, 1- np.sum(R[j])]), N-3)
            Q *=S
            factor += (-1)**(I+1)*Q
        Result += Fraction(factor)
    return (Result*Prefactor)

def mcmullen_volume(V):
    
    '''Computes the volume of the moduli space of gurves of genus 0
       
       :params V, list of integers'''
    
    N = len(V)
    
    Prefactor = Fraction((-4)**(N-3),math.factorial(N-2)) * sympy.pi**(N-3)

    Result = 0
    
    for P in range(3,N+1):
        R = list(subsets_k(V, P))
        factor = 0    
        for j in range(0,np.array(R, dtype="object").shape[0]):
            Q = 1
            for i in range(0,P):
                B = len(R[j][i])
                S = np.max([0,1-np.sum(R[j][i])])**(B-1)
                Q *=S
            factor += (-1)**(P+1) *math.factorial(P-3)*Q
        Result += factor
        
    return (Result*Prefactor)

class AnomalyTest():
    
    '''Anomaly test, is a class that performs a test to check wether the functions mandini_volume and
       mcmullen_volume achieve the same results
       
       :params length, is an integer representing the number of marked points in the moduli space
       :params total_tests, is an integer representing the number of tests to be achieved
       
       :method random_vector, returns a test vectors of integers
       :method run_test, performs the test.'''
    
    def __init__(self, length, total_tests):
        
            
        self.length = length    
        self.total_tests = total_tests
        
        if not isinstance(self.length, int):
            raise ValueError(f'lenght must be an integer, found {self.length}')
            
        elif not self.length >= 3:
            raise ValueError(f'length must be greater or equal than 3, found {self.length}')
        
        elif not isinstance(self.total_tests, int):
            raise ValueError(f'total_test must be ingeger, found {self.total_test}')
            
        
        self.random_vector()
        
        
        
    def random_vector(self):
        
        V = []                                 
    
        for i in range(0,self.length):
            n = random.randint(1,30)
            V.append(n)
        
        V = list(np.asarray(V)*Fraction(2)/np.sum(V))
        
        self.test_vector = V
    
        
        
    def run_test(self):
        
        
        test_iterations, anomalies = np.zeros((1,), dtype=int), np.zeros((1,), dtype=int)

        while test_iterations < self.total_tests: 
            
            V = self.test_vector

            check= any(entry > 1 for entry in V)

            if check: 
                continue

            test_iterations+=1

            Vol_1 = mandini_volume(V)
            Vol_2 = mcmullen_volume(V) 
            
            if Vol_1 == 0 and Vol_2 == 0: 
                Ratio = 1.0
            else:
                Ratio = float(Vol_1 / Vol_2) 

            if Ratio != 1: 
                anomalies += 1
                print (V)
                print('Vol_1, Vol_2:', Vol_1, Vol_2)
                print('The ratio of the two values of Vol(M_d):', r) 

        print('Total anomalies', anomalies)   