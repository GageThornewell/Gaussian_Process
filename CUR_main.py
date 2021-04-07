import numpy as np
import math as m 
import random as rd
import matplotlib.pyplot as plt 
import itertools as it

#random algorithm 

n = 4
t = [i for i in range(0,n)]

def norm(a,b):
	v1 = np.subtract(a,b)
	v2 = np.multiply(v1,v1)
	v3 = np.sum(v2)
	return np.sqrt(v3)

def f(a,b):
	return np.exp(a-b**2)**2  

a = np.zeros((n,n))

for i in range(0,n):
	for k in range(0,n):
		a[i,k] = f(i,k)

#print(a)





#rd.sample(range(1, 10), 4)

#print(np.random.randint(10, size=9))


# M is the matrix, r is number of rows and columns to take and u^2 is randomness threshold
def CUR(M,r,u):
	randvecs1 = []
	randvecs2 = []
	sp = []
	determinants = []

	   
	
	for i in range(u):
		h = rd.sample(range(0,len(M[:,1])),r)
		randvecs1.append(sorted(h))

	for i in range(u):
		h = rd.sample(range(0,len(M[:,1])),r)
		randvecs2.append(sorted(h))


	for i in range(u):
		h = (randvecs1[i],randvecs2[i])
		sp.append(h)

	for i in range(u):
		for k in range(u):
			
			C = M[:,sp[i][0]]
			U = C[sp[k][1],:]
			y = np.linalg.det(U)
			determinants.append(y)

	h = np.argmax(determinants)
	l = np.argmin(determinants)

	v = [abs(determinants[h]),abs(determinants[l])]
	v1 = [h,l]
	v2 = np.argmax(v)
	v3 = v1[v2]

	h1 = m.floor(v3/u)
	h2 = (v3%u) 
	
	C = M[:,randvecs1[h1]]
	U = C[randvecs2[h2],:]
	R = M[randvecs2[h2],:]
	U1 = np.linalg.inv(U)
	h5 = np.linalg.det(U)
	X = np.dot(U1,R)

	return np.dot(C,X)


 
	
print(a)

#i = CUR(a,3,3)


#print(a)
#print(CUR(a,2,100))

#print(norm(a,i))















