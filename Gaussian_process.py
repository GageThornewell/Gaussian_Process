import numpy as np
import math as m 
import random as rd
import matplotlib.pyplot as plt 
from scipy.optimize import minimize




x_data = [1,2,3,4]
y_data = [1,1/2,1/3,1/44]




y = np.array([c for c in y_data])

def kronecker(a,b):
	if a == b:

		return 1

	else: 

		return 0 


def k(x,y,z):
	l = z[0]
	sigma = z[1]
	sigma_n = z[2]

	return (sigma**2)* np.exp((-1*(x-y)**2)/2*l**2)  +  sigma_n*kronecker(x,y)

def cov_matrix(z):
	K = np.zeros((len(x_data),len(y_data)))
	for i in range(len(x_data)):
		for j in range(len(x_data)):
			K[i,j] = k(x_data[i],y_data[j],z)

	return K



def loss_function(z):
	y = np.array([i for i in y_data])
	alpha = np.linalg.solve(cov_matrix(z),y)

	(s,d) = np.linalg.slogdet(cov_matrix(z))
	

	

	n1 = -(1/2) * np.dot(y,alpha)
	n2 = -(1/2) * s
	n3 = - (len(y_data)/2) * np.log(m.pi * 2)


	return -1*(n1+n2+n3)



#print(loss_function((1,1,1)))




#res = minimize(loss_function, x0=[2,-1, 1], method = 'CG',options={'maxiter': 100})


#print(res.x)
#print(cov_matrix((res.x)))


def GP(x_star):

	K_star = np.array([k(x_star,i,(1,1,0.1)) for i in x_data])
	t = np.dot(K_star,np.linalg.inv(cov_matrix((1,1.37,.5))))
	y_star = np.dot(t,(y))

	return y_star



print(GP(1.5))






'''



plt.plot(x_data,y_data,  label = 'plot 1  ' )
#plt.plot(x,newy,'ro',label =' Full GP')
plt.title('x vs y')
plt.xlabel('x_data')
plt.ylabel('y_data')
plt.legend(loc = 5)
plt.grid(True)
plt.show()


'''
  


	
#print(np.linalg.det(A))

	 

 

	








