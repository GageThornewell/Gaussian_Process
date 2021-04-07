import numpy as np
import math as m 
import random as rd
import matplotlib.pyplot as plt 
from scipy.optimize import minimize


x_data = [-5 + i for i in range(0,10)]
y_data = [-i**4 + i**3 + -3*i**2 + i - 5 for i in x_data ]


plt.plot(x_data,y_data,'ro')
plt.title('Training Data for polynomial -x^4 + x^3 - 3x^2 + x - 2 ')
plt.show()

def kernel(x,y,pars):
	sigma = pars[0]
	l = pars[1] 
	a = (x-y)**2

	return sigma**2 * np.exp(-a/(2*l**2))


def cov_matrix(pars):
	K = np.zeros((len(x_data),len(x_data)))
	for i in range(len(x_data)):
		for j in range(len(x_data)):
			K[i,j] = kernel(x_data[i],x_data[j],pars)

	return K

#print(kernel(1,1,(1,1)))			
#print(cov_matrix((1,1)))
#(s,d) = np.linalg.slogdet(cov_matrix((1,1)))
#print(d)

	
def log_likelyhood(pars):
	y = np.array([c for c in y_data])
	alpha = np.linalg.solve(cov_matrix(pars),y)
	(s,d) = np.linalg.slogdet(cov_matrix(pars))
	n1 = -(1/2) * np.dot(y,alpha)
	n2 = -(1/2) * d
	n3 = - (len(x_data)/2) * np.log(m.pi * 2)

	return -1*(n1+n2+n3)

#print(log_likelyhood((1,1)))

res = minimize(log_likelyhood, x0=[2,-1], method = 'CG',options={'maxiter': 100})

#print(res.x)

def GP(x):
	y = np.array([c for c in y_data])
	K_star = np.array([kernel(x,i,(res.x)) for i in x_data]) 
	t = np.linalg.inv(cov_matrix(res.x))
	t1 = np.dot(t,y)

	return np.dot(K_star,t1)



xnew = np.linspace(-6, 6, 300)
ynew = [GP(i) for i in xnew]

xactual = np.linspace(-6,6,300)
yactual = [-i**4 + i**3 + -3*i**2 + i - 5 for i in xactual ]





plt.plot(x_data,y_data, 'ro', label = 'Training data ' )
plt.plot(xnew,ynew,'k' ,label =' Full GP ')
plt.plot(xactual,yactual, label = 'actual polynomial')
plt.title('Gaussian Process Regression for squared exponential kernel')
plt.xlabel('x data')
plt.ylabel('y data')
plt.legend(loc = 7)

plt.show()






    

    









