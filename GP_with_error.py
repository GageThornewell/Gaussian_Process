import numpy as np
import math as m 
import random as rd
import matplotlib.pyplot as plt 
from scipy.optimize import minimize
import pylab as pl

plt.style.use('classic')

q = 3 # lenght of nplinspace 
p = 10 # number of points taken

D = 1
a = 1
r_e = 1 






x = np.linspace(0,3,100)
y = (1-np.exp(-a*(x-r_e)))**2
#dx = 0.3 
#dy = 0.5
x_data = np.linspace(0,q,p)
y_data = (1-np.exp(-a*(x_data-r_e)))**2


xnew = np.linspace(0, q, p)
ynew = (1-np.exp(-a*(xnew-r_e)))**2

'''
plt.plot(x,y,label='Morse Potential')
#plt.plot(x_data,y_data,'ro')
plt.title("Morse Potential For Diatomic Molecule")
plt.xlabel('Radius')
plt.ylabel('Potential')
plt.ylim(-0.2,3.5)
plt.xlim(-1,11)
plt.legend(loc = 1)
plt.show()
'''

def kroneker(a,b):
	if a == b:
		return 1

	else:
		return 0


def kernel(x,y,pars):
	sigma = pars[0]
	l = pars[1] 
	delta = pars[2]
	
	a = (x-y)**2


	return sigma**2 * np.exp(-a/(2*l**2)) + kroneker(x,y)*delta**2

def cov_matrix(pars):
	K = np.zeros((len(xnew),len(xnew)))
	for i in range(len(xnew)):
		for j in range(len(xnew)):
			K[i,j] = kernel(xnew[i],xnew[j],pars)

	return K


def log_likelyhood(pars):
	y = np.array([c for c in ynew])
	alpha = np.linalg.solve(cov_matrix(pars),y)
	(s,d) = np.linalg.slogdet(cov_matrix(pars))
	n1 = -(1/2) * np.dot(y,alpha)
	n2 = -(1/2) * d
	n3 = - (len(x_data)/2) * np.log(m.pi * 2)

	return -1*(n1+n2+n3)


res = minimize(log_likelyhood, x0=[2,-1,2], method = 'CG',options={'maxiter': 100})

#print(res.x)


def GP(x):
	y = np.array([c for c in ynew])
	K_star = np.array([kernel(x,i,(res.x)) for i in x_data]) 
	K_starstar = kernel(x,x,(res.x))
	t = np.linalg.inv(cov_matrix(res.x))
	t1 = np.dot(t,y)

	y_star = np.dot(K_star,t1)
	r = np.dot(K_star,t)
	y_var = K_starstar - np.dot(r,np.transpose(K_star))

	return (y_star,y_var)



xnew1 = np.linspace(0, q, p)
ynew1 = [GP(i)[0] for i in xnew1]
dy_new = [GP(i)[1] for i in xnew1]

uppery = [ynew1[i] + dy_new[i] for i in range(len(xnew1))]
lowery = [ynew1[i] - dy_new[i] for i in range(len(xnew1))]



#xactual = np.linspace(-6,6,300)
#yactual = [-i**4 + i**3 + -3*i**2 + i - 5 for i in xactual ]






plt.plot(x,y ,'b--', label = 'Morse Potential')
plt.plot(x_data,y_data,'ro',label = 'Training Data 10 Points')
#plt.plot(xnew1,ynew1,label='GP Regression')
#plt.errorbar(x,ynew,yerr = dy_new, label = 'Guassian Process fit')
plt.fill_between(xnew1, lowery , uppery , color='gray', alpha=1 , label = "Guassian Process Fit With Error")
plt.title("Guassian Process Regression Applied To Morse Potential")
#plt.ylim(0,8)
#plt.xlim(-6,6)
plt.ylim(-0.2,3.5)
plt.xlim(-1,6)
plt.legend(loc = 1)
plt.show()









