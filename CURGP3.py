import numpy as np
import math as m 
import random as rd
import matplotlib.pyplot as plt 
from scipy.optimize import minimize
import pylab as pl

u1 = 2500 #threshold of CUR algorithm , u1 = 10^2
r1 = 50 # number of rows and columns to take 
p = 10 # number of points to take in grid.  # size of matrix is then p^2 by p^2 



#equilibrium data

r1eq = 1.10064 #equilibrium length of radius
r2eq =  1.10065 # equilibrium length of radius
r3eq = 1.120296 #equilibrium length for radius
theta1eq = 121.65 * m.pi / 180 #equilibrium angle in radians
theta2eq = 121.65 * m.pi / 180 #equilibrium angle in radians
phieq = m.pi  #equilibrium angle in radians



#potential in equlibrium 
#The input distance is in bohr and angles are in radian, and the return energy is in cm_-1

def pot(r_1,r_2):
	r1 = (r_1 - r1eq)/r1eq #unitless
	r2 = (r_2 - r2eq)/r2eq #unitless
	r3 = (r3eq - r3eq)/r3eq #unitless
	r4 = theta1eq - theta1eq #unitless
	r5 = theta2eq- theta2eq #unitless
	r6 = phieq - phieq  #unitless

	V = (0.680236 * r1 * r1 + #         f11   
         0.680236*r2*r2+ #                f22
        2.139126*r3*r3+ #... %,   &    !  f33 
        0.143918*r4*r4+ #... %,   &    !  f44 
        0.143918*r5*r5+ #... %,   &    !  f55 
        0.029988*r6*r6+ #... %,   &    !  f66 
        0.023043*r1*r2+ #... %,   &    !  f12 
        0.176269*r1*r3+ #... %,   &    !  f13 
        0.176269*r2*r3+ #... %,   &    !  f23 
        -0.032230*r1*r4+ #... %,   &    !  f14 
        -0.032230*r2*r5+ #... %,   &    !  f25 
        -0.064329*r1*r5+ #... %,   &    !  f15 
        -0.064329*r2*r4+ #... %,   &    !  f24 
        0.117855*r3*r4+ #... %,   &    !  f34 
        0.117855*r3*r5+ #... %,   &    !  f35 
        0.100287*r4*r5+ #... %,   &    !  f45       
        -0.169688*r1*r1*r1+ #... %,   &    !  f111 
        -0.169688*r2*r2*r2+ #... %,   &    !  f222 
        -1.800515*r3*r3*r3+ #... %,   &    !  f333 
        -0.007582*r4*r4*r4+ #... %,   &    !  f444 
        -0.007582*r5*r5*r5+ #... %,   &    !  f122 
        0.204245*r1*r1*r3+ #... %,   &    !  f113 
        0.204245*r2*r2*r3+ #... %,   &    !  f223 
        -0.250697*r1*r3*r3+ #... %,   &    !  f133      
        -0.250697*r2*r3*r3+ #... %,   &    !  f233 
        0.003235*r1*r1*r4+ #... %,   &    !  f114 
        0.003235*r2*r2*r5+ #... %,   &    !  f225 
        -0.024683*r1*r4*r4+ #... %,   &    !  f144 
        -0.024683*r2*r5*r5+ #... %,   &    !  f255 
        0.074283*r1*r1*r5+ #... %,   &    !  f115 
        0.074283*r2*r2*r4+ #... %,   &    !  f224 
        -0.037226*r1*r5*r5+ #... %,   &    !  f155 
        -0.037226*r2*r4*r4+ #... %,   &    !  f244 
        -0.020924*r1*r6*r6+ #... %,   &    !  f166 
        -0.020924*r2*r6*r6+ #... %,   &    !  f266 
        -0.026493*r3*r3*r4+ #... %,   &    !  f334 
        -0.026493*r3*r3*r5+ #... %,   &    !  f335 
        -0.229630*r3*r4*r4+ #... %,   &    !  f344 
        -0.229630*r3*r5*r5+ #... %,   &    !  f355 
        -0.084395*r3*r6*r6+ #... %,   &    !  f366 
        0.069178*r4*r4*r5+ #... %,   &    !  f445 
        0.069178*r4*r5*r5+ #... %,   &    !  f455 
        0.025829*r4*r6*r6+ #... %,   &    !  f466 
        0.025829*r5*r6*r6+ #... %,   &    !  f566 
        0.509035*r1*r2*r3+ #... %,   &    !  f123 
        0.069632*r1*r2*r4+ #... %,   &    !  f124 
        0.069632*r1*r2*r5+ #... %,   &    !  f125 
        -0.171974*r1*r3*r4+ #... %,   &    !  f134 
        -0.171974*r2*r3*r5+ #... %,   &    !  f235 
        -0.001865*r1*r3*r5+ #... %,   &    !  f135 
        -0.001865*r2*r3*r4+ #... %,   &    !  f234 
        -0.090346*r1*r4*r5+ #... %,   &    !  f145 
        -0.090346*r2*r4*r5+ #... %,   &    !  f245 
        0.120409*r3*r4*r5+ #... %,   &    !  f345       
        -0.499589*r1*r1*r1*r1+ #... %,   &    !  f1111 
        -0.499589*r2*r2*r2*r2+ #... %,   &    !  f2222 
        -0.689528*r3*r3*r3*r3+ #... %,   &    !  f3333 
        0.005005*r4*r4*r4*r4+ #... %,   &    !  f4444 
        0.005005*r5*r5*r5*r5+ #... %,   &    !  f5555 
        0.005770*r6*r6*r6*r6+ #... %,   &    !  f6666 
        0.055208*r1*r1*r1*r2+ #... %,   &    !  f1112 
        0.055208*r1*r2*r2*r2+ #... %,   &    !  f1222 
        0.273116*r1*r1*r2*r2+ #... %,   &    !  f1122 
        -0.795177*r1*r1*r1*r3+ #... %,   &    !  f1113 
        -0.795177*r2*r2*r2*r3+ #... %,   &    !  f2223 
        0.632579*r1*r1*r3*r3+ #... %,   &    !  f1133 
        0.632579*r2*r2*r3*r3+ #... %,   &    !  f2233 
        -0.137121*r1*r3*r3*r3+ #... %,   &    !  f1333 
        -0.137121*r2*r3*r3*r3+ #... %,   &    !  f2333 
        0.004848*r1*r1*r1*r4+ #... %,   &    !  f1114 
        0.004848*r2*r2*r2*r5+ #... %,   &    !  f2225 
        -0.181627*r1*r1*r4*r4+ #... %,   &    !  f1144 
        -0.181627*r2*r2*r5*r5+ #... %,   &    !  f2255 
        0.013648*r1*r4*r4*r4+ #... %,   &    !  f1444 
        0.013648*r2*r5*r5*r5+ #... %,   &    !  f2555 
        -0.026697*r1*r1*r1*r5+ #... %,   &    !  f1115 
        -0.026697*r2*r2*r2*r4+ #... %,   &    !  f2224 
        -0.056350*r1*r1*r5*r5+ #... %,   &    !  f1155 
        -0.056350*r2*r2*r4*r4+ #... %,   &    !  f2244 
        -0.007536*r1*r5*r5*r5+ #... %,   &    !  f1555 
        -0.007536*r2*r4*r4*r4+ #... %,   &    !  f2444 
        -0.017089*r1*r1*r6*r6+ #... %,   &    !  f1166 
        -0.017089*r2*r2*r6*r6+ #... %,   &    !  f2266 
        -0.139056*r3*r3*r3*r4+ #... %,   &    !  f3334 
        -0.139056*r3*r3*r3*r5+ #... %,   &    !  f3335 
        0.075517*r3*r3*r4*r4+ #... %,   &    !  f3344 
        0.075517*r3*r3*r5*r5+ #... %,   &    !  f3355 
        0.067468*r3*r4*r4*r4+ #... %,   &    !  f3444 
        0.067468*r3*r5*r5*r5+ #... %,   &    !  f3555       
        0.212949*r3*r3*r6*r6+ #... %,   &    !  f3366 
        0.013397*r4*r4*r4*r5+ #... %,   &    !  f4445 
        0.013397*r4*r5*r5*r5+ #... %,   &    !  f4555 
        0.092667*r4*r4*r5*r5+ #... %,   &    !  f4455 
        -0.006321*r4*r4*r6*r6+ #... %,   &    !  f4466 
        -0.006321*r5*r5*r6*r6+ #... %,   &    !  f5566 
        1.464673*r1*r1*r2*r3+ #... %,   &    !  f1123 
        1.464673*r1*r2*r2*r3+ #... %,   &    !  f1223 
        -1.276774*r1*r2*r3*r3+ #... %,   &    !  f1233 
        0.076275*r1*r1*r2*r4+ #... %,   &    !  f1124 
        0.076275*r1*r2*r2*r5+ #... %,   &    !  f1225 
        0.083466*r1*r2*r2*r4+ #... %,   &    !  f1224 
        0.083466*r1*r1*r2*r5+ #... %,   &    !  f1125 
        -0.058302*r1*r2*r4*r4+ #... %,   &    !  f1244 
        -0.058302*r1*r2*r5*r5+ #... %,   &    !  f1255 
        -0.030976*r1*r2*r6*r6+ #... %,   &    !  f1266 
        -0.170630*r1*r1*r3*r4+ #... %,   &    !  f1134 
        -0.170630*r2*r2*r3*r5+ #... %,   &    !  f2235 
        0.221103*r1*r3*r3*r4+ #... %,   &    !  f1334 
        0.221103*r2*r3*r3*r5+ #... %,   &    !  f2335 
        0.080599*r1*r3*r4*r4+ #... %,   &    !  f1344 
        0.080599*r2*r3*r5*r5+ #... %,   &    !  f2355 
        0.021261*r1*r1*r3*r5+ #... %,   &    !  f1135 
        0.021261*r2*r2*r3*r4+ #... %,   &    !  f2234 
        -0.011458*r1*r3*r3*r5+ #... %,   &    !  f1335 
        -0.011458*r2*r3*r3*r4+ #... %,   &    !  f2334 
        0.079214*r1*r3*r5*r5+ #... %,   &    !  f1355 
        0.079214*r2*r3*r4*r4+ #... %,   &    !  f2344 
        -0.442191*r1*r3*r6*r6+ #... %,   &    !  f1366 
        -0.442191*r2*r3*r6*r6+ #... %,   &    !  f2366 
        0.001912*r1*r1*r4*r5+ #... %,   &    !  f1145 
        0.001912*r2*r2*r4*r5+ #... %,   &    !  f2245 
        -0.018075*r1*r4*r4*r5+ #... %,   &    !  f1445 
        -0.018075*r2*r4*r5*r5+ #... %,   &    !  f2455 
        -0.022295*r1*r4*r5*r5+ #... %,   &    !  f1455 
        -0.022295*r2*r4*r4*r5+ #... %,   &    !  f2445 
        0.004889*r1*r4*r6*r6+ #... %,   &    !  f1466 
        0.004889*r2*r5*r6*r6+ #... %,   &    !  f2566 
        -0.069140*r1*r5*r6*r6+ #... %,   &    !  f1566 
        -0.069140*r2*r4*r6*r6+ #... %,   &    !  f2466 
        -0.737176*r3*r3*r4*r5+ #... %,   &    !  f3345 
        -0.028934*r3*r4*r4*r5+ #... %,   &    !  f3445 
        -0.028934*r3*r4*r5*r5+ #... %,   &    !  f3455 
        0.018220*r3*r4*r6*r6+ #... %,   &    !  f3466 
        0.018220*r3*r5*r6*r6+ #... %,   &    !  f3566 
        0.020535*r4*r5*r6*r6+ #... %,   &    !  f4566 
        -0.121751*r1*r2*r3*r4+ #... %,   &    !  f1234 
        -0.121751*r1*r2*r3*r5+ #... %,   &    !  f1235 
        -0.079185*r1*r2*r4*r5+ #... %,   &    !  f1245 
        -0.055581*r1*r3*r4*r5+ #... %,   &    !  f1345 
        -0.055581*r2*r3*r4*r5+ #... %,   &    !  f2345      
        -0.042230*r4*r4*r4*r6*r6+ #... %,   &    !  f44466 
        -0.042230*r5*r5*r5*r6*r6+ #... %,   &    !  f55566 
        0.035046*r4*r4*r5*r6*r6+ #... %,   &    !  f44566 
        0.035046*r4*r5*r5*r6*r6 )#%/        !  f45566 
        
	return V




#norm of a matrix or a row/columns
def Fnorm(a):
	e1 = np.multiply(a,a)  #squares each element by itself
	e3 = np.sum(e1)         #sums the squares
	return np.sqrt(e3)      #returns sqrt of the sum of the squares



#difference of two matrices

def norm(a,b):
    e1 = np.subtract(a,b)
    e2 = np.multiply(e1,e1)
    e3 = np.sum(e2)
    return np.sqrt(e3)



#finds moore penrose inverse of matrix, i.e., the left inverse
def MP(M): 
	e = np.transpose(M)  #transpose of M
	e1 = np.dot(M,e)     # multiply M and M transpose
	e2 = np.linalg.inv(e1)  # inverts the above step
	e3 = np.dot(e,e2)      # multiplies transpose of matrix M with inverse matrix

	return e3

#left inverse
def MP1(M): 
    e = np.transpose(M)  #transpose of M
    e1 = np.dot(e,M)     # multiply M and M transpose
    e2 = np.linalg.inv(e1)  # inverts the above step
    e3 = np.dot(e2,e)      # multiplies transpose of matrix M with inverse matrix

    return e3

#kroneker delta function - needed for algorithm
def kroneker(a,b):
	if a == b:
		return 1

	else:
		return 0



# M is the matrix, r is number of rows and columns to take and u^2 is randomness threshold
def approxinv(M,r,u):
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

	#h1 = m.floor(v3/u)
	h1 = v3//u
	h2 = (v3%u) 
	
	C = M[:,randvecs1[h1]]
	U = C[randvecs2[h2],:]
	R = M[randvecs2[h2],:]
	U1 = np.linalg.inv(U)
	#h5 = np.linalg.det(U)
	X = np.dot(U1,R)
	B = MP(X)
	A = MP(C)

	
    
	return (np.dot(B,A))
	
	






#main plot for comparison too and such 
x = np.linspace(0,4,100)
y = np.linspace(0,4,100)
X, Y = np.meshgrid(x, y)
Z1 = X
Z2 = Y
Z = pot(Z1,Z2)


'''
plt.figure()
contour = plt.contour(X, Y, Z, 5, colors='k')
plt.clabel(contour, cmap='Blues', fmt = '%2.1f', fontsize=12)
contour_filled = plt.contourf(X, Y, Z,cmap='Blues')
clb = plt.colorbar()
plt.title('Formaldehyde PES')
plt.xlabel('r1 (unitless)')
plt.ylabel('r2 (unitless)')
plt.text(1.25, 0.5, "Potential Energy cm^-1", {'color': 'black', 'fontsize': 10}, #this makes the right y axis label
         horizontalalignment='left',
         verticalalignment='center',
         rotation=270,
         clip_on=False,
         transform=plt.gca().transAxes)
plt.show()
'''



#This routine here makes the grid to do GP on 

x1_data = np.linspace(-1,4,p)   #grid for r_1 
x2_data = np.linspace(-1,4,p)   #grid for r_2 

xnew = []
ynew = []

for i in x1_data:
    for j in x2_data:
        a = (i,j)
        xnew.append(a)

for i in x1_data:
    for j in x2_data:
        a = pot(i,j)
        ynew.append(a)



#Parameters optimization Algorithm 
def kernel(x,y,pars):
    sigma_1 = pars[0]
    l_1 = pars[1] 
    sigma_2 = pars[2]
    l_2 = pars[3]
    delta = pars[4]
    
    a_1 = (x[0]-y[0])**2
    a_2 = (x[1]-y[1])**2


    return sigma_1**2 * np.exp(-a_1/(2*l_1**2)) * sigma_2**2 * np.exp(-a_2/(2*l_2**2)) + kroneker(x,y)*delta**2

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
	n3 = - (len(xnew)/2) * np.log(m.pi * 2)

	return -1*(n1+n2+n3)


res = minimize(log_likelyhood, x0=[2,-1,2,1,2], method = 'CG',options={'maxiter': 100})


#print(res.x)


matrix = cov_matrix(res.x)
t = approxinv(matrix,r1,u1)





def GP(x11,x22):
    xx = (x11,x22)
    y = np.array([c for c in ynew])
    K_star = np.array([kernel(xx,i,(res.x)) for i in xnew]) 
    K_starstar = kernel(xx,xx,(res.x))
    
    t1 = np.dot(t,y)

    y_star = np.dot(K_star,t1)
    r = np.dot(K_star,t)
    y_var = K_starstar - np.dot(r,np.transpose(K_star))

    return (y_star)



n = 100 # number of points to do GP with 

#GP plot of Formaldehyde potential

a = np.linspace(0,4,n)
b = np.linspace(0,4,n)
A, B = np.meshgrid(a, b)
A1 = A
B1 = B 





C = np.zeros((n,n))

for i in range(n):
    for k in range(n):
        C[i,k] = GP(a[i],b[k])


G = Z-C




plt.figure()
contour = plt.contour(A, B, C, 5, colors='k')
plt.clabel(contour, cmap='Blues', fmt = '%2.1f', fontsize=12)
contour_filled = plt.contourf(A, B, C,cmap='Blues')
clb = plt.colorbar()
plt.title('Formaldehyde PES Using GPCUR with r=50')
plt.xlabel('r1 (unitless)')
plt.ylabel('r2 (unitless)')
plt.text(1.25, 0.5, "Potential Energy cm^-1", {'color': 'black', 'fontsize': 10},
         horizontalalignment='left',
         verticalalignment='center',
         rotation=270,
         clip_on=False,
         transform=plt.gca().transAxes)
plt.show()



plt.figure()
contour = plt.contour(A, B, G, 5, colors='k')
plt.clabel(contour, cmap='Blues', fmt = '%2.1f', fontsize=12)
contour_filled = plt.contourf(A, B, G,cmap='Blues')
clb = plt.colorbar()
plt.title('Difference Between Potential and GPCUR r=50')
plt.xlabel('r1 (unitless)')
plt.ylabel('r2 (unitless)')
plt.text(1.25, 0.5, "Difference cm^-1", {'color': 'black', 'fontsize': 10},
         horizontalalignment='left',
         verticalalignment='center',
         rotation=270,
         clip_on=False,
         transform=plt.gca().transAxes)
plt.show()



















