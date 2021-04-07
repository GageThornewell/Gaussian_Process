import numpy as np
import math as m 
import random as rd
import matplotlib.pyplot as plt 
import itertools as it 
#import operator

n = 4 # size of square matrix
#r = 2   # number of combinations to take 
t = [i for i in range(0,n)]
#t1 = [i for i in range(0,r)]

#function to create entries in matrix
def f(a,b):
	return np.sin((a-b**2))


#makes the matrix

a = np.zeros((n,n))

for i in range(0,n):
	for k in range(0,n):
		a[i,k] = f(i,k)

print(a)

def random_combination(iterable, m):   #gives an arbitrary combinations
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(rd.sample(range(n), m))
    return tuple(pool[i] for i in indices)


def index(Y):   #returns a tuple from a matrix with (max element in modulus, row index, column index)
    b = np.amax(Y)
    c = np.amin(Y)
    if abs(b) >= abs(c):
    	b1 = m.floor(np.argmax(Y)/r)
    	return (b,b1,np.argmax(Y)%r)
    else: 
    	b2 = m.floor(np.argmin(Y)/r)
    	return(c,b2,np.argmin(Y)%r)


    u1 = m.floor(np.argmax(Y)/r)
    return (u,u1,np.argmax(Y)%r)


#difference of two matrices

def norm(a,b):
	v1 = np.subtract(a,b)
	v2 = np.multiply(v1,v1)
	v3 = np.sum(v2)
	return np.sqrt(v3)

#frobinius norm 

def Fnorm(a):
	v1 = np.multiply(a,a)
	v3 = np.sum(v1)
	return np.sqrt(v3)

#print(Fnorm(a))

def MP(M): 
	e = np.transpose(M)
	e1 = np.dot(e,M)
	e2 = np.linalg.inv(e1)
	e3 = np.dot(e2,e)

	return e3




#this will compute the CUR decomposition. I eventually want it to give me an approximate matrix close enough to A such
#the frobenius norm is very small. After I can use this decomposition to compute an approximate inverse
#using the penrose inverse. 




def CUR(M,r):
	probs1 = []
	columns = []
	probs2 = []
	rows = []
	for i in range(n):
		a = Fnorm(M[:,i])
		b = a/Fnorm(M)
		probs1.append(b)

	sl1 = sorted(probs1)
	sl2 = sl1[::-1]

	for i in range(r):
		columns.append(probs1.index(sl2[i]))

	C = M[:,columns]

	
	for i in range(n):
		a = Fnorm(M[i,:])
		b = a/Fnorm(M)
		probs2.append(b)

	sl3 = sorted(probs2)
	sl4 = sl3[::-1]

	for i in range(r):
		rows.append(probs2.index(sl4[i]))

	R = M[rows,:]

	U = np.linalg.inv(R[:,columns])

	X = np.dot(C,U)
	X1 = np.dot(X,R)

		


	
		

	return X1

def approxinv(M,r):
	probs1 = []
	columns = []
	probs2 = []
	rows = []
	for i in range(len(M[:,1])):
		a = Fnorm(M[:,i])
		b = a/Fnorm(M)
		probs1.append(b)

	sl1 = sorted(probs1)
	sl2 = sl1[::-1]

	for i in range(r):
		columns.append(probs1.index(sl2[i]))

	C = M[:,columns]

	
	for i in range(len(M[:,1])):
		a = Fnorm(M[i,:])
		b = a/Fnorm(M)
		probs2.append(b)

	sl3 = sorted(probs2)
	sl4 = sl3[::-1]

	for i in range(r):
		rows.append(probs2.index(sl4[i]))

	R = M[rows,:]

	U = np.linalg.inv(R[:,columns])

	X = np.dot(C,U)   #CU^-1
	X1 = np.dot(X,R)  #R 
	C = MP(R)
	B = MP(X)

	return np.dot(C,B)

	
	
norms = []	


def approxinv1(M,r):
	probs1 = []
	columns = []
	probs2 = []
	rows = []
	for i in range(len(M[:,1])):
		a = Fnorm(M[:,i])
		b = a/Fnorm(M)
		probs1.append(b)

	sl1 = sorted(probs1)
	sl2 = sl1[::-1]

	for i in range(r):
		columns.append(probs1.index(sl2[i]))

	C = M[:,columns]

	X = np.dot(MP(C),M)

	B = MP(X)
	A = MP(C)

	return(np.dot(B,A))


v = (approxinv1(a,2))

print(np.dot(a,v))

for i in t:
	g = CUR(a,i)
	G = norm(g,a)
	norms.append(G)


#plt.plot(t,norms,'bo')

plt.show()

#print(norm(a,CUR(a,1)))


'''


    Q = np.dot(np.transpose(M),M)
	eigvals, eigvecs = np.linalg.eig(M)
	V = np.transpose(eigvecs )          #right singular matrix
	#print(V)

	stats = []
	columns=[]

	def NSLS(c):
		e = np.multiply(c,c)
		return (1/len(c)) * np.sum(e)

	for i in range(n):
		stats.append(NSLS(V[:,i]))

	sl=sorted(stats)

	for i in range(r):
		columns.append(stats.index(sl[i]))

	C = M[:,columns]
	X = np.dot(MP(C),M)
	return (np.dot(C,X))



	
'''






	


#print(a)
#print(CUR(a))
#print(norm(a,CUR(a)))
#print(np.dot(penrose(CUR(a)),a))
    
    

     



#CUR(a)
#print(norm(a,CUR(a)))
#norm(a,CUR(a))
#norm(a,CUR(a))




'''
def CUR(M,delta):
	#step 0
	a1 = random_combination(t,r) # some random combinations
	C = M[:,a1]                  #random n by r matrix
	a2 = random_combination(t,r) # creates 
	A = C[a2,:] 
	print(np.linalg.det(A))

	 

	perm = [i for i in range(0,n)]
	for i in range(r):
		perm[i] = a2[i]

	C[t, :] = C[perm, :] 

	#step 1 

	B = np.dot(C,np.linalg.inv(A))
	b = index(B)[0]
	

	#step 2 

	if b <= 1 + delta:
		print(A)
	else:

		while b > 1 + delta:
			B[[index(B)[1],index(B)[2]],:] = B[[index(B)[2],index(B)[1]],:]   #swaps columns and rows of B 
		
			newA = B[t1,:]
			B_new = np.dot(C,np.linalg.inv(newA))
		
			b = index(B_new)[0]
			#break
			print(np.linalg.det(newA))

		#print(newA)
		#print(np.linalg.det(newA))

	#now we have a good submatrix new_A , the next problem is to keep track of the rows and columns and construct our
	#C U and R matrices. 

	#works ocassionaly but also blows up computer. 
'''		



	

          
		                 

#CUR(a,1)
#print(a)

#p = np.array([[1,2,3,5],[3,4,3,1],[2,0,0,1],[9,2,1,3]])

#print(p)

#p[[1,3],:] = p[[3,1],:]  swaps rows 3 and 1 
#print(p)
#arr[[frm, to],:] = arr[[to, frm],:]







	

    	

    
  

	
	






	 
















'''
	determinants = []
	for i in it.combinations(t,m):
		for k in it.combinations(t,m):
			B = M[i,:]
			U = B[:,k]  # m by m matrix that we want to optimize
			det = np.linalg.det(U)
			determinants.append(det)

'''



    
    




			

			









	





