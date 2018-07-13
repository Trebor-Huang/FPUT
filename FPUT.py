import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.animation as anime

"""
The FPUT system
X[-1] = X[N-1] = 0
"""

N = 100

A = np.sqrt(2/N) * np.array([[np.sin(i*j*np.pi/N) for i in range(1,N)]for j in range(1,N)])
L = np.array([4*np.sin(i*np.pi/(2*N))**2 for i in range(1,N)])
U = np.diag([2.]*(N-1))+np.diag([-1.]*(N-2), k=1)+np.diag([-1.]*(N-2), k=-1)

#print(U)
#print(np.allclose(A.T @ A, np.eye(N-1)))
#print(np.allclose((A.T @ U @ A),np.diag(L)))

X = np.zeros((N-1,))
Y = np.zeros((N-1,))

Y[0] = 10
X = A.T @ Y

"""
Always true:
Y = A @ X
E[i] = (m*(dY[i]/dt)**2 + k*L[i,i]Y[i]**2)/2
"""

k=1.
m=1.
r = k/m
alpha= 0.1
dt = .1

E = []

dX = np.zeros((N-1,))

for num in range(400000):
    PdL = np.concatenate(([0],X))
    PdR = np.concatenate((X,[0]))
    Ld = PdL[:-1]-X
    Rd = X-PdR[1:]
    y = Ld-Rd
    Z = 1 + alpha*(Ld+Rd)
    ddX = y*Z*r
    dX += dt*ddX
    X += dt*dX

    if num%1000 == 0:
        Y = A @ X
        dY = A @ dX
        E.append([])
        for i in range(N-1):
            E[-1].append((m*(dY[i])**2 + k*L[i]*Y[i]**2)/2)

E = np.array(E)
Etot = np.sum(E, axis = 1)

plt.plot(E[...,0],'r', E[...,1],'g', E[...,2],'b', Etot, 'r.')
plt.show()

"""dX = np.array([0.]*N)

fig = plt.figure()
ax = fig.add_subplot(111,ylim = (-1,1))

line, = ax.plot(X)
lx,ly = line.get_data()

#plt.show()

k=1.
m=1.
r = k/m
alpha= .5
dt = .1

def animate(i):
    global X, dX
    #print("s")
    PdL = np.concatenate(([0],X))
    PdR = np.concatenate((X,[0]))
    Ld = PdL[:-1]-X
    Rd = X-PdR[1:]
    Y = Ld-Rd
    Z = 1 + alpha*(Ld+Rd)
    ddX = Y*Z*r
    dX += dt*ddX
    X += dt*dX
    line.set_data(lx,X)
    "if i==0:
        hist[0].append(m1)
        hist[1].append(m2)
        hist[2].append(m3)"
    return line#,dots

an = anime.FuncAnimation(fig,animate,frames = 2,interval = 1)
plt.show()

#print(hist)
#kk = range(len(hist[0]))
#plt.plot(kk,hist[0],'r.',kk,hist[1],'g.',kk,hist[2],'b.')
#plt.show()
"""
