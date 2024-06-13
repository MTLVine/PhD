import numpy as np
import sympy as sy
import functions
import matplotlib.pyplot as plt

###############################################################################
#Section One: Deriving Matrix Coefficients Using SymPy

#Parent wavevector components
kWx, kWz = sy.symbols('kWx kWz')

#Disturbance wavevector components
alpha,beta,gamma,n = sy.symbols('alpha beta gamma n')

#Parent wave amplitudes
uW, rhoW = sy.symbols('uW rhoW')

#Dependent variables (n = 2)
u1x, u1y, u1z = sy.symbols('u1x u1y u1z')
u2x, u2y, u2z = sy.symbols('u2x u2y u2z')
u3x, u3y, u3z = sy.symbols('u3x u3y u3z')

rho1 = sy.symbols('rho1')
rho2 = sy.symbols('rho2')
rho3 = sy.symbols('rho3')

#Primary wave frequency
omega = kWx

#Operators
G2 = omega*(gamma + n)*1j
L2 = - alpha**2 - beta**2 - (gamma + n)**2

#Pressure
p2 = ((alpha*kWx/kWz - (gamma + n))*rho2*1j + 2*alpha*uW*(u1z - u3z))/L2

#Equation determining matrix coefficients for each row
R1 = G2*u2x - alpha*uW*(u1x + u3x)*1j - alpha*p2*1j + kWx*rho2/kWz - uW*(u1z - u3z)*1j
R2 = G2*u2z - alpha*uW*(u1z + u3z)*1j - (gamma + n)*p2*1j - rho2
R3 = G2*rho2 - alpha*uW*(rho1 + rho3)*1j + kWz*(kWz*u2z - kWx*u2x) - rhoW*(u1z + u3z)*1j

#Allows for extraction of coefficients below
R = sy.zeros(3)

R[0] = sy.expand(R1)
R[1] = sy.expand(R2)
R[2] = sy.expand(R3)

#Forming the submatrices
M1 = sy.zeros(3,3)
M2 = sy.zeros(3,3)
M3 = sy.zeros(3,3)

for i in range(3):
    
    #Lower diagonal submatrix  
    M1[i,0] = R[i].coeff(u1x)
    M1[i,1] = R[i].coeff(u1z)
    M1[i,2] = R[i].coeff(rho1)
    
    #Main diagonal submatrix
    M2[i,0] = R[i].coeff(u2x)
    M2[i,1] = R[i].coeff(u2z)
    M2[i,2] = R[i].coeff(rho2)
    
    #Upper diagonal submatrix
    M3[i,0] = R[i].coeff(u3x)
    M3[i,1] = R[i].coeff(u3z)
    M3[i,2] = R[i].coeff(rho3)

###############################################################################
#Section Two: Establishing System Parameters

#Amplitude parameter
a = 0.1

#Wave propagation angle
theta = np.pi/4

#Parent wavevector components
KWx = np.cos(theta)
KWz = np.sin(theta)

#Disturbance wavevector components
nAlpha = 100
Alpha = np.linspace(1E-10,4,nAlpha)
Beta = 0

#Floquet modulation parameter
Gamma = 0

#Number of Fourier modes
nN = 21

#n values of Fourier modes
N = np.linspace(-(nN - 1)/2,(nN - 1)/2,nN,dtype = int)

#Parent wave amplitudes
UW = functions.Ua(a,theta,0)
RhoW = functions.Rhoa(a)

###############################################################################
#Section Three: Computing growth rates using SymPy

#Inputting system parameters into submatrices
M1 = M1.subs([(kWx,KWx), (kWz,KWz), (beta,Beta), (gamma,Gamma), (uW,UW), (rhoW,RhoW)])
M2 = M2.subs([(kWx,KWx), (kWz,KWz), (beta,Beta), (gamma,Gamma), (uW,UW), (rhoW,RhoW)])
M3 = M3.subs([(kWx,KWx), (kWz,KWz), (beta,Beta), (gamma,Gamma), (uW,UW), (rhoW,RhoW)])

#Growth rates
growth = np.zeros(nAlpha)

for k in range(nAlpha):
    
    m1 = M1.subs(alpha,Alpha[k])
    m2 = M2.subs(alpha,Alpha[k])
    m3 = M3.subs(alpha,Alpha[k])
    
    #Overall matrix 
    M = np.zeros((3*nN,3*nN), dtype = complex)
    
    for i in range(nN):     
        for j in range(nN): 
            if i == j:
                M[3*i:3*i+3,3*j:3*j+3] = m2.subs(n,N[i])
            if j == i + 1:
                M[3*i:3*i+3,3*j:3*j+3] = m3.subs(n,N[i])
            if j == i - 1:
                M[3*i:3*i+3,3*j:3*j+3] = m1.subs(n,N[i])
      
    #All eigenvalues and eigenvectors
    eig = np.linalg.eig(M)
    
    #All eigenvalues
    eigval = eig[0]
    
    #All eigenvectors
    eigvec = eig[1]
    
    #Growth of each mode
    resigma = np.real(eigval)
    
    #Fastest growing mode
    growth[k] = max(resigma)
    
    #Adjusts values on the basis that decaying modes do not grow
    if growth[k] < 0:
        growth[k] = 0
        
    print(k)

#Growth rate against alpha
np.savez('growth_1d_theta_piby4', growth = growth, a = a, theta = theta, beta = beta, gamma = gamma, nN = nN)


    
        
        












