import numpy as np
import sympy as sy
import functions

###############################################################################
#Section One: Deriving Matrix Coefficients Using SymPy

#Background magnetic field strength
b0 = sy.symbols('b0')

#Background magnetic field components
b0x, b0z = sy.symbols('b0x b0z')

#Parent wavevector components
kWx, kWz = sy.symbols('kWx kWz')

#Disturbance wavevector components
alpha, beta, gamma, n = sy.symbols('alpha beta gamma n')

#Parent wave amplitudes
uW, bW, rhoW = sy.symbols('uW bW rhoW')

#Dependent variables (n = 2)
u1x, u1y, u1z = sy.symbols('u1x u1y u1z')
u2x, u2y, u2z = sy.symbols('u2x u2y u2z')
u3x, u3y, u3z = sy.symbols('u3x u3y u3z')

b1x, b1y, b1z = sy.symbols('b1x b1y b1z')
b2x, b2y, b2z = sy.symbols('b2x b2y b2z')
b3x, b3y, b3z = sy.symbols('b3x b3y b3z')

rho1 = sy.symbols('rho1')
rho2 = sy.symbols('rho2')
rho3 = sy.symbols('rho3')

#Primary wave frequency
omega = (kWx**2 + b0z**2)**0.5

#Operators
G2 = omega*(gamma + n)*1j
L2 = - alpha**2 - beta**2 - (gamma + n)**2

#Scalar products
kb0 = alpha*b0x + (gamma + n)*b0z

#Pressure
p2 = ((alpha*kWx/kWz - (gamma + n))*rho2*1j + 2*alpha*uW*(u1z - u3z) - 2*alpha*bW*(b1z - b3z))/L2

#Equation determining matrix coefficients for each row
R1 = G2*u2x + kb0*b2x*1j - alpha*uW*(u1x + u3x)*1j + alpha*bW*(b1x + b3x)*1j - uW*(u1z - u3z)*1j + bW*(b1z - b3z)*1j - alpha*p2*1j + kWx*rho2/kWz 
R2 = G2*u2z + kb0*b2z*1j - alpha*uW*(u1z + u3z)*1j + alpha*bW*(b1z + b3z)*1j                                         - (gamma + n)*p2*1j - rho2
R3 = G2*b2x + kb0*u2x*1j + alpha*bW*(u1x + u3x)*1j - alpha*uW*(b1x + b3x)*1j - bW*(u1z - u3z)*1j + uW*(b1z - b3z)*1j
R4 = G2*b2z + kb0*u2z*1j + alpha*bW*(u1z + u3z)*1j - alpha*uW*(b1z + b3z)*1j
R5 = G2*rho2           - alpha*uW*(rho1 + rho3)*1j                         - rhoW*(u1z + u3z)*1j                     + kWz*(kWz*u2z - kWx*u2x)

#Allows for extraction of coefficients below
R = sy.zeros(5)

R[0] = sy.expand(R1)
R[1] = sy.expand(R2)
R[2] = sy.expand(R3)
R[3] = sy.expand(R4)
R[4] = sy.expand(R5)

#Forming the submatrices
M1 = sy.zeros(5,5)
M2 = sy.zeros(5,5)
M3 = sy.zeros(5,5)

for i in range(5):
    
    #Lower diagonal submatrix  
    M1[i,0] = R[i].coeff(u1x)
    M1[i,1] = R[i].coeff(u1z)
    M1[i,2] = R[i].coeff(b1x)
    M1[i,3] = R[i].coeff(b1z)
    M1[i,4] = R[i].coeff(rho1)
    
    #Main diagonal submatrix
    M2[i,0] = R[i].coeff(u2x)
    M2[i,1] = R[i].coeff(u2z)
    M2[i,2] = R[i].coeff(b2x)
    M2[i,3] = R[i].coeff(b2z)
    M2[i,4] = R[i].coeff(rho2)
    
    #Upper diagonal submatrix
    M3[i,0] = R[i].coeff(u3x)
    M3[i,1] = R[i].coeff(u3z)
    M3[i,2] = R[i].coeff(b3x)
    M3[i,3] = R[i].coeff(b3z)
    M3[i,4] = R[i].coeff(rho3)

###############################################################################
#Section Two: Establishing System Parameters

#Amplitude parameter
a = 0.1

#Wave propagation angle
theta = np.pi/4

#Parent wavevector components
KWx = np.cos(theta)
KWz = np.sin(theta)

#Background magnetic field strength
B0 = 0.1

#Background magnetic field orientation
zeta = 0

#Background magnetic field components (original frame)
B0x1 = B0*np.cos(zeta)
B0z1 = B0*np.sin(zeta)

#Background magnetic field components (transformed frame)
B0x2 = functions.Bx(B0x1,B0z1,theta)
B0z2 = functions.Bz(B0x1,B0z1,theta)

#Disturbance wavevector components
nAlpha = 100
Alpha = np.linspace(1E-10,4,nAlpha)
Beta = 0

#Floquet modulation parameter
Gamma = 0

#Number of Fourier modes
nN = 41

#n values of Fourier modes
N = np.linspace(-(nN - 1)/2,(nN - 1)/2,nN,dtype = int)

#Parent wave amplitudes
UW = functions.Ua(a,theta,B0z2)
BW = functions.Ba(a,theta,B0z2)
RhoW = functions.Rhoa(a)

###############################################################################
#Section Three: Computing growth rates using SymPy

#Inputting system parameters into submatrices
M1 = M1.subs([(kWx,KWx), (kWz,KWz), (b0x,B0x2), (b0z,B0z2), (beta,Beta), (gamma,Gamma), (uW,UW), (bW,BW), (rhoW,RhoW)])
M2 = M2.subs([(kWx,KWx), (kWz,KWz), (b0x,B0x2), (b0z,B0z2), (beta,Beta), (gamma,Gamma), (uW,UW), (bW,BW), (rhoW,RhoW)])
M3 = M3.subs([(kWx,KWx), (kWz,KWz), (b0x,B0x2), (b0z,B0z2), (beta,Beta), (gamma,Gamma), (uW,UW), (bW,BW), (rhoW,RhoW)])

#Growth rates
growth = np.zeros(nAlpha)

for k in range(nAlpha):
    
    m1 = M1.subs(alpha,Alpha[k])
    m2 = M2.subs(alpha,Alpha[k])
    m3 = M3.subs(alpha,Alpha[k])
    
    #Overall matrix 
    M = np.zeros((5*nN,5*nN), dtype = complex)
    
    for i in range(nN):     
        for j in range(nN): 
            if i == j:
                M[5*i:5*i+5,5*j:5*j+5] = m2.subs(n,N[i])
            if j == i + 1:
                M[5*i:5*i+5,5*j:5*j+5] = m3.subs(n,N[i])
            if j == i - 1:
                M[5*i:5*i+5,5*j:5*j+5] = m1.subs(n,N[i])
      
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

#Growth rate against alpha
np.savez('growth_1d_b0_0.1', growth = growth, a = a, theta = theta, zeta = zeta, B0 = B0, beta = beta, gamma = gamma, nN = nN)