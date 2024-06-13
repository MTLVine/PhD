import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import sympy as sy
import functions

###############################################################################
#Section One: Deriving Solvability Conditions Using SymPy

#Background magnetic field
b0x, b0z = sy.symbols('b0x b0z')

#Parent wavevector components
kWx, kWz = sy.symbols('kWx kWz')

#Duaghter wavevector components
kAx, kAy, kAz = sy.symbols('kAx kAy kAz')
kBx, kBy, kBz = sy.symbols('kBx kBy kBz')

#Primary wave frequency
omegaW = sy.symbols('omegaW')

#Daughter wave freqeuncies
omegaA, omegaB = sy.symbols('omegaA omegaB')

#Parent wave eigenfunctions
uWx, uWz, bWx, bWz, rhoW = sy.symbols('uWx uWz bWx bWz rhoW')

#Daughter wave eigenfunctions
uAx, uAy, uAz, bAx, bAy, bAz, rhoA = sy.symbols('uAx uAy uAz bAx bAy bAz rhoA')
uBx, uBy, uBz, bBx, bBy, bBz, rhoB = sy.symbols('uBx uBy uBz bBx bBy bBz rhoB')

#Coefficients
A, B = sy.symbols('A B')
dA, dB = sy.symbols('dA dB')

#Scalar products
kAuW = kAx*uWx + kAz*uWz
kBuW = kBx*uWx + kBz*uWz
kAbW = kAx*bWx + kAz*bWz
kBbW = kBx*bWx + kBz*bWz
kWuA = kWx*uAx + kWz*uAz
kWuB = kWx*uBx + kWz*uBz
kWbA = kWx*bAx + kWz*bAz
kWbB = kWx*bBx + kWz*bBz
kAb0 = kAx*b0x + kAz*b0z
kBb0 = kBx*b0x + kBz*b0z

#Forcing terms
fUxA = dA*uAx + B*kBuW*uBx*1j - B*kBbW*bBx*1j - B*kWuB*uWx*1j + B*kWbB*bWx*1j
fUyA = dA*uAy + B*kBuW*uBy*1j - B*kBbW*bBy*1j
fUzA = dA*uAz + B*kBuW*uBz*1j - B*kBbW*bBz*1j - B*kWuB*uWz*1j + B*kWbB*bWz*1j

fUxB = dB*uBx + A*kAuW*uAx*1j - A*kAbW*bAx*1j + A*kWuA*uWx*1j - A*kWbA*bWx*1j
fUyB = dB*uBy + A*kAuW*uAy*1j - A*kAbW*bAy*1j
fUzB = dB*uBz + A*kAuW*uAz*1j - A*kAbW*bAz*1j + A*kWuA*uWz*1j - A*kWbA*bWz*1j

fBxA = dA*bAx + B*kBuW*bBx*1j - B*kBbW*uBx*1j - B*kWuB*bWx*1j + B*kWbB*uWx*1j
fByA = dA*bAy + B*kBuW*bBy*1j - B*kBbW*uBy*1j
fBzA = dA*bAz + B*kBuW*bBz*1j - B*kBbW*uBz*1j - B*kWuB*bWz*1j + B*kWbB*uWz*1j

fBxB = dB*bBx + A*kAuW*bAx*1j - A*kAbW*uAx*1j + A*kWuA*bWx*1j - A*kWbA*uWx*1j
fByB = dB*bBy + A*kAuW*bAy*1j - A*kAbW*uAy*1j
fBzB = dB*bBz + A*kAuW*bAz*1j - A*kAbW*uAz*1j + A*kWuA*bWz*1j - A*kWbA*uWz*1j

fRhoA = dA*rhoA + B*kBuW*rhoB*1j + B*kWuB*rhoW*1j
fRhoB = dB*rhoB + A*kAuW*rhoA*1j + A*kWuA*rhoW*1j

#Solvability conditions

#Velocity forcing term
GuzA = kAz*omegaA*(kAx*fUxA + kAy*fUyA)*1j - omegaA*(kAx**2 + kAy**2)*fUzA*1j
GuzB = kBz*omegaB*(kBx*fUxB + kBy*fUyB)*1j - omegaB*(kBx**2 + kBy**2)*fUzB*1j

#Magnetic forcing term
GbzA = kAb0*(kAx**2 + kAy**2 + kAz**2)*fBzA*1j
GbzB = kBb0*(kBx**2 + kBy**2 + kBz**2)*fBzB*1j

#Density forcing term
GrhozA = - (kAx**2 + kAy**2)*fRhoA
GrhozB = - (kBx**2 + kBy**2)*fRhoB

#Coefficients of EA and EB
gA = GuzA + GbzA + GrhozA
gB = GuzB + GbzB + GrhozB

#Allows for extraction of coefficients ci below
gA = gA.expand()
gB = gB.expand()

#Amplitude equation coefficients
c1 = gA.coeff(dA)
c2 = gA.coeff(B)
c3 = gB.coeff(dB)
c4 = gB.coeff(A)

###############################################################################
#Section Two: Establishing System Parameters

#Parent wave amplitude
a = 0.1

#Parent wave phase velocity angle
theta = np.pi/4

#Background magnetic field orientation
zeta = 0

#Background magnetic field strength
B0 = 1

#Background magnetic field components
B0x1 = B0*np.cos(zeta)
B0z1 = B0*np.sin(zeta)

#Transformed magnetic field components
B0x2 = functions.Bx(B0x1,B0z1,theta)
B0z2 = functions.Bz(B0x1,B0z1,theta)

#Parent wavevector components
KWx = np.cos(theta)
KWz = np.sin(theta)

#Parent wave frequency
OmegaW = (KWx**2 + B0z2**2)**0.5

#Fourier modes
nn = 8
n = -np.linspace(1,nn,nn)

#Disturbance wavevector components
alpha = np.zeros(nn)
beta = 0

#Floquet modulation parameter
gamma = 0

#Parent wave eigenfunctions
UWx = functions.uxa(a,theta,B0z2)
UWz = functions.uza(a,theta,B0z2)
BWx = functions.bxa(a,theta,B0z2)
BWz = functions.bza(a,theta,B0z2)
RhoW = functions.rhoa(a,theta)

###############################################################################
# Section Three: Determining alpha values satisfying exact resonance

#Expressing all quantities as functions of alpha

for i in range(nn):
         
    #Function for which the root satisfies exact resonance
    def freqres(alpha):
        kAx,kAy,kAz = (gamma + n[i])*KWx + alpha*KWz, beta, (gamma + n[i])*KWz - alpha*KWx
        kBx,kBy,kBz = -kAx + KWx, -kAy, -kAz + KWz
        omegaAagw = ((kAx**2 + kAy**2)/(kAx**2 + kAy**2 + kAz**2) + (kAx*B0x1 + kAz*B0z1)**2)**0.5
        omegaBagw = ((kBx**2 + kBy**2)/(kBx**2 + kBy**2 + kBz**2) + (kBx*B0x1 + kBz*B0z1)**2)**0.5
        omegaAaw = (kAx*B0x1 + kAz*B0z1)
        omegaBaw = (kBx*B0x1 + kBz*B0z1)
        return OmegaW - omegaAagw - omegaBagw
    
    #Finding the alpha value satisfying exact resonance for each n value
    print(i)
    #sol = sp.optimize.root_scalar(freqres,bracket = [0,12])
    sol = sp.optimize.root_scalar(freqres, method = 'secant', x0 = 0.98, x1 = 1)  
    alpha[i] = sol.root

#Matrix for depositing growth rates
growth = np.zeros(nn)

#Matrices for depositing frequencies
freq = np.zeros(nn)

###############################################################################
# Section Four: Predicting the growth rate at exact resonances

for i in range(nn):
      
    # Daughter wavevector components
    KAx = -(gamma + n[i])*KWx - alpha[i]*KWz
    KAy = -beta
    KAz = -(gamma + n[i])*KWz + alpha[i]*KWx
    
    KBx = KAx + KWx
    KBy = KAy
    KBz = KAz + KWz
        
    #Daughter wave frequencies           
    OmegaA = functions.omegaAagw(KAx,KAy,KAz,B0x1,B0z1)
    OmegaB = functions.omegaBagw(KBx,KBy,KBz,B0x1,B0z1)
    
    #Verifying that frequency resonance condition is satisfied
    freq[i] = OmegaW + OmegaA - OmegaB 
    
    #Daughter wave velocities
    UAx,UAy,UAz = functions.uA(KAx,KAy,KAz)
    UBx,UBy,UBz = functions.uB(KBx,KBy,KBz)
    
    #Daughter wave magnetic field
    BAx,BAy,BAz = functions.bA(KAx,KAy,KAz,B0x1,B0z1)
    BBx,BBy,BBz = functions.bB(KBx,KBy,KBz,B0x1,B0z1)
    
    #Daughter wave densities
    RhoA = functions.rhoAagw(KAx,KAy,KAz,B0x1,B0z1)
    RhoB = functions.rhoBagw(KBx,KBy,KBz,B0x1,B0z1)
    
    #Amplitude equation coefficients
    C1 = c1.subs([(kAx,KAx), (kAy,KAy), (kAz,KAz),
                  (uAx,UAx), (uAy,UAy), (uAz,UAz),
                  (bAx,BAx), (bAy,BAy), (bAz,BAz),
                  (rhoA,RhoA), 
                  (omegaA,OmegaA),
                  (b0x,B0x1),
                  (b0z,B0z1)])
    
    C3 = c3.subs([(kBx,KBx), (kBy,KBy), (kBz,KBz),
                  (uBx,UBx), (uBy,UBy), (uBz,UBz),
                  (bBx,BBx), (bBy,BBy), (bBz,BBz),
                  (rhoB,RhoB), 
                  (omegaB,OmegaB),
                  (b0x,B0x1),
                  (b0z,B0z1)])
    
    C2 = c2.subs([(kAx,KAx), (kAy,KAy), (kAz,KAz), 
                  (kBx,KBx), (kBy,KBy), (kBz,KBz),
                  (kWx,KWx),            (kWz,KWz),
                  (uAx,UAx), (uAy,UAy), (uAz,UAz),
                  (uBx,UBx), (uBy,UBy), (uBz,UBz),
                  (uWx,UWx),            (uWz,UWz),
                  (bAx,BAx), (bAy,BAy), (bAz,BAz),
                  (bBx,BBx), (bBy,BBy), (bBz,BBz),
                  (bWx,BWx),            (bWz,BWz),
                  (rhoA,RhoA), 
                  (rhoB,RhoB), 
                  (rhoW,RhoW),
                  (omegaA,OmegaA), 
                  (omegaB,OmegaB),
                  (b0x,B0x1),
                  (b0z,B0z1)])
    
    C4 = c4.subs([(kAx,KAx), (kAy,KAy), (kAz,KAz), 
                  (kBx,KBx), (kBy,KBy), (kBz,KBz),
                  (kWx,KWx),            (kWz,KWz),
                  (uAx,UAx), (uAy,UAy), (uAz,UAz),
                  (uBx,UBx), (uBy,UBy), (uBz,UBz),
                  (uWx,UWx),            (uWz,UWz),
                  (bAx,BAx), (bAy,BAy), (bAz,BAz),
                  (bBx,BBx), (bBy,BBy), (bBz,BBz),
                  (bWx,BWx),            (bWz,BWz),
                  (rhoA,RhoA), 
                  (rhoB,RhoB), 
                  (rhoW,RhoW),
                  (omegaA,OmegaA), 
                  (omegaB,OmegaB),
                  (b0x,B0x1),
                  (b0z,B0z1)])
    
    #Growth rate
    S = ((C2*C4)/(C1*C3))**0.5
    growth[i] = 2*sy.re(S)
    
    
###############################################################################
# Section Four: Taking growth rate predictions at the resonant alpha values and 
# matching them to numerical growth rate peaks

#Numerical data    
data = np.load('growth_1d_b0_0.0.npz', allow_pickle = True)
 
#Growth rate as a function of alpha  
Sigma = data['growth']

#Corresponding alpha values
nalpha = np.size(Sigma)
Alpha = np.linspace(0,4,nalpha)

#Setting up the figure
plt.figure(2)

#Growth rate as a function of alpha
plt.plot(Alpha,Sigma,'black')
plt.xlabel(r'$\alpha$', fontsize = 16)
plt.ylabel(r'$\sigma$', fontsize = 16)
plt.xlim(0,4)

#Resonant alpha as a function of n plotted as vertical dashed lines
for i in range(nn):
    plt.vlines(alpha[i], min(Sigma), max(Sigma), color = 'darkorchid', linestyles = 'dashed')
    
#Predicted growth rate in wavenumber space
plt.scatter(alpha,growth, c = 'red', marker = 's')