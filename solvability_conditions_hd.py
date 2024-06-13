import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import sympy as sy
import functions

# Purpose of script: Takes parameter values a and theta, fixes beta and gamma at
# zero and finds one resonant alpha value for each Fourier mode n in a specified
# range of Fourier modes

###############################################################################
#Section One: Deriving Solvability Conditions Using SymPy

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
uWx, uWz, rhoW = sy.symbols('uWx uWz rhoW')

#Daughter wave eigenfunctions
uAx, uAy, uAz, rhoA = sy.symbols('uAx uAy uAz rhoA')
uBx, uBy, uBz, rhoB = sy.symbols('uBx uBy uBz rhoB')

#Coefficients
A, B = sy.symbols('A B')
dA, dB = sy.symbols('dA dB')

#Scalar products
kAuW = kAx*uWx + kAz*uWz
kBuW = kBx*uWx + kBz*uWz
kWuA = kWx*uAx + kWz*uAz
kWuB = kWx*uBx + kWz*uBz

#Forcing terms
fUxA = dA*uAx + B*kBuW*uBx*1j - B*kWuB*uWx*1j
fUyA = dA*uAy + B*kBuW*uBy*1j 
fUzA = dA*uAz + B*kBuW*uBz*1j - B*kWuB*uWz*1j

fUxB = dB*uBx + A*kAuW*uAx*1j + A*kWuA*uWx*1j
fUyB = dB*uBy + A*kAuW*uAy*1j
fUzB = dB*uBz + A*kAuW*uAz*1j + A*kWuA*uWz*1j

fRhoA = dA*rhoA + B*kBuW*rhoB*1j + B*kWuB*rhoW*1j
fRhoB = dB*rhoB + A*kAuW*rhoA*1j + A*kWuA*rhoW*1j

#Solvability conditions

#Uz forcing term
GuzA = kAz*omegaA*(kAx*fUxA + kAy*fUyA)*1j - omegaA*(kAx**2 + kAy**2)*fUzA*1j
GuzB = kBz*omegaB*(kBx*fUxB + kBy*fUyB)*1j - omegaB*(kBx**2 + kBy**2)*fUzB*1j

#Rhoz forcing term
GrhozA = - (kAx**2 + kAy**2)*fRhoA
GrhozB = - (kBx**2 + kBy**2)*fRhoB

#Coefficients of EA and EB
gA = GuzA + GrhozA
gB = GuzB + GrhozB

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

#Parent wavevector components
KWx = np.cos(theta)
KWz = np.sin(theta)

#Parent wave frequency
OmegaW = KWx

#Fourier modes
nn = 8
n = -np.linspace(2,nn + 1,nn)

#Disturbance wavevector components
alpha = np.zeros(nn)
beta = 0

#Floquet modulation parameter
gamma = 0

#Parent wave eigenfunctions
UWx = functions.uxa(a,theta,0)
UWz = functions.uza(a,theta,0)
RhoW = functions.rhoa(a,theta)

###############################################################################
# Section Three: Determining alpha values satisfying exact resonance

#Expressing all quantities as functions of alpha
def freqres(alpha, n):
    kAx,kAy,kAz = (gamma + n)*KWx + alpha*KWz, beta, (gamma + n)*KWz - alpha*KWx
    kBx,kBy,kBz = (gamma + n + 1)*KWx + alpha*KWz, beta, (gamma + n + 1)*KWz - alpha*KWx
    omegaA = ((kAx**2 + kAy**2)/(kAx**2 + kAy**2 + kAz**2))**0.5
    omegaB = ((kBx**2 + kBy**2)/(kBx**2 + kBy**2 + kBz**2))**0.5
    return OmegaW - omegaA - omegaB

for i in range(nn):
        
    #Finding the alpha value satisfying exact resonance for each n value
    alpha[i] = sp.optimize.fsolve(freqres, x0 = 0.68, args = (n[i]))

#Matrix for depositing growth rates
growth = np.zeros(nn)

#Matrices for depositing frequencies
freq = np.zeros(nn)

###############################################################################
# Section Four: Predicting the growth rate at exact resonances
    
for i in range(nn):
        
    #Transformed frame
    KAxT = alpha[i]
    KAyT = beta
    KAzT = gamma + n[i]

    KBxT = alpha[i]
    KByT = beta
    KBzT = gamma + n[i] + 1

    #Original frame
    KAx = (KWx*KAzT + KWz*KAxT)
    KAy = KAyT
    KAz = (KWz*KAzT - KWx*KAxT)

    KBx = (KWx*KBzT + KWz*KBxT)
    KBy = KByT
    KBz = (KWz*KBzT - KWx*KBxT)

    OmegaA = -((KAx**2 + KAy**2)/(KAx**2 + KAy**2 + KAz**2))**0.5
    OmegaB = ((KBx**2 + KBy**2)/(KBx**2 + KBy**2 + KBz**2))**0.5

    freq[i] = abs(OmegaW - OmegaA - OmegaB)
        
    #Daughter wave velocities
    UAx,UAy,UAz = functions.uA(KAx,KAy,KAz)
    UBx,UBy,UBz = functions.uB(KBx,KBy,KBz)
    
    #Daughter wave densities
    RhoA = functions.rhoA(KAx,KAy,KAz)
    RhoB = functions.rhoB(KBx,KBy,KBz)
    
    #Amplitude equation coefficients
    C1 = c1.subs([(kAx,KAx), (kAy,KAy), (kAz,KAz),
                  (uAx,UAx), (uAy,UAy), (uAz,UAz), 
                  (rhoA,RhoA), 
                  (omegaA,OmegaA)])
    
    C3 = c3.subs([(kBx,KBx), (kBy,KBy), (kBz,KBz),
                  (uBx,UBx), (uBy,UBy), (uBz,UBz), 
                  (rhoB,RhoB), 
                  (omegaB,OmegaB)])
    
    C2 = c2.subs([(kAx,KAx), (kAy,KAy), (kAz,KAz), 
                  (kBx,KBx), (kBy,KBy), (kBz,KBz),
                  (kWx,KWx),            (kWz,KWz),
                  (uAx,UAx), (uAy,UAy), (uAz,UAz),
                  (uBx,UBx), (uBy,UBy), (uBz,UBz),
                  (uWx,UWx),            (uWz,UWz),
                  (rhoA,RhoA), 
                  (rhoB,RhoB), 
                  (rhoW,RhoW),
                  (omegaA,OmegaA), 
                  (omegaB,OmegaB)])
    
    C4 = c4.subs([(kAx,KAx), (kAy,KAy), (kAz,KAz), 
                  (kBx,KBx), (kBy,KBy), (kBz,KBz),
                  (kWx,KWx),            (kWz,KWz),
                  (uAx,UAx), (uAy,UAy), (uAz,UAz),
                  (uBx,UBx), (uBy,UBy), (uBz,UBz),
                  (uWx,UWx),            (uWz,UWz),
                  (rhoA,RhoA), 
                  (rhoB,RhoB), 
                  (rhoW,RhoW),
                  (omegaA,OmegaA), 
                  (omegaB,OmegaB)])
    
    #Growth rate
    S = ((C2*C4)/(C1*C3))**0.5
    growth[i] = 2*sy.re(S)
    
###############################################################################
# Section Four: Taking growth rate predictions at the resonant alpha values and 
# matching them to numerical growth rate peaks

#Numerical data    
data = np.load('growth_1d_theta_piby4.npz', allow_pickle = True)
 
#Growth rate as a function of alpha and beta   
Sigma = data['growth']

#Corresponding alpha values
nalpha = 100
Alpha = np.linspace(1E-10,4,nalpha)

#Setting up the figure
plt.figure(2)

#Growth rate as a function of alpha
plt.plot(Alpha,Sigma, 'black')
plt.xlabel(r'$\alpha$', fontsize = 16)
plt.ylabel(r'$\sigma$', fontsize = 16)
plt.xlim(0,4)

#Resonant alpha as a function of n plotted as vertical dashed lines
for i in range(nn):
    plt.vlines(alpha[i], min(Sigma), max(Sigma), color = 'darkorchid', linestyles = 'dashed')
    
#Predicted growth rate in wavenumber space
plt.scatter(alpha,growth, c = 'red', marker = 's')
