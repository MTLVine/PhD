import numpy as np

###############################################################################

# Section One: Fundamentals

#Horizontal wavevector component
def kx(theta):
    return np.cos(theta)

#Vertical wavevector component
def kz(theta):
    return np.sin(theta)

#IGW dispersion relation
def omegaG(theta):
    return kx(theta)

#Alfven wave dispersion relation
def omegaA(B0z):
    return B0z

#Alfven-gravity wave dispersion relation
def omega(theta,B0z):
    return (omegaG(theta)**2 + omegaA(B0z)**2)**0.5

###############################################################################

# Section Two: The Coordinate Transformation

#Spatial variables

def X(x,z,theta):
    return x*kz(theta) - z*kx(theta)

def Y(y):
    return y

def Z(x,z,t,theta,B0z):
    return kx(theta)*x + kz(theta)*z - omega(theta,B0z)*t

def T(t):
    return t

#Velocity transformation

def Ux(ux,uz,theta):
    return ux*kz(theta) - uz*kx(theta)

def Uy(uy):
    return uy

def Uz(ux,uz,theta,B0z):
    return ux*kx(theta) + uz*kz(theta) - omega(theta,B0z)

#Magnetic field transformation

def Bx(bx,bz,theta):
    return kz(theta)*bx - kx(theta)*bz

def By(by):
    return by

def Bz(bx,bz,theta):
    return kx(theta)*bx + kz(theta)*bz

#Density transformation

def Rho(rho,theta):
    return -rho*kz(theta)

#Reverse velocity transformation

def ux(Ux,Uz,theta,B0z):
    return Ux*kz(theta) + (Uz + omega(theta,B0z))*kx(theta)

def uy(Uy):
    return Uy

def uz(Ux,Uz,theta,B0z):
    return -Ux*kx(theta) + (Uz + omega(theta,B0z))*kz(theta)

#Reverse magnetic field transformation

def bx(Bx,Bz,theta):
    return Bz*kx(theta) + Bx*kz(theta)

def by(By):
    return By

def bz(Bx,Bz,theta):
    return Bz*kz(theta) - Bx*kz(theta)

#Reverse density transformation

def rho(Rho,theta):
    return -Rho/kz(theta)

###############################################################################

# Section Three: Primary Wave Quantities

#Original basic state amplitudes

def uxa(a,theta,B0z):
    return 0.5*a*omega(theta,B0z)/np.cos(theta)

def uza(a,theta,B0z):
    return -0.5*a*omega(theta,B0z)/np.sin(theta)

def bxa(a,theta,B0z):
    return -0.5*a*B0z/np.cos(theta)

def bza(a,theta,B0z): 
    return 0.5*a*B0z/np.sin(theta)

def rhoa(a,theta):
    return -0.5*a/np.sin(theta)

#Original basic state

def uxw(x,z,t,a,theta,B0z):
    phi = Z(x,z,t,theta,B0z)
    uxa = a*omega(theta,B0z)/kx(theta)
    return 2*uxa*np.cos(phi)

def uzw(x,z,t,a,theta,B0z):
    phi = Z(x,z,t,theta,B0z)
    uza = -a*omega(theta,B0z)/kz(theta)
    return 2*uza*np.cos(phi)

def bxw(x,z,t,a,theta,B0z):
    phi = Z(x,z,t,theta,B0z)
    bxa = (-a/kx(theta))*B0z
    return 2*bxa*np.cos(phi)

def bzw(x,z,t,a,theta,B0z):
    phi = Z(x,z,t,theta,B0z)
    bza = (a/kz(theta))*B0z
    return 2*bza*np.cos(phi)

def rhow(x,z,t,a,theta,B0z):
    phi = Z(x,z,t,theta,B0z)
    rhoa = a/kz(theta)
    return 2*rhoa*np.sin(phi)

#Transformed basic state amplitudes

def Ua(a,theta,B0z):
    return a*omega(theta,B0z)/(2*kx(theta)*kz(theta))

def Ba(a,theta,B0z):
    return -a*omegaA(B0z)/(2*kx(theta)*kz(theta))
    
def Rhoa(a):
    return -a*1j/2

#Transformed basic state

def Uxw(x,z,t,a,theta,B0z):
    phi = Z(x,z,t,theta,B0z)
    return 2*Ua(a,theta,B0z)*np.cos(phi)

def Uzw(theta,B0z):
    return -omega(theta,B0z)

def Bxw(x,z,t,a,theta,B0z):
    phi = Z(x,z,t,theta,B0z)
    return 2*Ba(a,theta,B0z)*np.cos(phi)

def Rhow(x,z,t,a,theta,B0z):
    phi = Z(x,z,t,theta,B0z)
    return 2*a*np.sin(phi)

#Transformed basic state derivatives

def dUxw(phi,a,theta,B0z):
    return -2*Ua(a,theta,B0z)*np.sin(phi)

def dBxw(phi,a,theta,B0z):
    return -2*Ba(a,theta,B0z)*np.sin(phi)

def dRhow(phi,a,theta,B0z):
    return 2*Rhoa(a)*np.cos(phi)

###############################################################################

# Section Four: Verifying the Implementation

#Transformed disturbance wavenumbers
def Kxprime(theta,alpha,gamma,n):
    return (gamma + n)*kx(theta) + alpha*kz(theta)

def Kyprime(beta):
    return beta

def Kzprime(theta,alpha,gamma,n):
    return (gamma + n)*kz(theta) - alpha*kx(theta)

#Frequency of perturbations

def freq(theta,zeta,B0,alpha,beta,gamma,n):
    kx = Kxprime(theta,alpha,gamma,n)
    ky = Kyprime(beta)
    kz = Kzprime(theta,alpha,gamma,n)
    b0x = B0*np.cos(zeta)
    b0z = B0*np.sin(zeta)
    B0z = Bz(b0x,b0z,theta)
    OmegaG = (kx**2 + ky**2)/(kx**2 + ky**2 + kz**2)
    OmegaA = (kx*b0x + kz*b0z)**2
    
    #Alfven-gravity wave doppler shift
    dopshift = omega(theta,B0z)*(gamma + n)
    
    #Pair of Alfven-gravity wave frequencies
    freq1 = (OmegaG + OmegaA)**0.5 + dopshift
    freq2 = -(OmegaG + OmegaA)**0.5 + dopshift
    
    #Pair of Alfven wave frequencies
    freq3 = OmegaA**0.5 + dopshift
    freq4 = -OmegaA**0.5 + dopshift
    
    #Zero frequency solution (still needs to be doppler shifted)
    freq5 = dopshift
    
    #Need to apply the same doppler shift to all 4 frequencies 
    #(since we only consider a single primary wave)
    
    return freq1,freq2,freq3,freq4,freq5

#Adrian's mathematica frequencies (only works for gamma = n = 0)

def Freq(theta,zeta,B0,alpha,beta,gamma):
    kx = Kxprime(theta,alpha,gamma,0)
    ky = Kyprime(beta)
    kz = Kzprime(theta,alpha,gamma,0)
    b0x = B0*np.cos(zeta)
    b0z = B0*np.sin(zeta)
    kb0 = kx*b0x + kz*b0z
    kperp2 = kx**2 + ky**2
    k2 = kx**2 + ky**2 + kz**2 
    
    #Alfven wave frequencies
    freq1 = kb0
    freq2 = -kb0
    
    #Alfven-gravity wave frequencies
    freq3 = (kb0**2 + kperp2/k2)**0.5
    freq4 = -(kb0**2 + kperp2/k2)**0.5
    
    return freq1,freq2,freq3,freq4 

###############################################################################

# Section Five: Asymptotics

#Daughter wavenumbers satisfying resonance
def kB(kAx,kAy,kAz,kx,kz):
    return kAx + kx, kAy, kAz + kz

#Daughter wave frequencies

def omegaAhd(kAx,kAy,kAz):
    return -((kAx**2 + kAy**2)/(kAx**2 + kAy**2 + kAz**2))**0.5

def omegaBhd(kBx,kBy,kBz):
    return ((kBx**2 + kBy**2)/(kBx**2 + kBy**2 + kBz**2))**0.5

def omegaAagw(kAx,kAy,kAz,b0x,b0z):
    return -((kAx**2 + kAy**2)/(kAx**2 + kAy**2 + kAz**2) + (kAx*b0x + kAz*b0z)**2)**0.5

def omegaBagw(kBx,kBy,kBz,b0x,b0z):
    return ((kBx**2 + kBy**2)/(kBx**2 + kBy**2 + kBz**2) + (kBx*b0x + kBz*b0z)**2)**0.5

def omegaAaw(kAx,kAz,b0x,b0z):
    return - kAx*b0x - kAz*b0z

def omegaBaw(kBx,kBz,b0x,b0z):
    return kBx*b0x + kBz*b0z

#Daughter wave velocities

def uA(kAx,kAy,kAz):
    return -kAx*kAz, -kAy*kAz, kAx**2 + kAy**2

def uB(kBx,kBy,kBz):
    return -kBx*kBz, -kBy*kBz, kBx**2 + kBy**2

#Daughter wave magnetic field components

def bA(kAx,kAy,kAz,b0x,b0z):
    kAb0 = kAx*b0x + kAz*b0z
    omegaA = omegaAagw(kAx,kAy,kAz,b0x,b0z)
    bAx = kAx*kAz*kAb0/omegaA
    bAy = kAy*kAz*kAb0/omegaA
    bAz = -(kAx**2 + kAy**2)*kAb0/omegaA
    return bAx, bAy, bAz

def bB(kBx,kBy,kBz,b0x,b0z):
    kBb0 = kBx*b0x + kBz*b0z
    omegaB = omegaBagw(kBx,kBy,kBz,b0x,b0z)
    bBx = kBx*kBz*kBb0/omegaB
    bBy = kBy*kBz*kBb0/omegaB
    bBz = -(kBx**2 + kBy**2)*kBb0/omegaB
    return bBx, bBy, bBz

#Daughter wave densities

def rhoA(kAx,kAy,kAz):
    return (kAx**2 + kAy**2 + kAz**2)*omegaAhd(kAx,kAy,kAz)*1j

def rhoB(kBx,kBy,kBz):
    return (kBx**2 + kBy**2 + kBz**2)*omegaBhd(kBx,kBy,kBz)*1j

def rhoAagw(kAx,kAy,kAz,b0x,b0z):
    return (kAx**2 + kAy**2)*1j/omegaAagw(kAx,kAy,kAz,b0x,b0z)

def rhoBagw(kBx,kBy,kBz,b0x,b0z):
    return (kBx**2 + kBy**2)*1j/omegaBagw(kBx,kBy,kBz,b0x,b0z)

#Coefficients required for forming an expression for s from amplitude equations

#Amplitude equations: c1dA/dtau + c2B = 0 and c3dB/dtau + c4A = 0

#HD coefficients

def c1(kAx,kAy,kAz):
    
    OmegaA = omegaAhd(kAx,kAy,kAz)
    
    return -2*OmegaA*(kAx**2 + kAy**2)*(kAx**2 + kAy**2 + kAz**2)*1j

def c2(a,theta,kAx,kAy,kAz,B0z):
    
    kx = np.cos(theta)
    kz = np.sin(theta)
    Uax = uxa(a,theta,B0z)
    Uaz = uza(a,theta,B0z)
    Rhoa = rhoa(a,theta)
    kBx,kBy,kBz = kB(kAx,kAy,kAz,kx,kz)
    uBx,uBy,uBz = uB(kBx,kBy,kBz)
    RhoB = rhoB(kBx,kBy,kBz) 
    OmegaA = omegaAhd(kAx,kAy,kAz)

    t1 = OmegaA*(uBx*kx + uBz*kz)*(kAz*(kBx*Uax + kBz*Uaz) - (kAx**2 + kAy**2 + kAz**2)*Uaz)
    t2 = OmegaA*(Uax*kBx + Uaz*kBz)*(kAz*(kx*uBx + kz*uBz) + (kAx**2 + kAy**2 + kAz**2)*uBz)
    t3 = (uBx*kx + uBz*kz)*(kAx**2 + kAy**2)*Rhoa*1j
    t4 = (Uax*kBx + Uaz*kBz)*(kAx**2 + kAy**2)*RhoB*1j
    
    return t1 + t2 - t3 - t4

def c3(a,theta,kAx,kAy,kAz):
    
    kx = np.cos(theta)
    kz = np.sin(theta)
    kBx,kBy,kBz = kB(kAx,kAy,kAz,kx,kz)
    OmegaB = omegaBhd(kBx,kBy,kBz)
    
    return -2*OmegaB*(kBx**2 + kBy**2)*(kBx**2 + kBy**2 + kBz**2)*1j

def c4(a,theta,kAx,kAy,kAz,B0z):
    
    kx = np.cos(theta)
    kz = np.sin(theta)
    Uax = uxa(a,theta,B0z)
    Uaz = uza(a,theta,B0z)
    Rhoa = rhoa(a,theta)
    kBx,kBy,kBz = kB(kAx,kAy,kAz,kx,kz)
    uAx,uAy,uAz = uA(kAx,kAy,kAz)
    RhoA = rhoA(kAx,kAy,kAz) 
    OmegaB = omegaBhd(kBx,kBy,kBz)

    t1 = OmegaB*(Uax*kAx + Uaz*kAz)*(kBz*(kx*uAx + kz*uAz) - (kBx**2 + kBy**2 + kBz**2)*uAz)
    t2 = OmegaB*(uAx*kx + uAz*kz)*(kBz*(kAx*Uax + kAz*Uaz) - (kBx**2 + kBy**2 + kBz**2)*Uaz)
    t3 = (Uax*kAx + Uaz*kAz)*(kBx**2 + kBy**2)*RhoA*1j
    t4 = (uAx*kx + uAz*kz)*(kBx**2 + kBy**2)*Rhoa*1j
    
    return - t1 - t2 - t3 - t4

def s(a,theta,kAx,kAy,kAz,B0z):
    
    C1 = c1(kAx,kAy,kAz)
    C2 = c2(a,theta,kAx,kAy,kAz,B0z)
    C3 = c3(a,theta,kAx,kAy,kAz)
    C4 = c4(a,theta,kAx,kAy,kAz,B0z)
    
    return ((C2*C4)/(C1*C3))**0.5
    
#MHD Coefficients

#Gz method (only works for AGW-AGW interactions)

def c1agw(kAx,kAy,kAz,b0x,b0z):
    omegaA = omegaAagw(kAx,kAy,kAz,b0x,b0z)
    return -2*omegaA*(kAx**2 + kAy**2)*(kAx**2 + kAy**2 + kAz**2)*1j

def c2agw(a,theta,kAx,kAy,kAz,B0x,B0z,b0x,b0z):
    
    #Parent wave
    kx = np.cos(theta)
    kz = np.sin(theta)
    Uax = uxa(a,theta,B0z)
    Uaz = uza(a,theta,B0z)
    Bax = bxa(a,theta,b0x,b0z)
    Baz = bza(a,theta,b0x,b0z)
    Rhoa = rhoa(a,theta)
    
    #Daughter waves
    kBx,kBy,kBz = kB(kAx,kAy,kAz,kx,kz)
    uAx,uAy,uAz = uA(kAx,kAy,kAz)
    uBx,uBy,uBz = uB(kBx,kBy,kBz)
    bAx,bAy,bAz = bA(kAx,kAy,kAz,b0x,b0z)
    bBx,bBy,bBz = bB(kBx,kBy,kBz,b0x,b0z)
    RhoA = rhoAagw(kAx,kAy,kAz,b0x,b0z)
    RhoB = rhoBagw(kBx,kBy,kBz,b0x,b0z) 
    OmegaA = omegaAagw(kAx,kAy,kAz,b0x,b0z)
    OmegaB = omegaBagw(kBx,kBy,kBz,b0x,b0z)
    
    #Scalar products
    kBuW = kBx*Uax + kBz*Uaz
    kBbW = kBx*Bax + kBz*Baz
    kWuB = kx*uBx + kz*uBz
    kWbB = kx*bBx + kz*bBz
    uAuB = uAx*uBx + uAy*uBy + uAz*uBz
    uAbB = uAx*bBx + uAy*bBy + uAz*bBz
    uAuW = uAx*Uax + uAz*Uaz
    uAbW = uAx*Bax + uAz*Baz
    kAb0 = kAx*b0x + kAz*b0z

    t1 = kBuW*(OmegaA*uAuB - kAb0*(kAx**2 + kAy**2 + kAz**2)*bBz - (kAx**2 + kAy**2)*RhoB*1j)
    t2 = kWuB*(OmegaA*uAuW - kAb0*(kAx**2 + kAy**2 + kAz**2)*Baz + (kAx**2 + kAy**2)*Rhoa*1j)
    t3 = kBbW*(OmegaA*uAbB - kAb0*(kAx**2 + kAy**2 + kAz**2)*uBz)
    t4 = kWbB*(OmegaA*uAbW - kAb0*(kAx**2 + kAy**2 + kAz**2)*Uaz)
    
    return t1 - t2 - t3 + t4

def c3agw(a,theta,kAx,kAy,kAz,b0x,b0z):
    
    #Parent wave
    kx = np.cos(theta)
    kz = np.sin(theta)
    
    #Daughter waves
    kBx,kBy,kBz = kB(kAx,kAy,kAz,kx,kz)
    OmegaB = omegaBagw(kBx,kBy,kBz,b0x,b0z)
    
    return -2*OmegaB*(kBx**2 + kBy**2)*(kBx**2 + kBy**2 + kBz**2)*1j

def c4agw(a,theta,kAx,kAy,kAz,B0x,B0z,b0x,b0z):
    
    #Parent wave
    kx = np.cos(theta)
    kz = np.sin(theta)
    Uax = uxa(a,theta,B0z)
    Uaz = uza(a,theta,B0z)
    Bax = bxa(a,theta,b0x,b0z)
    Baz = bza(a,theta,b0x,b0z)
    Rhoa = rhoa(a,theta)
    
    #Daughter waves
    kBx,kBy,kBz = kB(kAx,kAy,kAz,kx,kz)
    uAx,uAy,uAz = uA(kAx,kAy,kAz)
    uBx,uBy,uBz = uB(kBx,kBy,kBz)
    bAx,bAy,bAz = bA(kAx,kAy,kAz,b0x,b0z)
    bBx,bBy,bBz = bB(kBx,kBy,kBz,b0x,b0z)
    RhoA = rhoAagw(kAx,kAy,kAz,b0x,b0z)
    RhoB = rhoBagw(kBx,kBy,kBz,b0x,b0z) 
    OmegaA = omegaAagw(kAx,kAy,kAz,b0x,b0z)
    OmegaB = omegaBagw(kBx,kBy,kBz,b0x,b0z)
    
    #Scalar products
    kAuW = kAx*Uax + kAz*Uaz
    kAbW = kAx*Bax + kAz*Baz
    kWuA = kx*uAx + kz*uAz
    kWbA = kx*bAx + kz*bAz
    uBuA = uBx*uAx + uBy*uAy + uBz*uAz
    uBbA = uBx*bAx + uBy*bAy + uBz*bAz
    uBuW = uBx*Uax + uBz*Uaz
    uBbW = uBx*Bax + uBz*Baz
    kBb0 = kBx*b0x + kBz*b0z
    
    t1 = kAuW*(OmegaB*uBuA - kBb0*(kBx**2 + kBy**2 + kBz**2)*bAz - (kBx**2 + kBy**2)*RhoA*1j)
    t2 = kWuA*(OmegaB*uBuW - kBb0*(kBx**2 + kBy**2 + kBz**2)*Baz - (kBx**2 + kBy**2)*Rhoa*1j)
    t3 = kAbW*(OmegaB*uBbA - kBb0*(kBx**2 + kBy**2 + kBz**2)*uAz)
    t4 = kWbA*(OmegaB*uBbW - kBb0*(kBx**2 + kBy**2 + kBz**2)*Uaz)
    
    return t1 + t2 - t3 - t4

def sagw(a,theta,kAx,kAy,kAz,B0x,B0z,b0x,b0z):
    
    C1 = c1agw(kAx,kAy,kAz,b0x,b0z)
    C2 = c2agw(a,theta,kAx,kAy,kAz,B0x,B0z,b0x,b0z)
    C3 = c3agw(a,theta,kAx,kAy,kAz,b0x,b0z)
    C4 = c4agw(a,theta,kAx,kAy,kAz,B0x,B0z,b0x,b0z)
    
    return ((C2*C4)/(C1*C3))**0.5

#Vertical Vorticity Method (only works for AW-AW interactions)

def c1aw(kAx,kAy,kAz,b0x,b0z):
       
    OmegaA = omegaAaw(kAx,kAy,b0x,b0z)
    
    return -2*OmegaA*(kAx**2 + kAy**2)

def c2aw(a,theta,kAx,kAy,kAz,B0x,B0z,b0x,b0z):
    
    #Parent wave
    kx = np.cos(theta)
    kz = np.sin(theta)
    Uax = uxa(a,theta,B0z)
    Bax = bxa(a,theta,b0x,b0z)
    
    #Daughter waves
    kBx,kBy,kBz = kB(kAx,kAy,kAz,kx,kz)
    OmegaA = omegaAaw(kAx,kAy,b0x,b0z)
    
    #Scalar products
    kBuW = kBx*Uax
    kBbW = kBx*Bax
    kAkB = kAx*kBx + kAy*kBy
    
    return -OmegaA*(kBuW + kBbW)*kAkB*1j

def c3aw(theta,kAx,kAy,kAz,b0x,b0z):
    
    #Parent wave
    kx = np.cos(theta)
    kz = np.sin(theta)
    
    #Daughter waves
    kBx,kBy,kBz = kB(kAx,kAy,kAz,kx,kz)
    OmegaB = omegaBaw(kBx,kBy,b0x,b0z)
    
    return -2*OmegaB*(kBx**2 + kBy**2)

def c4aw(a,theta,kAx,kAy,kAz,B0x,B0z,b0x,b0z):
    
    #Parent wave
    kx = np.cos(theta)
    kz = np.sin(theta)
    Uax = uxa(a,theta,B0z)
    Bax = bxa(a,theta,b0x,b0z)
    
    #Daughter waves
    kBx,kBy,kBz = kB(kAx,kAy,kAz,kx,kz)
    OmegaB = omegaBaw(kBx,kBy,b0x,b0z)
    
    #Scalar products
    kAuW = kAx*Uax
    kAbW = kAx*Bax
    kAkB = kAx*kBx + kAy*kBy
    
    return -OmegaB*(kAuW + kAbW)*kAkB*1j

def saw(a,theta,kAx,kAy,kAz,B0x,B0z,b0x,b0z):
    
    C1 = c1aw(kAx,kAy,kAz,b0x,b0z)
    C2 = c2aw(a,theta,kAx,kAy,kAz,B0x,B0z,b0x,b0z)
    C3 = c3aw(theta,kAx,kAy,kAz,b0x,b0z)
    C4 = c4aw(a,theta,kAx,kAy,kAz,B0x,B0z,b0x,b0z)
    
    return ((C2*C4)/(C1*C3))**0.5
    

    



    
    


    


    