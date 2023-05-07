# %%

import math
import random
import numpy as np 
import matplotlib.pyplot as plt
import copy
import sys
from contextlib import redirect_stdout

# NPT Ensemble : Atom Displacement and Volume Rearrangement Move

# Function to Calculate long range corrections
def Long_range_corr(rcut, dens, N):
    sr3 = (1.0/rcut)**3
    sr9 = (sr3**3)
    vlrc12 = (8.0*math.pi*dens*N*sr9)/9.0
    vlrc6 = (8.0*math.pi*dens*N*sr3)/3.0
    
    wlrc12 = 4.0*vlrc12
    wlrc6 = 2.0*vlrc6
    
    vlrc = vlrc12 + vlrc6
    wlrc = wlrc12 + wlrc6
    
    return vlrc, wlrc, vlrc12, vlrc6, wlrc12, wlrc6

# Calculates PE of the system
def PE_config(rcut, rmin, box, rx, ry, rz, N):
    overlap = False
    rcutsq = rcut*rcut
    rminsq = rmin*rmin
    boxinv = 1.0/ box
    
    v12 = 0.0
    v6 = 0.0
    w12 = 0.0
    w6 = 0.0
    
    for i in range(1, N-1):
        rxi = rx[i]
        ryi = ry[i]
        rzi = rz[i]
        
        for j in range(i+1, N):
            rxij = rxi - rx[j]
            ryij = ryi - ry[j]
            rzij = rzi - rz[j]
            
            rxij = rxij - int(rxij*boxinv)*box
            ryij = ryij - int(ryij*boxinv)*box
            rzij = rzij - int(rzij*boxinv)*box
            
            rijsq = (rxij**2) + (ryij**2) + (rzij**2)
            
            if rijsq < rminsq :
                overlap = True
                return
            elif rijsq < rcutsq :
                sr2 = 1.0/rijsq
                sr6 = sr2**3
                vij12 = sr6**2
                v12 = v12 + vij12
                v6 = v6 +vij6
                w12 = w12 + vij12
                w6 = w6 + (vij6*0.5)
    
    v12 = 4.0 * v12
    v6 = 4.0 * v6
    w12 = (48.0 * w12)/ 3.0
    w6 = (48.0 *w6)/ 3.0
    
    return v12, v6, w12, w6, overlap

# Calculates PE of i th particle
def PE_part(rxi, ryi, rzi, i, rcut, box, N):
    
    rcutsq = rcut**2
    boxinv = 1.0/box
    
    v12 = 0.0
    v6 = 0.0
    w12 = 0.0
    w6 = 0.0
    
    for j in range(1, N):
        if i != j :
            rxij = rxi - rx[j]
            ryij = ryi - ry[j]
            rzij = rzi - rz[j]
            
            rxij = rxij - (int(rxij*boxinv))*box
            ryij = ryij - (int(ryij*boxinv))*box
            rzij = rzij - (int(rzij*boxinv))*box
            
            rijsq = (rxij**2) + (ryij**2) + (rzij**2)
            
            if rijsq < rcutsq :
                sr2 = 1.0/rijsq
                sr6 = sr2**3
                vij12 = sr6**2
                vij6 = (-1)*sr6
                v12 = v12 +vij12
                v6 = v6 + vij6
                w12 = w12 + vij12
                w6 = w6 +(vij6*0.5)
                
    v12 = 4.0*v12
    v6 = 4.0*v6
    w12 = (48.0*w12)/3.0
    w6 = (48.0*w6)/3.0
    
    return v12, v6, w12, w6 

# Arrange atoms in a Lattice
def Lattice(Npart, Length):
    #Npart = int(input("Enter Npart:"))
    #Box = int(input("Enter box length:"))
    
    rx = []
    ry = []
    rz = []
    K = []

    N = int((Npart)**(1/3))+1
    if N == 0:
        N = 1     
    Del = Length/float(N)
    Itel = 0
    Dx = -Del
    #print(Itel);
    for I in range(0, N, 1):
        Dx = Dx + Del
        Dy = -Del
        for J in range(0, N, 1):
            Dy = Dy + Del
            Dz = -Del
            for K in range(0, N, 1):
                Dz = Dz + Del
                if Itel < Npart:
                    Itel = Itel + 1
                    #print(Itel)
                    rx.append(Dx)
                    ry.append(Dy)
                    rz.append(Dz)  
    #K= X
    #print(K)  
    return rx, ry, rz

# %%
# ------------------------------------------------------------------------------------------------------------------------------
# Main MC code

nstep = int(input('Enter the total number of MC steps:  '))
rcut = float(input('Enter cutoff distance for LJ Potential:   '))
Press = float(input('Enter the desired pressure of the system:   '))
Temp = float(input('Enter the desired temperature of the system:   '))
N = int(input('Enter number of particles:   '))
dens = float(input('Enter desired Density of the system:   '))

vol = float(N/dens)
box = vol**(1/3)
boxinv = 1.0/box

if rcut > (0.5*box):
    rcut = float(input('Cutoff too large enter new rcut:  '))

dboxmx = box/40.0
drmax = 0.15
rmin = 0.7
beta = 1.0/Temp

# Interval for various operations
iprint = 50     # Print Interval
iratio = 100    # Ratio update interval for atoms
iratb = 50      # Ratio update interval for box

# Setting Accumulators to zero
acm = 0
acatma = 0
acboxa = 0

acv = 0
acp = 0
acd = 0

acvsq = 0
acpsq = 0
acdsq = 0

flv = 0
flp = 0
fld = 0

# Arrange all the atoms in a Lattice
rx, ry, rz = Lattice(N, box)

# Calculate Long Range Corrections
vlrc, wlrc, vlrc12, vlrc6, wlrc12, wlrc6 = Long_range_corr(rcut, dens, N)

# Calculate initial Energy and Virial Pressure
v12, v6, w12, w6, overlap = PE_config(rcut, rmin, box, rx, ry, rz, N)

if overlap == True:
    sys.exit('Overlap in the system, cannot proceed further')

vs = (v12 + v6 + vlrc)/N
ws = (w12 + w6 + wlrc)/N
ps = (dens* Temp) +(ws/vol)

v12 = v12 + vlrc12
v6 = v6 + vlrc6
w12 = w12 + wlrc12
w6 = w6 + wlrc6

print('----------------Initial Conditon of the system--------------------------')
print(' V/n  =  ', vs)
print(' W/n =   ', ws)
print(' P =   ', ps)

#----------------------------------------------------------------------------------------------------------------------------------------------------------
# Main MC Loop
for j in range(1, nstep):
    # Atom displacement move
    for i in range(1, N):
        rxiold = rx[i]
        ryiold = ry[i]
        rziold = rz[i]

        v12old, v6old, w12old, w6old = PE_part(rx[i], ry[i], rz[i], i, rcut, box, N)
        
        rxinew = rxiold + ((2*random.random() - 1.0)*drmax) 
        ryinew = ryiold + ((2*random.random() - 1.0)*drmax) 
        rzinew = rziold + ((2*random.random() - 1.0)*drmax) 

        v12new, v6new, w12new, w6new = PE_part(rxinew, ryinew, rzinew, i, rcut, box, N)
        
        # Checking for Acceptance
        delv12 = v12new - v12old
        delv6 = v6new - v6old
        delw12 = w12new - w12old
        delw6 = w6new - w6old
        deltv = delv12 + delv6
        deltvb = beta*deltv
        
        if deltvb < 75.0 :
            if deltv <= 0.0:
                v12 = v12 + delv12
                v6 = v6 + delv6
                w12 = w12 + delw12
                w6 = w6 + delw6
                
                rx[i] = rxinew
                ry[i] = ryinew
                rz[i] = rzinew
                
                acatma = acatma + 1.0
                
            elif (np.exp(-deltvb) > random.random()):
                
                v12 = v12 + delv12
                v6 = v6 + delv6
                w12 = w12 +delw12
                w6 = w6 + delw6
                
                rx[i] = rxinew
                ry[i] = ryinew
                rz[i] = rzinew
                acatma = acatma + 1.0
                
        vn = (v12 + v6)/N
        Press = (dens*Temp) + ((w12 + w6)/vol)
    
    # Increment Accumulators
    
        acm = acm + 1.0
        acv = acv + vn
        acp = acp + Press
        acd = acd + dens
    
        acvsq = acvsq + (vn**2)
        acpsq = acpsq + (Press**2)
        acdsq = acdsq + (dens**2)
    
# Volume Rearrangement Move

    boxnew = box + ((2*random.random() - 1)*dboxmx)
    ratbox = box/boxnew
    rrbox = 1.0/ratbox
    rcutn = rcut* rrbox
    
    # Calculating scaling parameters
    rat6 = ratbox**6
    rat12 = rat6*rat6
    
    v12new = v12 * rat12
    v6new = v6 * rat6
    w12new = w12 * rat12
    w6new = w6 * rat6
    
    deltv = v12new + v6new - v12 - v6 
    dpv = Press * (boxnew**3  - vol)
    dvol = 3 * Temp * N * np.log(ratbox)
    delthb = (beta)*(deltv + dpv + dvol)
    
    # Check for acceptance
    if delthb < 75.0 :
        if delthb <= 0:
            v12 = v12new
            v6 = v6new
            w12 = w12new
            w6 = w6new
            
            for i in range(1, N):
                
                rx[i] = rx[i]*rrbox
                ry[i] = ry[i]*rrbox
                rz[i] = rz[i]*rrbox
                
            box = boxnew
            rcut = rcutn
            acboxa = acboxa + 1.0
            
        elif np.exp(-delthb) > random.random() :
            
            v12 = v12new
            v6 = v6new
            w12 = w12new
            w6 = w6new
            
            for i in range(1, N):
                
                rx[i] = rx[i]*rrbox
                ry[i] = ry[i]*rrbox
                rz[i] = rz[i]*rrbox
            
            box = boxnew
            rcut = rcutn
            acboxa = acboxa + 1.0
    
    boxinv = 1.0 / box
    vol = box**3
    dens = N/vol
    
    vn = (v12 + v6)/N
    Press = (dens*Temp) + (w12 + w6)/vol
    
    acm = acm + 1.0
    acv = acv + vn
    acp = acp + Press
    acd = acd + dens
    
    acvsq = acvsq + (vn**2)
    acpsq = acpsq + (Press**2)
    acdsq = acdsq + (dens**2)
    
# Perform Periodic Operations
    if j == iratio :
    # Adjust atom displacement move
    
        ratio = acatma/(N*iratio)
        
        if ratio > 0.5 :
            drmax = drmax * 1.05
        
        else:
            drmax = drmax*0.95
        
        acatma = 0
        
    if j == iratb :
        
        bratio = acboxa/N
        if bratio > 0.5 :
            dboxmx = dboxmx * 1.05
        
        else:
            dboxmx = dboxmx * 0.95
        
        acboxa = 0
    
    if j == iprint :
        #print Info
        print('Iteration:  ', j, 'V/n:  ', vn, 'Pressure:  ', Press, 'Density:  ', dens, 'Box length:  ', box)
        
# Main Loop Ends
print('------------------End of MC Loop------------------')

# Calculate Final Averages

norm = acm
avv = acv/norm
avp = acp/norm
avd = acd/norm

acvsq = (acvsq/norm) - (avv**2)
acpsq = (acpsq/norm) - (avp**2)
acdsq = (acdsq/norm) - (avd**2)

if acvsq > 0: flv = np.sqrt(acvsq)
if acpsq > 0: flp = np.sqrt(acpsq)
if acdsq > 0: fld = np.sqrt(acdsq)

print('Average Energy:  ', avv)
print('Average Pressure:  ', avp)
print('Average Density:  ', avd)

print('Fluctuations in Energy:  ', flv)
print('Fluctuations in Pressure:  ', flp)
print('Fluctuations in density:  ', fld)

# %%   
    
        
    
    
            