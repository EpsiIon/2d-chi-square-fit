#!/usr/bin/python

import sys
import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import *
from scipy.optimize import root

from mpl_toolkits.axes_grid.inset_locator import inset_axes #plot in a plot

import matplotlib.gridspec as gridspec  # for unequal plot boxes

#upload data: t1-time; s1=signal; d1-stdv
t1, s1, d1 = np.loadtxt('table.csv', skiprows=1, unpack=True)

#parameters description
k1  = 0.5
k2  = 2

#system of ODEs 
def syst(r, t=0):
  x,y = r
  x_t = k1*x
  y_t = k2*x

  return np.array([x_t, y_t])

#calculus of integration curves 
from scipy import integrate

t  = np.linspace(np.min(t1), np.max(t1), 20)            # time
I0 = np.array([1, 0])                  # initials conditions: 1 'x' and 0 'y'
X, infodict = integrate.odeint(syst, I0, t, full_output=True)
print infodict['message']              #integration report

#*******************searching for the best fit******************************
dk    = 0.1 #step of surfing 

k1min = 0   #range of surfing
k1max = 5

k2min = 0 
k2max = 5
#*******************Chi-squared criterion***********************************
#matrix where to keep chi^2 values
flw = np.zeros((int((k1max-k1min)/dk), int((k1max-k1min)/dk)))

#scanning loop
for i in np.arange(int((k1max-k1min)/dk)):
  for j in np.arange(int((k2max-k2min)/dk)):
    k1 = k1min+i*dk
    k2 = k2min+j*dk
    X, infodict = integrate.odeint(syst, I0, t1, full_output=True)
    x, y = X.T
    chi=0       #begin the counting of chi^values at each point
    for p in np.arange(np.size(t1)):
      chi = (y[p]-s1[p])**2/(d1[p]**2)+chi
    flw[i,j]= math.log(chi/(np.size(t1)-2)) #normalize  

#return the indexes of the minimum values of a matrix
def minimum(a): 
  minimum = np.min(a)
  index_of_minimum = np.where(a == minimum)
  return index_of_minimum
  
#******************pritn out parameters********************************

k1 = k1min + int(minimum(flw)[0])*dk
k2 = k2min + int(minimum(flw)[1])*dk
print 'Parameter analysis:'
print 'k1-value................ {0:0.2f}'.format(k1)     #best k1 fit
print 'k2-value................ {0:0.2f}'.format(k2)     #best k2 fit
print 'error...................',dk
#parametric flexibility: gradient componebts at the MIN!!!
ddk1 = (flw[int(minimum(flw)[0])+1,int(minimum(flw)[1])]-
	flw[int(minimum(flw)[0]),int(minimum(flw)[1])])/dk

ddk2 = (flw[int(minimum(flw)[0]),int(minimum(flw)[1])+1]-
	flw[int(minimum(flw)[0]),int(minimum(flw)[1])])/dk
norm = np.sqrt(ddk1**2+ddk2**2)

print 'k1-flexibility.......... {0:0.2f}'.format(ddk1/norm) 
print 'k2-flexibility.......... {0:0.2f}'.format(ddk2/norm) 

#calculata residuals
X, infodict = integrate.odeint(syst, I0, t1, full_output=True)
x, y = X.T

resids = y - s1

#smooth curve 
R, infodict = integrate.odeint(syst, I0, t, full_output=True)
x0, y0 = R.T

#lets plot!
fig = plt.figure(2, figsize=(8,8))
plt.rc('font', size=14)

gs = gridspec.GridSpec(2, 2,width_ratios=[1, 2],height_ratios=[3, 1])
gs1 = gridspec.GridSpec(2, 2,width_ratios=[1, 2],height_ratios=[1, 1])

# Top plot: data and fit
ax2 = fig.add_subplot(gs[1])
ax2.plot(t, y0)
ax2.errorbar(t1, s1, yerr=d1, fmt='or', ecolor='black')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('RFU [a.u.]')

# Bottom plot: residuals
ax4 = fig.add_subplot(gs[3])
ax4.errorbar(t1, resids, yerr = d1, ecolor="black", fmt="ro")
ax4.axhline(color="gray", zorder=-1)
ax4.set_xlabel('RFU [a.u.]')
ax4.set_ylabel('Residuals')

#*******************plotting gradient matrix *************************

dflw = np.zeros((int((k1max-k1min)/dk), int((k1max-k1min)/dk)))

for i in np.arange(int((k1max-k1min)/dk)-1):
  for j in np.arange(int((k2max-k2min)/dk)-1):
    #partial derivative at each point
    dflw[i,j] = (flw[i+1,j]-2*flw[i,j]+flw[i,j+1])/dk 

dflw1 = np.zeros((int((k1max-k1min)/dk)-2, int((k1max-k1min)/dk)-2))
#the edge cutout
for i in np.arange(int((k1max-k1min)/dk)-2):
  for j in np.arange(int((k2max-k2min)/dk)-2):
    dflw1[i,j]=dflw[i+1,j+1]

ax1 = fig.add_subplot(gs1[0])
plt.imshow(np.flipud(dflw1), cmap=plt.cm.Blues,
	   interpolation='none',extent=[k1min,k1max,k2min,k2max])
ax1 = plt.gca()
ax1.set_xlabel('k1 coordinate')
ax1.set_ylabel('k2 coordinate')
plt.title('Defferential space $\Delta k$ = 0.1' )
plt.colorbar()

ax1 = fig.add_subplot(gs1[2])
plt.imshow(np.flipud(flw), cmap=plt.cm.Blues,
	   interpolation='none',extent=[k1min,k1max,k2min,k2max])
ax2 = plt.gca()
ax2.set_xlabel('k1 coordinate')
ax2.set_ylabel('k2 coordinate')
plt.title('Fitting space $\Delta k$ = 0.1' )
plt.colorbar()


#ZOOM
dk    = 0.01 #step of surfing 

k1min = 0    #range of surfing
k1max = 1

k2min = 1.5 
k2max = 2.5


flw = np.zeros((int((k1max-k1min)/dk), int((k1max-k1min)/dk)))
#scan loop
for i in np.arange(int((k1max-k1min)/dk)):
  for j in np.arange(int((k2max-k2min)/dk)):
    k1 = k1min+i*dk
    k2 = k2min+j*dk
    X, infodict = integrate.odeint(syst, I0, t1, full_output=True)
    x, y = X.T
    chi=0       #begin the counting of chi^values at each point
    for p in np.arange(np.size(t1)):
      chi = (y[p]-s1[p])**2/(d1[p]**2)+chi
    flw[i,j]= math.log(chi/(np.size(t1)-2))
    
# this is an inset axes over the main axes
inset_axes = inset_axes(ax2, 
                    width="50%", # width = 30% of parent_bbox
                    height=2.0, # height : 1 inch
                    loc=1)
plt.imshow(np.flipud(flw), cmap=plt.cm.Blues,
	   interpolation='none',extent=[k1min,k1max,k2min,k2max])
plt.title('Zoomed minimum')


plt.show()  


