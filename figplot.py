import os
import numpy as np
import time
import h5py
import matplotlib.pyplot as plt

from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.extras import plot_tools

import time
#from IPython import display

# Read in the data
f = h5py.File('./snapshots/snapshots_s1.h5','r')
a = h5py.File('./analysis_tasks/analysis.h5','r')
z = f['/scales/z/1.0'][:]
x = f['/scales/x/1.0'][:]
T = f['tasks']['T'][:]
u = f['tasks']['u'][:]
w = f['tasks']['w'][:]
#eq2a = a['tasks']['eq2a'][:]

print(T.shape)
print(u.shape)
print(w.shape)
#print(Diss.shape)

variables = f.items()

sh = u.shape
di = sh[0]
print(di)

for i in range(0, di):
    print(i)
    Tpl=T[i,:,:]
    fig = plt.figure(figsize=(6, 3.2))
    ax = fig.add_subplot(111)
    ax.set_title('Temperature')
    plt.pcolormesh(x,z,Tpl.transpose(), cmap= 'bwr', vmin = np.min(Tpl), vmax = np.max(Tpl),shading='gouraud')
    print("Temperature",np.min(Tpl),np.max(Tpl))
    ax.set_aspect('equal')
    nom1 = 'Temp'
    nom2 = str("%03d"%i)
    nom = nom1 + nom2 + '.png'
    plt.savefig(nom, dpi=600)
    plt.close()
    """
    upl=u[i,:,:]
    fig = plt.figure(figsize=(6, 3.2))
    ax =fig.add_subplot(111)
    ax.set_title('Vitesse u')
    plt.pcolormesh(x,z,upl.transpose(), cmap= 'rainbow', vmin = np.min(upl), vmax = np.max(upl),shading='gouraud')
    ax.set_aspect('equal')
    print("Vitesse",np.min(upl),np.max(upl))
    nom1 = 'Vitu'
    nom2 = str("%03d"%i)
    nom = nom1 + nom2 + '.png'
    plt.savefig(nom, dpi=600)
    plt.close()
    
    epl=eq2a[i,:,:]
    fig = plt.figure(figsize=(6, 3.2))
    ax =fig.add_subplot(111)
    ax.set_title('Eq2a')
    plt.pcolormesh(x,z,upl.transpose(), cmap= 'rainbow', vmin = np.min(epl), vmax = np.max(epl),shading='gouraud')
    ax.set_aspect('equal')
    print("Eq",np.min(epl),np.max(epl))
    nom1 = 'Eq'
    nom2 = str("%03d"%i)
    nom = nom1 + nom2 + '.png'
    plt.savefig(nom, dpi=600)
    plt.close()
    """
f.close()

