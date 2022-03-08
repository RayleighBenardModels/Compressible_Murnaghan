"""
Dedalus script for 2D Rayleigh-Benard compressible convection.

This script uses a Fourier basis in the x direction with periodic boundary
conditions and a Chebishev basis in the z direction. The complete theory and
results are detailed in 
Ricard et al., Fully compressible convection for planetary mantles,
 Geophys. J. Int., in press.

This script can be ran serially or in parallel, and uses the built-in analysis
framework to save data snapshots in HDF5 files. After the run, merging of the
output files is done using postprocess.py and figures can be done using 
figplot.py:
To run, merge, and plot using 4 processes, for instance, you could use:
    $ mpiexec -n 4 python3 FC_Murnaghan.py
    $ mpiexec -n 4 python3 postprocess.py
    $ python3 figplot.py

If the file restart.h5 exists, the run continues from that.

Input parameters are specified in the par.toml file.

"""

import os
import numpy as np
from scipy.integrate import odeint
from mpi4py import MPI
import time
import sys
import h5py
import math
import pathlib
from numpy.polynomial import Chebyshev as cheby

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)
import toml

# par file contains parameters to be changed by user
par = toml.load("par.toml")

# Parameters
Lx, Lz = (par['Lx'], 1.)
Rayleigh = par['Rayleigh']
nn = par['nn']
epsilon = par['epsilon']
Di = par['Di']
r = par['r']
gamma0 = par['gamma0']
beta = par ['beta']
BCtop = par['BCtop']
BCbot = par['BCbot']
NFou = par['NFourier']
NCheb = par['NCheb']
cadprop = par['cadprop']
cadprint = max([par['cadprint'], cadprop])
iterdiag = par['iterdiag']

# Now compute parameters for reference isentropic profiles
rhoa0 = 1 ; r0 = 0.
Ta0 = 1 ; T1 = 0.

T12 = (1+r)/2+beta
tol = 1e-12
while np.abs(r0-rhoa0)>tol and np.abs(T1-Ta0) > tol :
    T1 = Ta0
    r1=rhoa0
    def model(rho,zd):
        cpcv = 1+epsilon*gamma0*T1/rho**(nn+1)*math.exp(gamma0*(1/r1-1/rho))
        dydt = Di/gamma0/rho**(nn-2)/cpcv
        return dydt
    zd = np.linspace(0,0.5)
    rho = odeint(model,r1,zd)
    Tc=T1*math.exp(gamma0 * (1 /r1 - 1 / rho[-1]))
    Ta0=Ta0+(T12-Tc)/10
    rhoa0=(1 - nn * epsilon * (Ta0 - 1)) ** (1/nn)

logger.info('Adiabatic Top temperature = {}'.format(Ta0))
logger.info('Adiabatic Top density = {}'.format(rhoa0))
logger.info('Adiabatic Top pressure = {}'.format(Rayleigh * gamma0 / Di
                *((rhoa0**nn - 1) / nn + epsilon * (Ta0 - 1))))

# Create bases and domain
x_basis = de.Fourier('x', NFou, interval=(0, Lx), dealias=3/2)
z_basis = de.Chebyshev('z', NCheb, interval=(0, 1), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

# Compute reference ("adiabatic") density profile
# First compute it by numerical integration
zd = np.linspace(0, 1, 200)
zdm1= list(reversed(zd))
rho = odeint(model, rhoa0, zd)
# Expand it on Chebychev polynomials
rho_cheb = cheby.fit(zdm1, rho[:, 0], NCheb)
x, z = domain.all_grids()
# Define a new field that depends only on z
rhoa = domain.new_field()
rhoa.meta['x']['constant'] = True #Tells dedalus this field doesn't vary in x and can go on LHS of eqns
rhoa['g'] = rho_cheb(z)

# Compute values at the top
rhoa1=rho_cheb(0)
Ta1 = Ta0* math.exp(gamma0 * (1 / rhoa0 - 1 / rhoa1))

logger.info('Adiabatic Bottom temperature = {}'.format(Ta1))
logger.info('Adiabatic Bottom density = {}'.format(rhoa1))

# 2D Boussinesq hydrodynamics
problem = de.IVP(domain, variables=['rho', 'T', 'u', 'w', 'Tz', 'uz', 'wz'])
problem.parameters['R'] = Rayleigh
problem.parameters['epsilon'] = epsilon
problem.parameters['nn'] = nn
problem.parameters['Tb'] = r
problem.parameters['Ta1'] = Ta1
problem.parameters['Ta0'] = Ta0
# problem.parameters['rhoa1'] = rhoa1
problem.parameters['Di'] = Di
problem.parameters['rhoa0'] = rhoa0
problem.parameters['gamma0'] = gamma0

problem.parameters['rhoa'] = rhoa

problem.substitutions['Ta']   = "Ta0 * exp(gamma0 * (1 /rhoa0 - 1 / rhoa))"
problem.substitutions['C']   = "1 + gamma0 * epsilon * Ta / rhoa**(nn+1)"
problem.substitutions['drsr'] = "-Di/gamma0/rhoa**(nn-1)/C"
problem.substitutions['ddrsr'] = "dz(drsr)"
problem.substitutions['Lapa'] = "dz(dz(Ta))"
problem.substitutions['Pa']   = "R*gamma0/Di*((rhoa**nn-1)/nn+epsilon*(Ta-1))"
problem.substitutions['div']  = "dx(u)+wz"
problem.substitutions['diss'] = "2*(dx(u))**2+(dx(w)+uz)**2+2*(wz)**2-2/3*(dx(u)+wz)**2"
problem.substitutions['AA']   = "div+w*drsr"
problem.substitutions['LapT'] = "(dx(dx(T)) + dz(Tz))"
problem.substitutions['enthalp'] = "(T+Ta)*(1+gamma0/(rho+rhoa))+ (rho+rhoa)**(nn-1)/(nn-1)/epsilon"
problem.substitutions['P'] = "R*gamma0/Di*(((rho+rhoa)**nn-rhoa**nn)/nn+epsilon*T)"
problem.substitutions['entrop'] = "log(1+T/Ta)-gamma0*rho/(rho+rhoa)/rhoa"
#
# mass conservation
problem.add_equation("dt(rho) + rhoa * AA = -rho*div-u*dx(rho)-w*dz(rho)")

# temperature equation
problem.add_equation("dt(T) - LapT/ rhoa=-u*dx(T)-w*Tz+gamma0/rhoa**2*drsr*w*(rhoa*T-Ta*rho)\
+1/(rhoa+rho)*(Di/R/epsilon*diss+Lapa)\
-gamma0*(Ta+T)/(rhoa+rho)*AA\
+rho/rhoa/(rhoa+rho)*(-(rhoa*T-Ta*rho)*gamma0*drsr/rhoa*w-LapT)")

# x momentum equation
problem.add_equation("dx(dx(u))+dz(uz)+1/3*dx(div)-gamma0*R/Di*(rhoa**(nn-1)*dx(rho)+epsilon*dx(T))=gamma0*R/Di*((rhoa+rho)**(nn-1)-rhoa**(nn-1))*dx(rho)")

# y momentum equation
problem.add_equation("dx(dx(w)) + dz(wz)- 1/3*drsr*wz-1/3*w*ddrsr-gamma0*R/Di*(rhoa**(nn-1)*dz(rho)+epsilon*Tz)-R*rho*(1+(nn-1)*drsr*gamma0/Di*rhoa**(nn-1))=-1/3*dz(AA)+gamma0*R/Di*((rhoa+rho)**(nn-1)-rhoa**(nn-1))*dz(rho)+R*gamma0/Di*((rhoa+rho)**(nn-1)-rhoa**(nn-1)-(nn-1)*rhoa**(nn-2)*rho)*drsr*rhoa")

# vertical derivatives
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("wz - dz(w) = 0")
problem.add_equation("Tz - dz(T) = 0")

problem.add_bc("left(uz) = 0.")
problem.add_bc("right(uz) = 0.", condition="(nx != 0)")
problem.add_bc("right(u) = 0.", condition="(nx == 0)")
problem.add_bc("left(w) = 0.")
problem.add_bc("right(w) = 0.0")
problem.add_bc("left(T) = Tb-Ta1")
problem.add_bc("right(T) = 1.-Ta0")

# Build solver
solver = problem.build_solver(de.timesteppers.RK443)
logger.info('Solver built')

# Initial conditions or restart
if not pathlib.Path('restart.h5').exists():

    # Initial conditions
    x, z = domain.all_grids()
    T = solver.state['T']
    Tz = solver.state['Tz']

    # Random perturbations, initialized globally for same results in parallel
    gshape = domain.dist.grid_layout.global_shape(scales=1)
    slices = domain.dist.grid_layout.slices(scales=1)
    rand = np.random.RandomState(seed=44)
    noise = rand.standard_normal(gshape)[slices]

    # Linear background + perturbations damped at walls
    zb, zt = z_basis.interval
    pert =  1e-2 * noise * (zt - z) * (z - zb)
    T['g'] = pert
    T.differentiate('z', out=Tz)

    # Timestepping and output
    dt = 1e-5
    stop_sim_time = par['max_time'] 
    fh_mode = 'overwrite'

else:
    # Restart
    write, last_dt = solver.load_state('restart.h5', -1)

    # Timestepping and output
    write, last_dt = solver.load_state('restart.h5', -1)
    with h5py.File('restart.h5') as h5f:
        last_time = h5f['scales']['sim_time'][-1]
    stop_sim_time = last_time + par['max_time']
    dt = last_dt
    fh_mode = 'append'

# Integration parameters
solver.stop_sim_time = stop_sim_time

sim_dt1=par['max_time']/30
sim_dt2=par['max_time']/300

solver.evaluator.vars['Lx'] = Lx
solver.evaluator.vars['Lz'] = Lz

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=sim_dt1, max_writes=500000, mode=fh_mode)
snapshots.add_system(solver.state)
analysis = solver.evaluator.add_file_handler('analysis_tasks', sim_dt=sim_dt2, max_writes=500000,mode=fh_mode)

analysis.add_task("R", name='R')
analysis.add_task("Ta", name='Ta_profile')
analysis.add_task("rhoa", name='rhoa_profile')
analysis.add_task("integ(-Tz,'x')/Lx", name='Fluxcond')
analysis.add_task("-gamma0*Ta/rhoa*drsr", name='Fluxadia')
analysis.add_task("integ((rho+rhoa)*w*(T+Ta),'x')/Lx", name='FluxconvT')
analysis.add_task("integ((rho+rhoa)*w*enthalp,'x')/Lx", name='Fluxconv')
analysis.add_task("integ((rho+rhoa)*w*(T+Ta)*entrop,'x')/Lx",name='Fluxentrop')
analysis.add_task("integ(-Di/R/epsilon*(u*(uz+dx(w))+w*(4/3*wz-2/3*dx(u))),'x')/Lx",name='Fluxw')
analysis.add_task("integ(T,'x')/Lx", name='T')
analysis.add_task("integ(rho,'x')/Lx", name='Rho')

#analysis.add_task("dz(integ(w*b,'x')/Lx)",name='T0')
#analysis.add_task("dz(integ(bz,'x')/Lx)", name='T3')
analysis.add_task("integ(diss,'x')/Lx", name='Diss_profile')

# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=0.5,
                     max_change=1.005, min_change=0.2)
CFL.add_velocities(('u', 'w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("sqrt(u*u + w*w) / R", name='Re')

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.proceed:
        dt = CFL.compute_dt()
        dt = solver.step(dt)
        if (solver.iteration-1) % 200 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max Re = %f' %flow.max('Re'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))

