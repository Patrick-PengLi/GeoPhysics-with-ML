# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 14:37:26 2022

@author: pli3
"""

from Inversion import *
from RockPhysics import *
import numpy as np
import pandas as pd  # DataFrames
import matplotlib.pyplot as plt  # plotting
from Geostats import *
import scipy.io as sio

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device('cude:0'if torch.cuda.is_available() else 'cpu')

# Rock physics parameters

# solid phase
Kmat = 30
Gmat = 60
Rho_sol = 2.6
Vp_sol = 6

# fluid phase
Kw = 2.5
Ko = 0.7
Rho_w = 1.03
Rho_o = 0.7
Rho_fl = 1
Vp_fl = 1.5
Sw = 0.8

# granular media theory parameters
critporo = 0.4
coordnum = 9
press = 0.04

# Seismic parameter
freq = 45
dt = 0.001
ntw = 64
theta = np.array([0])


# Loading 1D dataset
# Depth, Time, TimeSeis, Phi, Sw, Vp,Vs,Rho, Sfar, Smid, Snear

# dataset_1d = sio.loadmat('1DdataWell.mat')   

# Depth_1d = dataset_1d['Depth']
# Time_1d = dataset_1d['Time']

# Phi_1d = dataset_1d['Phi']
# Sw_1d = dataset_1d['Sw']

# Vp_1d = dataset_1d['Vp']
# Vs_1d = dataset_1d['Vs']
# Rho_1d = dataset_1d['Rho']

# dm=np.hstack((Depth_1d, Time_1d,Phi_1d,Sw_1d,Vp_1d,Vs_1d,Rho_1d))
# df_1d = pd.DataFrame(dm,columns = ['Depth','Time','Phi','SW','Vp','Vs','Rho'])

# Loading 2D dataset
dataset_2D = sio.loadmat('Goliat2D_Torstein.mat') 
# Depth, trace, phi, sw, Vclay, Vp, Vs, Rho,

Depth_2D  = dataset_2D['depth']
Trace_2D  = dataset_2D['trace']

Phi_2D  = dataset_2D['Phi']
Sw_2D  = dataset_2D['Sw']
Vclay_2D  = dataset_2D['Vclay']

Vp_2D  = dataset_2D['Vp']
Vs_2D  = dataset_2D['Vs']
Rho_2D  = dataset_2D['Rho']

ntr = Phi_2D.shape[1]
ns = Phi_2D.shape[0]
t0 = 1.8


# Depth to time, assume t0 = 1.8, taking Phi as an example
# m in the shape of ns =m.shape[0], ntr = m.shape[1]

def DeptoTime(m,Depth,Vp,t0,dt):
    ns = m.shape[0]
    ntr = m.shape[1]     
    Time =np.zeros([ns-1,ntr],dtype = np.float64)
    
    for i in range(ntr):
      Time[:,i]= (t0 + np.cumsum(2*np.ediff1d(Depth)/(1000*(Vp[1:,i]+Vp[0:-1,i])/2)))
        
    TimeSeis = np.arange(np.max(Time[0,:]),np.min(Time[-1,:]),dt)
    TimeSeis = np.matlib.repmat(TimeSeis.reshape(len(TimeSeis),1),1,ntr)
    Time = np.concatenate((TimeSeis-dt/2,TimeSeis[-1:]+dt/2),axis=0)
    
    t1 = TimeSeis[0,0]-dt/2
    t2 = TimeSeis[-1,0]+dt/2
    t = np.arange(t1,t2,(t2-t1)/ns)
    
    mTime =np.array([[0]*ntr]*(Time.shape[0]),dtype = np.float64)

    for i in range(ntr):
        mTime[:,i] = np.interp(Time[:,i],t,m[:,i])

    return mTime

SwTime = DeptoTime(Sw_2D,Depth_2D,Vp_2D,1.8,0.001)
VclayTime = DeptoTime(Vclay_2D,Depth_2D,Vp_2D, 1.8, 0.001)
# 2d array to 3D array
# a: np array
def Addaxis(a):
    from numpy import newaxis
    b = a[:, newaxis,:]      # newaxis can be anywhere
    return b

# example: aa = Addaxis(df.to_numpy())

# phi_sgs = Addaxis(Phi.T)
seis_sgs_noise = Addaxis(Seis_noise.T)
seis_sgs_noise = seis_sgs_noise[:,:,::4]
#
np.save('seis_noise.npy',seis_sgs_noise)
# np.save('seis_sgs_witherr.npy',seis_sgs)


def Normalize(x, mean_val, std_val):

    n = (x - mean_val) / std_val

    return n


def Denormalize(x, mean_val, std_val):
    dn = x * std_val + mean_val

    return dn


# Generating 2D grid for the use of SGS from SeReMpy
# (Xcoords, dcoords, dz, zmean, zvar, l, krigtype, krig)
def GenMeasureD(m,nt,ns):
    """
    Generate dcoordinate and dvalue for SGS
    """
    trace_idx = np.arange(nt).reshape(1,-1)
    trace_idxmat = np.tile(trace_idx,(ns,1))   # Construct an array by repeating A the number of times given by n.
    sample_idx = np.arange(ns).reshape(-1,1)
    sample_idxmat = np.tile(sample_idx,(1,nt))
    dx = np.reshape(trace_idxmat,-1)
    dy = np.reshape(sample_idxmat,-1)
    dcoords = np.transpose(np.vstack([dx.reshape(-1), dy.reshape(-1)]))
    dz = m.reshape(-1,1)
    
    return dcoords, dz

# dcoords, dz = GenMeasureD(PhiTime)


def EnlargeGrid(nt,ns,tlarge,slarge):
    
    X_trace = np.linspace(0,nt-1,(nt-1)*tlarge+1).reshape(1,-1)   # trace enlarge 4 times
    X_trace_idxmat = np.tile(X_trace,(ns*tlarge,1))
    X = np.reshape(X_trace_idxmat,-1)
    
    Y_sample = np.linspace(0,ns-1,ns*slarge).reshape(-1,1)   # sample enlarge 4 times
    Y_sample_mat= np.tile(Y_sample,(1,(nt-1)*slarge+1))
    Y = np.reshape(Y_sample_mat,-1)
    Xcoords = np.transpose(np.vstack([X.reshape(-1), Y.reshape(-1)]))
    
    return Xcoords,X_trace.shape[1],Y_sample.shape[0]

# Xcoords, X_trace_num, Y_sample_num= EnlargeGrid(nt, ns,tlarge,slarge)

# using sgs to generate high resolution synthetic porosity
def ApplySGS(m,tlarge,slarge):
    """
    Applying SGS for enlargement
    
    Parameters
    ----------
    m : 2D array
        Shape[0] should be the samples.
        Shape[1] should be the traces
    tlarge : int
        the number of enlargement in trace direction.
    slarge : int
        the number of enlargement in sample direction.

    Returns
    -------
    sgs : 2D array
        the sgs results of 2D section.

    """
    
    ns = m.shape[0]  #number of samples
    nt = m.shape[1]   #number of traces
    
    dcoords, dz = GenMeasureD(m,nt,ns)
    
    xmean = np.mean(dz)
    xvar = np.var(dz)
    xstd = np.std(dz)
    l = 1
    krigtype = 'gau'
    krig = 1
        
    d = Normalize(dz, xmean, xstd)
    
    xcoords,X_trace_num,Y_sample_num = EnlargeGrid(nt, ns, tlarge, slarge)
    
    sim = SeqGaussianSimulation(xcoords, dcoords, d, xmean, xvar, l, krigtype, krig)
    
    d_out = Denormalize(sim, xmean, xstd)
    sgs = d_out.reshape(Y_sample_num,X_trace_num)       # column first, then row
    
    return sgs

sgs_sw = ApplySGS(SwTime, 4, 4)
sgs_Vclay = ApplySGS(VclayTime, 4, 4)
# not tested
def StochasticPerturbationModel(x,z,vp,stdvp,nsamp):

    # perturbation
     # Gaussian
    # vppert = vp+stdvp*randn(size(vp));
    # Uniform
    vppert = vp+stdvp*np.rand(np.shape(vp));
# subsampling
    ind = np.randi(len(vp),nsamp, 1);
    vpsub = vppert(ind,1);
    xsub = x(ind,1);
    zsub = z(ind,1);
# interpolation
    F = scatteredInterpolant(xsub,zsub,vpsub);
    vpmod = F[x,z];


# Phi = np.load('phi_sgs.npy')
# Phi = Phi[:,0,:].T
# Synthetic 2D seismic with only one (Phi,in time domain) variable, and using softsand model and 0 incident angle

def SynSeisLinearVpZeroIncident(Phi,a,b):
    
    ns = Phi.shape[0]
    ntr = Phi.shape[1]
    # Roch physics model
    Rho_syn = DensityModel(Phi, Rho_sol, Rho_fl)
    Vp_syn = a*Phi +b
    
    # Seismic model
    w, tw = RickerWavelet(freq, dt, ntw)
    
    Seis_syn = np.zeros((ns - 1, ntr))
    
    for i in range(ntr):
        Seis = SeismicModelZeroincidentAngle(Vp_syn[:, i].reshape(-1, 1), Rho_syn[:, i].reshape(-1, 1), theta, w)
        # err = np.sqrt(0.2 * np.var(Seis.flatten())) * np.random.randn(len(Seis.flatten()))
        Seis_syn[:, i] = Seis.flatten() 
        # + err
    
    return Seis_syn


def SynSeisRaymerZeroIncident(Phi,a,b):
    
    ns = Phi.shape[0]
    ntr = Phi.shape[1]
    
    # Roch physic parameters
    Vp_sol = 6
    Vp_fl = 1.5
    Rho_sol = 2.6
    Rho_fl = 1
    
    # Roch physics model
    Rho_syn = DensityModel(Phi, Rho_sol, Rho_fl)
    Vp_syn = RaymerModel(Phi,Vp_sol,Vp_fl)
    
    # Seismic model
    w, tw = RickerWavelet(freq, dt, ntw)
    
    Seis_syn = np.zeros((ns - 1, ntr))
    
    for i in range(ntr):
        Seis = SeismicModelZeroincidentAngle(Vp_syn[:, i].reshape(-1, 1), Rho_syn[:, i].reshape(-1, 1), theta, w)
        # err = np.sqrt(0.2 * np.var(Seis.flatten())) * np.random.randn(len(Seis.flatten()))
        Seis_syn[:, i] = Seis.flatten() 
        # + err
    
    return Seis_syn


def SynSeisSoftsandZeroincidentAngle2D(Phi):
    
    ns = Phi.shape[0]
    ntr = Phi.shape[1]
    
    Sw = 0.8
    Kfl = Sw * Kw + (1 - Sw) * Ko
    
    Rho_syn = DensityModel(Phi, Rho_sol, Rho_fl)
    Vp_syn, _ = SoftsandModel(Phi, Rho_syn, Kmat, Gmat, Kfl, critporo, coordnum, press)
    
    w, tw = RickerWavelet(freq, dt, ntw)
    
    Seis_syn = np.zeros((ns - 1, ntr))
    
    for i in range(ntr):
        Seis = SeismicModelZeroincidentAngle(Vp_syn[:, i].reshape(-1, 1), Rho_syn[:, i].reshape(-1, 1), theta, w)
        # err = np.sqrt(0.2 * np.var(Seis.flatten())) * np.random.randn(len(Seis.flatten()))
        Seis_syn[:, i] = Seis.flatten() 
        # + err
    
    return Seis_syn
#
Phi = np.load('por_sgs.npy')
Seis_noise_free = SynSeisSoftsandZeroincidentAngle2D(Phi)


def SynSeisFullRockphysics2D(Phi,Vclay,Sw):   # based on granular media theory+softsand+ aki-richard approximation
    
    n = len(Phi)

    ## rock phsyics parameters
    # solid phase (quartz and clay)
    Kclay = 21
    Kquartz = 33
    Gclay = 15
    Gquartz = 36
    Rhoclay = 2.45
    Rhoquartz = 2.65
    # fluid phase (water and gas)
    Kwater = 2.25
    Kgas = 0.1
    Rhowater = 1.05
    Rhogas = 0.1
    patchy = 0
    # granular media theory parameters
    criticalporo=0.4
    coordnumber=7
    pressure=0.02

    ## seismic parameters
    # angles
    # theta = [15, 30, 45]
    theta = [0]

    # wavelet
    # # time interval
    # dt = 0.001
    # # initial time (random value for synthetic data)
    # t0 = 1.5
    dt = 0.001
    freq = 45
    ntw = 64
    wavelet, tw = RickerWavelet(freq, dt, ntw)

    ## solid and fluid phases
    ns = Phi.shape[0]
    nt = Phi.shape[1]
    
    Kmat = np.zeros([ns,nt])
    Gmat = np.zeros([ns,nt])
    Rhomat =np.zeros([ns,nt])
    Kfl = np.zeros([ns,nt])
    Rhofl = np.zeros([ns,nt])
    
    for i in range(nt):
        Kmat[:,i], Gmat[:,i], Rhomat[:,i], Kfl[:,i], Rhofl[:,i] = MatrixFluidModel(np.array([Kclay, Kquartz]), np.array([Gclay, Gquartz]), np.array([Rhoclay,Rhoquartz]),np.array([Vclay[:,i], 1-Vclay[:,i]]).T, np.array([Kwater, Kgas]), np.array([Rhowater,Rhogas]), np.array([Sw[:,i],1-Sw[:,i]]).T, patchy)


    ## Density
    Rho = DensityModel(Phi, Rhomat, Rhofl)

    ## Soft sand model
    Vp, Vs = SoftsandModel(Phi, Rho, Kmat, Gmat, Kfl, criticalporo, coordnumber, pressure)

    ## Seismic
    Snear = np.zeros([ns-1,nt])
    Smid = np.zeros([ns-1,nt])
    Sfar = np.zeros([ns-1,nt])
    Seis_syn = np.zeros([ns-1,nt])
    for i in range(nt):
        Seis = SeismicModelAkiRichard(Vp[:,i].reshape(-1, 1), Vs[:,i].reshape(-1, 1), Rho[:,i].reshape(-1, 1), theta, wavelet)
        # Snear[:,i] = Seis[:ns-1].flatten()  # ns-1 not include acorrding to python role
        # Smid[:,i] = Seis[ns-1:2*(ns-1)].flatten()
        # Sfar[:,i] = Seis[2*(ns-1):].flatten()
        Seis_syn[:,i] = Seis.flatten()

    # return Snear, Smid, Sfar
    return Seis_syn

# Snear, Smid, Sfar = SynSeisFullRockphysics2D(Phi_2D, Vclay_2D,Sw_2D)

Phi = np.load('por_sgs.npy')
S = SynSeisFullRockphysics2D(Phi, sgs_Vclay,sgs_sw)


# SRN dB calculation
# measure the power P(x) of a signal x(n), P(x) = 1/N*(sum x(n)**2)
# SNR via power ratio, SNR = P_signal/P_noise
# SNR in dB, SNR_db = 10log10(P_signal/P_noise)

def SignalPower(d):
    return np.mean(d**2).astype('float64')

def SNR_db(d_noise,d_noise_free):
    S_pow = SignalPower(d_noise_free).astype('float64')
    N_pow = SignalPower(d_noise).astype('float64')
    SNR_dB = 10*np.log10(S_pow/(N_pow-S_pow))
    return SNR_dB

# d_noise_free = np.load('seis_sgs.npy')
# d_noise = np.load('seis_sgs_witherr.npy')

# snr = SNR_db(d_noise, d_noise_free)


# 1D plot
# fig1=plt.figure(figsize=(2.5,6)) 
# plt.plot(seis_sgs_noise_free[:,0].T[:,1],list(range(seis_sgs_noise_free[:,0].T.shape[0])),c='k',label='Poststack')
# plt.plot(seis_sgs_noise[:,0].T[:,1],list(range(seis_sgs_noise[:,0].T.shape[0])),c='r',label='Snear',linestyle ='dashed')
# # plt.plot(Smid[:,0],list(range(ns-1)),c='g',label='Smid',linestyle ='dashed')
# # plt.plot(Sfar[:,0],list(range(ns-1)),c='b',label='Sfar',linestyle ='dashed')
# # plt.xlabel('Porosity (v/v)')
# # plt.ylabel('Depth (km)')
# # plt.xlim([0,0.4])
# plt.gca().invert_yaxis()
# plt.grid()
# plt.legend(loc='upper left')
# plt.show()
# fig1.tight_layout()


# 2D plot
fig7 = plt.figure()
plt.imshow(S, cmap='gray', aspect='auto',
            interpolation='bilinear' , extent=[0,337,1.95363,1.80063] ) #Extent defines the left and right limits, and the bottom and top limits, e.g. extent=[horizontal_min,horizontal_max,vertical_min,vertical_max]
fig7.set_size_inches(9, 4)
cor = plt.colorbar()
# cor.ax.set_title('Sw')
# plt.clim(vmin = 0, vmax=1)
plt.xlabel('Traces')
plt.ylabel('Time (s)')
fig7.tight_layout()
fig7.show()


# train loss
# plt.plot(train_loss)
# plt.show()

# np.load('Ip_rand_err_5.npy')
# fig7.savefig('Fig7.tiff',dpi=200)
# np.save("Seis_1d_witherr.npy", Seis_1d_witherr)
# mae = np.mean(np.abs(a,b))
# np.clip(a,vmin,v_max) truncate a into v_min, v_max