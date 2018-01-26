import numpy as np
import sys
from iSDRmodule import iSDR
import scipy.io as sp
Data = sp.loadmat('../examples/S1_p1_MEGT800.mat')
n_c = Data['n_c'][0][0]
n_s = Data['n_s'][0][0]
n_t = Data['n_t'][0][0]
m_p = Data['m_p'][0][0]
n_t_s = n_t + m_p - 1
G = Data['G']
SC = Data['SC']
M = Data['M'] 
samples = Data['time'][0]
delta_t = samples[1]-samples[0]
A = np.eye(n_s)
clt = iSDR(30)#,1e-7,1e4,100, True, True)
clt.fit(G, M, A, SC)

