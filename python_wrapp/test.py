import numpy as np

from iSDRmodule import iSDR
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sp
Data = sp.loadmat('/home/bbelaouc/Wokspace/Data/Results_iSDR_cpp/data/S1_p3_MEGT800.mat')

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
A= np.zeros((n_s, n_s*3))
A[:, 2*n_s:] = np.eye(n_s)
clt = iSDR(15)#,1e-7,1e4,100, True, True)
clt.fit(G, M, A, SC)
sns.set()
plt.plot(delta_t*np.arange(n_t_s) , clt.J)
plt.xlim([0,delta_t*n_t_s])
plt.xlabel('Time (s)')
plt.ylabel('Magnitude Am')
plt.show()

print clt.MAR
