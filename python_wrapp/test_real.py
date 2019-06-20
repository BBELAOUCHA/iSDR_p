import numpy as np
from scipy import linalg
import mne
from mne.datasets import sample
from mne.viz import plot_sparse_source_estimates


data_path = sample.data_path()
fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
ave_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-shrunk-cov.fif'
subjects_dir = data_path + '/subjects'
condition = 'Left Auditory'

# Read noise covariance matrix
noise_cov = mne.read_cov(cov_fname)
# Handling average file
evoked = mne.read_evokeds(ave_fname, condition=condition, baseline=(None, 0))
evoked.crop(tmin=0.04, tmax=0.18)

evoked = evoked.pick_types(eeg=False, meg=True)
# Handling forward solution
forward = mne.read_forward_solution(fwd_fname)


import sys 
caffe_root = '/home/bbelaouc/Wokspace/Project/iSDR_p/python_wrapp/'  # this file is expected to be in {caffe_root}/examples
sys.path.append(caffe_root)
import PyiSDR as isdr

def apply_solver_isdr(solver, evoked, forward, noise_cov, loose=0.2, depth=0.8):
    # Import the necessary private functions
    from mne.inverse_sparse.mxne_inverse import \
        (_prepare_gain, _check_loose_forward, is_fixed_orient,
         _reapply_source_weighting, _make_sparse_stc)

    all_ch_names = evoked.ch_names

    loose, forward = _check_loose_forward(loose, forward)

    # put the forward solution in fixed orientation if it's not already
    if loose == 0. and not is_fixed_orient(forward):
        forward = mne.convert_forward_solution(
            forward, surf_ori=True, force_fixed=True, copy=True, use_cps=True)

    # Handle depth weighting and whitening (here is no weights)
    gain, gain_info, whitener, source_weighting, mask = _prepare_gain(
        forward, evoked.info, noise_cov, pca=False, depth=depth,
        loose=loose, weights=None, weights_min=None)
    print source_weighting
    # Select channels of interest
    sel = [all_ch_names.index(name) for name in gain_info['ch_names']]
    M = evoked.data[sel]

    # Whiten data
    M = np.dot(whitener, M).astype(np.double)
    active_set = np.zeros(gain.shape[1], dtype=bool)
    gain = gain[:, :].astype(np.double)
    n_orient = 1 if is_fixed_orient(forward) else 3
    SC = np.eye(np.shape(gain)[1]).astype(np.int32)
    GA = gain.copy().astype(np.double)
    X, n_active_set, coef = solver_isdr(M, gain, GA, SC)
    active_set[n_active_set] = True
    #X = _reapply_source_weighting(X, source_weighting, active_set, n_orient)
    #Z = np.zeros((np.shape(gain)[1], np.shape(M)[1]))
    #Z[np.array(n_active_set), :] = X[:np.shape(M)[1], :len(n_active_set)].T
    #stc = _make_sparse_stc(Z, active_set, forward, tmin=evoked.times[0], tstep=1. / evoked.info['sfreq'])

    return X#stc





def solver_isdr(M, G, GA, SC):
    n_c, n_s = np.shape(G)
    print np.shape(G)
    print np.shape(SC)
    _, n_t = np.shape(M)
    m_p = np.shape(GA)[1]/n_s
    n_t_s = n_t + m_p - 1
    print "n_t", n_t
    G = G.T.reshape(-1).astype(np.double) 
    SC = SC.T.reshape(-1).astype(np.int32)
    M = M.T.reshape(-1).astype(np.double) 
    GA = GA.T.reshape(-1).astype(np.double)
    J = np.ones((1,(n_t+m_p-1)*n_s), dtype=np.double).reshape(-1)
    Coef = np.ones((1,n_s*n_s*m_p), dtype=np.double).reshape(-1)
    Active = np.ones((1,n_s), dtype=np.int32).reshape(-1)
    n_active = isdr.run(G, SC, M, GA, J, Coef, Active,25.0,10000,100, True, True)
    return J.reshape((n_t_s, n_s), order="F"), Active, Coef


loose, depth = 1., 0.  # corresponds to free orientation
stc_isdr = apply_solver_isdr(solver_isdr, evoked, forward, noise_cov, loose, depth)

import matplotlib.pyplot as plt

plt.plot(stc_isdr)
plt.show()
