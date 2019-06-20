# -*- coding: utf-8 -*-
'''
###############################################################################
#
# This code is used as a wrapper to the iSDR c++ implementation to reconstruct
# the brain activity and effective connectivity between the brain regions.
#
//// \author Brahim Belaoucha, INRIA <br>
////         Copyright (c) 2017 <br>
//// If you used this function, please cite one of the following:
//// (1) Brahim Belaoucha, Théodore Papadopoulo. Large brain effective network
//// from EEG/MEG data and dMR information. PRNI 2017 – 7th International
//// Workshop on Pattern Recognition in NeuroImaging, Jun 2017, Toronto, Canada. 
//// (2) Brahim Belaoucha, Mouloud Kachouane, Théodore Papadopoulo. Multivariate
//// Autoregressive Model Constrained by Anatomical Connectivity to Reconstruct
//// Focal Sources. 2016 38th Annual International Conference of the IEEE
//// Engineering in Medicine and Biology Society (EMBC), Aug 2016, Orlando,
//// United States. 2016.
###############################################################################

'''
import numpy as np
import PyiSDRcpp as pciSDR
import time
class iSDR:
    # an interface class to the iSDR c++ implementation
    def __init__(self, alpha=30, d_w_tol = 1e-7,n_mxne=1e4,n_iSDR=100,
        initial=True, verbose=True):
        '''
            alpha: the regularization parameter in the iSDR framework
            d_w_tol: tolerance used to stop the MxNE estimate
            n_mxne: the number of MxNE iterations
            n_iSDR: the number of iSDR iterations
            initial: true: use the previous estimated brain activity to re-
            estimate J
            verbose: true: display results
        '''
        self.alpha = alpha
        self.d_w_tol = np.double(d_w_tol)
        self.n_mxne = np.int64(n_mxne)
        self.n_iSDR = np.int64(n_iSDR)
        self.initial = initial
        self.verbose = verbose;


    def fit(self,G, M, A, SC):
        '''
        This function solve the following two functionals iteratively:
            1  min_J   \sum_{t=0} ^ T ||M_t-Gxsum_{i=1} ^ p A_i J_{t-i}||_2^2
                                               +
                                    alpha x ||J_t=0^T||_{21}
            2  min_A   \sum_{t=p} ^ T ||J_t - sum_{i=1} ^ p A_i J_{t-i}||_2^2 
            
            G: Lead field matrix
            M: Measurements EEG and/or MEG
            A: initial multivariate autoregressive model [A_p,..,A_1]
            results are saved in:
                self.J: only active source time courses
                self.MAR: effective connectivity between the active sources
                self.Active_set: indices of active sources
                self.Weights: weights used to normalize the rows of MAR
        '''
        if SC.shape[0] != SC.shape[1]:
            raise ValueError('SC has to be squared matrix but {}'.format(SC.shape))
        start_time = time.time()
        n_c, n_s = np.shape(G)
        m_p = np.shape(A)[1]/n_s
        _, n_t = np.shape(M)
        n_t_s = n_t + m_p - 1
        GxA = np.dot(G, A)
        Gtmp = G.T.reshape(-1).astype(np.double)
        Mtmp = M.T.reshape(-1).astype(np.double)
        SCtmp = SC.reshape(-1).astype(np.int32)
        MAR = np.zeros((1,n_s*n_s*m_p), dtype=np.double).reshape(-1)
        Active_set = np.zeros((1,n_s), dtype=np.int32).reshape(-1)
        J = np.zeros((1,n_t_s*n_s), dtype=np.double).reshape(-1)
        clt = pciSDR.PyiSDRcpp(self.n_mxne, self.n_iSDR, self.alpha, self.d_w_tol,
        self.initial, self.verbose)
        GxA = np.array(GxA.T.reshape(-1), dtype=np.double) 
        Wt = np.zeros((1,n_s*n_s*m_p), dtype=np.double).reshape(-1)
        n_active = clt.fit(Gtmp, SCtmp, Mtmp, GxA, J, MAR, Active_set, Wt)
        MAR =  np.array(MAR[:m_p*n_active**2].reshape((n_active, n_active*m_p), order='F')) 
        X =  np.array(J.reshape((n_t_s, n_s), order='F'))
        self.J = X[:, :n_active]
        self.MAR = MAR
        self.Active_set = Active_set[:n_active]
        self.Weights = Wt[:n_active]
        self.time_execution = time.time() - start_time
