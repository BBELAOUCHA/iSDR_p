#pragma once
//#ifndef USE_CXXLAPACK
//#define USE_CXXLAPACK
//#endif

#include <cxxstd/iostream.h>
#include <flens/flens.cxx>
#include "matio.h"
#include <cmath>
//#include <omp.h>
#include <vector>
#include "MxNE.h"
#include <stdio.h>

////============================================================================
////============================================================================
/////
///// \file iSDR.cpp
/////
///// \brief Compute the modefied mixed norm estimate MxNE, with multivariate
/////        autoregressive model (MVAR) ([A1,..,Ap]) on the source dynamics
/////        using blockcoordinate descent. That I call: iterative sources and
/////        dynamics reconstructions (iSDR(MARp)).
/////        Reorder_G: Reorder the GxA from [GxA1,..,GxAp] to [GA1|s=1,..,
/////                   GAp|s=1, .., GA1|s=n,..,GAp|s=n]
/////        Reduce_G: reduce gain matrix from n_cxn_s to n_cxn_s_i, where n_s_i
/////                  is the number of active sources.
/////        Reduce_SC: reduce the structural connectivity matrix SC from
/////                   n_sxn_s to n_s_ixn_s_i
/////        G_times_A: compute GxA  reordered.
/////        A_step_lsq: compute the MVAR model of only active sources using
/////                    least square solution.
/////        Zero_non_zero: get the set of active sources i.e. norm(X) > 0.0.
/////        iSDR_solve: core function which compute the MxNE and MVAR
/////                    iteratively
///// \author Brahim Belaoucha, INRIA <br>
/////         Copyright (c) 2017 <br>
///// If you used this function, please cite one of the following:
//// (1) Brahim Belaoucha, Théodore Papadopoulo. Large brain effective network
//// from EEG/MEG data and dMR information. PRNI 2017 – 7th International
//// Workshop on Pattern Recognition in NeuroImaging, Jun 2017, Toronto, Canada. 
//// (2) Brahim Belaoucha, Mouloud Kachouane, Théodore Papadopoulo. Multivariate
//// Autoregressive Model Constrained by Anatomical Connectivity to Reconstruct
//// Focal Sources. 2016 38th Annual International Conference of the IEEE
//// Engineering in Medicine and Biology Society (EMBC), Aug 2016, Orlando,
//// United States. 2016.
////
////
////============================================================================
////============================================================================

class iSDR {
    private:
        int n_t;
        int n_c;
        int m_p;
        //int n_t_s;
        double alpha;
        double mar_th;
        double n_mxne;
        double n_isdr;
        double d_w_tol;
        bool verbose;
    public:
        std::vector<double> Re {0,0,0};
        int n_t_s;
        int n_s;
        iSDR(int n_s, int n_c, int n_t, int p, double alpha, double n_iter_mxne,
            double n_iter_iSDR, double d_w_tol, double mar_th, bool ver);
        ~iSDR();
        int iSDR_solve(double *G_o, int *SC, const double *M, double *G,
        double * J, double * Acoef, int * Active, bool initial);
        void Reorder_G(const double *GA, double *G_reorder)const;
        void Reduce_G(const double * G, double *G_n, std::vector<int> ind)const;
        void G_times_A(const double * G, const double *A, double *GA_reorder)
                    const;
        void Reduce_SC(const int * SC, int *SC_n, std::vector<int> ind)const;
        void A_step_lsq(const double * S,const  int * A_scon,const double tol,
                    double * VAR)const;
        std::vector<int> Zero_non_zero(const double * S)const;
        void GA_removeDC(double * GA) const;
};
