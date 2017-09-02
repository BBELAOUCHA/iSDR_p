#pragma once

#include <cxxstd/iostream.h>
#include <flens/flens.cxx>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <omp.h>
#include <string>
#include <vector>
////============================================================================
////============================================================================
/////
///// \file MxNE.h
/////
///// \brief Compute the modefied mixed norm estimate MxNE, with multivariate
/////        autoregressive model (MVAR) on the source dynamics using blockcoor-
/////        dinate descent.
/////        Compute_dX: compute the update of each time source course.
/////        Compute_mu: compute the gradient step for each source s.
/////        absmax: compute max(abs(X)) needed to stop MxNE iteration.
/////        update_r: update the residual i.e. M-G*X for each source update.
/////        duality_gap: compute the duality gap =Fp-Fd which is used to stop
/////                     iterating the MxNE.
/////        MxNE_solve: the core function which is used to estimate the source
/////                    activity using MVAR model.
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

class MxNE {
    private:
        int n_t;
        int n_c;
        int n_t_s;
        int m_p;
        double d_w_tol;
        bool verbose;
    public:
        MxNE(int n_sources, int n_sensors, int Mar_model, int n_samples,
             double d_w_tol, bool ver);
        ~MxNE();
        int n_s;
        int MxNE_solve(const double *M, double *G_reorder, double *J, double
        alpha, int n_iter, double &dual_gap_, double &tol,bool initial) const;
        void Compute_dX(const double *G_ptr,const double *R_ptr, double *X,
        const int n_source) const;
        void Compute_mu(const double *G, double *mu) const;
        double absmax(const double *X) const;
        void update_r(const double *G_reorder, double *R,const double *dX,
        const int n_source) const;
        double duality_gap(const double* G, const double *M, double *R,
        double * J, const double alpha) const;
        void Compute_GtR(const double *G, const double * R, double *GtR) const;
};
