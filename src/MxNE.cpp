#include "MxNE.h"
//#include <omp.h>

////============================================================================
////============================================================================
/////
///// \file MxNE.cpp
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

MxNE::MxNE(int n_sources, int n_sensors, int Mar_model, int n_samples,
           double d_w_tol, bool ver){
    this-> n_t = n_samples;
    this-> n_c = n_sensors;
    this-> n_s = n_sources;
    this-> m_p = Mar_model;
    this-> n_t_s = n_t + m_p - 1;
    this-> d_w_tol= d_w_tol;
    this-> verbose = ver;
}

MxNE::~MxNE(){}
double MxNE::absmax(const double *X) const {
    // compute max(abs(X))
    double si = 0;
    for (int i = 0;i < n_t_s; ++i){
        double s = std::abs(X[i]);
        if (s > si)
            si = s;
    }
    return si;
}

void MxNE::Compute_mu(const double *G, double *mu) const {
    // compute the gradient step mu for each block coordinate i.e. Source
    // mu = ||G_s||_F
    //#pragma omp parallel for
    for(int i = 0;i < n_s; ++i)
        cxxblas::nrm2(n_c*m_p, &G[i*m_p*n_c], 1, mu[i]);
}

void MxNE::Compute_dX(const double *G_ptr,const double *R_ptr, double *X,
                    const int n_source) const {
    // compute the update of X i.e. X^{i+1} = X^{i} + mu dX
    double GtR[m_p*n_t];
    const int ix = n_source*n_c*m_p;
    const double * G_ptr_x = &G_ptr[ix];
    cxxblas::gemm(cxxblas::ColMajor,cxxblas::Trans, cxxblas::NoTrans, m_p,
    n_t, n_c, 1.0, G_ptr_x, n_c, R_ptr, n_c, 0.0, &GtR[0], m_p);
    for (int j = 0; j < n_t; ++j){
        for (int k = 0;k < m_p; ++k) 
            X[k + j] += GtR[j * m_p + k];
    }
}

void MxNE::update_r(const double *G_reorder, double *R,const double *dX,
                    const int n_source) const {
    // recompute the residual for each updated source, s, activation 
    // R = R + G_s * (X^{i-1} - X^i) = R = R - G_s * ( X^i - X^{i-1})
    const int x_n = n_source * n_c * m_p;
    //#pragma omp parallel for
    for (int j = 0; j < n_t; ++j){
        double* Rj = &R[j*n_c];
        const double* Gp = &G_reorder[x_n];
        cxxblas::gemm(cxxblas::ColMajor,cxxblas::NoTrans, cxxblas::NoTrans, n_c,
        1, m_p, 1.0, Gp, n_c, &dX[j], m_p, 1.0, Rj, n_c);
    }
}

void MxNE::Compute_GtR(const double *G, const double * R, double *GtR)const{
    // This function compute the multiplication of G (gainxMAR model) by 
    // The residual R
    //   Input:
    //         G (n_c x (n_s x m_p)): Gain x [A1, .., Ap] reordered 
    //         R (n_c x n_t): residual matrix (M-GJ)
    //   Output:
    //          GtR : ((n_t x m_p) x n_s)
    //#pragma omp parallel for
    for(int i=0;i<n_s; ++i)
        cxxblas::gemm(cxxblas::ColMajor,cxxblas::Trans, cxxblas::NoTrans, n_t,
        m_p, n_c, 1.0, &R[0], n_c, &G[i*m_p*n_c], n_c, 0.0, &GtR[i*m_p*n_t],
        n_t);
}

double MxNE::duality_gap(const double* G,const double *M, double *R,
                         double * J,  double alpha) const {
    // compute the duality gap for mixed norm estimate gap = Fp-Fd;
    double GtR[n_s*m_p*n_t];
    std::fill(&GtR[0],&GtR[n_s*m_p*n_t], 0.0);
    Compute_GtR(&G[0], &R[0], &GtR[0]);
    // this part was taking from scikit-learn package (MultiTaskLasso) 
    double norm_GtR = 0.0;
    for (int ii =0; ii < n_s; ++ii){
        double GtR_axis1norm = 0.0;
        cxxblas::nrm2(n_t*m_p, &GtR[0] + ii*n_t*m_p, 1, GtR_axis1norm);
        if (GtR_axis1norm > norm_GtR)
            norm_GtR = GtR_axis1norm;
    }
    double R_norm, w_norm, gap, con;
    cxxblas::nrm2(n_t * n_c, &R[0], 1, R_norm);
    cxxblas::nrm2(n_t_s * n_s, &J[0], 1, w_norm);
    if (norm_GtR > alpha){
        con =  alpha / norm_GtR;
        double A_norm = R_norm * con;
        gap = 0.5 * (R_norm * R_norm + A_norm *A_norm);
    }
    else{
        con = 1.0;
        gap = R_norm * R_norm;
    }
    double ry_sum = 0.0;
    cxxblas::dot(n_c*n_t, &M[0], 1, &R[0], 1, ry_sum);
    double l21_norm = 0.0;
    for (int i =0; i<n_s; ++i){
        double r = 0.0;
        cxxblas::nrm2(n_t_s, &J[0] + i*n_t_s, 1, r);
        l21_norm += r;
    }
    gap += alpha * l21_norm - con * ry_sum;
    return gap;
}

int MxNE::MxNE_solve(const double *M, double *G_reorder, double *J,
                     double alpha, int n_iter, double &dual_gap_,
                     double &tol, bool initial) const {
    // Compute the mixed norm estimate i.e.
    // Objective F(X) = \sum_{t=1-T}||M_t-sum_i{1-p} G_i X_{t-i}|| +
    //                                                         alpha ||X||_{21}
    double d_w_ii = 0, d_w_max = 0, W_ii_abs_max = 0, w_max  = 0.0;
    double *R = new double [n_c*n_t];
    double *mu = new double [n_s];
    double n_x;
    if (not initial)
        std::fill(&J[0], &J[n_t_s*n_s], 0.0);
    cxxblas::nrm2(n_t * n_c, &M[0], 1, n_x);
    cxxblas::copy(n_t * n_c, &M[0], 1, &R[0], 1);// initialize Residual R = M-0
    Compute_mu(&G_reorder[0], &mu[0]);
    tol = d_w_tol*n_x*n_x;
    dual_gap_ = 0.0;
    if (not initial)
        std::fill(&J[0], &J[n_t_s*n_s], 0.0);
    else {
        for (int k=0;k<n_t;k++){
            double * Rj = &R[k*n_c];
            for (int j=0;j<n_s;j++){
                cxxblas::gemm(cxxblas::ColMajor,cxxblas::NoTrans,
                cxxblas::NoTrans, n_c, 1, m_p, -1.0, &G_reorder[j*n_c*m_p],
                n_c, &J[j*n_t_s + k], m_p, 1.0, Rj, n_c);
            }
        }
    }
    int j;
    for (j = 0; j <n_iter; ++j){
        w_max = 0.0;
        d_w_max = 0.0;
        for (int i = 0; i < n_s; ++i){
            double dX[n_t_s];
            double wii[n_t_s];
            double* Ji = &J[i*n_t_s];
            std::fill(&dX[0], &dX[n_t_s], 0.0);
            cxxblas::copy(n_t_s, Ji, 1, &wii[0], 1);
            Compute_dX(&G_reorder[0], &R[0], &dX[0], i);
            cxxblas::axpy(n_t_s, mu[i], &dX[0], 1, Ji, 1);
            double nn;
            cxxblas::nrm2(n_t_s, Ji, 1, nn);
            double s_t = 0.0;
            double s_ = std::max(nn, mu[i] * alpha);
            if (s_ != 0.0)
                s_t = std::max(1.0 - (mu[i] * alpha) / s_, 0.0);
            cxxblas::scal(n_t_s, s_t, Ji, 1);
            cxxblas::axpy(n_t_s,-1.0, Ji, 1, wii, 1); // wii = X^i - X^{i-1}
            d_w_ii = absmax(&wii[0]);
            W_ii_abs_max = absmax(Ji);
            if (d_w_ii != 0.0)
                update_r(&G_reorder[0], &R[0], &wii[0], i);
            if (d_w_ii > d_w_max)
                d_w_max = d_w_ii;
            if (W_ii_abs_max > w_max)
                w_max = W_ii_abs_max;
        }
        if (w_max == 0.0 or d_w_max / w_max < d_w_tol or j == n_iter - 1){
            dual_gap_ = duality_gap(&G_reorder[0], &M[0], &R[0], &J[0], alpha);
            if (dual_gap_ < tol)
                break;
        }
    }
    if (verbose){
        if (dual_gap_ > tol){
            printf( "\n Objective did not converge, you might want to increase");
            printf("\n the number of iterations (%d)", n_iter);
            printf( "\n Duality gap = %f | tol = %f \n", dual_gap_, tol);
        }
    }
    return j+1;
}


