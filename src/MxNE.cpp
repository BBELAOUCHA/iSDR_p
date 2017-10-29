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
    R = new double [n_c*n_t];
    mu = new double [n_s];
}

MxNE::~MxNE(){
    //delete[] R;
    }
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

void MxNE::Compute_mu(const double *G) const {
    // compute the gradient step mu for each block coordinate i.e. Source
    // mu = ||G_s||_F^{-1}
    int mp2 = m_p*m_p;
    int mpc = m_p*n_c;
    double *X = new double [mp2];
    for(int i = 0;i < n_s; ++i){
        mu[i] = 0.0;
        double x = 0.0;
        const double * G_ptr = &G[i*mpc];
        cxxblas::gemm(cxxblas::ColMajor,cxxblas::Trans, cxxblas::NoTrans, m_p,
        m_p, n_c, 1.0, G_ptr, n_c, G_ptr, n_c, 0.0, &X[0], m_p);
        for (int k = 0; k < mp2; k++)
            x += X[k]*X[k];
        if (x != 0.0)
            mu[i] = 1.0/x;
        else
            printf("\nSilent source detected (%d) i.e. columns of G =0.0", i);
    }
    delete[] X;
}

void MxNE::Compute_dX(const double *G, double *X, const int n_source) const {
    // compute the update of X i.e. X^{i+1} = X^{i} + mu dX for source with an 
    // indice n_source
    double * GtR = new double [m_p*n_t];
    const double * G_ptr = &G[n_source*n_c*m_p];
    cxxblas::gemm(cxxblas::ColMajor,cxxblas::Trans, cxxblas::NoTrans, m_p,
    n_t, n_c, 1.0, G_ptr, n_c, &R[0], n_c, 0.0, GtR, m_p);
    for (int j = 0; j < n_t; ++j){
        for (int k = 0;k < m_p; ++k) 
            X[k + j] += GtR[j * m_p + k];
    }
    delete[] GtR;
}

void MxNE::update_r(const double *G_reorder,const double *dX,
                    const int n_source) const {
    // recompute the residual for each updated source, s, of indice n_source
    // activation R = R + G_s * (X^{i-1} - X^i) = R = R - G_s * ( X^i - X^{i-1})
    const double* Gp = &G_reorder[n_source * n_c * m_p];
    for (int j = 0; j < n_t; ++j)
        cxxblas::gemm(cxxblas::ColMajor,cxxblas::NoTrans, cxxblas::NoTrans, n_c,
        1, m_p, 1.0, Gp, n_c, &dX[j], m_p, 1.0, &R[j*n_c], n_c);
}

void MxNE::Compute_GtR(const double *G, const double* Rx, double *GtR)const{
    // This function compute the multiplication of G (gainxMAR model) by 
    // The residual R
    //   Input:
    //         G (n_c x (n_s x m_p)): Gain x [A1, .., Ap] reordered 
    //         R (n_c x n_t): residual matrix (M-GJ)
    //   Output:
    //          GtR : ((n_t x m_p) x n_s)
    const double * R_ptr = &Rx[0];
    const int x = m_p*n_c;
    const int y = m_p*n_t;
    for(int i=0;i<n_s; ++i)
        cxxblas::gemm(cxxblas::ColMajor,cxxblas::Trans, cxxblas::NoTrans, n_t,
        m_p, n_c, 1.0, R_ptr, n_c, &G[i*x], n_c, 0.0, &GtR[i*y], n_t);
}

double MxNE::Compute_alpha_max(const double *G, const double *M) const{
    double *GtM = new double [n_s*m_p*n_t];
    double norm_GtM = 0.0;
    Compute_GtR(G, M, GtM);
    for(int i=0;i<n_s; ++i){
        double GtM_axis1norm;
        cxxblas::nrm2(n_t*m_p, &GtM[i*n_t*m_p], 1, GtM_axis1norm);
        if (GtM_axis1norm > norm_GtM)
            norm_GtM = GtM_axis1norm;
    }
    delete[] GtM;
    return norm_GtM;
}

void MxNE::Compute_Me(const double *G, const double * J, double *Me)const{
    // This function compute the multiplication of G (gainxMAR model) by 
    // The estimated brain activity J i.e. explained data
    //   Input:
    //         G (n_c x (n_s x m_p)): Gain x [A1, .., Ap] reordered 
    //         J (n_s x n_t_s): residual matrix (M-GJ)
    //   Output:
    //          Me : ((n_t x n_s)
/*    for(int i=0;i<n_t;i++)
        for(int j=0;j<n_c;j++)
            for (int l=0;l<m_p;l++)
                for (int k=0;k<n_s;k++)
                    Me[i*n_c + j] += G[k*n_c*m_p + j + l*n_c] * J[k*n_t_s + i + l]; 
*/
    std::fill(&Me[0], &Me[n_c*n_t], 0.0);
    for(int i=0;i<n_s; ++i)
        for(int j=0;j<n_t; ++j)
            cxxblas::gemm(cxxblas::ColMajor,cxxblas::NoTrans, cxxblas::NoTrans,
            n_c, 1, m_p, 1.0, &G[i*n_c*m_p], n_c, &J[i*n_t_s + j], m_p, 1.0,
            &Me[j*n_c], n_c);
}

double MxNE::duality_gap(const double* G,const double *M, const double * J,
                         double alpha) const {
    // compute the duality gap for mixed norm estimate gap = Fp-Fd;
    double *GtR=new double [n_s*m_p*n_t];
    std::fill(&GtR[0],&GtR[n_s*m_p*n_t], 0.0);
    Compute_GtR(G,&R[0], GtR);
    double norm_GtR = 0.0;
    for (int ii =0; ii < n_s; ii++){
        double GtR_axis1norm = 0.0;
        cxxblas::nrm2(n_t*m_p, &GtR[ii*n_t*m_p], 1, GtR_axis1norm);
        if (GtR_axis1norm > norm_GtR)
            norm_GtR = GtR_axis1norm;
    }
    double R_norm, gap, s;
    cxxblas::nrm2(n_t*n_c, &R[0], 1, R_norm);
     if (norm_GtR > alpha){
        s =  alpha / norm_GtR;
        double A_norm = R_norm * s;
        gap = 0.5 * (R_norm * R_norm + A_norm * A_norm);
    }
    else{
        s = 1.0;
        gap = R_norm * R_norm;
    }
    double ry_sum = 0.0;
    cxxblas::dot(n_c*n_t, &M[0], 1, &R[0], 1, ry_sum);
    
    double l21_norm = 0.0;
    for (int i =0; i<n_s; ++i){
        double r = 0.0;
        cxxblas::nrm2(n_t_s, &J[i*n_t_s], 1, r);
        l21_norm += r;
    }
    delete[] GtR;
    gap += alpha * l21_norm - s * ry_sum;
    return gap;
}

int MxNE::MxNE_solve(const double *M, double *G_reorder, double *J,
                     double alpha, int n_iter, double &dual_gap_,
                     double &tol, bool initial) const {
    // Compute the mixed norm estimate i.e.
    // Objective F(X) = \sum_{t=1-T}||M_t-sum_i{1-p} G_i X_{t-i}|| +
    //                                                         alpha ||X||_{21}
    // check reference papers
    double d_w_ii = 0, d_w_max = 0, W_ii_abs_max = 0, w_max  = 0.0;
    double n_x;
    cxxblas::nrm2(n_t*n_c, M, 1, n_x);
    cxxblas::copy(n_t*n_c, M, 1, &R[0], 1);// initialize Residual R = M-0
    Compute_mu(G_reorder);
    tol = d_w_tol*n_x;//*n_x;
    dual_gap_ = 0.0;
    if (not initial)
        std::fill(&J[0], &J[n_t_s*n_s], 0.0);
    else{
        double * Me = new double [n_t*n_c];
        Compute_Me(G_reorder, &J[0], Me);
        cxxblas::axpy(n_t*n_c, -1.0, Me, 1, &R[0], 1);
        delete[] Me;
    }
    double *mu_alpha = new double [n_s];
    for (int i=0;i<n_s;i++)
        mu_alpha[i] = mu[i]*alpha;
    int ji;
    for (ji = 0; ji < n_iter; ji++){
        w_max = 0.0;
        d_w_max = 0.0;
        for (int i = 0; i < n_s; ++i){
            double * dX = new double [n_t_s];
            double * wii= new double [n_t_s];
            double * Ji = &J[i*n_t_s];
            std::fill(&dX[0], &dX[n_t_s], 0.0);
            cxxblas::copy(n_t_s, Ji, 1, wii, 1);
            Compute_dX(G_reorder, dX, i);
            cxxblas::axpy(n_t_s, mu[i], dX, 1, Ji, 1);
            double nn;
            cxxblas::nrm2(n_t_s, Ji, 1, nn);
            double s_t = 0.0;
            double s_ = std::max(nn, mu_alpha[i]);
            if (s_!= 0.0)
                s_t = std::max(1.0 - mu_alpha[i]/s_, 0.0);
            cxxblas::scal(n_t_s, s_t, Ji, 1);
            cxxblas::axpy(n_t_s,-1.0, Ji, 1, wii, 1); // wii = X^i - X^{i-1}
            d_w_ii = absmax(wii);
            W_ii_abs_max = absmax(Ji);
            if (d_w_ii != 0.0)
                update_r(G_reorder, wii, i);
            delete[] dX;
            delete[] wii;
            if (d_w_ii > d_w_max)
                d_w_max = d_w_ii;
            if (W_ii_abs_max > w_max)
                w_max = W_ii_abs_max;
        }
        if ((w_max == 0.0) || (d_w_max / w_max <= d_w_tol) || (ji == n_iter-1)){
            dual_gap_ = duality_gap(G_reorder, M, J, alpha);
            if (dual_gap_ <= tol)
                break;
        }
    }
    if (verbose){
        if (dual_gap_ > tol){
            printf("\n Objective did not converge, you might want to increase");
            printf("\n the number of iterations (%d)", n_iter);
            printf("\n Duality gap = %.2e | tol = %.2e \n", dual_gap_, tol);
        }
        else{
            printf("\n Objective converges");
            printf("\n the number of iterations (%d)", ji+1);
            printf("\n Duality gap = %.2e | tol = %.2e \n", dual_gap_, tol);
        }
    }
    delete[] mu_alpha;
    return ji+1;
}
