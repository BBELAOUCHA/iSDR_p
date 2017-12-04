#include "MxNE.h"
//#include <omp.h>
#include "Matrix.h"
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

MxNE::~MxNE(){
    }
double MxNE::absmax(const Maths::DVector &X) const {
    // compute max(abs(X)) for i in [1, length(dX)]
    double si = X(1);
    for (int i = 2;i <= n_t_s; ++i){
        double s = std::abs(X(i));
        if (s > si)
            si = s;
    }
    return si;
}

void MxNE::Compute_mu(const Maths::DMatrix &G, Maths::DVector &mu) const {
    // compute the gradient step mu for each block coordinate i.e. Source
    // mu = ||G_s||_F^{-1}
    int mp2 = m_p*m_p;
    using namespace flens;
    using namespace std;
    Underscore<Maths::DMatrix::IndexType> _;
    for(int i = 0;i < n_s; ++i){
        Maths::DMatrix X(m_p, m_p);
        Maths::DMatrix Xz(n_c, m_p);
        Maths::DMatrix Xw(n_c, m_p);
        Xz = G(_, _(i*m_p+1, m_p*(i+1)));
        Xw = G(_, _(i*m_p+1, m_p*(i+1)));
        X = transpose(Xz)*Xw;
        mu.data()[i] = 0.0;
        double x = 0.0;
        for (int k = 0; k < mp2; k++)
            x += X.data()[k]*X.data()[k];
        if (x != 0.0)
            mu.data()[i] = 1.0/x;
        else
            printf("\nSilent source detected (%d) i.e. columns of G =0.0", i);
    }
}

void MxNE::Compute_dX(const Maths::DMatrix &G, Maths::DMatrix &R, Maths::DVector &X,
    const int n_source) const {
    /* compute the update of X i.e. X^{i+1} = X^{i} + mu dX for source with an 
    // indice n_source
    * where dX = G^T|_n_source x R;
    * Input:
    *       G: (n_c x (m_p x n_s)) a matrix containg the dynamic lead field matrix
    *       R: (n_c x n_t) a matrix containing the residual between the MEG/EEG 
    *           measurements and the reconstructed one.
    *       n_source: the source index 
    * Output:
    *       dX: (n_t_s) a vector containg GtR that correspends to n_source.
    */
    using namespace flens;
    Underscore<Maths::DMatrix::IndexType> _;
    Maths::DMatrix GtR(m_p, n_t);
    Maths::DMatrix Gx(n_c, m_p);
    Gx = G(_, _(n_source*m_p+1, (n_source + 1)*m_p));
    GtR = transpose(Gx)*R;
    for (int j = 0; j < n_t; ++j){
        for (int k = 0;k < m_p; ++k) 
            X.data()[k + j] += GtR(k+1, j+1);
    }
}

void MxNE::update_r(const Maths::DMatrix &G_reorder, const Maths::DVector &dX,
    Maths::DMatrix &R, const int n_source) const {
    // recompute the residual for each updated source, s, of indice n_source
    // activation R = R + G_s * (X^{i-1} - X^i) = R = R - G_s * ( X^i - X^{i-1})
    using namespace flens;
    Underscore<Maths::DMatrix::IndexType> _;
    Maths::DMatrix Gs(n_c, m_p);
    Gs = G_reorder(_, _(n_source*m_p+1, (n_source + 1)*m_p));
    for (int j = 1; j <= m_p; ++j)
        R += Gs(_, j)*transpose(dX(_(j,j+n_t-1)));
}

void MxNE::Compute_GtR(const Maths::DMatrix &G, const Maths::DMatrix &Rx,
    Maths::DMatrix &GtR)const{
    // This function compute the multiplication of G (gainxMAR model) by 
    // The residual R
    //   Input:
    //         G (n_c x (n_s x m_p)): Gain x [A1, .., Ap] reordered 
    //         R (n_c x n_t): residual matrix (M-GJ)
    //   Output:
    //          GtR : ((n_t x m_p) x n_s)
    //   due to computation we compute RtG = transpose(GtR);
    using namespace flens;
    GtR = transpose(Rx)*G;
}

double MxNE::Compute_alpha_max(const Maths::DMatrix &G, const Maths::DMatrix &M) const{
    /* a function used to compute the alpha_max which correspend to the minimum
    // alpha that results to an empty active set of regions/sources.
    // Input:
    //       G: (n_cx(m_pxn_s)) a matrix that contain dynamic gain matrix i.e. 
    //           G(lead field)x MVAR matrix A.
    //       M: (n_cxn_t) a matrix that contain the EEG and or MEG measurements.
    // Output:
    //       norm_GtM: a double scaler which correspend to alpha_max = 
    //                 norm_inf(norm_2(G^TM)).
    //
    * */
    using namespace flens;
    using namespace std;
    double norm_GtM = 0.0;
    Maths::DMatrix GtM(n_t, n_s*m_p);
    Compute_GtR(G, M, GtM);
    for(int i=0;i<n_s; ++i){
        double GtM_axis1norm;
        cxxblas::nrm2(n_t*m_p, &GtM.data()[i*n_t*m_p], 1, GtM_axis1norm);
        if (GtM_axis1norm > norm_GtM)
            norm_GtM = GtM_axis1norm;
    }
    return norm_GtM;
}

void MxNE::Compute_Me(const Maths::DMatrix &G, const Maths::DMatrix &J,
        Maths::DMatrix &Me)const{
    /* This function compute the multiplication of G (gainxMAR model) by 
    // The estimated brain activity J i.e. explained data
    //   Input:
    //         G (n_c x (n_s x m_p)): Gain x [A1, .., Ap] reordered 
    //         J (n_s x n_t_s): residual matrix (M-GJ)
    //   Output:
    //          Me : ((n_t x n_s)
    */
    using namespace flens;
    Underscore<Maths::DMatrix::IndexType> _;
    Maths::DMatrix Gs(n_c, m_p);
    Me = 0;
    for(int i=0;i<n_s; ++i){
        Gs = G(_, _(i*m_p+1, (i + 1)*m_p));
        for(int j=1;j<=m_p; ++j)
            Me += Gs(_, j) * transpose(J(_(j, j + n_t - 1), i+1));
    }
}

double MxNE::duality_gap(const Maths::DMatrix &G,const Maths::DMatrix &M,
    Maths::DMatrix &J, Maths::DMatrix &R, double alpha) const {
    /* compute the duality gap for mixed norm estimate gap = Fp-Fd;
    // between the primal and dual functions. Check reference papers for more
    // details.
    */
    Maths::DMatrix GtR(n_t, m_p*n_s);
    Compute_GtR(G, R, GtR);
    double norm_GtR = 0.0;
    for (int ii =0; ii < n_s; ii++){
        double GtR_axis1norm = 0.0;
        cxxblas::nrm2(n_t*m_p, &GtR.data()[ii*n_t*m_p], 1, GtR_axis1norm);
        if (GtR_axis1norm > norm_GtR)
            norm_GtR = GtR_axis1norm;
    }
    double R_norm, gap, s;
    cxxblas::nrm2(n_t*n_c, &R.data()[0], 1, R_norm);
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
    cxxblas::dot(n_c*n_t, &M.data()[0], 1, &R.data()[0], 1, ry_sum);
    
    double l21_norm = 0.0;
    for (int i =0; i<n_s; ++i){
        double r = 0.0;
        cxxblas::nrm2(n_t_s, &J.data()[i*n_t_s], 1, r);
        l21_norm += r;
    }
    gap += alpha * l21_norm - s * ry_sum;
    return gap;
}

int MxNE::MxNE_solve(const Maths::DMatrix &M, const Maths::DMatrix &GA,
    Maths::DMatrix &J, double alpha, int n_iter, double &dual_gap_,
    double &tol, bool initial) const {
    /* Compute the mixed norm estimate i.e.
    // Objective F(X) = \sum_{t=1-T}||M_t-sum_i{1-p} G_i X_{t-i}|| +
    //                                                         alpha ||X||_{21}
    // check reference papers
    */
    using namespace flens;
    using namespace std;
    Underscore<Maths::DMatrix::IndexType> _;
    double d_w_ii = 0, d_w_max = 0, W_ii_abs_max = 0, w_max  = 0.0;
    double n_x;
    Maths::DMatrix R(n_c, n_t);
    Maths::DVector mu(n_s);
    R = M;
    cxxblas::nrm2(n_t*n_c, &M.data()[0], 1, n_x);
    Compute_mu(GA, mu);
    tol = d_w_tol*n_x;
    dual_gap_ = 0.0;
    if (not initial)
        J = 0.0;
    else{
        Maths::DMatrix Me(n_c, n_t);
        Compute_Me(GA, J, Me);
        R -= Me;
    }
    Maths::DVector mu_alpha(n_s);
    for (int i=1;i<=n_s;i++)
        mu_alpha(i) = mu(i)*alpha;
    int ji;
    for (ji = 0; ji < n_iter; ji++){
        w_max = 0.0;
        d_w_max = 0.0;
        for (int i = 1; i <= n_s; ++i){
            Maths::DVector dX(n_t_s);
            Maths::DVector wii(n_t_s);
            wii = J(_, i);
            Compute_dX(GA, R, dX, i-1);
            J(_, i) += mu(i)*dX;
            double nn;
            cxxblas::nrm2(n_t_s, &J.data()[(i-1)*n_t_s], 1, nn);
            double s_t = 0.0;
            double s_ = std::max(nn, mu_alpha(i));
            if (s_!= 0.0)
                s_t = std::max(1.0 - mu_alpha(i)/s_, 0.0);
            J(_, i) *= s_t;
            wii -= J(_, i); // wii = X^{i-1} - X^i
            d_w_ii = absmax(wii);
            W_ii_abs_max = absmax(J(_, i));
            if (d_w_ii != 0.0)
                update_r(GA, wii, R, i-1);
            if (d_w_ii > d_w_max)
                d_w_max = d_w_ii;
            if (W_ii_abs_max > w_max)
                w_max = W_ii_abs_max;
        }
        if ((w_max == 0.0) || (d_w_max / w_max <= d_w_tol) || (ji == n_iter-1)){
            dual_gap_ = duality_gap(GA, M, J, R, alpha);
            if (dual_gap_ <= tol)
                break;
        }
    }
    if (verbose){
        if (dual_gap_ > tol)
            printf("\n Objective did not converge, you might want to increase");
        else
            printf("\n Objective converges");

        printf("\n the number of iterations (%d)", ji+1);
        printf("\n Duality gap = %.2e | tol = %.2e \n", dual_gap_, tol);
    }
    return ji+1;
}


