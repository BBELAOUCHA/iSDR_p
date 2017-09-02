#include "iSDR.h"
//#include <omp.h>

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

iSDR::iSDR(int n_sources, int n_sensors, int Mar_model, int n_samples,
           double alpha, double n_iter_mxne, double n_iter_iSDR, double d_w_tol, 
           double mar_th, bool ver){
    this-> n_t = n_samples;
    this-> n_c = n_sensors;
    this-> n_s = n_sources;
    this-> m_p = Mar_model;
    this-> n_t_s = n_t + m_p - 1;
    this-> alpha = alpha;
    this-> n_mxne = n_iter_mxne;
    this-> n_isdr = n_iter_iSDR;
    this-> d_w_tol= d_w_tol;
    this-> mar_th = mar_th;
    this-> verbose = ver;
    std::vector<double> Re {0,0,0};
    return;
}
iSDR::~iSDR() { }
void iSDR::Reorder_G(const double *GA, double *G_reorder)const{
    // reorder GxA from [GxA1,..,GxAp] to [GA1|s=1,..,GAp|s=1, .., GA1|s=n,..,
    //                                                                  GAp|s=n]
    // Input:
    //       GA (n_c x (n_s x m_p)): matrix containing Gx[A1,..,Ap]
    // Output:
    //       G_reorder (n_c x (n_s x m_p)):  reordered GA
    for(int y = 0;y < n_s; ++y)
        for (int x = 0; x < m_p; ++x)
            for(int i = 0; i < n_c; ++i)
                G_reorder[i + (y*m_p + x)* n_c] = GA[i + (y + x*n_s)*n_c];
}

void iSDR::Reduce_G(const double * G, double *G_n, std::vector<int> ind)const{
    // n_s_i number of active sources
    // G (n_c x n_s)   ----->  G_n (n_c x n_s_i)
    // Input:
    //       G (n_c x (n_s x p)): G*A
    //       ind (1x n_s_i): vector containg the number of active sources
    // Output:
    //        G_n (n_c x n_s_i): gain matrix columns that correspends to active
    //                           sources.
    //        ind (1xn_s_i): vector containing label of active sources
    //#pragma omp parallel for
    for (unsigned int i=0;i<ind.size(); ++i){
        int x = ind[i];
        cxxblas::copy(n_c, &G[x*n_c], 1, &G_n[i*n_c], 1);
    }
}

void iSDR::Reduce_SC(const int * SC, int *SC_n, std::vector<int> ind)const{
    // reduce the strucural connectivity matrix by considering only active
    // sources
    // Input:
    //       SC (n_sxn_s): structural connectivity matrix
    // Output:
    //       SC_n (n_s_i x n_s_i): reduced structural connectivity matrix
    //       ind (1 x n_s_i): vector containing active sources.
    //#pragma omp parallel for
    for (unsigned int i=0;i<ind.size(); ++i){
        int x = ind[i];
        for (unsigned int j=0;j<ind.size(); ++j){
            int y = ind[j];
            SC_n[i*ind.size()+j] = SC[x*n_s + y];
        }
    }
}

void iSDR::G_times_A(const double * G, const double *A, double *GA_reorder)
                    const{
    // compute Gx[A1,..,Ap] and then reorder it to:
    // [GA1|s=1,..,GAp|s=1, .., GA1|s=n,..,GAp|s=n]
    // Input:
    //       G (n_c x n_s): gain matrix which project brain activity into
    //                      eeg/meg sensor space.
    //       A (n_sx(n_sxm_p)): MVAR coefficients.
    // Output:
    //       GA_reorder (n_c x (n_sx m_p)): matrix containg GxA reordered.
    using namespace flens;
    typedef GeMatrix<FullStorage<double, ColMajor> > GeMatrix;
    GeMatrix GA(n_c, n_s);
    //#pragma omp parallel for
    for (int i = 0 ;i< m_p; i++){
        cxxblas::gemm(cxxblas::ColMajor, cxxblas::NoTrans, cxxblas::NoTrans, n_c,
        n_s, n_s, 1.0, &G[0], n_c, &A[i*n_s*n_s], n_s, 0.0, &GA.data()[0], n_c);    
        for (int j =0; j< n_s; j++)
            cxxblas::copy(n_c, &GA.data()[j*n_c], 1, &GA_reorder[j*m_p*n_c + i*n_c], 1);
 
    }
}

void iSDR::A_step_lsq(const double * S,const int * A_scon,const double tol,
                    double * VAR)const{
    // Compute the MVAR coeficients of only active sources using least square 
    // Input:
    //       S (n_t_sxn_s): matrix containing the activation of sources (n_s)
    //                      estimated by MxNE module.
    //       A_scon (n_sxn_s): structural connectivity matrix between sources.
    //       tol (scaler) used to set small MVAR values to zero.
    // Output:
    //       VAR (n_sx(n_sxm_p)): matrix containing MVAR coefficients.
    // estimate results are written in VAR.
    using namespace flens;
    using namespace std;
    typedef GeMatrix<FullStorage<double, ColMajor> > GeMatrix;
    typedef DenseVector<Array<double> > DenseVector;
    typedef typename DenseVector::IndexType IndexType;
    const Underscore<IndexType>  _;

    int n_x = n_t_s - m_p;
    //#pragma omp parallel for
    for (int i =0;i < n_s; ++i){
        DenseVector b(n_x);
        std::vector<int> ind_X;
        for (int j=0; j < n_s; ++j){
            if (A_scon[i*n_s + j] != 0.0)
                ind_X.push_back(j);
        }
        int n_connect = ind_X.size();
        GeMatrix a(n_x, n_connect*m_p);
        //std::cerr<<" A.data()"<< a.data()<<std::endl;
        for (int j=0;j<n_x; ++j){   
            for (int l=0;l < m_p; ++l){
                for (int k=0;k<n_connect; ++k)
                    a(j+1, k + l*n_connect+1)= S[ind_X[k]*n_t_s + j + l];
            }
        }
        for (int j=0;j<n_x; ++j)
            b(j+1) = S[i*n_t_s + m_p + j];
        //cout<<a<<endl;
        auto x = b(_(1, n_connect*m_p));
        lapack::ls(NoTrans, a, b);
        //cout<<x<<endl;
        for(int j =0; j<n_connect*m_p; ++j){
            int block = j/n_connect;
            int source = j%n_connect;
            if (std::abs(x(j+1))< tol)
                x(j+1) = 0.0;
            VAR[i+ind_X[source]*n_s+block*n_s*n_s] = x(j+1);
        }
    }
}

std::vector<int> iSDR::Zero_non_zero(const double * S)const{
    // Get active sources
    // Input:
    //       S (n_t_s x n_s): the matrix containg brain activity.
    // Output:
    //       ind_x : vector containg the label of active sources.
    const double * Si = &S[0];
    std::vector<int> ind_x;
    for(int i=0;i<n_s; ++i){
        double ix;
        cxxblas::nrm2(n_t_s, Si+i*n_t_s, 1, ix);
        if (ix > 0.0){
            ind_x.push_back(i);
        }
    }
    return ind_x;
}

int iSDR::iSDR_solve(double *G_o, int *SC, const double *M, double *G,
                     double * J, double * Acoef, int * Active,bool initial){
    // Core function to compute iteratively the MxNE and MVAR coefficients.
    // Input:
    //       G_o (n_c x n_s): gain matrix M = G_o x J
    //       SC (n_s x n_s): structural connectivity between sources.
    //       M (n_c x n_t): EEG/MEG measurements.
    //       G (n_c x (n_s x m_p)): initial Gx[A1,..,Ap]
    //       initial (boolean): if true use precomputed J as initialization
    //                                to compute the residual of MxNE
    // Output:
    //        J (n_t_s x n_s): the brain activity estimated by iSDR.
    //        Acoef (n_s x (n_s x m_p)): MVAR coefficients
    //        Active (1 x n_s): label of only active sources
    std::vector<int> v1;
    for (int i=0;i<n_s; ++i)
        v1.push_back(i);
    std::vector<int> v2;
    const double * M_ptr = &M[0];
    double * G_ptr_o = &G_o[0];
    double * G_ptr = &G[0];
    double *GA_reorder;
    GA_reorder = new double [n_c*n_s*m_p];
    double * G_reorder_ptr = &GA_reorder[0];
    Reorder_G(G_ptr, G_reorder_ptr);// reorder gain matrix
    double *J_ptr = J;
    int *SC_ptr = &SC[0];
    int *SC_n = new int [n_s*n_s];
    double * G_tmp = new double [n_c*n_s];
    double * MVAR;
    MVAR = new double [n_s * n_s * m_p];
    MxNE _MxNE(n_s, n_c, m_p, n_t, d_w_tol, verbose);
    for (int ii = 0; ii < n_isdr; ++ii){
        v2.clear();
        double dual_gap_= 0.0;
        double tol = 0.0;
        _MxNE.MxNE_solve(M_ptr, G_reorder_ptr, J_ptr, alpha, n_mxne, dual_gap_,
                        tol, initial);
        std::vector<int> ind_x;
        ind_x = Zero_non_zero(J_ptr);
        int n_s_x = ind_x.size();
        for (int i=0; i < n_s_x;i++)
            v2.push_back(v1[ind_x[i]]);
        v1 = v2;
        _MxNE.n_s = n_s_x;
        Re[2] = dual_gap_;
        Re[1] = ii;
        Re[0] = n_s_x;
        if (n_s != n_s_x && n_s_x > 0){
            //#pragma omp parallel for
            for(int i = 0;i < n_s_x; ++i){
                if (i != ind_x[i]){
                    int ix = n_t_s*ind_x[i];
                    cxxblas::copy(n_t_s, J_ptr + ix, 1, J_ptr+i*n_t_s, 1);
                    std::fill(J_ptr + ix, J_ptr+ n_t_s + ix, 0.0);
                }
            }
            G_tmp = new double [n_c*n_s_x];
            Reduce_G(G_ptr_o, &G_tmp[0], ind_x);
            G_ptr_o = &G_tmp[0];
            SC_n = new int [n_s_x*n_s_x];
            Reduce_SC(SC_ptr, &SC_n[0], ind_x);
            SC_ptr = &SC_n[0];
            n_s = n_s_x;
            MVAR = new double [n_s * n_s * m_p];
            A_step_lsq(J_ptr, &SC_n[0], mar_th, MVAR); 
            G_reorder_ptr = new double [n_c*n_s*m_p];
            G_times_A(G_ptr_o, MVAR, G_reorder_ptr);
            for (int i = 0;i < n_s*n_s*m_p; i++)
                Acoef[i] = MVAR[i];
            for (int i = 0;i < n_s; i++)
                Active[i] = v1[i];
        }
        else if (n_s_x > 0){
            if (verbose)
                printf("Same active set (%d) is detected in 2 successive iterations.\n", n_s);
            break;
        }
        else {
            if (verbose)
                printf("No active source. You may decrease alpha = %f \n", alpha);
            break;
        }
    }
   return n_s; 
}

double iSDR::Eigen_CompanionMatrix(const double * MVAR) const{
    int n = n_s*n_s;
    using namespace flens;
    using namespace std;
    typedef GeMatrix<FullStorage<double, ColMajor> > GeMatrix;
    typedef DenseVector<Array<double> > DenseVector;
    typedef typename DenseVector::IndexType IndexType;
    const Underscore<IndexType>  _;
    GeMatrix Phi(n_s*m_p, n_s*m_p);
    for (int i=0;i<(m_p-1)*n_s; i++)
        Phi(n_s + i + 1, i+1) = 1;
/*
    GeMatrix A(n_s, n_s*m_p);
    for (int i=0;i<n*m_p; i++){
        A.data()[i] = MVAR[i];    
    }
    cout<<A<<endl;
*/
    for (int i=0;i<m_p; i++){
        int block = m_p - 1 - i;
        for (int j=0;j<n_s;j++){
            for(int k=0;k<n_s;k++){
               Phi(j+1, k+1+block*n_s) = MVAR[j + block*n + k*n_s];
            }
        }
    }
    DenseVector   work;
    GeMatrix  VL(n, n), VR(n, n);
    DenseVector   wr(n), wi(n), w(n);
    // cerr << "A = " << Phi << endl;
    // int     optSize = ev_wsq(true, true, A);
    // Vector  work(optSize);
    lapack::ev(true, true, Phi, wr, wi, VL, VR, work);
    w(1) = wr(1)*wr(1)+wi(1)*wi(1);
    double coef = w(1);
    for (int i=2;i<=n;i++){
        w(i) = wr(i)*wr(i)+wi(i)*wi(i);
        if (coef < w(i))
           coef = w(i);
    }
    //cout<<w<<endl;
    //std::cout<<coef<<std::endl;
    return coef;
}
