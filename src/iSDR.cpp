#include "iSDR.h"
//#include <cxxstd/iostream.h>
//#include <iostream>
//#include <stdio.h>
//#include <ctime>
//#include <time.h>
#include "Matrix.h"
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
    std::vector<double> Re {0, 0, 0};
    return;
}
iSDR::~iSDR() { }
void iSDR::Reorder_G(const Maths::DMatrix &GA, Maths::DMatrix &G_reorder)const{
    // reorder GxA from [GxA1,..,GxAp] to [GA1|s=1,..,GAp|s=1, .., GA1|s=n,..,
    //                                                                  GAp|s=n]
    // Input:
    //       GA (n_c x (n_s x m_p)): matrix containing Gx[A1,..,Ap]
    // Output:
    //       G_reorder (n_c x (n_s x m_p)):  reordered GA
    using namespace flens;
    //typedef GeMatrix<FullStorage<double> >   GeMatrix;
    typedef typename Maths::DMatrix::IndexType     IndexType;
    const Underscore<IndexType>  _;
    for(int y = 0;y < n_s; y++)
        for (int x = 0; x < m_p; x++){
            G_reorder(_, y*m_p+x+1) = GA(_, x*n_s+y+1);
            //for(int i = 0; i < n_c; i++)
            //   G_reorder.data()[i + (y*m_p + x)* n_c] = GA.data()[i + (y + x*n_s)*n_c];
        }
}

void iSDR::Reduce_G(const double *G, Maths::DMatrix &G_n, std::vector<int> ind)const{
    // n_s_i number of active sources
    // G (n_c x n_s)   ----->  G_n (n_c x n_s_i)
    // Input:
    //       G (n_c x (n_s x p)): G*A
    //       ind (1x n_s_i): vector containg the number of active sources
    // Output:
    //        G_n (n_c x n_s_i): gain matrix columns that correspends to active
    //                           sources.
    //        ind (1xn_s_i): vector containing label of active sources
    for (unsigned int i=0; i < ind.size(); ++i){
        int x = ind[i];
        cxxblas::copy(n_c, &G[x*n_c], 1, &G_n.data()[i*n_c], 1);
    }
}

void iSDR::Reduce_SC(const int *SC, Maths::IMatrix &SC_n, std::vector<int> ind)const{
    // reduce the strucural connectivity matrix by considering only active
    // sources
    // Input:
    //       SC (n_sxn_s): structural connectivity matrix
    // Output:
    //       SC_n (n_s_i x n_s_i): reduced structural connectivity matrix
    //       ind (1 x n_s_i): vector containing active sources.
    int si = ind.size();
    for (int i=0;i<si; ++i){
        int x = ind[i];
        for (int j=0;j<si; ++j){
            int y = ind[j];
            SC_n.data()[i*si+j] = SC[x*n_s + y];
        }
    }
}

void iSDR::G_times_A(const Maths::DMatrix &G, const Maths::DMatrix &A,
            Maths::DMatrix &GA_reorder) const{
    // compute Gx[A1,..,Ap] and then reorder it to:
    // [GA1|s=1,..,GAp|s=1, .., GA1|s=n,..,GAp|s=n]
    // Input:
    //       G (n_c x n_s): gain matrix which project brain activity into
    //                      eeg/meg sensor space.
    //       A (n_sx(n_sxm_p)): MVAR coefficients.
    // Output:
    //       GA_reorder (n_c x (n_sx m_p)): matrix containg GxA reordered.
    using namespace flens;
    typedef typename Maths::DMatrix::IndexType     IndexType;
    const Underscore<IndexType>  _;
    Maths::DMatrix GA(n_c, n_s);
    for (int i = 0 ;i< m_p; ++i){
        GA = G*A(_, _(i*n_s+1, n_s*(i+1)));  
        for (int j =0; j< n_s; ++j)
            GA_reorder(_, j*m_p+i+1) = GA(_, j+1);
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
    typedef typename Maths::DMatrix::IndexType     IndexType;
    const Underscore<IndexType>  _;
    int n_x = n_t_s - 2*m_p;
    for (int source = 0; source < n_s; ++source){
        std::vector<int> ind_X;
        for (int j=0; j < n_s; ++j){
            if (A_scon[source*n_s + j] != 0.0)
                ind_X.push_back(j);
        }
        int n_connect = ind_X.size();
        Maths::DMatrix A(n_x, n_connect*m_p);
        Maths::DMatrix A_(n_connect*m_p, n_x);
        int ixy = n_connect*m_p;
        Maths::DVector y(std::max(n_x, ixy));
        for (int j = 0;j < n_x; ++j){
            for (int l=0;l < m_p; ++l){
                int x = l*n_connect;
                int m = j + l + m_p;
                for (int k = 0;k < n_connect; k++){
                    A(j+1, k+1+x)= S[ind_X[k]*n_t_s + m];
                    //A_(k+1+x, j+1) = A(j+1, k+1+x);
                }
            }
            y(j+1) = S[source*n_t_s + 2*m_p + j];
        }
        A_ = transpose(A);
        Maths::DVector solution(ixy);
        if (n_connect > 1){
            Maths::DMatrix     ATA(ixy, ixy);
            Maths::DMatrix     ATA2(ixy, ixy);
            ATA = A_*A;
            ATA2 = A_*A;
            Maths::DMatrix L(ixy, ixy);
            for (int i=0;i<ixy; ++i)
                L(i+1, i+1) = 1.0;
            lapack::ls(NoTrans, ATA2, L);
            Maths::DMatrix Check(ixy, ixy);
            Check = ATA*L;
            Maths::DVector ATB(ixy);
            cxxblas::gemm(cxxblas::ColMajor,cxxblas::Trans, cxxblas::NoTrans,
            ixy, 1, n_x, 1.0, &A.data()[0], n_x, &y.data()[0], n_x, 0.0,
            &ATB.data()[0], ixy);
            solution = ATB*L;
        }
        else{
            lapack::ls(NoTrans, A, y);
            for (int q=1;q<=m_p;++q)
                solution(q) = y(q);
        }
        /*double max_z = 0;
        for (int q=1;q<=ixy;++q){
            double qz = std::abs(solution(q));
            if (qz > max_z){
                max_z = qz; 
            }
        }
        
        double threshold = tol*max_z;
        * */
        for (int j=0;j<m_p; ++j){
            int block = j*n_s*n_s;
            for (int k=0;k<n_connect; ++k){
                int s = ind_X[k];
                //if (std::abs(solution(k+j*n_connect + 1)) < threshold)
                //    solution(k+j*n_connect + 1) = 0;
                VAR[source+s*n_s + block] = solution(k+j*n_connect + 1);
            }
        }
    }
}

void iSDR::GA_removeDC(Maths::DMatrix &GA) const {
    using namespace flens;
    typedef typename Maths::DMatrix::IndexType     IndexType;
    const Underscore<IndexType>  _;
    for (int i=0;i<n_s; ++i){
        double x = 0;
        for (int j=0;j<n_c*m_p; ++j)
            x += GA.data()[i*n_c*m_p + j];
        x /= n_c*m_p;
        GA(_, _(i*m_p+1, (i+1)*m_p)) -= x;
    }
}

void iSDR::Depth_comp(Maths::DMatrix &GA) const {
    using namespace flens;
    typedef typename Maths::DMatrix::IndexType     IndexType;
    const Underscore<IndexType>  _;
    for (int i=0;i<n_s; ++i){
        double x;
        cxxblas::nrm2(n_c*m_p, &GA.data()[i*n_c*m_p], 1, x);
        GA(_, _(i*m_p+1, (i+1)*m_p)) /= x;
    }
}

std::vector<int> iSDR::Zero_non_zero(const Maths::DMatrix &S)const{
    // Get active sources
    // Input:
    //       S (n_t_s x n_s): the matrix containg brain activity.
    // Output:
    //       ind_x : vector containg the label of active sources.
    std::vector<int> ind_x;
    for(int i = 0; i < n_s; ++i){
        double ix;
        cxxblas::nrm2(n_t_s, &S.data()[i*n_t_s], 1, ix);
        if (ix > 0.0)
            ind_x.push_back(i);
    }
    return ind_x;
}

int iSDR::iSDR_solve(const Maths::DMatrix &G_o, const Maths::IMatrix &SC,
    const Maths::DMatrix &M, const Maths::DMatrix &G, Maths::DMatrix &J,
    Maths::DMatrix &Acoef, Maths::IVector &Active, bool initial,
    bool with_alpha){
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
    using namespace flens;
    using namespace std;
    Underscore<Maths::DMatrix::IndexType> _;
    std::vector<int> v1;
    for (int i = 0; i < n_s; i++)
        v1.push_back(i);
    std::vector<int> v2;
    Maths::DMatrix Gx(n_c, n_s*m_p);
    Reorder_G(G, Gx);// reorder gain matrix
    double * G_o_ptr = new double[n_c*n_s];
    int * SC_ptr = new int [n_s*n_s];
    cxxblas::copy(n_c*n_s, &G_o.data()[0], 1, &G_o_ptr[0], 1);
    cxxblas::copy(n_s*n_s, &SC.data()[0], 1, &SC_ptr[0], 1);
    MxNE _MxNE(n_s, n_c, m_p, n_t, d_w_tol, verbose);
    if (not with_alpha){
        double alpha_max = _MxNE.Compute_alpha_max(Gx, M);
        alpha_max *= 0.01;
        alpha *= alpha_max;
    }
    double n_M, n_Me, M_tol;
    cxxblas::nrm2(n_t*n_c, &M.data()[0], 1, n_M);
    n_Me = n_M;
    M_tol = 1e-2;
    double dual_gap_;
    double tol;
    Maths::DMatrix Jtmp(n_t_s, n_s);
    double * GA_i_ = new double [n_c*n_s*m_p];
    cxxblas::copy(n_c*n_s*m_p, &Gx.data()[0], 1, &GA_i_[0], 1);
    Jtmp = J;
    for (int ii = 0; ii < n_isdr; ii++){
        v2.clear();
        dual_gap_= 0.0;
        tol = 0.0;
        Maths::DMatrix GA_i(n_c, n_s*m_p);
        cxxblas::copy(n_c*n_s*m_p, &GA_i_[0], 1, &GA_i.data()[0], 1);
        Maths::DMatrix J_i(n_t_s, n_s);
        cxxblas::copy(n_t_s*n_s, &Jtmp.data()[0], 1, &J_i.data()[0], 1);
        _MxNE.MxNE_solve(M, GA_i, J_i, alpha, n_mxne, dual_gap_, tol, initial);
        std::vector<int> ind_x;
        ind_x = Zero_non_zero(J_i);
        int n_s_x = ind_x.size();
        for (int i=0; i < n_s_x;i++)
            v2.push_back(v1[ind_x[i]]);
        v1 = v2;
        Re[2] = dual_gap_;
        Re[1] = ii;
        Re[0] = n_s_x;
        if (n_s == n_s_x){
            J(_, _(1, n_s)) = J_i;
            n_s = n_s_x;
            if (verbose)
                printf("Same active set (%d) is detected in 2 successive iterations.\n", n_s);
            break;
        }
        if (n_s_x == 0) {
            n_s = 0;
            if (verbose)
                printf("No active source. You may decrease alpha = %2e \n", alpha);
            break;
        }
        Jtmp = 0;
        for(int i = 0;i < n_s_x; ++i){
            int ix = n_t_s*ind_x[i];
            cxxblas::copy(n_t_s, &J_i.data()[ix], 1, &Jtmp.data()[i*n_t_s], 1);
        }
        J = Jtmp;
        Maths::DMatrix G_tmp(n_c, n_s_x);
        Maths::IMatrix SC_n_x(n_s_x, n_s_x);
        Reduce_G(&G_o_ptr[0], G_tmp, ind_x);
        cxxblas::copy(n_c*n_s_x, &G_tmp.data()[0], 1, &G_o_ptr[0], 1);
        Reduce_SC(&SC_ptr[0], SC_n_x, ind_x);
        cxxblas::copy(n_s_x*n_s_x, &SC_n_x.data()[0], 1, &SC_ptr[0], 1);
        n_s = n_s_x;
        Maths::DMatrix MVAR(n_s, n_s*m_p);
        Maths::DMatrix J_(n_t_s, n_s);
        cxxblas::copy(n_t_s*n_s, &Jtmp.data()[0], 1, &J_.data()[0], 1);
        A_step_lsq(&J_.data()[0], &SC_ptr[0], mar_th, &MVAR.data()[0]);
        double EigMax = Phi_TransitionMatrix(MVAR);
        if (EigMax > 1)
            MVAR *= 1/(EigMax*(1+1e-6));
        EigMax = Phi_TransitionMatrix(MVAR);
        Maths::DMatrix Gt(n_c, n_s*m_p);
        G_times_A(G_tmp, MVAR, Gt);
        //GA_removeDC(Gt);
        //Depth_comp(Gt);
        cxxblas::copy(n_c*n_s*m_p, &Gt.data()[0], 1, &GA_i_[0], 1);
        cxxblas::copy(n_s*n_s*m_p, &MVAR.data()[0], 1, &Acoef.data()[0], 1);
        cxxblas::copy(n_s, &v1[0], 1, &Active.data()[0], 1);
        _MxNE.n_s = n_s;
        Maths::DMatrix Me(n_c, n_t);
        _MxNE.Compute_Me(Gt, Jtmp, Me);
        Me -= M;
        cxxblas::nrm2(n_t*n_c, &Me.data()[0], 1, n_Me);
        if (verbose){
            std::cout<<"Number of active regions/sources = "<<n_s<<std::endl;
            std::cout<<"Max Eigenvalue after norm "<< EigMax <<std::endl;
        }
        if ((n_Me/n_M) < M_tol){
            std::cout<<"Stop iSDR: small residual = "<<(n_Me/n_M)*100.<<" %"
            <<std::endl;
            break;
        }
    }
    delete[] G_o_ptr;
    delete[] SC_ptr;
    delete[] GA_i_;
    return n_s;
}


double iSDR::Phi_TransitionMatrix(Maths::DMatrix &MVAR)const{
    using namespace flens;
    using namespace std;
    Underscore<Maths::DMatrix::IndexType> _;
    int n_col = MVAR.numCols();
    int ns = MVAR.numRows();
    Maths::DMatrix Phi(n_col, n_col);
    for (int i=0;i<m_p;++i){
        int x = m_p-i-1;
        int y = m_p-i;
        Phi(_(1, ns), _(i*ns+1, (i+1)*ns)) = MVAR(_, _(x*ns+1, y*ns));
    }
    for (int i=ns+1;i<=m_p*ns;++i){
        Phi(i, i-ns) = 1.0;
    }
    int n=n_s*m_p;
    Maths::DMatrix   VL(n, n), VR(n, n);
    Maths::DVector   wr(n), wi(n);
    Maths::DVector   work;
    lapack::ev(true, true, Phi, wr, wi, VL, VR, work);
    double EigenMax = 0;
    for (int i=1;i<=n;++i){
        double a = wr(i)*wr(i) + wi(i)*wi(i);
        double z = std::sqrt(a);
        if (z > EigenMax)
            EigenMax = z;
    }
    return EigenMax;
}
