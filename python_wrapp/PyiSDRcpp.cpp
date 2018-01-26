#include "PyiSDRcpp.h"
#include "Matrix.h"
#include <flens/flens.cxx>
#include <stdio.h>
#include "iSDR.h"

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


PyiSDRcpp::PyiSDRcpp(int n_iter_mxne,int n_iter_iSDR, double alpha_x,double d_w_tol,
bool initial, bool ver){
    this-> n_iter_mxne = n_iter_mxne;
    this-> n_iter_iSDR = n_iter_iSDR;
    this-> initial = initial;
    this-> ver = ver;
    this-> alpha_x = alpha_x;
    this-> d_w_tol = d_w_tol;
}
int PyiSDRcpp::fit(double * go,int n_go, int * sc,int n_sc, double * meg, int n_meg, double * g,
int n_g, double * js, int n_js, double * coef, int n_coef, int * ac, int n_ac, double *wt, int n_wt){
    using namespace flens;
    using namespace std;
    int n_s = std::sqrt(n_sc);
    int n_c = n_go/n_s;
    int m_p = n_g/(n_c*n_s);
    int n_t = n_meg/n_c;
    int n_t_s = n_t + m_p - 1;
    if (ver){
        std::cout<<"N sources "<<n_s<<std::endl;
        std::cout<<"N sensors "<<n_c<<std::endl;
        std::cout<<"MAR order "<<m_p<<std::endl;
        std::cout<<"N samples "<<n_t<<std::endl;
        std::cout<<"alpha in % "<<alpha_x<<std::endl;
    }
    Maths::DMatrix G_o(n_c, n_s);
    Maths::IMatrix SC(n_s, n_s);
    Maths::DMatrix M(n_c, n_t);
    Maths::DMatrix G(n_c, n_s*m_p);
    Maths::DMatrix J(n_t_s, n_s);
    Maths::DMatrix Acoef(n_s, n_s*m_p);
    Maths::IVector Active(n_s);
    Maths::DVector Wt(n_s);
    for (int i=0; i<n_go; i++)
        G_o.data()[i] = go[i];
    for (int i=0; i<n_sc; i++)
        SC.data()[i] = (int)sc[i];
    for (int i=0; i<n_meg; i++)
        M.data()[i] = meg[i];
    for (int i=0; i<n_g; i++)
        G.data()[i] = g[i];
    double mar_th = 1e-3;
    bool with_alpha=false;
    iSDR _iSDR(n_s, n_c, m_p, n_t, alpha_x, n_iter_mxne, n_iter_iSDR, d_w_tol, mar_th, ver);
    int n_active = _iSDR.iSDR_solve(G_o, SC, M, G, J, Acoef, Active, Wt, initial, with_alpha);
    for (int i=0; i<n_coef; i++)
        coef[i] = Acoef.data()[i];
    for (int i=0; i<n_ac; i++)
        ac[i] = Active.data()[i];
    for (int i=0; i<n_js; i++)
        js[i] = J.data()[i];
    for (int i=0; i<n_active; i++)
        wt[i] = Wt.data()[i];
    return n_active;
}
