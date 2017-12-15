#include "MxNE.h"
#include "iSDR.h"
#include "CV_iSDR.h"
#include <omp.h>
#include "Matrix.h"
#include <random>
#include <chrono>
#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60
////============================================================================
////============================================================================
/////
///// \file CV_iSDR.cpp
/////
///// \brief Compute the K-Fold crossvalidation of the iSDR EEG/MEG inverse 
/////        solver 
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

CV_iSDR::CV_iSDR( int Kfold, double d_w_tol, bool verbose, bool use_mxne){
    this-> d_w_tol= d_w_tol;
    this-> verbose = verbose;
    this-> Kfold = Kfold;
    this-> use_mxne = use_mxne;
}
void CV_iSDR::printProgress (double percentage){
    int val = (int) (percentage * 100);
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
    fflush(stdout);
}

double CV_iSDR::Run_CV(const Maths::DMatrix &M, const Maths::DMatrix &G_o,
    const Maths::DMatrix &GA_initial, const Maths::IMatrix &SC, const Maths::DVector &ALPHA,
    Maths::DVector &alpha_real, Maths::DMatrix &cv_fit_data){
    using namespace flens;
    typedef typename Maths::DMatrix::IndexType     IndexType;
    const Underscore<IndexType>  _;
    int n_alpha = alpha_real.length();
    int n_c = M.numRows();
    int n_s = SC.numRows();
    int n_t = M.numCols();
    int m_p = (int) GA_initial.numCols()/n_s;
    int n_t_s = 0;
    n_t_s = n_t + m_p - 1;
    int n_iter_mxne = 10000;
    int n_iter_iSDR = 100;
    int n_Kfold = cv_fit_data.numCols();
    double mvar_th = 1e-3;
    int block = n_c / Kfold;
    iSDR _iSDR(n_s, n_c, m_p, n_t, 1.0, n_iter_mxne, n_iter_iSDR, d_w_tol,
    mvar_th, verbose);
    Maths::DMatrix GA_reorder(n_c, n_s*m_p);
    _iSDR.Reorder_G(GA_initial, GA_reorder);
    MxNE _MxNE(n_s, n_c, m_p, n_t, d_w_tol, verbose);
    double alpha_max = _MxNE.Compute_alpha_max(GA_reorder, M);
    for (int x=1;x<=n_alpha;x++)
        alpha_real(x) = 0.01*alpha_max*ALPHA(x);
    int n_cpu = omp_get_num_procs();
    int x, r_s;
    double m_norm;
    cxxblas::nrm2(n_t*n_c, &M.data()[0], 1, m_norm);
    m_norm *= m_norm/n_c;
    std::random_device rd;
    std::mt19937 generator(rd());
    int iter_i = 0;
    #pragma omp parallel for default(shared) private(r_s, x) collapse(2) \
    num_threads(n_cpu)
    for (r_s = 1; r_s<=n_Kfold; ++r_s){
        for (x = 1; x <= n_alpha ; ++x){
            double alpha = alpha_real(x);
            std::vector<int> sensor_list(n_c);
            std::iota(sensor_list.begin(), sensor_list.end(),0);
            std::shuffle(sensor_list.begin(),sensor_list.end(),generator);
            std::vector<int>::iterator fold = sensor_list.begin();
            double error_cv_alp = 0.0;
            for (int i=0; i<Kfold; i++){
                Maths::DMatrix J(n_t_s, n_s);
                Maths::DMatrix Acoef(n_s, n_s*m_p);
                Maths::IVector Active(n_s);
                std::vector<int> sensor_kfold;
                const int set = (i!=Kfold-1) ? block : std::distance(fold,sensor_list.end());
                std::vector<int>::iterator fold_end = fold+set;
                std::sort(fold, fold_end);
                int set_i = n_c - set;
                Maths::DMatrix Mn(set_i, n_t);
                Maths::DMatrix G_on(set_i, n_s);
                Maths::DMatrix GA_n(set_i,n_s*m_p);
                int z=1;
                for (int j=1; j<=set_i; j++){
                    if (not (std::find(fold, fold_end, j-1) != fold_end)){
                        Mn(z, _) = M(j, _);
                        GA_n(z, _) = GA_initial(j, _);
                        G_on(z, _) = G_o(j, _);
                        z += 1;
                    }
                }
                iSDR _iSDR_(n_s, set_i, m_p, n_t, alpha, n_iter_mxne,
                n_iter_iSDR, d_w_tol, mvar_th, false);
                int n_s_e = _iSDR_.iSDR_solve(G_on, SC, Mn, GA_n, J, Acoef,
                            Active, use_mxne, true);
                Maths::DMatrix  Mcomp(set, n_t);
                for (int k =0;k<set;k++)
                    Mcomp(k+1, _) = M(*(fold+k)+1, _);
                double cv_k;
                cxxblas::nrm2(n_t*set, &Mcomp.data()[0], 1, cv_k);
                if (n_s_e > 0){
                    Maths::DMatrix Gx(set, n_s_e);
                    for (int t =1;t<=n_s_e;++t)
                        for (int y=0;y<set;++y)
                            Gx(y+1, t) = G_o(*(fold+y)+1, Active(t)+1);
                    Maths::DMatrix GA_es(set, n_s_e);
                    Maths::DMatrix X(n_t_s, n_s_e);
                    Maths::DMatrix GA(set, n_s_e);
                    Maths::DMatrix MAR(n_s_e, n_s_e*m_p);
                    cxxblas::copy(n_s_e*n_s_e*m_p, &Acoef.data()[0], 1,
                    &MAR.data()[0], 1);
                    X = J(_, _(1, n_s_e));
                    for (int p =0;p<m_p;p++){
                        GA_es = Gx * MAR(_, _(p*n_s_e+1, (p+1)*n_s_e));
                        Mcomp -= GA_es*transpose(X(_(p+1, n_t+p),_));
                    }
                    cxxblas::nrm2(n_t*set, &Mcomp.data()[0], 1, cv_k);
                }
                cv_k *= cv_k/set;
                error_cv_alp += cv_k;
                fold += set;
            }
            sensor_list.clear();
            #pragma omp atomic
            iter_i += 1;
            error_cv_alp /= Kfold;
            cv_fit_data(x, r_s) = error_cv_alp;
            double tps = (double)iter_i/(n_alpha*n_Kfold);
            if (verbose)
                printProgress(tps);
        }
    }
    if (verbose){
        std::cout<<"\n          ***************************************************"<<std::endl;
        std::cout<<"          ****      iSDR cross validation finished       ****"<<std::endl;
        std::cout<<"          ***************************************************"<<std::endl;
    }
    return alpha_max;
}


int CV_iSDR::WriteData(const char *file_path, const Maths::DVector &alpha,
    const Maths::DMatrix &cv_fit_data, double alpha_max){
    /* This function write the results of the K-fold cross-validation into a
     * mat file.
     *      file_path: file name and location to where you wanna write results
     *      alpha: a 1D array (n_alpha) containing the alpha values used in the 
     *             crossvalidation
     *      cv_fit_data: 2D array (n_Kfoldxn_alpha) containing the fit error of
     *                   each alpha with different runs
     *      alpha_max: maximum value of alpha in which all sources/regions are
     *                 inactive
     *      n_alpha: length of 1D array "alpha"
     *      n_Kfold: number of runs for each alpha value.
     * */
    int n_alpha = cv_fit_data.numRows();
    int n_Kfold = cv_fit_data.numCols();
    double vec1[n_alpha];
    double mat1[n_alpha][n_Kfold];
    double sca1[1];
    int j,i;
    for(j=0;j<n_alpha;j++){
      vec1[j] = alpha(j+1);
    }
    for(j=0;j<n_alpha;j++){
        for(i=0;i<n_Kfold;i++)
            mat1[j][i] = cv_fit_data(j+1, i+1);
    }
    sca1[0] = alpha_max;
    /* setup the output */
    mat_t *mat;
    matvar_t *matvar;
    size_t dim1d[1] = {(unsigned int)n_alpha};
    size_t dim2d[1] = {(unsigned int)1};
    size_t dims2[2] = {(unsigned int)n_Kfold, (unsigned int)n_alpha};
    mat = Mat_Create(file_path, NULL);
    if(mat != NULL){
        matvar = Mat_VarCreate("Alpha",MAT_C_DOUBLE,MAT_T_DOUBLE,1, dim1d,
        &vec1,0);
        Mat_VarWrite( mat, matvar, MAT_COMPRESSION_NONE);
        Mat_VarFree(matvar);
        matvar = Mat_VarCreate("CV data fit", MAT_C_DOUBLE, MAT_T_DOUBLE,2,
        dims2, &mat1,0);
        Mat_VarWrite( mat, matvar, MAT_COMPRESSION_NONE);
        Mat_VarFree(matvar);
        matvar = Mat_VarCreate("Alpha max",MAT_C_DOUBLE,MAT_T_DOUBLE,1, dim2d,
        &sca1,0);
        Mat_VarWrite( mat, matvar, MAT_COMPRESSION_NONE);
        Mat_VarFree(matvar);
        Mat_Close(mat);
    }
    else{
        printf("Failed to save results in %s\n", file_path);
        printf("Wrong path or you cannt write\n");
        return 1;
    }
    return 0;
}
