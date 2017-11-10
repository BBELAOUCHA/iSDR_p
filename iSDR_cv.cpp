#include <cxxstd/iostream.h>
#include <flens/flens.cxx>
#include "matio.h"
#include <cmath>
#include <vector>
#include "iSDR.h"
#include <stdlib.h>
#include "ReadWriteMat.h"
#include <omp.h>
#include <algorithm>
#include <random>   // for default_random_engine & uniform_int_distribution<int>
#include <chrono>   // to provide seed to the default_random_engine
#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60
using namespace flens;
using namespace std;


void printProgress (double percentage){
    int val = (int) (percentage * 100);
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
    fflush(stdout);
}

default_random_engine dre (chrono::steady_clock::now().time_since_epoch().count());     // provide seed
int RANDOM (int lim)
{
    uniform_int_distribution<int> uid {0,lim};   // help dre to generate nos from 0 to lim (lim included);
    return uid(dre);    // pass dre as an argument to uid to generate the random no
}

int WriteData(const char *file_path, Maths::DVector &alpha, Maths::DMatrix &cv_fit_data,
                double alpha_max){
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
    mat = Mat_Create(file_path,NULL);
    if(mat){
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
        printf("Failed to save results in %s", file_path);
        return 1;
    }
    return 0;
}
void print_args(const int argc,char* argv[]) {
    for (int i=0;i<argc;++i)
        std::cerr << argv[i] << ' ';
    std::cerr << std::endl;
}
void explain_para(){
    printf( " ./iSDR  arg1 arg2 arg3 arg4 arg5 arg6 arg7 arg8\n");
    printf( "      arg1 : path to mat file that contains MEG/EEG, G, GA\n");
    printf( "      arg2 : min value of regularization parameter >0. \n");
    printf( "      arg3 : max value of regularization parameter <100. \n");
    printf( "      arg4 : N of reg parameters in the range [alpha_min, alpha_max]. \n");
    printf( "      arg5 : N of KFold. \n");
    printf( "      arg6 : N of repetions of the KFold for each alpha_i. \n");
    printf( "      arg7 : where to save results. \n");
    printf( "      arg8 : verbose. \n");
}

void printHelp(){
    printf("\n--help or -h of the iterative source and dynamics reconstruction algorithm.\n");
    printf(" This code uses KFold cross-validation technique to choose the regularization parameter\n");
    printf(" of iSDR. The actual version of the code needs 9 inputs:\n");
    printf(" The output of this function is a mat file that contains the following:\n");
    printf("    Alpha: vector (arg4) contains the alpha's used in the kFold\n");
    printf("    CV data fit: a matrix (arg5xarg4) which contains the crossvalidation fit error value\n");
    printf("    Alpha max: a scaler. the smallest alpha that gives an empty active set\n");
    explain_para();
}

void print_param(int n_s, int n_t, int n_c, int m_p, double alpha,
                 double d_w_tol){
    printf(" N of sensors %d\n", n_c);
    printf(" N of sources %d\n", n_s);
    printf(" N of samples %d\n", n_t);
    printf(" MAR model    %d\n", m_p);
    printf(" iSDR tol   %.3e\n", d_w_tol);
    printf(" iSDR (p : =  %d with alpha : = %.2f%%\n", m_p, alpha);
}


int main(int argc, char* argv[]){
    std::string str1 ("-h");
    std::string str2 ("--help");
    if (str1.compare(argv[1]) == 0 || str2.compare(argv[1]) == 0){
        printHelp();
        return 1;
    }
    if(argc < 9){
        printf("Missing arguments:\n");
        explain_para();
        return 1;
    }
    print_args(argc,argv);
    const bool verbose = (atoi(argv[8]) == 1) ? true : false;
    Underscore<Maths::DMatrix::IndexType> _;
    int n_c = 306;int n_s = 600;int m_p = 3;int n_t = 297;
    int n_iter_mxne = 10000;int n_iter_iSDR = 100;
    const char *file_path = argv[1];
    double alpha_min = atof(argv[2]);
    double alpha_max_ = atof(argv[3]);
    int n_alpha = atoi(argv[4]);
    int Kfold = atoi(argv[5]);
    int n_Kfold = atoi(argv[6]);
    const char *save_path = argv[7];
    double d_w_tol=1e-7;
    int re_use = 1;
    int n_t_s = n_t + m_p - 1;
    ReadWriteMat _RWMat(n_s, n_c, m_p, n_t);
    _RWMat.Read_parameters(file_path);
    n_s = _RWMat.n_s;
    n_c = _RWMat.n_c;
    m_p = _RWMat.m_p;
    n_t = _RWMat.n_t;
    n_t_s = _RWMat.n_t_s;
    int block = n_c / Kfold;
    if (verbose){
        std::cerr<<n_alpha <<" values of alpha in["<<alpha_min<<", "<< alpha_max_<<"]"<<std::endl;
        std::cerr<<"KFold = "<<Kfold<<std::endl;
        std::cerr<<"Input file: "<<file_path<<std::endl;
        std::cerr<<"Output file: "<<save_path<<std::endl;
        std::cerr<<"Block size = "<<block<<std::endl;
    }
    Maths::DMatrix G_o(n_c,n_s);
    Maths::DMatrix GA_initial(n_c, n_s*m_p);
    Maths::DMatrix M(n_c, n_t);
    Maths::IMatrix SC(n_s,n_s);
    bool use_mxne = false;
    if (re_use==1)
        use_mxne = true;
    _RWMat.ReadData(file_path, G_o, GA_initial, M, SC);
    double mvar_th = 1e-3;
    Maths::DVector ALPHA(n_alpha);
    double alp_step = (alpha_max_ - alpha_min)/(float)n_alpha;
    for (int y=1; y<= n_alpha;y++){
        ALPHA(y) = alpha_min + (y-1)*alp_step;
    }
    Maths::DMatrix GA_reorder(n_c, n_s*m_p);
    iSDR _iSDR(n_s, n_c, m_p, n_t, 1.0, n_iter_mxne, n_iter_iSDR, d_w_tol,
    mvar_th, verbose);
    _iSDR.Reorder_G(GA_initial, GA_reorder);// reorder gain matrix
    MxNE _MxNE(n_s, n_c, m_p, n_t, d_w_tol, verbose);
    double alpha_max = _MxNE.Compute_alpha_max(GA_reorder, M);
    Maths::DMatrix cv_fit_data(n_alpha, n_Kfold);
    Maths::DVector  alpha_real(n_alpha);
    for (int x=1;x<=n_alpha;x++)
        alpha_real(x) = 0.01*alpha_max*ALPHA(x);
    int n_cpu = omp_get_num_procs();

    cerr<<"Cross-validation of iSDR uses "<<n_cpu<<" cpus"<<std::endl;
    int x, r_s;
    double m_norm;
    cxxblas::nrm2(n_t*n_c, &M.data()[0], 1, m_norm);
    m_norm *= m_norm/n_c;

    std::random_device rd;
    std::mt19937 generator(rd());

    int iter_i = 0;
    #pragma omp parallel for default(shared) private(r_s, x) collapse(2) \
    num_threads(n_cpu)
    for (r_s = 0; r_s<n_Kfold; ++r_s){
        for (x = 0; x< n_alpha ; ++x){
            double alpha = alpha_real(x+1);
            std::vector<int> sensor_list(n_c);
            std::iota(sensor_list.begin(),sensor_list.end(),0);
            std::shuffle(sensor_list.begin(),sensor_list.end(),generator);
            std::vector<int>::iterator fold = sensor_list.begin();
            double error_cv_alp = 0.0;
            for (int i=0; i<Kfold; i++){
                Maths::DMatrix J(n_t_s, n_s);
                Maths::DMatrix Acoef(n_s, n_s*m_p);
                Maths::IVector Active(n_s);

                std::vector<int> sensor_kfold;
                const unsigned set = (std::distance(fold,sensor_list.end())>=block) ? block : std::distance(fold,sensor_list.end());
                std::vector<int>::iterator fold_end = fold+set;
                std::sort(fold, fold_end);
                for (std::vector<int>::iterator kk=fold;kk!=fold_end;++kk)
                    std::cerr << *kk << ' ';
                std::cerr << endl;
                int set_i = n_c - set;
                Maths::DMatrix Mn(set_i, n_t);
                Maths::DMatrix G_on(set_i, n_s);
                Maths::DMatrix GA_n(set_i,n_s*m_p);
                int z=1;
                for (int j=0; j<n_c; j++){
                    if (not (std::find(fold, fold_end, j) != fold_end)){
                        Mn(z, _) = M(j+1, _);
                        GA_n(z, _) = GA_initial(j+1, _);
                        G_on(z, _) = G_o(j+1, _);
                        z += 1;
                    }
                }
                iSDR _iSDR_(n_s, set_i, m_p, n_t, alpha, n_iter_mxne,
                n_iter_iSDR, d_w_tol, mvar_th, false);
                int n_s_e = _iSDR_.iSDR_solve(G_on, SC, Mn, GA_n, J,
                &Acoef.data()[0], &Active.data()[0], use_mxne, true);
                Maths::DMatrix  Mcomp(set, n_t);
                for (int k =0;k<set;k++)
                    Mcomp(k+1, _) = M(*(fold+k)+1, _);
                double cv_k;
                cxxblas::nrm2(n_t*set, &Mcomp.data()[0], 1, cv_k);
                if (n_s_e > 0){
                    Maths::DMatrix Gx(set, n_s_e);
                    for (int t =0;t<n_s_e;++t)
                        for (int y=0;y<set;++y)
                            Gx(y+1, t+1) = G_o(*(fold+y)+1, Active(t+1)+1);
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
            }
            #pragma omp atomic
            iter_i += 1;
            error_cv_alp /= Kfold;
            cv_fit_data(x+1, r_s+1) = error_cv_alp;
            double tps = (double)iter_i/(n_alpha*n_Kfold);
            if (verbose)
                printProgress(tps);
        }
    }
    if (verbose){
        cout<<"\n          ***************************************************"<<endl;
        cout<<"          ****      iSDR cross validation finished       ****"<<endl;
        cout<<"          ***************************************************"<<endl;
    }
    WriteData(save_path, alpha_real, cv_fit_data, alpha_max);
    return 0;
}
