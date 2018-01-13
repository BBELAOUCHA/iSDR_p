#include <cxxstd/iostream.h>
#include <flens/flens.cxx>
#include "matio.h"
#include <cmath>
#include <vector>
#include "iSDR.h"
#include "CV_iSDR.h"
#include <stdlib.h>
#include "ReadWriteMat.h"
#include "Matrix.h"
#include <omp.h>
#include <algorithm>
#include <random>   // for default_random_engine & uniform_int_distribution<int>
#include <chrono>   // to provide seed to the default_random_engine
#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60
using namespace flens;
using namespace std;
 

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
    print_args(argc, argv);
    bool verbose = false;
    if (atoi(argv[8]) == 1)
        verbose = true; 
    int n_c=0, n_s=0, m_p=0, n_t=0;
    const char *file_path = argv[1];
    double alpha_min = atof(argv[2]);
    double alpha_max_ = atof(argv[3]);
    int n_alpha = atoi(argv[4]);
    int Kfold = atoi(argv[5]);
    int n_Kfold = atoi(argv[6]);
    const char *save_path = argv[7];
    bool use_mxne = false;
    int re_use = 1;
    if (re_use==1)
        use_mxne = true;
    double d_w_tol=1e-7;
    ReadWriteMat _RWMat(n_s, n_c, m_p, n_t);
    if (_RWMat.Read_parameters(file_path))
        return 1;
    n_s = _RWMat.n_s;
    n_c = _RWMat.n_c;
    m_p = _RWMat.m_p;
    n_t = _RWMat.n_t;

    int block = n_c / Kfold;
    int n_cpu = omp_get_num_procs();
    Maths::DVector ALPHA(n_alpha);
    double alp_step = (alpha_max_ - alpha_min)/(float)n_alpha;
    for (int y=1; y<= n_alpha;y++)
        ALPHA(y) = alpha_min + (y-1)*alp_step;
          
    if (verbose){
        std::cerr<<n_alpha <<" values of alpha in["<<alpha_min<<", "<< alpha_max_<<"]"<<std::endl;
        std::cerr<<"KFold = "<<Kfold<<std::endl;
        std::cerr<<"Input file: "<<file_path<<std::endl;
        std::cerr<<"Output file: "<<save_path<<std::endl;
        std::cerr<<"Block size = "<<block<<std::endl;
        std::cerr<<"Cross-validation of iSDR uses "<<n_cpu<<" cpus"<<std::endl;
    }
    Maths::DMatrix G_o(n_c, n_s);
    Maths::DMatrix GA_initial(n_c, n_s*m_p);
    Maths::DMatrix M(n_c, n_t);
    Maths::IMatrix SC(n_s, n_s);
    Maths::DMatrix cv_fit_data(n_alpha, n_Kfold);
    Maths::DVector alpha_real(n_alpha);
    if (_RWMat.ReadData(file_path, G_o, GA_initial, M, SC))
        return 1;
    CV_iSDR _CV_iSDR(Kfold, d_w_tol, verbose, use_mxne);
    double alpha_max = _CV_iSDR.Run_CV(M, G_o, GA_initial, SC, ALPHA, alpha_real, cv_fit_data);
    _CV_iSDR.WriteData(save_path, alpha_real, cv_fit_data, alpha_max);
    return 0;
}
