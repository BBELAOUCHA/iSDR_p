#include <cxxstd/iostream.h>
#include <flens/flens.cxx>
#include "matio.h"
#include <cmath>
//#include <omp.h>
#include <vector>
#include "iSDR.h"
#include "ReadWriteMat.h"


using namespace flens;
using namespace std;

void Weight_MVAR(double *J, double * A, int n_s, int n_t_s){
    double x=0;
    for (int i=0;i<n_s;i++){
        cxxblas::nrm2(n_t_s, &J[i*n_t_s], 1, x);
        A[i] = x;
    }
}

double absf(double *a, int n_s){
    double x=std::abs(a[0]);
    for (int i=1;i<n_s;i++){
        double z = std::abs(a[i]);
        if (z > x)
            x = z;
    }
    return x;
}

void explain_para(){
    printf( " ./iSDR  arg1 arg2 arg3 arg4 arg5 arg6 arg7\n");
    printf( "      arg1 : path to mat file which includes MEG/EEG, G, GA\n");
    printf( "      arg2 : number of iSDR iterations\n");
    printf( "      arg3 : value of regularization parameter ]0, 100]. \n");
    printf( "      arg4 : tolerance to stop MxNE. \n");
    printf( "      arg5 : where to save results. \n");
    printf( "      arg6 : use previous MxNE estimate. \n");
    printf( "      arg7 : verbose. \n");
}

void printHelp(){
    printf("\n--help or -h of the iterative source and dynamics reconstruction algorithm.\n");
    printf(" This code uses MEG and/or EEG data to reconstruct brin acitivity.\n");
    printf(" The actual version of the code needs 6 inputs:\n");
    explain_para();
}

void print_param(int n_s, int n_t, int n_c, int m_p, double alpha, double d_w_tol){
    printf(" N of sensors %d\n", n_c);
    printf(" N of sources %d\n", n_s);
    printf(" N of samples %d\n", n_t);
    printf(" MAR model    %d\n", m_p);
    printf(" iSDR tol   %.3e\n", d_w_tol);
    printf(" iSDR (p : =  %d with alpha : = %.3f percent\n", m_p, alpha);
}

int main(int argc, char* argv[]){
    std::string str1 ("-h");
    std::string str2 ("--help");
    if (str1.compare(argv[1]) == 0 || str2.compare(argv[1]) == 0){
        printHelp();
        return 1;
    }
    if(argc < 7){
        printf("Missing arguments:\n");
        explain_para();
        return 1;
    }
    else{
        bool verbose = false;
        if (atoi(argv[6]) == 1)
            verbose = true; 
        int n_c = 306;
        int n_s = 600;
        int m_p = 3;
        double alpha = 0.005;
        int n_t = 297;
        alpha = atof(argv[3]);
        int n_iter_mxne = 10000;
        int n_iter_iSDR = atoi(argv[2]);
        double d_w_tol=atof(argv[4]);
        int re_use = atoi(argv[5]);
        const char *file_path = argv[1];

        int n_t_s = n_t + m_p - 1;
        ReadWriteMat _RWMat(n_s, n_c, m_p, n_t);
        _RWMat.Read_parameters(file_path);
        n_s = _RWMat.n_s;
        n_c = _RWMat.n_c;
        m_p = _RWMat.m_p;
        n_t = _RWMat.n_t;
        n_t_s = _RWMat.n_t_s;
        //alpha *=n_c;
        if (verbose){
            print_param(n_s, n_t, n_c, m_p, alpha, d_w_tol);
            std::cout<< "Reading file: "<< file_path<<std::endl;
        }
        double *G_o = new double [n_c*n_s];
        double *GA_initial = new double [n_c*n_s*m_p];
        double *M = new double [n_c*n_t];
        int *SC = new int [n_s*n_s];
        if (GA_initial == NULL || M == NULL || G_o == NULL || SC == NULL ||
            SC == NULL) {
            printf( "\n ERROR: Can't allocate memory. Aborting...\n\n");
            return 1;
        }
        else{
            bool use_mxne = false;
            if (re_use==1)
                use_mxne = true;
            double *J= new double [n_s*n_t_s];
            std::fill(&J[0], &J[n_t_s*n_s], 0.0);
            double *Acoef= new double [n_s*n_s*m_p];
            int *Active= new int [n_s];
            _RWMat.ReadData(file_path, G_o, GA_initial, M, SC);
            double mvar_th = 1e-2;
            iSDR _iSDR(n_s, n_c, m_p, n_t, alpha, n_iter_mxne, n_iter_iSDR,
            d_w_tol, mvar_th, verbose);
            n_s = _iSDR.iSDR_solve(G_o, SC, M, GA_initial, J, &Acoef[0],
            &Active[0], use_mxne);
            double * W = new double [n_s];
            std::fill(&W[0], &W[n_s], 0.0);
            Weight_MVAR(&J[0], &W[0], n_s, n_t_s);
            double w_max = absf(&W[0], n_s);
            cxxblas::scal(n_s, 1.0/w_max, &W[0], 1);
            for (int i=0;i<n_s*m_p;i++){
                int s = i%n_s;
                cxxblas::scal(n_s, W[s], &Acoef[i*n_s], 1);
            }
            ReadWriteMat _RWMat(n_s, n_c, m_p, n_t);
            const char *save_path = argv[5];
            _RWMat.WriteData(save_path, &J[0], &Acoef[0], &Active[0], &W[0]);
            delete[] W;
            delete[] J;
            delete[] Acoef;
            delete[] Active;
        }
        delete[] G_o;
        delete[] GA_initial;
        delete[] M;
        delete[] SC;
    }
    return 0;
}
