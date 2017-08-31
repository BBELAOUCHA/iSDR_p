#include <cxxstd/iostream.h>
#include <flens/flens.cxx>
#include "matio.h"
#include <cmath>
#include <ctime>
//#include <omp.h>
#include <vector>
#include "iSDR.h"
#include "ReadWriteMat.h"


using namespace flens;
using namespace std;

void explain_para(){
    printf( " ./iSDR  arg1 arg2 arg3 arg4 arg5 arg6\n");
    printf( "      arg1 : path to mat file which includes MEG/EEG, G, GA\n");
    printf( "      arg2 : value of regularization parameter. \n");
    printf( "      arg3 : tolerance to stop MxNE. \n");
    printf( "      arg4 : where to save results. \n");
}

void printHelp(){
    printf("\n--help or -h of the iterative source and dynamics reconstruction algorithm.\n");
    printf(" This code uses MEG and/or EEG data to reconstruct brin acitivity.\n");
    printf(" The actual version of the code needs 6 inputs:\n");
    explain_para();
}


int main(int argc, char* argv[]){
    std::string str1 ("-h");
    std::string str2 ("--help");
    if (str1.compare(argv[1]) == 0 || str2.compare(argv[1]) == 0){
        printHelp();
        return 1;
    }
    if(argc < 5){
        printf("Missing arguments:\n");
        explain_para();
        return 1;
    }
    else{
        int n_c = 306;
        int n_s = 600;
        int m_p = 3;
        double alpha = 0.005;
        int n_t = 297;
        alpha = atof(argv[2]);
        int n_iter_mxne = 10000;
        int n_iter_iSDR = 10;
        double d_w_tol=atof(argv[3]);
        const char *file_path = argv[1];
        std::cout<< "Reading file: "<< file_path<<std::endl;

        int n_t_s = n_t + m_p - 1;
        ReadWriteMat _RWMat(n_s, n_c, m_p, n_t);
        _RWMat.Read_parameters(file_path);
        n_s = _RWMat.n_s;
        n_c = _RWMat.n_c;
        m_p = _RWMat.m_p;
        n_t = _RWMat.n_t;
        n_t_s = _RWMat.n_t_s;
        printf(" N of sensors %d\n", n_c);
        printf(" N of sources %d\n", n_s);
        printf(" N of samples %d\n", n_t);
        printf(" MAR model    %d\n", m_p);
        printf(" iSDR alpha   %.6f\n", alpha);
        alpha *=n_c;
        cout<<"iSDR (p : = "<< m_p<< ") with alpha : = "<<alpha<<endl;;
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
            double *J= new double [n_s*n_t_s];
            std::fill(&J[0],&J[n_t_s*n_s], 0.0);
            double *Acoef= new double [n_s*n_s*m_p];
            double *Active= new double [n_s];
            _RWMat.ReadData(file_path, G_o, GA_initial, M, SC);
            iSDR _iSDR(n_s, n_c, m_p, n_t, alpha, n_iter_mxne, n_iter_iSDR, d_w_tol, 0.001);
            n_s = _iSDR.iSDR_solve(G_o, SC, M, GA_initial, J, &Acoef[0], &Active[0]);
            ReadWriteMat _RWMat(n_s, n_c, m_p, n_t);
            const char *save_path = argv[4];
            _RWMat.WriteData(save_path, &J[0], &Acoef[0], &Active[0]);
        }
    }
    return 0;
}
