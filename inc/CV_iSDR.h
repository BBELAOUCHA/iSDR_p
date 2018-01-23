#ifndef CV_ISDR
#define CV_ISDR
//#include <cxxstd/iostream.h>
#include <flens/flens.cxx>
#include "MxNE.h"
#include "Matrix.h"
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

class CV_iSDR {
    private:
        int Kfold;
        double d_w_tol;
        bool verbose, use_mxne;
        void printProgress (double percentage);
    public:
        ~CV_iSDR(){};
        CV_iSDR(int Kfold, double d_w_tol, bool verbose, bool use_mxne);
        int WriteData(const char *file_path, const Maths::DVector &alpha, const Maths::DMatrix &cv_fit_data, double alpha_max);
        double Run_CV(const Maths::DMatrix &M, const Maths::DMatrix &G_o, const Maths::DMatrix &GA_initial, const Maths::IMatrix &SC, const Maths::DVector &ALPHA, Maths::DVector &alpha_real, Maths::DMatrix &cv_fit_data);
};
#endif //CV_ISDR
