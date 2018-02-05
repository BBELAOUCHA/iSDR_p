#pragma once

//#include <cxxstd/iostream.h>
#include <flens/flens.cxx>
#include "matio.h"
#include <cmath>
#include <ctime>
#include <algorithm>
//#include <omp.h>
#include <string>
#include <vector>
#include "Matrix.h"
//==============================================================================
//==============================================================================
///
/// \file ReadWriteMat.h
///
/// \author Brahim Belaoucha, INRIA <br>
///         Copyright (c) 2017  <br>
//==============================================================================
//==============================================================================

class ReadWriteMat {
public:
    long unsigned int n_t;
    long unsigned int n_c;
    long unsigned int n_t_s;
    long unsigned int m_p;
    long unsigned int n_s;

    ReadWriteMat(int n_sources, int n_sensors, int Mar_model, int n_samples);
    ~ReadWriteMat(){};

    int ReadData(const char *file_path, Maths::DMatrix &G_o, Maths::DMatrix &GA,
                 Maths::DMatrix &R, Maths::IMatrix &SC) const;
    int WriteData(const char *file_path, Maths::DMatrix &S, Maths::DMatrix &mvar,
                  Maths::IVector &A, Maths::DVector &w, double max_eigenvalue);
    int Read_parameters(const char *file_path);
};
