#include "ReadWriteMat.h"
#include <omp.h>
#include "Matrix.h"

ReadWriteMat::ReadWriteMat(int n_sources, int n_sensors, int Mar_model,
    int n_samples){
    this-> n_t = n_samples;
    this-> n_c = n_sensors;
    this-> n_s = n_sources;
    this-> m_p = Mar_model;
    this-> n_t_s = n_t + m_p - 1;
}

int ReadWriteMat::Read_parameters(const char *file_path){
    // use matio to read the .mat file
    mat_t *matfp = Mat_Open(file_path, MAT_ACC_RDONLY);
    if (matfp==nullptr) {
        std::cerr << "Cannot open file " << file_path << " for reading." << std::endl;
        return 1;
    }
    
    matvar_t *matvar = Mat_VarRead(matfp, "n_s") ;
    n_s = *static_cast<const int*>(matvar->data);

    matvar = Mat_VarRead(matfp, "n_c") ;
    n_c = *static_cast<const int*>(matvar->data);

    matvar = Mat_VarRead(matfp, "n_t") ;
    n_t = *static_cast<const int*>(matvar->data);

    matvar = Mat_VarRead(matfp, "m_p") ;
    m_p = *static_cast<const int*>(matvar->data);

    n_t_s = n_t + m_p - 1;

	return 0;
}

// Read data from mat file

int
ReadWriteMat::ReadData(const char *file_path, Maths::DMatrix &G_o, Maths::DMatrix &GA,
                       Maths::DMatrix &R, Maths::IMatrix &SC) const {

    mat_t* matfp = Mat_Open(file_path, MAT_ACC_RDONLY);
    if (matfp==nullptr) {
        std::cerr << "Cannot open file " << file_path << " for reading." << std::endl;
        return 1;
    }

	matvar_t* matvar = Mat_VarRead(matfp, "GA") ;
    const double* GAData = static_cast<const double*>(matvar->data);
    for (unsigned long y = 0;y < n_s*m_p; ++y)
        for (unsigned long x = 0; x < n_c; ++x)
            GA.data()[x + y*n_c] = GAData[x + y*n_c]; 
    Mat_VarFree(matvar);

    matvar = Mat_VarRead(matfp, "M") ;
    const double* RData = static_cast<const double*>(matvar->data);
    for (unsigned long y = 0;y < n_t; ++y)
        for (unsigned long x = 0; x < n_c; ++x)
            R.data()[x + y*n_c] = RData[x+y*n_c];
    Mat_VarFree(matvar);

    matvar = Mat_VarRead(matfp, "G") ;
    const double* GData = static_cast<const double*>(matvar->data);
    for (unsigned long y = 0;y < n_s; ++y)
        for (unsigned long x = 0; x < n_c; ++x)
            G_o.data()[x + y*n_c] = GData[x+y*n_c];
    Mat_VarFree(matvar);

    matvar = Mat_VarRead(matfp, "SC");
    const double* SCData = static_cast<const double*>(matvar->data);
    //#pragma omp parallel for
    for(unsigned long y = 0;y < n_s; y++)
        for (unsigned long x = 0; x < n_s; x++)
            SC.data()[x*n_s + y] = static_cast<int>(SCData[x*n_s+y]);
    Mat_VarFree(matvar);

    Mat_Close(matfp);

    return 0;
}


int ReadWriteMat::WriteData(const char *file_path, Maths::DMatrix &S, Maths::DMatrix &mvar,
                            Maths::IVector &A, Maths::DVector &w, double max_eigenvalue) {

    // Setup the output

    mat_t* mat = Mat_Create(file_path,NULL);
    if (mat==nullptr) {
        std::cerr << "Cannot create file " << file_path << " for writing" << std::endl;
        return 1;
	}

    double sca1[1] = { max_eigenvalue };
    size_t dims[1] = { 1U };

    matvar_t* matvar = Mat_VarCreate("Eigen max", MAT_C_DOUBLE, MAT_T_DOUBLE,1, dims, &sca1,0);
    Mat_VarWrite(mat, matvar, MAT_COMPRESSION_NONE);
    Mat_VarFree(matvar);

    // Estimated brain activation

    size_t dims1[2] = {n_t_s,n_s};
    double mat1[n_s][n_t_s];
    for(unsigned j=0;j<n_s;++j)
        for(unsigned i=0;i<n_t_s;++i)
            mat1[j][i] = S.data()[n_t_s*j+i];

    matvar = Mat_VarCreate("S estimate",MAT_C_DOUBLE,MAT_T_DOUBLE,2, dims1, &mat1,0);
    Mat_VarWrite( mat, matvar, MAT_COMPRESSION_NONE);
    Mat_VarFree(matvar);

    // Multivariate autoregresive model elements

    size_t dims2[2] = {n_s,n_s*m_p};
    double mat2[n_s*m_p][n_s];
    for(unsigned i=0;i<n_s * m_p;++i)
        for(unsigned j=0;j<n_s;++j)
            mat2[i][j] = mvar.data()[j + n_s*i];

    matvar = Mat_VarCreate("MVAR", MAT_C_DOUBLE,MAT_T_DOUBLE,2, dims2, &mat2,0);
    Mat_VarWrite( mat, matvar, MAT_COMPRESSION_NONE);
    Mat_VarFree(matvar);

    // Label of active sources/regions

    size_t dim1d[1] = {n_s};
    double mat3[n_s];
    for (unsigned j=0;j<n_s;++j)
        mat3[j] = A.data()[j];

    matvar = Mat_VarCreate("Active set",MAT_C_DOUBLE,MAT_T_DOUBLE,1, dim1d, &mat3,0);
    Mat_VarWrite( mat, matvar, MAT_COMPRESSION_NONE);
    Mat_VarFree(matvar);

    // weights used to normalize MVAR coeffitions

    double mat4[n_s];
    for (unsigned j=0;j<n_s;++j)
        mat4[j] = w.data()[j];

    matvar = Mat_VarCreate("Weights", MAT_C_DOUBLE, MAT_T_DOUBLE,1, dim1d, &mat4,0);
    Mat_VarWrite( mat, matvar, MAT_COMPRESSION_NONE);
    Mat_VarFree(matvar);
    Mat_Close(mat);

    return 0;
}
