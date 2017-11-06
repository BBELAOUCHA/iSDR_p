#include "ReadWriteMat.h"
#include <omp.h>

ReadWriteMat::ReadWriteMat(int n_sources, int n_sensors, int Mar_model, int n_samples){
    this-> n_t = n_samples;
    this-> n_c = n_sensors;
    this-> n_s = n_sources;
    this-> m_p = Mar_model;
    this-> n_t_s = n_t + m_p - 1;
}

void ReadWriteMat::Read_parameters(const char *file_path){
    mat_t *matfp; // use matio to read the .mat file
    matvar_t *matvar;
    matfp = Mat_Open(file_path, MAT_ACC_RDONLY);
    matvar = Mat_VarRead(matfp, "n_s") ;
    const int *x = static_cast<const int*>(matvar->data);
    n_s = x[0];
    matvar = Mat_VarRead(matfp, "n_c") ;
    x = static_cast<const int*>(matvar->data);
    n_c = x[0];
    matvar = Mat_VarRead(matfp, "n_t") ;
    x = static_cast<const int*>(matvar->data);
    n_t = x[0];
    matvar = Mat_VarRead(matfp, "m_p") ;
    x = static_cast<const int*>(matvar->data);
    m_p = x[0];
    n_t_s = n_t + m_p - 1;
}
void ReadWriteMat::ReadData(const char *file_path, double *G_o, double *GA,
                            double *R, int * SC) const {
    // read data from mat file
    mat_t *matfp; // use matio to read the .mat file
    matvar_t *matvar;
    matfp = Mat_Open(file_path, MAT_ACC_RDONLY);
    matvar = Mat_VarRead(matfp, "GA") ;
    const double *xData = static_cast<const double*>(matvar->data) ;
    //#pragma omp parallel for
    for(long unsigned int y = 0;y < n_s*m_p; ++y){
        for (long unsigned int x = 0; x < n_c; ++x)
            GA[x + y*n_c] = xData[x + y*n_c]; 
    }

    Mat_VarFree(matvar);
    matvar = Mat_VarRead(matfp, "M") ;
    const double *xData1 = static_cast<const double*>(matvar->data) ;
    //#pragma omp parallel for
    for(long unsigned int y = 0;y < n_t; ++y){
        for (long unsigned int x = 0; x < n_c; ++x)
            R[x + y*n_c] = xData1[x+y*n_c];
    }
    Mat_VarFree(matvar);
    matvar = Mat_VarRead(matfp, "G") ;
    const double *xData_ = static_cast<const double*>(matvar->data) ;
    //#pragma omp parallel for
    for(long unsigned int y = 0;y < n_s; ++y){
        for (long unsigned int x = 0; x < n_c; ++x)
            G_o[x + y*n_c] = xData_[x+y*n_c];
    }
    Mat_VarFree(matvar);
    matvar = Mat_VarRead(matfp, "SC") ;
    const double *xData2 = static_cast<const double*>(matvar->data) ;
    //#pragma omp parallel for
    for(long unsigned int y = 0;y < n_s; y++){
        for (long unsigned int x = 0; x < n_s; x++)
            SC[x*n_s + y] = (int)xData2[x*n_s+y];
    }

    xData_ = NULL;
    xData = NULL;
    xData1 = NULL;
    xData2 = NULL;
    Mat_VarFree(matvar);
    Mat_Close(matfp);
}


int ReadWriteMat::WriteData(const char *file_path, double *S, double *mvar,
                            int *A, double * w){
    double mat1[n_s][n_t_s];
    double mat2[n_s*m_p][n_s];
    double mat3[n_s];
    double mat4[n_s];
    unsigned int i, j;
    //#pragma omp parallel for
    for(j=0;j<n_s;j++){
        for(i=0;i<n_t_s;i++)
            mat1[j][i] = S[n_t_s*j+i];
    }
    //#pragma omp parallel for
    for(i=0;i<n_s * m_p;i++)
        for(j=0;j<n_s;j++)
            mat2[i][j] = mvar[j + n_s*i];

    for(j=0;j<n_s;j++){
      mat3[j] = A[j];
      mat4[j] = w[j];
    }
    /* setup the output */
    mat_t *mat;
    matvar_t *matvar;
    size_t dims1[2] = {n_t_s,n_s};
    size_t dims2[2] = {n_s,n_s*m_p};
    size_t dim1d[1] = {n_s};
    mat = Mat_Create(file_path,NULL);
    if(mat){
        /* Estimated brain activation */
        matvar = Mat_VarCreate("S estimate",MAT_C_DOUBLE,MAT_T_DOUBLE,2, dims1,
            &mat1,0);
        Mat_VarWrite( mat, matvar, MAT_COMPRESSION_NONE);
        Mat_VarFree(matvar);
        /* multivariate autoregresive model elements */
        matvar = Mat_VarCreate("S MVAR", MAT_C_DOUBLE,MAT_T_DOUBLE,2, dims2,
             &mat2,0);
        Mat_VarWrite( mat, matvar, MAT_COMPRESSION_NONE);
        Mat_VarFree(matvar);
      /* Label of active sources/regions */
      matvar = Mat_VarCreate("S Active",MAT_C_DOUBLE,MAT_T_DOUBLE,1, dim1d,
             &mat3,0);
      Mat_VarWrite( mat, matvar, MAT_COMPRESSION_NONE);
      Mat_VarFree(matvar);
      /* weights used to normalize MVAR coeffitions */
      matvar = Mat_VarCreate("Weights", MAT_C_DOUBLE, MAT_T_DOUBLE,1, dim1d,
             &mat4,0);
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
