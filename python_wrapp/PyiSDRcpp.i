%module PyiSDRcpp
%{
 #define SWIG_FILE_WITH_INIT
 #include "PyiSDRcpp.h"
%}
%include "numpy.i"
%init %{
    import_array();
%}
%apply (int* IN_ARRAY1, int DIM1) {(int* sc, int n_sc)}
%apply (double* IN_ARRAY1, int DIM1) {(double* go, int n_go), (double* meg, int n_meg), (double* g, int n_g)}
%apply (double* INPLACE_ARRAY1, int DIM1){(double* js, int n_js),(double* coef, int n_coef), (double *wt, int n_wt)}
%apply (int* INPLACE_ARRAY1, int DIM1){(int* ac, int n_ac)}
%include "PyiSDRcpp.h"
