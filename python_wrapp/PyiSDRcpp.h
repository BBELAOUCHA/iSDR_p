#ifndef PYISDRCPP
#define PYISDRCPP

class PyiSDRcpp {
    private:
        int n_iter_mxne, n_iter_iSDR;
        bool initial, ver;
        double alpha_x, d_w_tol;
    public:
        PyiSDRcpp(int n_iter_mxne,int n_iter_iSDR,double alpha_x, double d_w_tol, bool initial, bool ver);
        ~PyiSDRcpp(){};
        int fit(double * go,int n_go, int * sc,int n_sc, double * meg, int n_meg, double * g,
        int n_g, double * js, int n_js, double * coef, int n_coef, int * ac, int n_ac, double *wt, int n_wt);
  };
#endif //PYISDRCPP
