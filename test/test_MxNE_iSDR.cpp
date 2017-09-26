#include <cxxstd/iostream.h>
#include <flens/flens.cxx>
#include "matio.h"
#include <cmath>
#include <ctime>
#include <omp.h>
#include <vector>
#include "iSDR.h"
#include "MxNE.h"
#include <algorithm>
#include "ReadWriteMat.h"
using namespace flens;
using namespace std;
double Asum_x(double *a, double *b, int x){
    double asum_x = 0.0;
    for (int i =0;i<x;i++)
        asum_x += std::abs(a[i] - b[i]);
    return asum_x;
}
int test_Compute_mu(){
    int n_s = 3; int n_c = 2; int m_p = 3; int n_t = 100; double d_w_tol=1e-6; 
    MxNE _MxNE(n_s, n_c, m_p, n_t, d_w_tol, false);
    typedef GeMatrix<FullStorage<double, ColMajor> > GeMatrix;
    GeMatrix G(n_c, n_s*m_p);
    for(int j=0;j<n_s; ++j){
           for (int i =0; i<n_c; ++i){
               for(int k=0;k<m_p;k++)
                G(i+1,  j*m_p + 1 + k) = j+1;
           }
    }
    _MxNE.Compute_mu(G.data());
    for (int i=0;i<n_s;i++){
        double x = m_p*m_p*(n_c*(i+1)*(i+1))*(n_c*(i+1)*(i+1));
        if (_MxNE.mu[i] != 1.0/x)
            return 0;
    }
    return 1;
}

int test_Compute_dX(){
    int n_s = 3; int n_c = 2; int m_p = 2; int n_t = 10; double d_w_tol=1e-6;
    int n_t_s = n_t + m_p -1; 
    MxNE _MxNE(n_s, n_c, m_p, n_t, d_w_tol, false);
    typedef GeMatrix<FullStorage<double, ColMajor> > GeMatrix;
    typedef DenseVector<Array<double> >               DEVector;
    GeMatrix G(n_c, n_s*m_p);GeMatrix R(n_c, n_t);
    GeMatrix X(n_t_s, n_s);DEVector X_c(n_t_s);
    DEVector X_e(n_t_s);
    std::fill(&X.data()[0], &X.data()[n_t_s], 1);
    for(int j=0;j<n_s; ++j){
           for (int i =0; i<n_c; ++i){
               for(int k=0;k<m_p;k++)
                G(i+1,  j*m_p + 1 + k) = j + 1;
           }
    }
    for (int l=0;l<m_p;l++){
        for (int i=0;i<n_t; i++){
            for (int j =0; j<n_c;j++){
                 for (int k=0;k<n_s;k++){
                    R(j+1, i+1) += G(j+1, k*m_p + 1 + l) * X(i+l+1, k+1);
                 }
            }
       }
    }
    cxxblas::copy(n_c*n_t, &R.data()[0], 1, &_MxNE.R[0], 1);
    for (int i=0;i<n_s;i++){
        _MxNE.Compute_dX(&G.data()[0], &X_c.data()[0], i);
        std::fill(&X_e.data()[0], &X_e.data()[n_t_s], 8*(i+1));
        for (int j=0;j<m_p-1;j++){
             X_e(j+1) = 4*(i+1);
             X_e(n_t_s-j) = 4*(i+1);
        }
        double asum_x = Asum_x(X_c.data(), X_e.data(), n_t_s);
        if (asum_x != 0.0)
           return 0;
        std::fill(&X_c.data()[0], &X_c.data()[n_t_s], 0.0);
    }
    return 1; 
}


int test_update_r(){
    int n_s = 3; int n_c = 2; int m_p = 2; int n_t = 10; double d_w_tol=1e-6;
    int n_t_s = n_t + m_p -1; 
    MxNE _MxNE(n_s, n_c, m_p, n_t, d_w_tol, false);
    typedef GeMatrix<FullStorage<double, ColMajor> > GeMatrix;
    typedef DenseVector<Array<double> >               DEVector;
    DEVector X_i(n_t_s);DEVector X_i_1(n_t_s);DEVector X_i_2(n_t_s);
    GeMatrix R(n_c, n_t);GeMatrix G(n_c, n_s*m_p);
    for(int j=0;j<n_s; ++j){
           for (int i =0; i<n_c; ++i){
               for(int k=0;k<m_p;k++)
                G(i+1,  j*m_p + 1 + k) = 0.5*(j+1);
           }
    }
    std::fill(&X_i.data()[0], &X_i.data()[n_t_s], 1.0);
    std::fill(&X_i_2.data()[0], &X_i_2.data()[n_t_s], 2.0);
    cxxblas::copy(n_c*n_t, &R.data()[0], 1, &_MxNE.R[0], 1);
    _MxNE.update_r(G.data(), X_i.data(), 0);
    _MxNE.Compute_dX(G.data(), X_i_1.data(), 0);
    X_i_2(1) = 1;X_i_2(n_t_s) = 1;
    if (Asum_x(X_i_1.data(), X_i_2.data(), n_t_s) != 0.0)
        return 0;
    return 1;
}


int test_duality_gap(){
    int n_s = 3; int n_c = 2; int m_p = 2; int n_t = 10; double d_w_tol=1e-6;
    int n_t_s = n_t + m_p -1; 
    MxNE _MxNE(n_s, n_c, m_p, n_t, d_w_tol, false);
    typedef GeMatrix<FullStorage<double, ColMajor> > GeMatrix;
    GeMatrix M(n_c, n_t);GeMatrix R(n_c, n_t);GeMatrix G(n_c, n_s*m_p);
    GeMatrix J(n_t_s, n_s);double alpha = 1e-3;
    for(int j=0;j<n_s; ++j){
           for (int i =0; i<n_c; ++i){
               for(int k=0;k<m_p;k++)
                G(i+1,  j*m_p + 1 + k) = 0.5*(j+1);
           }
    }
    std::fill(&J.data()[0], &J.data()[n_t_s], 1.0);
    for (int l=0;l<m_p;l++){
        for (int i=0;i<n_t; i++){
            for (int j =0; j<n_c;j++){
                 for (int k=0;k<n_s;k++){
                    M(j+1, i+1) += G(j+1, k*m_p + 1 + l) * J(i+l+1, k+1);
                 }
            }
       }
    }
    double gap;
    gap = _MxNE.duality_gap(G.data(), M.data(), J.data(), alpha);
    if (gap!=alpha*std::sqrt(n_t_s))
        return 0;
    std::fill(&R.data()[0], &R.data()[n_t*n_c], 0.5);
    gap = _MxNE.duality_gap(G.data(), M.data(), J.data(), alpha);
    double gap2 = 0.0;GeMatrix GtR(n_s, n_t_s);
    _MxNE.Compute_GtR(G.data(), R.data(), GtR.data());  
    double norm_GtR = 0.0;
    for (int ii =0; ii < n_s; ++ii){
        double XtA_axis1norm = 0.0;
        cxxblas::nrm2(n_t_s, &GtR.data()[0] + ii*n_t_s, 1, XtA_axis1norm);
        if (XtA_axis1norm > norm_GtR)
            norm_GtR = XtA_axis1norm;
    }
    double R_norm = 0.0;
    cxxblas::nrm2(n_t*n_c, R.data(), 1, R_norm);
    double con = 1.0;
    if (norm_GtR > alpha){
        con =  alpha / norm_GtR;
        double A_norm = R_norm * con;
        gap2 = 0.5 * (R_norm * R_norm + A_norm *A_norm);
    }
    else{
        con = 1.0;
        gap2 = R_norm * R_norm;
    }
    double ry=0.0;
    for (int i=0;i<n_c*n_t;i++)
        ry += M.data()[i]*R.data()[i];
    gap2 += alpha*std::sqrt(n_t_s) -con * ry;
    if (gap !=gap2)
        return 0;
    return 1;
}

int test_Greorder(){
   int n_s=3;int n_c=2;int m_p=3;
   typedef GeMatrix<FullStorage<double, ColMajor> > GeMatrix;
   iSDR _iSDR(n_s, n_c, m_p, 100, 0.0, 1, 1, 1e-3, 0.001, false);
   GeMatrix GA(n_c, n_s*m_p);GeMatrix G_reorder(n_c, n_s*m_p);
   for (int k=0;k<m_p;++k){
       for(int j=0;j<n_s; ++j){
           for (int i =0; i<n_c; ++i)
                GA.data()[j*n_c + i + k*n_c*n_s] = j;
           }
       }
    _iSDR.Reorder_G(GA.data(), G_reorder.data()); 
    double asum_v[n_s];
    for (int i =0; i<n_s;i++){
        cxxblas::asum(n_c*m_p, &G_reorder.data()[i*n_c*m_p], 1, asum_v[i]);
        if (asum_v[i] != n_c*m_p*i)
           return  0;
    }   
    return 1;
}


int test_GxA(){
   int n_s=3;int n_c=2;int m_p=3;
   typedef GeMatrix<FullStorage<double, ColMajor> > GeMatrix;
   GeMatrix G(n_c, n_s);GeMatrix A(n_s, n_s*m_p);
   for(int j=0;j<n_s; ++j){
           for (int i =0; i<n_c; ++i)
                G(i+1,  j+1) = j;
   }
   for (int i =0; i<m_p;i++){
        for (int j =0; j<n_s;j++)
                A(j+1, j+1+i*n_s) = 1.0;
   }
   iSDR _iSDR(n_s, n_c, m_p, 100, 0.0, 1, 1, 1e-3, 0.001, false);
   GeMatrix GA(n_c, n_s*m_p);
   _iSDR.G_times_A(G.data(), A.data(), GA.data());
   double asum_v[n_s];
   for (int i =0; i<n_s;i++){
       cxxblas::asum(n_c*m_p, &GA.data()[i*n_c*m_p], 1, asum_v[i]);
       if (asum_v[i] != n_c*m_p*i)
          return  0;
   }
   return 1;
}

int test_Reduce_G(){
   int n_s=5;int n_c=2;int m_p=3;
   typedef GeMatrix<FullStorage<double, ColMajor> > GeMatrix;
   GeMatrix G(n_c, n_s);
    for(int j=0;j<n_s; ++j){
           for (int i =0; i<n_c; ++i)
                G(i+1,  j+1) = j;
    }
    iSDR _iSDR(n_s, n_c, m_p, 100, 0.0, 1, 1, 1e-3, 0.001, false);
    std::vector<int> ind;int x = 1;int y = 3;
    ind.push_back(x);ind.push_back(y);
    GeMatrix G_n(n_c, ind.size());
    _iSDR.Reduce_G(G.data(), G_n.data(), ind);
    for (unsigned int i=0;i<ind.size();i++){
        for (int j=0;j<n_c;j++){
            if (G_n(j+1, i+1) != G(j+1, ind[i]+1))
               return 0;
        }
    }
    return 1;
}

int test_Reduce_SC(){
   int n_s=6;int n_c=2;int m_p=3;
   typedef GeMatrix<FullStorage<int, ColMajor> > GeMatrix;
   GeMatrix SC(n_s, n_s);
   iSDR _iSDR(n_s, n_c, m_p, 100, 0.0, 1, 1, 1e-3, 0.001, false);
   std::vector<int> ind;
   int x = 1;int y = 3;int z = 5;
   ind.push_back(x-1);ind.push_back(y-1);ind.push_back(z-1);
   for (int i=0;i<n_s;i++)
       SC(i+1, i+1) = 1;
   SC(y, x) = 1;SC(z, x) = 1;GeMatrix SC_n(3, 3);
   _iSDR.Reduce_SC(SC.data(), SC_n.data(), ind);
   for (int i=0; i<3;i++){
       for (int j=0;j<3;j++){
           if (SC_n(i+1, j+1) != SC(ind[i]+1, ind[j]+1))
              return 0;       
        }
   }
   return 1;
}

int test_A_step_lsq(){
    int n_s=3;int n_c=2;int m_p=2;int n_t = 100;double acceptable = 1e-2;
    typedef GeMatrix<FullStorage<double, ColMajor> > GeMatrix1;
    typedef GeMatrix<FullStorage<int, ColMajor> > GeMatrix2;
    GeMatrix1 SC(n_s, n_s);
    int n_t_s = n_t;
    iSDR _iSDR(n_s, n_c, m_p, n_t, 0.0, 1, 1, 1e-3, 0.001, false);
    _iSDR.n_t_s = n_t_s;
    GeMatrix1 A(n_s, n_s*m_p);
    A(1,1) = 0.47599234;A(1,2) = 0.12215775;A(2,1) = -0.12460725;
    A(2,2) = 0.46669897;A(1,1+n_s) = 0.43268679;A(1,2+n_s) = 0.24245205;
    A(2,1+n_s) = -0.24120229;A(2,2+n_s) = 0.42778779;A(3,3+n_s) = 1.0;
    GeMatrix1 S(n_t_s, n_s);
    S(1,1) = 6.15700359;S(2,1) = 5;S(1,2) = -3.64289379;S(2,2) = -5;
    S(1, 3) = 1.0;S(2, 3) = 1.0;
    for (int i=m_p+1;i<=n_t_s;i++){
        for (int j=0;j<n_s;j++){
             for (int k=0;k<n_s;k++){
                 for (int l=0;l<m_p;l++)
                    S(i, j+1) += A(j+1, k+1 + n_s*l)*S(i-m_p+l, k+1); 
            }
        }
    }
    GeMatrix2 A_scon(n_s, n_s);
    for (int i=1;i<=n_s;i++)
        A_scon(i, i) = 1;
    A_scon(1, 2) = 1;A_scon(2, 1) = 1;
    GeMatrix1 VAR(n_s, n_s*m_p);
    _iSDR.A_step_lsq(&S.data()[0], &A_scon.data()[0], 1e-3, &VAR.data()[0]);
    GeMatrix1 Se(n_t_s, n_s);
    Se(1,1) = 6.15700359;Se(2,1) = 5;Se(1,2) = -3.64289379;Se(2,2) = -5;
    Se(1, 3) = 1.0;Se(2, 3) = 1.0;
    for (int i=m_p+1;i<=n_t_s;i++){
        for (int j=0;j<n_s;j++){
             for (int k=0;k<n_s;k++){
                 for (int l=0;l<m_p;l++)
                        Se(i, j+1) += VAR(j+1, k+1 + n_s*l)*Se(i-m_p+l, k+1); 
            }
        }
    }
    double asum_s = Asum_x(Se.data(),S.data(), n_s*n_t_s);
    double asum_a = Asum_x(A.data(),VAR.data(), n_s*n_s*m_p);
    if (asum_s > acceptable || asum_a>acceptable)
        return 0;
    return 1;
    }

int test_Zero_non_zero(){
    std::vector<int> ind;std::vector<int> ind_c;
    int n_s=30;int n_c=2;int m_p=2;int n_t = 300;
    typedef GeMatrix<FullStorage<double, ColMajor> > GeMatrix;
    GeMatrix S(n_t, n_s);
    iSDR _iSDR(n_s, n_c, m_p, n_t-m_p+1, 0.0, 1, 1, 1e-3, 0.001, false);
    ind = _iSDR.Zero_non_zero(S.data());
    if (ind.size()>0)
        return 0;
    int x = 10;int y = 20;
    S(x,x) = 1.0;S(y,y) = 1.0;
    ind_c.push_back(x-1);ind_c.push_back(y-1);
    ind = _iSDR.Zero_non_zero(S.data());
    if (ind.size()!=2)
        return 0;
    for (unsigned int i = 0; i<ind.size();i++){
        if (ind_c[i]!=ind[i])
           return 0;    
    }
        
    return 1;
}

int test_MxNE(){
    int n_s = 5; int n_c = 3; int m_p = 1; int n_t = 100; double d_w_tol=1e-6; 
    MxNE _MxNE(n_s, n_c, m_p, n_t, d_w_tol, false);
    typedef GeMatrix<FullStorage<double, ColMajor> > GeMatrix;
    GeMatrix G(n_c, n_s*m_p);
    for (int i=1; i<= n_c; i++)
        G(i, i) = 1.0;
    GeMatrix S(n_t + m_p -1, n_s);
    GeMatrix A(n_s, n_s);
    int x = 1; int y= 3;
    A(x,x) = 0.96;A(x,y) = 0.25;
    A(y,x)=-0.25;A(y,y) = 0.95;
    S(1,x) = 5; S(1,y) = -5;
    for (int i=m_p+1;i<n_t+1;i++){
        for (int j=1;j<=n_s;j++){
             for (int k=1;k<=n_s;k++){
                    S(i, j) += A(j, k)*S(i-1, k); 
            }
        }
    }
    GeMatrix M(n_c, n_t);GeMatrix J(n_t + m_p -1, n_s);
    for (int i=0;i<n_t; i++){
        for (int j =0; j<n_c;j++){
            for (int k=0;k<n_s;k++){
                M(j+1, i+1) += G(j + 1, k + 1) * S(i+1, k+1);
            }
        }
    }
    double alpha = 0.0001;double dual_gap = 0;double tol = 0.0;
    _MxNE.MxNE_solve(&M.data()[0], &G.data()[0], &J.data()[0], alpha, 10000,
                     dual_gap, tol, false);
    double re = Asum_x(&S.data()[0], &J.data()[0], n_s*(n_t+m_p-1));   
    if (re > 1e-2)
       return 0;
    return 1;
}

int main(){
    
    if (test_Compute_mu())
        printf( "MxNE.Compute_mu    ... Ok\n");
    else
        printf( "MxNE.Compute_mu    ... Failed\n");
    if (test_Compute_dX())
        printf( "MxNE.Compute_dX    ... Ok\n");
    else
        printf( "MxNE.Compute_dX    ... Failed\n");

    if (test_update_r())
        printf( "MxNE.update_r      ... Ok\n");
    else
        printf( "MxNE.update_r      ... Failed\n");
    //if (test_duality_gap())
   //     printf( "MxNE.duality_gap   ... Ok\n");
   // else
    //    printf( "MxNE.duality_gap   ... Failed\n");
    if (test_Greorder())
        printf( "iSDR.Reorder_G     ... Ok\n");
    else
        printf( "iSDR.Reorder_G     ... Failed\n");

    if (test_GxA())
        printf( "iSDR.G_times_A     ... Ok\n");
    else
        printf( "iSDR.G_times_A     ... Failed\n");
    if (test_Reduce_G()) 
        printf( "iSDR.Reduce_G      ... Ok\n");
    else
        printf( "iSDR.Reduce_G      ... Failed\n");
    if (test_Reduce_SC())
         printf( "iSDR.Reduce_SC     ... Ok\n");
    else
        printf( "iSDR.Reduce_SC     ... Failed\n");
    
    if (test_A_step_lsq())
         printf( "iSDR.A_step_lsq    ... Ok\n");
    else
        printf( "iSDR.A_step_lsq    ... Failed\n");
    if (test_Zero_non_zero())
         printf( "iSDR.Zero_non_zero ... Ok\n");
    else
        printf( "iSDR.Zero_non_zero ... Failed\n");
    //if (test_MxNE())
    //    printf( "MxNE.MxNE_solve   ... Ok\n");
    //else
    //    printf( "MxNE.MxNE_solve   ... Failed\n");
    return 0;
}
