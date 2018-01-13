#include <flens/flens.cxx>

namespace Maths {
    using namespace flens;

    typedef GeMatrix<FullStorage<double, ColMajor>> DMatrix;
    typedef typename DMatrix::IndexType   DIndex;
    typedef DenseVector<Array<double>>    DVector;

    typedef GeMatrix<FullStorage<int, ColMajor>>  IMatrix;
    typedef typename IMatrix::IndexType IIndex;
    typedef DenseVector<Array<int>>     IVector;
}
