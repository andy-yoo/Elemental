/*
   Copyright (c) 2009-2015, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "El.hpp"

namespace El {

template<typename F>
void DiagonalEquil
( Matrix<F>& A, Matrix<Base<F>>& d, bool progress )
{
    DEBUG_ONLY(CSE cse("DiagonalEquil"))
    // TODO: Ensure A is square
    const Int n = A.Height();
    Ones( d, n, 1 );
    if( progress )
        cout << "    Diagonal equilibration not yet enabled for dense "
                "matrices" << endl;
}

template<typename F>
void DiagonalEquil
( AbstractDistMatrix<F>& A, AbstractDistMatrix<Base<F>>& d, bool progress )
{
    DEBUG_ONLY(CSE cse("DiagonalEquil"))
    // TODO: Ensure A is square
    const Int n = A.Height();
    Ones( d, n, 1 );
    if( progress )
        cout << "    Diagonal equilibration not yet enabled for dense "
                "matrices" << endl;
}

template<typename F>
void DiagonalEquil( SparseMatrix<F>& A, Matrix<Base<F>>& d, bool progress )
{
    DEBUG_ONLY(CSE cse("DiagonalEquil"))
    typedef Base<F> Real;
    const Int n = A.Height();
    auto maxSqrtLambda = []( F delta ) 
                         { return Sqrt(Max(Abs(delta),Real(1))); };
    function<Real(F)> maxSqrt( maxSqrtLambda );
    GetMappedDiagonal( A, d, maxSqrt );
    if( progress )
    {
        const Real maxNorm = MaxNorm( d ); 
        cout << "    || d ||_max = " << maxNorm << endl;
    }
    DiagonalSolve( LEFT, NORMAL, d, A );
    DiagonalSolve( RIGHT, NORMAL, d, A );
}

template<typename F>
void DiagonalEquil
( DistSparseMatrix<F>& A, DistMultiVec<Base<F>>& d, 
  bool progress, bool time )
{
    DEBUG_ONLY(CSE cse("DiagonalEquil"))
    typedef Base<F> Real;
    mpi::Comm comm = A.Comm();
    const int commRank = mpi::Rank(comm);
    const Int n = A.Height();
    Timer timer;

    d.SetComm( comm );
    auto maxSqrtLambda = []( F delta )
                         { return Sqrt(Max(Abs(delta),Real(1))); };
    function<Real(F)> maxSqrt( maxSqrtLambda );
    if( commRank == 0 && time )
        timer.Start();
    GetMappedDiagonal( A, d, maxSqrt );
    if( commRank == 0 && time )
        cout << "    Get mapped diag time: " << timer.Stop() << endl;
    if( commRank == 0 && time )
        timer.Start();
    DiagonalSolve( LEFT, NORMAL, d, A );
    if( commRank == 0 && time )
        cout << "    Left diag solve time: " << timer.Stop() << endl;
    if( commRank == 0 && time )
        timer.Start();
    DiagonalSolve( RIGHT, NORMAL, d, A );
    if( commRank == 0 && time )
        cout << "    Right diag solve time: " << timer.Stop() << endl;
    if( progress )
    {
        const Real maxNorm = MaxNorm( d );
        if( commRank == 0 ) 
            cout << "    Diagonally equilibrated with || d ||_max = " 
                 << maxNorm << endl;
    }
}

#define PROTO(F) \
  template void DiagonalEquil \
  ( Matrix<F>& A, Matrix<Base<F>>& d, bool progress ); \
  template void DiagonalEquil \
  ( AbstractDistMatrix<F>& A,  AbstractDistMatrix<Base<F>>& d, \
    bool progress ); \
  template void DiagonalEquil \
  ( SparseMatrix<F>& A, Matrix<Base<F>>& d, bool progress ); \
  template void DiagonalEquil \
  ( DistSparseMatrix<F>& A, DistMultiVec<Base<F>>& d, \
    bool progress, bool time );

#define EL_NO_INT_PROTO
#include "El/macros/Instantiate.h"

} // namespace El
