/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ELEM_CONVEX_SVT_NORMAL_HPP
#define ELEM_CONVEX_SVT_NORMAL_HPP

#include "elemental/blas-like/level1/DiagonalScale.hpp"
#include "elemental/lapack-like/Norm/Max.hpp"
#include "elemental/lapack-like/Norm/Zero.hpp"
#include "elemental/lapack-like/SVD.hpp"
#include "elemental/convex/SoftThreshold.hpp"

namespace elem {

namespace svt {

template<typename F>
inline Int
Normal( Matrix<F>& A, BASE(F) tau, bool relative=false )
{
#ifndef RELEASE
    CallStackEntry cse("svt::Normal");
#endif
    typedef Base<F> Real;
    Matrix<F> U( A );
    Matrix<Real> s;
    Matrix<F> V;

    SVD( U, s, V );
    SoftThreshold( s, tau, relative );
    DiagonalScale( RIGHT, NORMAL, s, U );
    Gemm( NORMAL, ADJOINT, F(1), U, V, F(0), A );

    return ZeroNorm( s );
}

template<typename F>
inline Int
Normal( DistMatrix<F>& A, BASE(F) tau, bool relative=false )
{
#ifndef RELEASE
    CallStackEntry cse("svt::Normal");
#endif
    typedef Base<F> Real;
    DistMatrix<F> U( A );
    DistMatrix<Real,VR,STAR> s( A.Grid() );
    DistMatrix<F> V( A.Grid() );

    SVD( U, s, V );
    SoftThreshold( s, tau, relative );
    DiagonalScale( RIGHT, NORMAL, s, U );
    Gemm( NORMAL, ADJOINT, F(1), U, V, F(0), A );

    return ZeroNorm( s );
}

} // namespace svt
} // namespace elem

#endif // ifndef ELEM_CONVEX_SVT_NORMAL_HPP
