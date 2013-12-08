/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ELEM_LAPACK_HERMITIANFUNCTION_HPP
#define ELEM_LAPACK_HERMITIANFUNCTION_HPP

#include "elemental/lapack-like/HermitianEig.hpp"
#include "elemental/matrices/HermitianFromEVD.hpp"
#include "elemental/matrices/NormalFromEVD.hpp"

namespace elem {

//
// Modify the eigenvalues of A with the real-valued function f, which will 
// therefore result in a Hermitian matrix, which we store in-place.
//

template<typename F,class RealFunction>
inline void
RealHermitianFunction
( UpperOrLower uplo, Matrix<F>& A, RealFunction func )
{
#ifndef RELEASE
    CallStackEntry cse("RealHermitianFunction");
#endif
    if( A.Height() != A.Width() )
        LogicError("Hermitian matrices must be square");
    typedef Base<F> R;

    // Get the EVD of A
    Matrix<R> w;
    Matrix<F> Z;
    HermitianEig( uplo, A, w, Z );

    // Replace w with f(w)
    const Int n = w.Height();
    for( Int i=0; i<n; ++i )
    {
        const R omega = w.Get(i,0);
        w.Set(i,0,func(omega));
    }

    // A := Z f(Omega) Z^H
    HermitianFromEVD( uplo, A, w, Z );
}

template<typename F,class RealFunction>
inline void
RealHermitianFunction
( UpperOrLower uplo, DistMatrix<F>& A, RealFunction func )
{
#ifndef RELEASE
    CallStackEntry cse("RealHermitianFunction");
#endif
    EnsurePMRRR();
    if( A.Height() != A.Width() )
        LogicError("Hermitian matrices must be square");
    typedef Base<F> R;

    // Get the EVD of A
    const Grid& g = A.Grid();
    DistMatrix<R,VR,STAR> w(g);
    DistMatrix<F> Z(g);
    HermitianEig( uplo, A, w, Z );

    // Replace w with f(w)
    const Int numLocalEigs = w.LocalHeight();
    for( Int iLoc=0; iLoc<numLocalEigs; ++iLoc )
    {
        const R omega = w.GetLocal(iLoc,0);
        w.SetLocal(iLoc,0,func(omega));
    }

    // A := Z f(Omega) Z^H
    HermitianFromEVD( uplo, A, w, Z ); 
}

//
// Modify the eigenvalues of A with the complex-valued function f, which will
// therefore result in a normal (in general, non-Hermitian) matrix, which we 
// store in-place. At some point a version will be written which takes a real
// symmetric matrix as input and produces a complex normal matrix.
//

template<typename R,class Function>
inline void
ComplexHermitianFunction
( UpperOrLower uplo, Matrix<Complex<R> >& A, Function func )
{
#ifndef RELEASE
    CallStackEntry cse("ComplexHermitianFunction");
#endif
    if( A.Height() != A.Width() )
        LogicError("Hermitian matrices must be square");
    typedef Complex<R> C;

    // Get the EVD of A
    Matrix<R> w;
    Matrix<C> Z;
    HermitianEig( uplo, A, w, Z );

    // Form f(w)
    const Int n = w.Height();
    Matrix<C> fw( n, 1 );
    for( Int i=0; i<n; ++i )
    {
        const R omega = w.Get(i,0);
        fw.Set(i,0,func(omega));
    }

    // A := Z f(Omega) Z^H
    NormalFromEVD( A, fw, Z );
}

template<typename R,class Function>
inline void
ComplexHermitianFunction
( UpperOrLower uplo, DistMatrix<Complex<R> >& A, Function func )
{
#ifndef RELEASE
    CallStackEntry cse("ComplexHermitianFunction");
#endif
    EnsurePMRRR();
    if( A.Height() != A.Width() )
        LogicError("Hermitian matrices must be square");
    typedef Complex<R> C;

    // Get the EVD of A
    const Grid& g = A.Grid();
    DistMatrix<R,VR,STAR> w(g);
    DistMatrix<C> Z(g);
    HermitianEig( uplo, A, w, Z );

    // Form f(w)
    DistMatrix<C,VR,STAR> fw(g);
    fw.AlignWith( w.DistData() );
    fw.ResizeTo( w.Height(), 1 );
    const Int numLocalEigs = w.LocalHeight();
    for( Int iLoc=0; iLoc<numLocalEigs; ++iLoc )
    {
        const R omega = w.GetLocal(iLoc,0);
        fw.SetLocal(iLoc,0,func(omega));
    }

    // A := Z f(Omega) Z^H
    NormalFromEVD( A, fw, Z );
}

} // namespace elem

#endif // ifndef ELEM_LAPACK_HERMITIANFUNCTION_HPP
