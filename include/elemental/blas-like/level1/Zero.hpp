/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ELEM_BLAS_ZERO_HPP
#define ELEM_BLAS_ZERO_HPP

namespace elem {

template<typename T>
inline void
Zero( Matrix<T>& A )
{
#ifndef RELEASE
    CallStackEntry cse("Zero");
#endif
    const Int height = A.Height();
    const Int width = A.Width();
    PARALLEL_FOR
    for( Int j=0; j<width; ++j )
        MemZero( A.Buffer(0,j), height );
}

template<typename T,Distribution U,Distribution V>
inline void
Zero( DistMatrix<T,U,V>& A )
{
#ifndef RELEASE
    CallStackEntry cse("Zero");
#endif
    Zero( A.Matrix() );
}

} // namespace elem

#endif // ifndef ELEM_BLAS_ZERO_HPP
