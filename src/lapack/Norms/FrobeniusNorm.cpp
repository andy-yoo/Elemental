/*
   Copyright (c) 2009-2011, Jack Poulson
   All rights reserved.

   This file is part of Elemental.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

    - Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    - Neither the name of the owner nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
   ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
   LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.
*/
#include "elemental/lapack.hpp"
using namespace elemental;

template<typename R> // representation of a real number
R
lapack::FrobeniusNorm( const Matrix<R>& A )
{
#ifndef RELEASE
    PushCallStack("lapack::FrobeniusNorm");
#endif
    R normSquared = 0;
    for( int j=0; j<A.Width(); ++j )
    {
        for( int i=0; i<A.Height(); ++i )
        {
            R alpha = A.Get(i,j);
            normSquared += alpha*alpha;
        }
    }
    R norm = sqrt(normSquared);
#ifndef RELEASE
    PopCallStack();
#endif
    return norm;
}

#ifndef WITHOUT_COMPLEX
template<typename R> // representation of a real number
R
lapack::FrobeniusNorm( const Matrix< std::complex<R> >& A )
{
#ifndef RELEASE
    PushCallStack("lapack::FrobeniusNorm");
#endif
    R normSquared = 0;
    for( int j=0; j<A.Width(); ++j )
    {
        for( int i=0; i<A.Height(); ++i )
        {
            std::complex<R> alpha = A.Get(i,j);
            // The std::norm function is a field norm rather than a vector norm.
            normSquared += norm(alpha);
        }
    }
    R norm = sqrt(normSquared);
#ifndef RELEASE
    PopCallStack();
#endif
    return norm;
}
#endif

template<typename R> // representation of a real number
R
lapack::FrobeniusNorm( const DistMatrix<R,MC,MR>& A )
{
#ifndef RELEASE
    PushCallStack("lapack::FrobeniusNorm");
#endif
    R localNormSquared = 0;
    for( int j=0; j<A.LocalWidth(); ++j )
    {
        for( int i=0; i<A.LocalHeight(); ++i )
        {
            R alpha = A.GetLocalEntry(i,j);
            localNormSquared += alpha*alpha;
        }
    }

    // The norm squared is simply the sum of the local contributions
    R normSquared;
    wrappers::mpi::AllReduce
    ( &localNormSquared, &normSquared, 1, MPI_SUM, A.Grid().VCComm() );

    R norm = sqrt(normSquared);
#ifndef RELEASE
    PopCallStack();
#endif
    return norm;
}

#ifndef WITHOUT_COMPLEX
template<typename R> // representation of a real number
R
lapack::FrobeniusNorm( const DistMatrix<std::complex<R>,MC,MR>& A )
{
#ifndef RELEASE
    PushCallStack("lapack::FrobeniusNorm");
#endif
    R localNormSquared = 0;
    for( int j=0; j<A.LocalWidth(); ++j )
    {
        for( int i=0; i<A.LocalHeight(); ++i )
        {
            std::complex<R> alpha = A.GetLocalEntry(i,j);
            // The std::norm function is a field norm rather than a vector norm.
            localNormSquared += norm(alpha);
        }
    }

    // The norm squared is simply the sum of the local contributions
    R normSquared;
    wrappers::mpi::AllReduce
    ( &localNormSquared, &normSquared, 1, MPI_SUM, A.Grid().VCComm() );

    R norm = sqrt(normSquared);
#ifndef RELEASE
    PopCallStack();
#endif
    return norm;
}
#endif

template float elemental::lapack::FrobeniusNorm
( const Matrix<float>& A );
template double elemental::lapack::FrobeniusNorm
( const Matrix<double>& A );
#ifndef WITHOUT_COMPLEX
template float elemental::lapack::FrobeniusNorm
( const Matrix< std::complex<float> >& A );
template double elemental::lapack::FrobeniusNorm
( const Matrix< std::complex<double> >& A );
#endif

template float elemental::lapack::FrobeniusNorm
( const DistMatrix<float,MC,MR>& A );
template double elemental::lapack::FrobeniusNorm
( const DistMatrix<double,MC,MR>& A );
#ifndef WITHOUT_COMPLEX
template float elemental::lapack::FrobeniusNorm
( const DistMatrix<std::complex<float>,MC,MR>& A );
template double elemental::lapack::FrobeniusNorm
( const DistMatrix<std::complex<double>,MC,MR>& A );
#endif

