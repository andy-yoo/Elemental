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
#include "elemental/blas_internal.hpp"
using namespace std;
using namespace elemental;

// Template conventions:
//   G: general datatype
//
//   T: any ring, e.g., the (Gaussian) integers and the real/complex numbers
//   Z: representation of a real ring, e.g., the integers or real numbers
//   std::complex<Z>: representation of a complex ring, e.g. Gaussian integers
//                    or complex numbers
//
//   F: representation of real or complex number
//   R: representation of real number
//   std::complex<R>: representation of complex number

// Right Upper Normal (Non)Unit Trmm
//   X := X triu(U), and
//   X := X triuu(U)
template<typename T>
void
elemental::blas::internal::TrmmRUN
( Diagonal diagonal,
  T alpha, const DistMatrix<T,MC,MR>& U,
                 DistMatrix<T,MC,MR>& X )
{
#ifndef RELEASE
    PushCallStack("blas::internal::TrmmRUN");
    if( U.Grid() != X.Grid() )
        throw logic_error( "U and X must be distributed over the same grid." );
    if( U.Height() != U.Width() || X.Width() != U.Height() )
    {
        ostringstream msg;
        msg << "Nonconformal TrmmRUN: " << endl
            << "  U ~ " << U.Height() << " x " << U.Width() << endl
            << "  X ~ " << X.Height() << " x " << X.Width() << endl;
        throw logic_error( msg.str() );
    }
#endif
    const Grid& g = U.Grid();

    // Matrix views
    DistMatrix<T,MC,MR> 
        UTL(g), UTR(g),  U00(g), U01(g), U02(g),
        UBL(g), UBR(g),  U10(g), U11(g), U12(g),
                         U20(g), U21(g), U22(g);

    DistMatrix<T,MC,MR> XL(g), XR(g),
                        X0(g), X1(g), X2(g);

    // Temporary distributions
    DistMatrix<T,MR,  Star> U01_MR_Star(g);
    DistMatrix<T,Star,Star> U11_Star_Star(g); 
    DistMatrix<T,VC,  Star> X1_VC_Star(g);    
    DistMatrix<T,MC,  Star> D1_MC_Star(g);
    
    // Start the algorithm
    blas::Scal( alpha, X );
    LockedPartitionUpDiagonal
    ( U, UTL, UTR,
         UBL, UBR, 0 );
    PartitionLeft( X, XL, XR, 0 );
    while( XL.Width() > 0 )
    {
        LockedRepartitionUpDiagonal
        ( UTL, /**/ UTR,  U00, U01, /**/ U02,
               /**/       U10, U11, /**/ U12, 
         /*************/ /******************/
          UBL, /**/ UBR,  U20, U21, /**/ U22 );

        RepartitionLeft
        ( XL,     /**/ XR,
          X0, X1, /**/ X2 ); 

        U01_MR_Star.AlignWith( X0 );
        D1_MC_Star.AlignWith( X1 );
        D1_MC_Star.ResizeTo( X1.Height(), X1.Width() );
        //--------------------------------------------------------------------//
        X1_VC_Star = X1;
        U11_Star_Star = U11;
        blas::internal::LocalTrmm
        ( Right, Upper, Normal, diagonal, (T)1, U11_Star_Star, X1_VC_Star );
        X1 = X1_VC_Star;
 
        U01_MR_Star = U01;
        blas::internal::LocalGemm
        ( Normal, Normal, (T)1, X0, U01_MR_Star, (T)0, D1_MC_Star );
        X1.SumScatterUpdate( (T)1, D1_MC_Star );
       //--------------------------------------------------------------------//
        U01_MR_Star.FreeAlignments();
        D1_MC_Star.FreeAlignments();

        SlideLockedPartitionUpDiagonal
        ( UTL, /**/ UTR,  U00, /**/ U01, U02,
         /*************/ /******************/
               /**/       U10, /**/ U11, U12,
          UBL, /**/ UBR,  U20, /**/ U21, U22 );

        SlidePartitionLeft
        ( XL, /**/ XR,
          X0, /**/ X1, X2 );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template void elemental::blas::internal::TrmmRUN
( Diagonal diagonal,
  float alpha, const DistMatrix<float,MC,MR>& U,
                     DistMatrix<float,MC,MR>& X );

template void elemental::blas::internal::TrmmRUN
( Diagonal diagonal,
  double alpha, const DistMatrix<double,MC,MR>& U,
                      DistMatrix<double,MC,MR>& X );

#ifndef WITHOUT_COMPLEX
template void elemental::blas::internal::TrmmRUN
( Diagonal diagonal,
  scomplex alpha, const DistMatrix<scomplex,MC,MR>& U,
                        DistMatrix<scomplex,MC,MR>& X );

template void elemental::blas::internal::TrmmRUN
( Diagonal diagonal,
  dcomplex alpha, const DistMatrix<dcomplex,MC,MR>& U,
                        DistMatrix<dcomplex,MC,MR>& X );
#endif

