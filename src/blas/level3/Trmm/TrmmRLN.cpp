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

// Right Lower Normal (Non)Unit Trmm
//   X := X tril(L), and
//   X := X trilu(L)
template<typename T>
void
elemental::blas::internal::TrmmRLN
( Diagonal diagonal,
  T alpha, const DistMatrix<T,MC,MR>& L,
                 DistMatrix<T,MC,MR>& X )
{
#ifndef RELEASE
    PushCallStack("blas::internal::TrmmRLN");
    if( L.Grid() != X.Grid() )
        throw logic_error( "L and X must be distributed over the same grid." );
    if( L.Height() != L.Width() || X.Width() != L.Height() )
    {
        ostringstream msg;
        msg << "Nonconformal TrmmRLN: " << endl
            << "  L ~ " << L.Height() << " x " << L.Width() << endl
            << "  X ~ " << X.Height() << " x " << X.Width() << endl;
        throw logic_error( msg.str() );
    }
#endif
    const Grid& g = L.Grid();

    // Matrix views
    DistMatrix<T,MC,MR> 
        LTL(g), LTR(g),  L00(g), L01(g), L02(g),
        LBL(g), LBR(g),  L10(g), L11(g), L12(g),
                         L20(g), L21(g), L22(g);

    DistMatrix<T,MC,MR> XL(g), XR(g),
                        X0(g), X1(g), X2(g);

    // Temporary distributions
    DistMatrix<T,Star,Star> L11_Star_Star(g);
    DistMatrix<T,MR,  Star> L21_MR_Star(g);
    DistMatrix<T,VC,  Star> X1_VC_Star(g);
    DistMatrix<T,MC,  Star> D1_MC_Star(g);

    // Start the algorithm
    blas::Scal( alpha, X );
    LockedPartitionDownDiagonal
    ( L, LTL, LTR,
         LBL, LBR, 0 );
    PartitionRight( X, XL, XR, 0 );
    while( XR.Width() > 0 )
    {
        LockedRepartitionDownDiagonal
        ( LTL, /**/ LTR,  L00, /**/ L01, L02,
         /*************/ /******************/
               /**/       L10, /**/ L11, L12,
          LBL, /**/ LBR,  L20, /**/ L21, L22 );
 
        RepartitionRight
        ( XL, /**/ XR,
          X0, /**/ X1, X2 );

        L21_MR_Star.AlignWith( X2 );
        D1_MC_Star.AlignWith( X1 );
        D1_MC_Star.ResizeTo( X1.Height(), X1.Width() );
        //--------------------------------------------------------------------//
        X1_VC_Star = X1;
        L11_Star_Star = L11;
        blas::internal::LocalTrmm
        ( Right, Lower, Normal, diagonal, (T)1, L11_Star_Star, X1_VC_Star );
        X1 = X1_VC_Star;
 
        L21_MR_Star = L21;
        blas::internal::LocalGemm
        ( Normal, Normal, (T)1, X2, L21_MR_Star, (T)0, D1_MC_Star );
        X1.SumScatterUpdate( (T)1, D1_MC_Star );
       //--------------------------------------------------------------------//
        L21_MR_Star.FreeAlignments();
        D1_MC_Star.FreeAlignments();

        SlideLockedPartitionDownDiagonal
        ( LTL, /**/ LTR,  L00, L01, /**/ L02,
               /**/       L10, L11, /**/ L12,
         /*************/ /******************/
          LBL, /**/ LBR,  L20, L21, /**/ L22 );

        SlidePartitionRight
        ( XL,     /**/ XR,
          X0, X1, /**/ X2 );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template void elemental::blas::internal::TrmmRLN
( Diagonal diagonal,
  float alpha, const DistMatrix<float,MC,MR>& L,
                     DistMatrix<float,MC,MR>& X );

template void elemental::blas::internal::TrmmRLN
( Diagonal diagonal,
  double alpha, const DistMatrix<double,MC,MR>& L,
                      DistMatrix<double,MC,MR>& X );

#ifndef WITHOUT_COMPLEX
template void elemental::blas::internal::TrmmRLN
( Diagonal diagonal,
  scomplex alpha, const DistMatrix<scomplex,MC,MR>& L,
                        DistMatrix<scomplex,MC,MR>& X );

template void elemental::blas::internal::TrmmRLN
( Diagonal diagonal,
  dcomplex alpha, const DistMatrix<dcomplex,MC,MR>& L,
                        DistMatrix<dcomplex,MC,MR>& X );
#endif

