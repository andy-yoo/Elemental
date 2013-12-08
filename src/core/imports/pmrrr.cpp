/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "elemental.hpp"

#ifdef HAVE_PMRRR

extern "C" {

int PMRRR
( const char* jobz,  // 'N' ~ only eigenvalues, 'V' ~ also eigenvectors
  const char* range, // 'A'~all eigenpairs, 'V'~interval (vl,vu], 'I'~il-iu
  const int* n,      // size of matrix
        double* d,   // full diagonal of tridiagonal matrix [length n]
        double* e,   // full subdiagonal in first n-1 entries [length n]
  const double* vl,  // if range=='V', compute eigenpairs in (vl,vu]
  const double* vu,
  const int* il, // if range=='I', compute il-iu eigenpairs
  const int* iu,
  int* tryrac, // if nonzero, try for high relative accuracy
  MPI_Comm comm,
  int* nz,        // number of locally computed eigenvectors
  int* offset,    // the first eigenpair computed by our process
  double* w,      // eigenvalues corresponding to local eigenvectors [length nz]
  double* Z,      // local eigenvectors [size ldz x nz]
  const int* ldz, // leading dimension of Z
  int* ZSupp      // support of eigenvectors [length 2n]
);

} // extern "C"

#endif // ifdef HAVE_PMRRR

namespace elem {
namespace pmrrr {

// Return upper bounds on the number of (local) eigenvalues in the given range,
// (lowerBound,upperBound]
Estimate EigEstimate
( int n, double* d, double* e, double* w, mpi::Comm comm, 
  double lowerBound, double upperBound )
{
    DEBUG_ONLY(CallStackEntry cse("pmrrr::EigEstimate"))
    Estimate estimate;
#ifdef HAVE_PMRRR
    char jobz='C';
    char range='V';
    int il, iu;
    int highAccuracy=0;
    int nz, offset;
    int ldz=1;
    std::vector<int> ZSupport(2*n);
    int retval = PMRRR
    ( &jobz, &range, &n, d, e, &lowerBound, &upperBound, &il, &iu, 
      &highAccuracy, comm, &nz, &offset, w, 0, &ldz, ZSupport.data() );
    if( retval != 0 )
    {
        std::ostringstream msg;
        msg << "PMRRR returned " << retval; 
        RuntimeError( msg.str() );
    }

    estimate.numLocalEigenvalues = nz;
    estimate.numGlobalEigenvalues = mpi::AllReduce( nz, comm );
#else
    EnsurePMRRR();
#endif
    return estimate;
}

// Compute all of the eigenvalues
Info Eig( int n, double* d, double* e, double* w, mpi::Comm comm )
{
    DEBUG_ONLY(CallStackEntry cse("pmrrr::Eig"))
    Info info;
#ifdef HAVE_PMRRR
    char jobz='N';
    char range='A';
    double vl, vu;
    int il, iu;
    int highAccuracy=0; 
    int nz, offset;
    int ldz=1;
    std::vector<int> ZSupport(2*n);
    int retval = PMRRR
    ( &jobz, &range, &n, d, e, &vl, &vu, &il, &iu, &highAccuracy, comm,
      &nz, &offset, w, 0, &ldz, ZSupport.data() );
    if( retval != 0 )
    {
        std::ostringstream msg;        
        msg << "PMRRR returned " << retval;
        RuntimeError( msg.str() );
    }

    info.numLocalEigenvalues=nz;
    info.firstLocalEigenvalue=offset;
    info.numGlobalEigenvalues=n;
#else
    EnsurePMRRR();
#endif
    return info;
}

// Compute all of the eigenpairs
Info Eig
( int n, double* d, double* e, double* w, double* Z, int ldz, mpi::Comm comm )
{
    DEBUG_ONLY(CallStackEntry cse("pmrrr::Eig"))
    Info info;
#ifdef HAVE_PMRRR
    char jobz='V';
    char range='A';
    double vl, vu;
    int il, iu;
    int highAccuracy=0; 
    int nz, offset;
    std::vector<int> ZSupport(2*n);
    int retval = PMRRR
    ( &jobz, &range, &n, d, e, &vl, &vu, &il, &iu, &highAccuracy, comm,
      &nz, &offset, w, Z, &ldz, ZSupport.data() );
    if( retval != 0 )
    {
        std::ostringstream msg;        
        msg << "PMRRR returned " << retval;
        RuntimeError( msg.str() );
    }

    info.numLocalEigenvalues=nz;
    info.firstLocalEigenvalue=offset;
    info.numGlobalEigenvalues=n;
#else
    EnsurePMRRR();
#endif
    return info;
}

// Compute all of the eigenvalues in (lowerBound,upperBound]
Info Eig
( int n, double* d, double* e, double* w, mpi::Comm comm, 
  double lowerBound, double upperBound )
{
    DEBUG_ONLY(CallStackEntry cse("pmrrr::Eig"))
    Info info;
#ifdef HAVE_PMRRR
    char jobz='N';
    char range='V';
    int il, iu;
    int highAccuracy=0; 
    int nz, offset;
    int ldz=1;
    std::vector<int> ZSupport(2*n);
    int retval = PMRRR
    ( &jobz, &range, &n, d, e, &lowerBound, &upperBound, &il, &iu, 
      &highAccuracy, comm, &nz, &offset, w, 0, &ldz, ZSupport.data() );
    if( retval != 0 )
    {
        std::ostringstream msg;        
        msg << "PMRRR returned " << retval;
        RuntimeError( msg.str() );
    }

    info.numLocalEigenvalues=nz;
    info.firstLocalEigenvalue=offset;
    mpi::AllReduce( &nz, &info.numGlobalEigenvalues, 1, mpi::SUM, comm );
#else
    EnsurePMRRR();
#endif
    return info;
}

// Compute all of the eigenpairs with eigenvalues in (lowerBound,upperBound]
Info Eig
( int n, double* d, double* e, double* w, double* Z, int ldz, mpi::Comm comm, 
  double lowerBound, double upperBound )
{
    DEBUG_ONLY(CallStackEntry cse("pmrrr::Eig"))
    Info info;
#ifdef HAVE_PMRRR
    char jobz='V';
    char range='V';
    int il, iu;
    int highAccuracy=0; 
    int nz, offset;
    std::vector<int> ZSupport(2*n);
    int retval = PMRRR
    ( &jobz, &range, &n, d, e, &lowerBound, &upperBound, &il, &iu, 
      &highAccuracy, comm, &nz, &offset, w, Z, &ldz, ZSupport.data() );
    if( retval != 0 )
    {
        std::ostringstream msg;        
        msg << "PMRRR returned " << retval;
        RuntimeError( msg.str() );
    }

    info.numLocalEigenvalues=nz;
    info.firstLocalEigenvalue=offset;
    mpi::AllReduce( &nz, &info.numGlobalEigenvalues, 1, mpi::SUM, comm );
#else
    EnsurePMRRR();
#endif
    return info;
}

// Compute all of the eigenvalues with indices in [lowerBound,upperBound]
Info Eig
( int n, double* d, double* e, double* w, mpi::Comm comm, 
  int lowerBound, int upperBound )
{
    DEBUG_ONLY(CallStackEntry cse("pmrrr::Eig"))
    Info info;
#ifdef HAVE_PMRRR
    ++lowerBound;
    ++upperBound;
    char jobz='N';
    char range='I';
    double vl, vu;
    int highAccuracy=0; 
    int nz, offset;
    int ldz=1;
    std::vector<int> ZSupport(2*n);
    int retval = PMRRR
    ( &jobz, &range, &n, d, e, &vl, &vu, &lowerBound, &upperBound, 
      &highAccuracy, comm, &nz, &offset, w, 0, &ldz, ZSupport.data() );
    if( retval != 0 )
    {
        std::ostringstream msg;        
        msg << "PMRRR returned " << retval;
        RuntimeError( msg.str() );
    }

    info.numLocalEigenvalues=nz;
    info.firstLocalEigenvalue=offset;
    info.numGlobalEigenvalues=(upperBound-lowerBound)+1;
#else
    EnsurePMRRR();
#endif
    return info;
}

// Compute all of the eigenpairs with eigenvalues indices in 
// [lowerBound,upperBound]
Info Eig
( int n, double* d, double* e, double* w, double* Z, int ldz, mpi::Comm comm, 
  int lowerBound, int upperBound )
{
    DEBUG_ONLY(CallStackEntry cse("pmrrr::Eig"))
    Info info;
#ifdef HAVE_PMRRR
    ++lowerBound;
    ++upperBound;
    char jobz='V';
    char range='I';
    double vl, vu;
    int highAccuracy=0; 
    int nz, offset;
    std::vector<int> ZSupport(2*n);
    int retval = PMRRR
    ( &jobz, &range, &n, d, e, &vl, &vu, &lowerBound, &upperBound, 
      &highAccuracy, comm, &nz, &offset, w, Z, &ldz, ZSupport.data() );
    if( retval != 0 )
    {
        std::ostringstream msg;        
        msg << "PMRRR returned " << retval;
        RuntimeError( msg.str() );
    }

    info.numLocalEigenvalues=nz;
    info.firstLocalEigenvalue=offset;
    info.numGlobalEigenvalues=(upperBound-lowerBound)+1;
#else
    EnsurePMRRR();
#endif
    return info;
}

} // namespace pmrrr
} // namespace elem
