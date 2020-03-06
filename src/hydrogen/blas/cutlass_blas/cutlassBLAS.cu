#ifndef HYDROGEN_HAVE_CUTLASS_HPP_
#define HYDROGEN_HAVE_CUTLASS_HPP_

#include <cublas_v2.h>

#include <El/hydrogen_config.h>
#include <hydrogen/blas/BLAS_Common.hpp>
#include <hydrogen/device/gpu/CUDA.hpp>
#include <hydrogen/utils/HalfPrecision.hpp>
#include <hydrogen/utils/NumericTypeConversion.hpp>
#include <hydrogen/SyncInfo.hpp>

// Helper methods to check for errors
//#include "helper.h"
// Defines cutlass::gemm::device::Gemm, the generic Gemm computation template class.
#include "cutlass/gemm/device/gemm.h"

#include<vector>

namespace hydrogen
{

namespace gpu_cutlass_blas
{

void Gemm_cutlass_float(
    char transpA, char transpB,
    int m, int n, int k,
    float const& alpha,
    float const* A, int lda,
    float const* B, int ldb,
    float const& beta,
    float* C, int ldc){


/*
const int ldim = lda;
std::vector<float> host_A(m*k,  0);
cudaMemcpy(host_A.data(), A, m*k*sizeof(float), cudaMemcpyDeviceToHost);
printf("host_A\n");
for( int j=0; j<(int)m*k; ++j ){
        printf("%lf ", host_A[j]);
}
printf("\n");
*/

    using ColumnMajor = cutlass::layout::ColumnMajor;
    using RowMajor = cutlass::layout::RowMajor;

    if(transpA == 'N' && transpB == 'N'){
       using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                   ColumnMajor,  // Layout of A matrix
                                                   float,        // Data-type of B matrix
                                                   ColumnMajor,  // Layout of B matrix
                                                   //RowMajor,  // Layout of B matrix
                                                   float,        // Data-type of C matrix
                                                   ColumnMajor>; // Layout of C matrix
 
       // Define a CUTLASS GEMM type
       CutlassGemm gemm_operator;


       // Construct the CUTLASS GEMM arguments object.
       //
       // One of CUTLASS's design patterns is to define gemm argument objects that are constructible
       // in host code and passed to kernels by value. These may include pointers, strides, scalars,
       // and other arguments needed by Gemm and its components.
       //
       // The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
       // arguments to kernels and (2.) minimized initialization overhead on kernel entry.
       //
       CutlassGemm::Arguments args({m , n, k},  // Gemm Problem dimensions
                                {A, lda},    // Tensor-ref for source matrix A
                                {B, ldb},    // Tensor-ref for source matrix B
                                {C, ldc},    // Tensor-ref for source matrix C
                                {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                {alpha, beta}); // Scalars used in the Epilogue

  
       cutlass::Status status = gemm_operator(args);
    }
    else if(transpA == 'N' && transpB == 'T'){
       using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                   ColumnMajor,  // Layout of A matrix
                                                   float,        // Data-type of B matrix
                                                   RowMajor,  // Layout of B matrix
                                                   float,        // Data-type of C matrix
                                                   ColumnMajor>; // Layout of C matrix
 
       // Define a CUTLASS GEMM type
       CutlassGemm gemm_operator;

       // Construct the CUTLASS GEMM arguments object.
       CutlassGemm::Arguments args({m , n, k},  // Gemm Problem dimensions
                                {A, lda},    // Tensor-ref for source matrix A
                                {B, ldb},    // Tensor-ref for source matrix B
                                {C, ldc},    // Tensor-ref for source matrix C
                                {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                {alpha, beta}); // Scalars used in the Epilogue

  
       cutlass::Status status = gemm_operator(args);
    }
    else if(transpA == 'T' && transpB == 'N'){
       using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                   RowMajor,  // Layout of A matrix
                                                   float,        // Data-type of B matrix
                                                   ColumnMajor,  // Layout of B matrix
                                                   float,        // Data-type of C matrix
                                                   ColumnMajor>; // Layout of C matrix
 
       // Define a CUTLASS GEMM type
       CutlassGemm gemm_operator;

       // Construct the CUTLASS GEMM arguments object.
       CutlassGemm::Arguments args({m , n, k},  // Gemm Problem dimensions
                                {A, lda},    // Tensor-ref for source matrix A
                                {B, ldb},    // Tensor-ref for source matrix B
                                {C, ldc},    // Tensor-ref for source matrix C
                                {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                {alpha, beta}); // Scalars used in the Epilogue

       cutlass::Status status = gemm_operator(args);
    }
    else if(transpA == 'T' && transpB == 'T'){
       using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                   RowMajor,  // Layout of A matrix
                                                   float,        // Data-type of B matrix
                                                   RowMajor,  // Layout of B matrix
                                                   float,        // Data-type of C matrix
                                                   ColumnMajor>; // Layout of C matrix
 
       // Define a CUTLASS GEMM type
       CutlassGemm gemm_operator;

       // Construct the CUTLASS GEMM arguments object.
       CutlassGemm::Arguments args({m , n, k},  // Gemm Problem dimensions
                                {A, lda},    // Tensor-ref for source matrix A
                                {B, ldb},    // Tensor-ref for source matrix B
                                {C, ldc},    // Tensor-ref for source matrix C
                                {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                {alpha, beta}); // Scalars used in the Epilogue

       cutlass::Status status = gemm_operator(args);
    }

/*
std::vector<float> host_C(m*n,  0);
cudaMemcpy(host_C.data(), C, m*n*sizeof(float), cudaMemcpyDeviceToHost);
printf("host_C\n");
for( int j=0; j<(int)m*k; ++j ){
        printf("%lf ", host_C[j]);
}
printf("\n");
*/


}


void Gemm_cutlass_double(
    char transpA, char transpB,
    int m, int n, int k,
    double const& alpha,
    double const* A, int lda,
    double const* B, int ldb,
    double const& beta,
    double* C, int ldc){

    using ColumnMajor = cutlass::layout::ColumnMajor;
    using RowMajor = cutlass::layout::RowMajor;
    if(transpA == 'N' && transpB == 'N'){
       using CutlassGemm = cutlass::gemm::device::Gemm<double,   // Data-type of A matrix
                                                   ColumnMajor,  // Layout of A matrix
                                                   double,       // Data-type of B matrix
                                                   ColumnMajor,  // Layout of B matrix
                                                   double,       // Data-type of C matrix
                                                   ColumnMajor>; // Layout of C matrix
 
       // Define a CUTLASS GEMM type
       CutlassGemm gemm_operator;

       // Construct the CUTLASS GEMM arguments object.
       CutlassGemm::Arguments args({m , n, k},  // Gemm Problem dimensions
                                {A, lda},    // Tensor-ref for source matrix A
                                {B, ldb},    // Tensor-ref for source matrix B
                                {C, ldc},    // Tensor-ref for source matrix C
                                {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                {alpha, beta}); // Scalars used in the Epilogue

  
       cutlass::Status status = gemm_operator(args);
    }
    else if(transpA == 'N' && transpB == 'T'){
       using CutlassGemm = cutlass::gemm::device::Gemm<double,   // Data-type of A matrix
                                                   ColumnMajor,  // Layout of A matrix
                                                   double,       // Data-type of B matrix
                                                   RowMajor,  // Layout of B matrix
                                                   double,       // Data-type of C matrix
                                                   ColumnMajor>; // Layout of C matrix
 
       // Define a CUTLASS GEMM type
       CutlassGemm gemm_operator;

       // Construct the CUTLASS GEMM arguments object.
       CutlassGemm::Arguments args({m , n, k},  // Gemm Problem dimensions
                                {A, lda},    // Tensor-ref for source matrix A
                                {B, ldb},    // Tensor-ref for source matrix B
                                {C, ldc},    // Tensor-ref for source matrix C
                                {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                {alpha, beta}); // Scalars used in the Epilogue

  
       cutlass::Status status = gemm_operator(args);
    }
    else if(transpA == 'T' && transpB == 'N'){
       using CutlassGemm = cutlass::gemm::device::Gemm<double,   // Data-type of A matrix
                                                   RowMajor,  // Layout of A matrix
                                                   double,       // Data-type of B matrix
                                                   ColumnMajor,  // Layout of B matrix
                                                   double,       // Data-type of C matrix
                                                   ColumnMajor>; // Layout of C matrix
 
       // Define a CUTLASS GEMM type
       CutlassGemm gemm_operator;

       // Construct the CUTLASS GEMM arguments object.
       CutlassGemm::Arguments args({m , n, k},  // Gemm Problem dimensions
                                {A, lda},    // Tensor-ref for source matrix A
                                {B, ldb},    // Tensor-ref for source matrix B
                                {C, ldc},    // Tensor-ref for source matrix C
                                {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                {alpha, beta}); // Scalars used in the Epilogue

  
       cutlass::Status status = gemm_operator(args);
    }
    else if(transpA == 'T' && transpB == 'T'){
       using CutlassGemm = cutlass::gemm::device::Gemm<double,   // Data-type of A matrix
                                                   RowMajor,  // Layout of A matrix
                                                   double,       // Data-type of B matrix
                                                   RowMajor,  // Layout of B matrix
                                                   double,       // Data-type of C matrix
                                                   ColumnMajor>; // Layout of C matrix
 
       // Define a CUTLASS GEMM type
       CutlassGemm gemm_operator;

       // Construct the CUTLASS GEMM arguments object.
       CutlassGemm::Arguments args({m , n, k},  // Gemm Problem dimensions
                                {A, lda},    // Tensor-ref for source matrix A
                                {B, ldb},    // Tensor-ref for source matrix B
                                {C, ldc},    // Tensor-ref for source matrix C
                                {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                {alpha, beta}); // Scalars used in the Epilogue

  
       cutlass::Status status = gemm_operator(args);
    }
   



}

void Gemm_cutlass_half(
    char transpA, char transpB,
    int m, int n, int k,
    cutlass::half_t const& alpha,
    cutlass::half_t const* A, int lda,
    cutlass::half_t const* B, int ldb,
    cutlass::half_t const& beta,
    cutlass::half_t* C, int ldc){

    using ColumnMajor = cutlass::layout::ColumnMajor;
    using RowMajor = cutlass::layout::RowMajor;
    if(transpA == 'N' && transpB == 'N'){
       using CutlassGemm = cutlass::gemm::device::Gemm<cutlass::half_t, // Data-type of A matrix
                                                   ColumnMajor,  // Layout of A matrix
                                                   cutlass::half_t,        // Data-type of B matrix
                                                   ColumnMajor,  // Layout of B matrix
                                                   cutlass::half_t,        // Data-type of C matrix
                                                   ColumnMajor>; // Layout of C matrix
 
       // Define a CUTLASS GEMM type
       CutlassGemm gemm_operator;

       // Construct the CUTLASS GEMM arguments object.
       CutlassGemm::Arguments args({m , n, k},  // Gemm Problem dimensions
                                {A, lda},    // Tensor-ref for source matrix A
                                {B, ldb},    // Tensor-ref for source matrix B
                                {C, ldc},    // Tensor-ref for source matrix C
                                {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                {alpha, beta}); // Scalars used in the Epilogue

       cutlass::Status status = gemm_operator(args);
    }
    else if(transpA == 'N' && transpB == 'T'){
       using CutlassGemm = cutlass::gemm::device::Gemm<cutlass::half_t, // Data-type of A matrix
                                                   ColumnMajor,  // Layout of A matrix
                                                   cutlass::half_t,        // Data-type of B matrix
                                                   RowMajor,  // Layout of B matrix
                                                   cutlass::half_t,        // Data-type of C matrix
                                                   ColumnMajor>; // Layout of C matrix
 
       // Define a CUTLASS GEMM type
       CutlassGemm gemm_operator;

       // Construct the CUTLASS GEMM arguments object.
       CutlassGemm::Arguments args({m , n, k},  // Gemm Problem dimensions
                                {A, lda},    // Tensor-ref for source matrix A
                                {B, ldb},    // Tensor-ref for source matrix B
                                {C, ldc},    // Tensor-ref for source matrix C
                                {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                {alpha, beta}); // Scalars used in the Epilogue

       cutlass::Status status = gemm_operator(args);
    }
    else if(transpA == 'T' && transpB == 'N'){
       using CutlassGemm = cutlass::gemm::device::Gemm<cutlass::half_t, // Data-type of A matrix
                                                   RowMajor,  // Layout of A matrix
                                                   cutlass::half_t,        // Data-type of B matrix
                                                   ColumnMajor,  // Layout of B matrix
                                                   cutlass::half_t,        // Data-type of C matrix
                                                   ColumnMajor>; // Layout of C matrix
 
       // Define a CUTLASS GEMM type
       CutlassGemm gemm_operator;

       // Construct the CUTLASS GEMM arguments object.
       CutlassGemm::Arguments args({m , n, k},  // Gemm Problem dimensions
                                {A, lda},    // Tensor-ref for source matrix A
                                {B, ldb},    // Tensor-ref for source matrix B
                                {C, ldc},    // Tensor-ref for source matrix C
                                {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                {alpha, beta}); // Scalars used in the Epilogue

       cutlass::Status status = gemm_operator(args);
    }
    else if(transpA == 'T' && transpB == 'T'){
       using CutlassGemm = cutlass::gemm::device::Gemm<cutlass::half_t, // Data-type of A matrix
                                                   RowMajor,  // Layout of A matrix
                                                   cutlass::half_t,        // Data-type of B matrix
                                                   RowMajor,  // Layout of B matrix
                                                   cutlass::half_t,        // Data-type of C matrix
                                                   ColumnMajor>; // Layout of C matrix
 
       // Define a CUTLASS GEMM type
       CutlassGemm gemm_operator;

       // Construct the CUTLASS GEMM arguments object.
       CutlassGemm::Arguments args({m , n, k},  // Gemm Problem dimensions
                                {A, lda},    // Tensor-ref for source matrix A
                                {B, ldb},    // Tensor-ref for source matrix B
                                {C, ldc},    // Tensor-ref for source matrix C
                                {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                {alpha, beta}); // Scalars used in the Epilogue

       cutlass::Status status = gemm_operator(args);
    }



}

  template <typename T>
  void Gemm_cutlass(
    char transpA, char transpB,
    int m, int n, int k,
    T const& alpha,
    T const* A, int lda,
    T const* B, int ldb,
    T const& beta,
    T* C, int ldc){

    if( std::is_same<T, float>::value){
  
	Gemm_cutlass_float(transpA, transpB, m, n, k,
    		(float)alpha,
    		(float *)A,  lda,
    		(float *)B,  ldb,
    		(float)beta,
    		(float *)C, ldc);
    }
    else if( std::is_same<T, double>::value){
  
	Gemm_cutlass_double(transpA, transpB, m, n, k,
    		(double)alpha,
    		(double *)A,  lda,
    		(double *)B,  ldb,
    		(double)beta,
    		(double *)C, ldc);

    }
    else if( std::is_same<T, __half>::value){
  
	Gemm_cutlass_half(transpA, transpB, m, n, k,
    		(cutlass::half_t)alpha,
    		(cutlass::half_t *)A,  lda,
    		(cutlass::half_t *)B,  ldb,
    		(cutlass::half_t)beta,
    		(cutlass::half_t *)C, ldc);

    }
    else
	return;

/*
    using ColumnMajor = cutlass::layout::ColumnMajor;
    using CutlassGemm = cutlass::gemm::device::Gemm<T,        // Data-type of A matrix
                                                   ColumnMajor,  // Layout of A matrix
                                                   T,        // Data-type of B matrix
                                                   ColumnMajor,  // Layout of B matrix
                                                   T,        // Data-type of C matrix
                                                   ColumnMajor>; // Layout of C matrix
 
    // Define a CUTLASS GEMM type
    CutlassGemm gemm_operator;


    // Construct the CUTLASS GEMM arguments object.
    //
    // One of CUTLASS's design patterns is to define gemm argument objects that are constructible
    // in host code and passed to kernels by value. These may include pointers, strides, scalars,
    // and other arguments needed by Gemm and its components.
    //
    // The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
    // arguments to kernels and (2.) minimized initialization overhead on kernel entry.
    //
    CutlassGemm::Arguments args({m , n, k},  // Gemm Problem dimensions
                                {A, lda},    // Tensor-ref for source matrix A
                                {B, ldb},    // Tensor-ref for source matrix B
                                {C, ldc},    // Tensor-ref for source matrix C
                                {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                {alpha, beta}); // Scalars used in the Epilogue

    //
    // Launch the CUTLASS GEMM kernel.
    //
  
    printf("calling cutlass::gemm::device::Gemm<...>.operator(args)...\n");
    cutlass::Status status = gemm_operator(args);

    //
    // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
    //

    //if (status != cutlass::Status::kSuccess) {
    //  return cudaErrorUnknown;
    //}
*/
  }


template void Gemm_cutlass(
    char transpA, char transpB,
    int m, int n, int k,
    float const& alpha,
    float const* A, int lda,
    float const* B, int ldb,
    float const& beta,
    float* C, int ldc);

template void Gemm_cutlass(
    char transpA, char transpB,
    int m, int n, int k,
    double const& alpha,
    double const* A, int lda,
    double const* B, int ldb,
    double const& beta,
    double* C, int ldc);

template void Gemm_cutlass(
    char transpA, char transpB,
    int m, int n, int k,
    __half const& alpha,
    __half const* A, int lda,
    __half const* B, int ldb,
    __half const& beta,
    __half* C, int ldc);
}// gpu_cutlass_blas
}// namespace hydrogen
#endif // HYDROGEN_HAVE_CUTLASS_HPP_

