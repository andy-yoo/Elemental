#ifndef HYDROGEN_HAVE_CUTLASS_HPP_
#define HYDROGEN_HAVE_CUTLASS_HPP_

#include <cublas_v2.h>

#include <El/hydrogen_config.h>
#include <hydrogen/blas/BLAS_Common.hpp>
#include <hydrogen/device/gpu/CUDA.hpp>
#include <hydrogen/utils/HalfPrecision.hpp>
#include <hydrogen/utils/NumericTypeConversion.hpp>
#include <hydrogen/SyncInfo.hpp>


namespace hydrogen
{

namespace gpu_cutlass_blas
{

  template <typename T>
  void Gemm_cutlass(
    char transpA, char transpB,
    int m, int n, int k,
    T const& alpha,
    T const* A, int lda,
    T const* B, int ldb,
    T const& beta,
    T* C, int ldc);

}// gpu_cutlass_blas
}// namespace hydrogen
#endif // HYDROGEN_HAVE_CUTLASS_HPP_

