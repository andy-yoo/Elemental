#ifndef HYDROGEN_IMPORTS_CUBLAS_HPP_
#define HYDROGEN_IMPORTS_CUBLAS_HPP_

#include <cublas_v2.h>

#include <El/hydrogen_config.h>
#include <hydrogen/blas/BLAS_Common.hpp>
#include <hydrogen/device/gpu/CUDA.hpp>
#include <hydrogen/utils/HalfPrecision.hpp>
#include <hydrogen/utils/NumericTypeConversion.hpp>
#include <hydrogen/SyncInfo.hpp>

namespace hydrogen
{

#define ADD_ENUM_TO_STRING_CASE(enum_value) \
    case enum_value:                        \
        return #enum_value

/** \class cuBLASError
 *  \brief Exception class for cuBLAS errors.
 */
struct cuBLASError : std::runtime_error
{
    static std::string get_error_string_(cublasStatus_t status)
    {
        switch (status)
        {
        ADD_ENUM_TO_STRING_CASE(CUBLAS_STATUS_SUCCESS);
        ADD_ENUM_TO_STRING_CASE(CUBLAS_STATUS_NOT_INITIALIZED);
        ADD_ENUM_TO_STRING_CASE(CUBLAS_STATUS_ALLOC_FAILED);
        ADD_ENUM_TO_STRING_CASE(CUBLAS_STATUS_INVALID_VALUE);
        ADD_ENUM_TO_STRING_CASE(CUBLAS_STATUS_ARCH_MISMATCH);
        ADD_ENUM_TO_STRING_CASE(CUBLAS_STATUS_MAPPING_ERROR);
        ADD_ENUM_TO_STRING_CASE(CUBLAS_STATUS_EXECUTION_FAILED);
        ADD_ENUM_TO_STRING_CASE(CUBLAS_STATUS_INTERNAL_ERROR);
        ADD_ENUM_TO_STRING_CASE(CUBLAS_STATUS_NOT_SUPPORTED);
        ADD_ENUM_TO_STRING_CASE(CUBLAS_STATUS_LICENSE_ERROR);
        default:
            return "unknown cuBLAS error";
        }
    }

    std::string build_error_string_(
        cublasStatus_t status, char const* file, int line)
    {
        std::ostringstream oss;
        oss << "cuBLAS error (" << file << ":" << line << "): "
            << get_error_string_(status);
        return oss.str();
    }

    cuBLASError(cublasStatus_t status, char const* file, int line)
        : std::runtime_error{build_error_string_(status,file,line)}
    {}
};// struct cublasError

#undef ADD_ENUM_TO_STRING_CASE

#define H_FORCE_CHECK_CUBLAS(cublas_call)                              \
    do                                                                  \
    {                                                                   \
        /* Check for earlier asynchronous errors. */                    \
        H_FORCE_CHECK_CUDA(cudaSuccess);                               \
        {                                                               \
            /* Make cuBLAS call and check for errors. */                \
            const cublasStatus_t status_CHECK_CUBLAS = (cublas_call);   \
            if (status_CHECK_CUBLAS != CUBLAS_STATUS_SUCCESS)           \
            {                                                           \
              cudaDeviceReset();                                        \
              throw cuBLASError(status_CHECK_CUBLAS,__FILE__,__LINE__); \
            }                                                           \
        }                                                               \
        {                                                               \
            /* Check for CUDA errors. */                                \
            cudaError_t status_CHECK_CUBLAS = cudaDeviceSynchronize();  \
            if (status_CHECK_CUBLAS == cudaSuccess)                     \
                status_CHECK_CUBLAS = cudaGetLastError();               \
            if (status_CHECK_CUBLAS != cudaSuccess)                     \
            {                                                           \
                cudaDeviceReset();                                      \
                throw CudaError(                                        \
                    status_CHECK_CUBLAS,__FILE__,__LINE__,false);       \
            }                                                           \
        }                                                               \
    } while (0)

#define H_FORCE_CHECK_CUBLAS_NOSYNC(cublas_call)                       \
    do                                                                  \
    {                                                                   \
        /* Make cuBLAS call and check for errors without */             \
        /* synchronizing. */                                            \
        const cublasStatus_t status_CHECK_CUBLAS = (cublas_call);       \
        if (status_CHECK_CUBLAS != CUBLAS_STATUS_SUCCESS)               \
        {                                                               \
            cudaDeviceReset();                                          \
            throw cuBLASError(status_CHECK_CUBLAS,__FILE__,__LINE__);   \
        }                                                               \
    } while (0)

#ifdef HYDROGEN_RELEASE_BUILD
#define H_CHECK_CUBLAS(cublas_call)            \
    H_FORCE_CHECK_CUBLAS_NOSYNC(cublas_call)
#else
#define H_CHECK_CUBLAS(cublas_call)            \
    H_FORCE_CHECK_CUBLAS(cublas_call)
#endif // #ifdef HYDROGEN_RELEASE_BUILD

namespace cublas
{

/** @name cuBLAS utility functions. */
///@{

/** @brief Initialize CUBLAS.
 *
 *  This must be called after `MPI_Init` is called with
 *  MVAPICH2-GDR. Effectively, this creates the global cuBLAS library
 *  handle.
 */
void Initialize();

/** @class NativeType
 *  @brief Metafunction mapping type names to CUDA/cuBLAS equivalents.
 *
 *  The mapping should provide bitwise equivalence.
 *
 *  @note This belongs at this level because rocBLAS defines types (or
 *        names of types) that are local to the BLAS
 *        implementation. Additionally, it's feasible to conceive of
 *        custom types on the GPU that would, likewise, need to be
 *        mapped to the types that cuBLAS knows about.
 *
 *  @todo Add static assertions to ensure only valid types get mapped.
 */
template <typename T>
struct NativeTypeT;

// Built-in types are their own native types
template <> struct NativeTypeT<float> { using type = float; };
template <> struct NativeTypeT<double> { using type = double; };
template <>
struct NativeTypeT<cuComplex> { using type = cuComplex; };
template <>
struct NativeTypeT<cuDoubleComplex> { using type = cuDoubleComplex; };

// Complex and Double-Complex types require conversion
template <>
struct NativeTypeT<std::complex<float>> { using type = cuComplex; };
template <>
struct NativeTypeT<std::complex<double>> { using type = cuDoubleComplex; };

// Half precision requires conversion as well
#ifdef HYDROGEN_GPU_USE_FP16
template <> struct NativeTypeT<__half> { using type = __half; };
#ifdef HYDROGEN_HAVE_HALF
template <> struct NativeTypeT<cpu_half_type> { using type = __half; };
#endif // HYDROGEN_HAVE_HALF
#endif // HYDROGEN_GPU_USE_FP16

/** @brief Convenience wrapper for NativeTypeT */
template <typename T>
using NativeType = typename NativeTypeT<T>::type;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace meta_details
{
template <typename T>
auto Try_HasNativeType(int) -> SubstitutionSuccess<NativeType<T>>;
template <typename T>
auto Try_HasNativeType(...) -> std::false_type;
}// namespace meta_details
#endif // DOXYGEN_SHOULD_SKIP_THIS

/** @struct HasNativeType
 *  @brief Predicate that determines if a type is mappable to a
 *         library-native type.
 */
template <typename T>
struct HasNativeType : decltype(meta_details::Try_HasNativeType<T>(0)) {};

/** @class IsSupportedType_Base
 *  @brief Predicate indicating that a type is supported within cuBLAS
 *         for the given operation.
 *
 *  This is used to map internal cuBLAS types to the operations that
 *  are supported. For example, `float` is always supported but
 *  `__half` only has support in a few functions.
 */
template <typename T, BLAS_Op op>
struct IsSupportedType_Base : std::false_type {};

template <BLAS_Op op>
struct IsSupportedType_Base<float, op> : std::true_type {};
template <BLAS_Op op>
struct IsSupportedType_Base<double, op> : std::true_type {};
template <BLAS_Op op>
struct IsSupportedType_Base<cuComplex, op> : std::true_type {};
template <BLAS_Op op>
struct IsSupportedType_Base<cuDoubleComplex, op> : std::true_type {};

// No need to further test CUDA because this file isn't included if
// either we don't have GPUs at all or we don't have CUDA support.
#ifdef HYDROGEN_GPU_USE_FP16
template <>
struct IsSupportedType_Base<__half, BLAS_Op::AXPY> : std::true_type {};
template <>
struct IsSupportedType_Base<__half, BLAS_Op::GEMM> : std::true_type {};
template <>
struct IsSupportedType_Base<__half, BLAS_Op::SCAL> : std::true_type {};
#endif // HYDROGEN_GPU_USE_FP16

/** @class IsSupportedType
 *  @brief Predicate indicating that the given type is compatible with
 *         cuBLAS.
 *
 *  This is true when either the type is a compatible cuBLAS type
 *  (e.g., float) or when it is binarily equivalent to one (e.g.,
 *  std::complex<float>)..
 */
template <typename T, BLAS_Op op, bool=HasNativeType<T>::value>
struct IsSupportedType
    : IsSupportedType_Base<NativeType<T>, op>
{};

template <typename T, BLAS_Op op>
struct IsSupportedType<T,op,false> : std::false_type {};

/** @brief cuBLAS uses ints to represent sizes. */
using SizeT = int;

/** @brief Convert a value to the size type expected by the cuBLAS
 *         library.
 *
 *  If `HYDROGEN_DO_BOUNDS_CHECKING` is defined, this will do a
 *  "safe cast" (it will verify that `val` is in the dynamic range of
 *  `int`. Otherwise it will do a regular static_cast.
 */
template <typename T>
#ifdef HYDROGEN_DO_BOUNDS_CHECKING
SizeT ToSizeT(T const& val)
{
    return narrow_cast<SizeT>(val);
}
#else
SizeT ToSizeT(T const& val) noexcept
{
    return static_cast<SizeT>(val);
}
#endif // HYDROGEN_DO_BOUNDS_CHECKING

/** @brief Overload to prevent extra work in the case of dynamic range checking. */
inline SizeT ToSizeT(SizeT const& val) noexcept
{
    return val;
}

/** @brief Convert an TransposeMode to the cuBLAS operation type. */
inline cublasOperation_t
ToNativeTransposeMode(TransposeMode const& orient) noexcept
{
    switch (orient)
    {
    case TransposeMode::TRANSPOSE:
        return CUBLAS_OP_T;
    case TransposeMode::CONJ_TRANSPOSE:
        return CUBLAS_OP_C;
    default: // TransposeMode::NORMAL
        return CUBLAS_OP_N;
    }
}

/** @brief Convert a SideMode to the cuBLAS side mode type. */
inline cublasSideMode_t
ToNativeSideMode(SideMode const& side) noexcept
{
    if (side == SideMode::LEFT)
        return CUBLAS_SIDE_LEFT;

    return CUBLAS_SIDE_RIGHT;
}

/** @brief Get the cuBLAS library handle. */
cublasHandle_t GetLibraryHandle() noexcept;

/** @class SyncManager
 *  @brief Manage stream synchronization within cuBLAS.
 */
class SyncManager
{
public:
    SyncManager(cublasHandle_t handle, SyncInfo<Device::GPU> const& si);
    ~SyncManager();
private:
    cudaStream_t orig_stream_;
};// class SyncManager

///@}
/** @name BLAS-1 Routines */
///@{

#define ADD_AXPY_DECL(ScalarType)               \
    void Axpy(cublasHandle_t handle,            \
              int n, ScalarType const& alpha,   \
              ScalarType const* X, int incx,    \
              ScalarType* Y, int incy)

#define ADD_COPY_DECL(ScalarType)                       \
    void Copy(cublasHandle_t handle,                    \
              int n, ScalarType const* X, int incx,     \
              ScalarType* Y, int incy)

#define ADD_SCALE_DECL(ScalarType)                       \
    void Scale(cublasHandle_t handle,                    \
               int n, ScalarType const& alpha,           \
               ScalarType* X, int incx)

#ifdef HYDROGEN_GPU_USE_FP16
ADD_AXPY_DECL(__half);
#endif // HYDROGEN_GPU_USE_FP16
ADD_AXPY_DECL(float);
ADD_AXPY_DECL(double);
ADD_AXPY_DECL(cuComplex);
ADD_AXPY_DECL(cuDoubleComplex);

ADD_COPY_DECL(float);
ADD_COPY_DECL(double);
ADD_COPY_DECL(cuComplex);
ADD_COPY_DECL(cuDoubleComplex);

#ifdef HYDROGEN_GPU_USE_FP16
ADD_SCALE_DECL(__half);
#endif // HYDROGEN_GPU_USE_FP16
ADD_SCALE_DECL(float);
ADD_SCALE_DECL(double);
ADD_SCALE_DECL(cuComplex);
ADD_SCALE_DECL(cuDoubleComplex);

///@}
/** @name BLAS-2 Routines */
///@{

#define ADD_GEMV_DECL(ScalarType)                       \
    void Gemv(                                          \
        cublasHandle_t handle,                          \
        cublasOperation_t transpA, int m, int n,        \
        ScalarType const& alpha,                        \
        ScalarType const* A, int lda,                   \
        ScalarType const* x, int incx,                  \
        ScalarType const& beta,                         \
        ScalarType* y, int incy)

ADD_GEMV_DECL(float);
ADD_GEMV_DECL(double);
ADD_GEMV_DECL(cuComplex);
ADD_GEMV_DECL(cuDoubleComplex);

///@}
/** @name BLAS-3 Routines */
///@{

#define ADD_GEMM_DECL(ScalarType)               \
    void Gemm(                                  \
        cublasHandle_t handle,                  \
        cublasOperation_t transpA,              \
        cublasOperation_t transpB,              \
        int m, int n, int k,                    \
        ScalarType const& alpha,                \
        ScalarType const* A, int lda,           \
        ScalarType const* B, int ldb,           \
        ScalarType const& beta,                 \
        ScalarType* C, int ldc)

#ifdef HYDROGEN_GPU_USE_FP16
ADD_GEMM_DECL(__half);
#endif // HYDROGEN_GPU_USE_FP16
ADD_GEMM_DECL(float);
ADD_GEMM_DECL(double);
ADD_GEMM_DECL(cuComplex);
ADD_GEMM_DECL(cuDoubleComplex);

///@}
/** @name BLAS-like Extension Routines */
///@{

// We use this for Axpy2D, Copy2D, and Transpose
#define ADD_GEAM_DECL(ScalarType)               \
    void Geam(cublasHandle_t handle,            \
              cublasOperation_t transpA,        \
              cublasOperation_t transpB,        \
              int m, int n,                     \
              ScalarType const& alpha,          \
              ScalarType const* A, int lda,     \
              ScalarType const& beta,           \
              ScalarType const* B, int ldb,     \
              ScalarType* C, int ldc)

#define ADD_DGMM_DECL(ScalarType)               \
    void Dgmm(cublasHandle_t handle,            \
              cublasSideMode_t side,            \
              int m, int n,                     \
              ScalarType const* A, int lda,     \
              ScalarType const* X, int incx,    \
              ScalarType* C, int ldc)

ADD_GEAM_DECL(float);
ADD_GEAM_DECL(double);
ADD_GEAM_DECL(cuComplex);
ADD_GEAM_DECL(cuDoubleComplex);

ADD_DGMM_DECL(float);
ADD_DGMM_DECL(double);
ADD_DGMM_DECL(cuComplex);
ADD_DGMM_DECL(cuDoubleComplex);

///@}

}// namespace cublas
}// namespace hydrogen
#endif // HYDROGEN_IMPORTS_CUBLAS_HPP_
