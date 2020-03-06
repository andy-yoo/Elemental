#ifndef TOM_CUDA_HELPERS_HPP_
#define TOM_CUDA_HELPERS_HPP_

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <forward_list>
#include <iostream>
#include <sstream>
#include <stdexcept>

#ifndef NUM_ITERS
#define NUM_ITERS 100
#endif

#define CHECK_CUDA(cmd)                                                 \
    do                                                                  \
    {                                                                   \
        auto error = cmd;                                               \
        if (error != cudaSuccess)                                       \
        {                                                               \
            std::ostringstream oss;                                     \
            oss << __FILE__ << ":" << __LINE__ << ": "                  \
                << "Command \"" << #cmd << "\" failed.\n\n"             \
                << "Error code: " << error << "\n"                      \
                << "Error msg: " << cudaGetErrorString(error) << "\n"   \
                << std::endl;                                           \
            throw std::runtime_error(oss.str());                        \
        }                                                               \
    } while (false)

#define CHECK_CUBLAS(cmd)                                        \
    do                                                           \
    {                                                            \
        auto err = cmd;                                          \
        if (err != CUBLAS_STATUS_SUCCESS)                        \
        {                                                        \
            if (err == CUBLAS_STATUS_NOT_INITIALIZED)            \
                std::cerr << "CUBLAS_NOT_INITIALIZED!" << std::endl; \
            else                                                     \
                std::cerr << "cublas_status = " << err << std::endl;    \
            throw std::runtime_error("cuBLAS problem detected.");       \
        }                                                               \
    } while (false)


namespace andy
{

using gpuEvent_t = cudaEvent_t;
using gpuStream_t = cudaStream_t;
using gpuBLASHandle_t = cublasHandle_t;

class GPU_Session
{
public:
    GPU_Session();
    ~GPU_Session();

    gpuStream_t get_managed_stream() const;
    gpuEvent_t get_managed_event() const;
    void* get_managed_memory(size_t size) const;

private:
    struct gpuBLAS_Session
    {
        gpuBLAS_Session();
        ~gpuBLAS_Session();
        void init();
    };

    gpuBLAS_Session gpublas_;

    mutable std::forward_list<gpuStream_t> streams_;
    mutable std::forward_list<gpuEvent_t> events_;
    mutable std::forward_list<void*> ptrs_;
};// GPU_Session

template <typename F, typename... Args>
void LaunchCudaKernel(
    F kernel, dim3 const& gridDim, dim3 const& blkDim,
    size_t sharedMem, gpuStream_t stream,
    Args&&... kernel_args)
{
    void* args[] = { reinterpret_cast<void*>(&kernel_args)... };
    CHECK_GPU(
        cudaLaunchKernel(
            (void const*) kernel,
            gridDim, blkDim, args, sharedMem, stream));

}

// Returns the total time in ms
template <typename F>
float run_kernel_test(
    F kernel_caller, cudaStream_t stream,
    cudaEvent_t start, cudaEvent_t stop,
    std::string const& name)
{
    // Warmup
    for (int ii = 0; ii < 2; ++ii)
        kernel_caller(stream);

    float elapsed_time;
    cudaEventRecord(start, stream);
    for (int ii = 0; ii < NUM_ITERS; ++ii)
        kernel_caller(stream);

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);

     std::cout << "Total " << name << " time (" << NUM_ITERS << " calls): "
              << elapsed_time  << "ms"
              << std::endl;
    return elapsed_time;
}

gpuBLASHandle_t get_gpublas_handle() noexcept;

void CopyToHost(
    void const* src, void* dest, size_t size, gpuStream_t stream);
void CopyToDevice(
    void const* src, void* dest, size_t size, gpuStream_t stream);
void SyncStream(gpuStream_t stream);

}// namespace andy
#endif // TOM_CUDA_HELPERS_HPP_
