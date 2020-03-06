#include "cuda_helpers.hpp"

//#define USE_TENSOR_MATH_IF_AVAILABLE

namespace hydrogen
{

GPU_Session::GPU_Session()
{
    int localRank = -1;
    char* env = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (env)
        localRank = std::atoi(env);
    else
        localRank = 0;

    int num_devices = 0;
    CHECK_CUDA(cudaGetDeviceCount(&num_devices));

    int device_id = localRank % num_devices;
    CHECK_CUDA(cudaSetDevice(device_id));
}

GPU_Session::~GPU_Session()
{
    for (auto&& s : streams_)
        if (s) cudaStreamDestroy(s);

    for (auto&& e : events_)
        if (e) cudaEventDestroy(e);

    for (auto&& p : ptrs_)
        if (p) cudaFree(p);
}

cudaStream_t GPU_Session::get_managed_stream() const
{
    streams_.emplace_front();
    CHECK_CUDA(
        cudaStreamCreateWithFlags(
            &streams_.front(), cudaStreamNonBlocking));
    return streams_.front();
}

cudaEvent_t GPU_Session::get_managed_event() const
{
    events_.emplace_front();
    CHECK_CUDA(cudaEventCreate(&events_.front()));
    return events_.front();
}

void* GPU_Session::get_managed_memory(size_t size) const
{
    ptrs_.emplace_front();
    CHECK_CUDA(cudaMalloc(&ptrs_.front(), size));
    return ptrs_.front();
}

cublasHandle_t handle_;

GPU_Session::gpuBLAS_Session::gpuBLAS_Session()
{
    CHECK_CUBLAS(cublasCreate(&handle_));
#ifdef USE_TENSOR_MATH_IF_AVAILABLE
    CHECK_CUBLAS(cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH));
#endif // USE_TENSOR_MATH_IF_AVAILABLE
}

GPU_Session::gpuBLAS_Session::~gpuBLAS_Session()
{
    cublasDestroy(handle_);
}

cublasHandle_t get_gpublas_handle() noexcept { return handle_; }

void CopyToHost(
    void const* src, void* dest, size_t size, cudaStream_t stream)
{
    CHECK_CUDA(
        cudaMemcpyAsync(
            dest, src, size, cudaMemcpyDeviceToHost, stream));
}

void CopyToDevice(
    void const* src, void* dest, size_t size, cudaStream_t stream)
{
    CHECK_CUDA(
        cudaMemcpyAsync(
            dest, src, size, cudaMemcpyHostToDevice, stream));
}

void SyncStream(cudaStream_t stream)
{
    CHECK_CUDA(cudaStreamSynchronize(stream));
}

}// namespace hydrogen
