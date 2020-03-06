#include <El/hydrogen_config.h>
#include <hydrogen/device/gpu/CUDA.hpp>
#include <cuda_runtime.h>

#include <nvshmem.h>
#include <nvshmemx.h>
#include "mpi.h"


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




__host__ __device__
int grid_row_rank(int p, int d0, int d1);

__host__ __device__
int grid_col_rank(int p, int d0, int d1);

__host__ __device__
int local_height_mcmr(int p, int m, int n, int d0, int d1);

__host__ __device__
int local_width_mcmr(int p, int m, int n, int d0, int d1);

__host__ __device__
int local_height_vcstar(int p, int m, int n, int d0, int d1);

__host__ __device__
int local_width_vcstar(int p, int m, int n, int d0, int d1);

__host__ __device__
int local_height_vrstar(int p, int m, int n, int d0, int d1);

__host__ __device__
int local_width_vrstar(int p, int m, int n, int d0, int d1);

__host__ __device__
int local_to_global_row_index_mcmr(int p, int k, int m, int n, int d0, int d1);

__host__ __device__
int local_to_global_col_index_mcmr(int p,
        int k,
        int m,
        int n,
        int grid_height,
        int grid_width
        );

//
// [VR,*] TRANSPOSE TO [*,VR]
//
__host__ __device__
int my_vr_row_rank(int p, int d0, int d1);

__host__ __device__
int my_vr_col_rank(int p, int d0, int d1);

__host__ __device__
int vcstar_to_vrstar_to_pid(int p, int d0, int d1);

__host__ __device__
int vcstar_to_vrstar_from_pid(int p, int d0, int d1);

//namespace
//{
__device__
void WaitOnAllSet(int const volatile* const watched_mem, int const value, int const num_inds);

__device__
void _WaitOnAll(int me, int const volatile* const watched_mem, int const value,
               int const* const inds, int const num_inds);
__device__
void WaitOnAll(int const volatile* const watched_mem, int const value,
               int const* const inds, int const num_inds);

__device__
void WaitOnSingle(int const volatile* watched_mem, int const value);

__device__
void NotifyAll(int * const workspace, int const value,
               int const* pes, int const npes,
               int const my_rank);

__device__
void WaitOnSelective(int * const watched_mem, int const value,
	       int const* ranks_set, int const set_len, 
               int const* pes);

__device__
void NotifyOne (int* row_wspace, int const value,
	       int const target_col, int const my_row_rank, int my_col_rank,
               int const* pes, int const npes, int const grid_height, int const grid_width);

__device__
void NotifySelective(int * const workspace, int const value,
	       int const* ranks_set, int const set_len, int const my_rank_in_set,
               int const* pes, int const npes,
               int const my_rank);

__device__
int FindLocation(int const* const list, int const list_size, int const value);

template <typename T>
__device__
T const* PutNonLocal(T const* sbuf, T* rbuf,
                     int const total_size,
                     int const* const pes, int const npes, int const my_rank);

// Here 'me' is my MPI rank, not nvshmem pe rank
template <typename T>
__device__
T const* PutNonLocalv(T const* sbuf, T* rbuf,
    		     int const* sdispls, int const* rdispls,
                     int const* pes, int const npes, int const my_pe_rank);

template <typename T>
__device__
void local_addition(T* dest_buf, T const* src_buf, int const len);

// 'my_pe_rank' is the PE rank of me 
template <typename T>
__device__
T const* PutGathervNonLocal(int const my_pe_rank, int const grid_height, int const grid_width,
		T const* sbuf, T* rbuf,
                int const my_displs, int const my_count,
                int const* pes, int const npes);

template <typename T>
__global__
void interleave_starvr_to_starmr_kernel(int const m, int const n, int const grid_height, int const grid_width,
	T* dest_buf, T const* src_buf, int const* dev_star_mr_buffer_displs, int const size);

template<typename T>
__global__
void lower_data_put_kernel(
    int const m, int const n, int const my_pe_rank, 
    int const grid_height, int const grid_width,
    T* __restrict__ sbuf, T* __restrict__ rbuf, int const local_buf_len,
    int const my_row_rank, int my_col_rank,
    int const low_pow, int const offset,
    int const* pe_set, int const pe_set_len,
    int const* pes, int const npes, int* row_wspace);

// 'me' is my PE number in given (sub)communicator.
// Need to translate this to actual PE number when communication is required
template <typename T>
__global__
void pack_mcmr_to_vc_star(int m, int n, int me, int grid_height, int grid_width, int* send_displs, T* data_buffer, T* send_buffer);

// Unpack data in SDO recv_buffer to another SDO local_buffer
// 'me' is my PE number in given (sub)communicator.
template <typename T>
__global__
void unpack_mcmr_to_vc_star(int m, int n, int me, int grid_height, int grid_width, int* recv_displs, T* local_buffer, T* recv_buffer);

// Here 'my_pe_rank' is my pe rank, not MPI rank, and needs to be mapped to actual
// PE via pes before communication.
template <typename T>
__global__
void Allgatherv_put_kernel(
    int const my_pe_rank, int const grid_height, int const grid_width,
    T* __restrict__ sbuf, T* __restrict__ rbuf,
    int const displacement, int const size,
    int const* pes, int const npes, int* const workspace);

template<typename T>
__global__
void recursive_halving_put_kernel(
    int const m, int const n, int const my_pe_rank, 
    int const grid_height, int const grid_width,
    T* __restrict__ sbuf, T* __restrict__ rbuf, int const local_buf_len,
    int const my_row_rank, int my_col_rank,
    int const low_pow, int const offset,
    int const* pe_set, int const pe_set_len,
    int const* pes, int const npes, int* row_wspace);

template<typename T>
__global__
void  finalize_gemm_kernel(int const my_row_rank, int const my_col_rank, int const grid_height, int const grid_width, T* local_buffer, T* dev_reduction_buf, int const local_buffer_length, T* dev_final_buf, int const final_buf_length, int const* dev_location_find_displs, int const* dev_scatter_counter, int const* dev_scatter_displs, int const chunk_unit, int const* pes, int npes, int* row_wspace);



/*
// Here 'me' is my pe rank, not MPI rank
template <typename T>
__global__
void Alltoallv_put_kernel(
    T const* __restrict__ sbuf, T* __restrict__ rbuf, 
    int const* sdispls, int const* rdispls,
    int const* pes, int const npes, int const my_pe_rank, int const* workspace);
*/
template <typename T>
__global__
void Alltoallv_put_kernel(
int total_recv,
    T const* __restrict__ sbuf, T* __restrict__ rbuf,
    int const* sdispls, int const* rdispls,
    int const* pes, int const npes, int const my_pe_rank, 
    int const sync_counter, int* workspace);


template<typename T>
__global__
void transpose_vrstar_to_starvr_kernel(int const my_pe_rank, T* local_buffer, int const local_size, T* dev_tmp_buffer, int const m, int const n, int const grid_size, int const vr_row_rank, int const H, int const W, int const* pes, int const npes, int* const workspace);


// NOTE: All the PE numbers used in this function are those in given subcommunicator.
// Hence, it is required to translate these PE numbers to corresponding actual numbers
// when communication is needed.
template <typename T>
__global__
void Sendrecv_put_kernel(T const* __restrict__  local_buf, int local_size, int to_pe, 
	T* __restrict__ dev_recv_buf, int from_pe, 
	//T* __restrict__ dev_recv_buf, int recv_size, int from_pe, 
	int const my_pe_rank, int const* pes, int const npes, int const sync_counter, int* const workspace);

template <typename T>
__global__
void Alltoall_put_kernel_boring(
    T const* __restrict__ sbuf, T* __restrict__ rbuf, int const size,
    int const* pes, int const npes, int const me, int* const workspace);

__global__
void Global_sync_kernel(
    int const* pes, int const npes, int const my_pe_rank,
    int const  sync_counter, int* workspace);
//}// namespace <anon>

#if 1
namespace hydrogen
{

void Global_sync (
int me,
                  int const* pes, int const npes,
                  int const sync_counter, int* sync_space,
                  cudaStream_t const stream);
void counts_mcmr_to_vc_star(int grid_height, int grid_width, int myrank, int m, int n,
        std::vector<int>& send_counts, std::vector<int>& send_displs, int* total_send,
        std::vector<int>& recv_counts, std::vector<int>& recv_displs, int* total_recv);

template<typename T>
void mcmr_to_vrstar(MPI_Comm mpi_comm,
        int m,
        int n,
        int grid_height,
        int grid_width,
        T* dev_B_buffer,
	T* dev_target_buffer,
	cudaStream_t stream);
void mcmr_to_vrstar(MPI_Comm, int, int, int, int, float*, float*, cudaStream_t);

// Convert a Matrix (of size m x n) in [MC,MR] format to the one in [VC,*] format
// 'me' is my nvshmem pe, not MPI rank; it is expected that my PE number is computed prior 
// A function prepare_for_conversion must be called before any of DistMatrix redistribution
// functions
// 'me' is my PE number in the given subcommunicator. 
// Mapping from this to actual PE number is needed ONLY WHEN actual communication is 
// performed.
template <typename T>
void convert_mcmr_to_vcstar(int m, int n, int me, int grid_height, int grid_width,
        T* local_buffer, 
	int local_buffer_size,
        int const* dev_pes, // PE mapping information in device
        int const npes,
        int* sync_space, cudaStream_t const stream);

void convert_mcmr_to_vcstar_float(int m, int n, int me, int grid_height, int grid_width,
        float* local_buffer, 
        int const* dev_pes, 
        int const npes,
        int* const sync_space, cudaStream_t const stream);



template <typename T>
void Alltoall_put(T const* sbuf, T* rbuf, int const size,
                  int const* pes, int const npes,
                  int* const sync_space, cudaStream_t const stream);


template <typename T>
void Alltoallv_put(
int total_recv,
		  int me,
                  T const* sbuf, T* rbuf,
                  int const* sdispls, int const* rdispls,
                  int const* pes, int const npes,
                  int const sync_counter, int* sync_space, 
		  cudaStream_t const stream);


template <typename T>
void Sendrecv_put(T const* local_buf, int const local_size, int const to_pe,
		  T* dev_recv_buf, int const from_pe,
		  //T* dev_recv_buf, int const recv_size, int const from_pe,
                  int const my_pe_rank, int const* pes, int const npes,
                  int const sync_counter, int* const sync_space, cudaStream_t const stream);

// Convert a Matrix (of size m x n) in [VC,*] format to the one in [VR,*] format
// 'my_pe_rank' is my nvshmem pe, PE number in given (sub)communicator, not MPI rank; 
// It is expected that my PE number is computed prior 
template <typename T>
void convert_vcstar_to_vrstar(int m, int n, int my_pe_rank,
        int grid_height, int grid_width,
        T* local_buffer,
        int const* pes, int const npes,
        int* const sync_space, cudaStream_t const stream);
void convert_vcstar_to_vrstar_float(int m, int n, int my_pe_rank,
        int grid_height, int grid_width,
        float* local_buffer,
        int const* pes, int const npes,
        int* const sync_space, cudaStream_t const stream);


// Convert a Matrix (of size m x n) in [VR,*] format to the one in [*,VR] format; Basically, this is done by transposing
// given [VR,*] matrix into [*,VR] matrix; Hence, the dimension of inputt matrix is changed from (m x n) to (n x m) after the
// conversion.
// 'my_pe_rank' is my nvshmem pe, not MPI rank; it is expected that my PE number is computed prior 
template<typename T>
void convert_vrstar_to_starvr(T* local_buffer, int const local_size, int const m, int const n, int const grid_height, int const grid_width, int const my_pe_rank, int const* pes, int const npes, int* const sync_space, cudaStream_t const stream);

// NOTE: 'my_pe_rank' is my PE number in (sub)communicator. Need to translate this to actual PE via pes

void counts_mcmr_to_vc_star(
	int my_pe_rank,
	int k, int n, int grid_height, int grid_width,
	std::vector<int>& mr_msg_cnts, std::vector<int>& star_mr_buffer_displs,
	int *sum);

// mr_msg_cnt and star_mr_buffer_displs are vectors of grid_height and grid_height+1 elements, respectively.
void counts_starvr_to_starmr(int const my_pe_rank, int const m, int const n, int const grid_height, int const grid_width, std::vector<int>& mr_msg_cnts, std::vector<int>& star_mr_buffer_displs, int* sum);

// Convert a Matrix (of size m x n) in [*,VR] format to the one in [*,MR] format.
// 'my_pe_rank' is my PE number in given communicator, which MUSTT be mapped to actual
// PE number for communication
template <typename T>
void Allgatherv_put(int const my_pe_rank, int const grid_height, int const grid_width,
	T* dev_send_buf, T* dev_recv_buf,
	int const offset, int const size,  
        int const* pes, int const npes,
        int* const sync_space, cudaStream_t const stream);

// NOTE: Here, 'my_pe_rank' is my PE number in  given (sub)communicator.
// All PE ranks used in this call tree are interpreted as same and hence,
// they MUST be translated to their corresponding PE number when COMMUNICATION is required.
template <typename T>
void convert_starvr_to_starmr(int m, int n, int my_pe_rank, 
		int grid_height, int grid_width,
        	T* local_buffer, // local_buffer is my data buffer in [*,VR] format. Also, it will be output data buffer.
                	         // It is assumed to reside on Host at this point, but it will change 
        	int const* dev_pes, // PE mapping information in device
        	int const npes,
        	int* const sync_space, cudaStream_t const stream);
// NOTE: Here, 'my_pe_rank' is my PE number in  given (sub)communicator.
// All PE ranks used in this call tree are interpreted as same and hence,
// they MUST be translated to their corresponding PE number when COMMUNICATION is required.
template <typename T>
void axpy(int m, int n, int my_pe_rank, 
	int grid_height, int grid_width,
        T* local_buffer, // local_buffer is my data buffer in [MR,*] format. 
        int const local_buffer_length,
        int const* dev_pes, // PE mapping information in device
        int const npes,
        int* const sync_space, cudaStream_t const stream);


}// namespace hydrogen
#endif
