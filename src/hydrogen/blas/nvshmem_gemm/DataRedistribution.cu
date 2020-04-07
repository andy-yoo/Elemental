#include <hydrogen/blas/nvshmem_gemm/NVSHMEM_Gemm.hpp>
#include <hydrogen/blas/nvshmem_gemm/DataRedistribution.hpp>
#include <El/hydrogen_config.h>
#include <hydrogen/device/gpu/CUDA.hpp>
#include <cuda_runtime.h>

#include <nvshmem.h>
#include <nvshmemx.h>
#include "mpi.h"

#define MAX_PES	1025

namespace El
{
  template <typename T>
  class Complex;
}

__host__ __device__
int pe_from_rank_coord(int r, int c, int d0, int d1){
    return c*d0+r;
}

__host__ __device__
int grid_row_rank(int p, int d0, int d1){
    return p%d0;
}

__host__ __device__
int grid_col_rank(int p, int d0, int d1){
    return p/d0;
}

__host__ __device__
int local_height_mcmr(int p, int m, int n, int d0, int d1){
    return (m/d0 + ((grid_row_rank(p, d0, d1) < m%d0)?1:0));
}

__host__ __device__
int local_width_mcmr(int p, int m, int n, int d0, int d1){
    return (n/d1 + ((grid_col_rank(p, d0, d1) < n%d1)?1:0));
}

__host__ __device__
int local_height_starmr(int p, int m, int n, int d0, int d1){
    return m;
}

__host__ __device__
int local_width_starmr(int p, int m, int n, int d0, int d1){
    return (n/d1 + ((grid_col_rank(p, d0, d1) < n%d1)?1:0));
}

__host__ __device__
int local_height_mcstar(int p, int m, int n, int d0, int d1){
    return (m/d0 + ((grid_row_rank(p, d0, d1) < m%d0)?1:0));
}

__host__ __device__
int local_width_mcstar(int p, int m, int n, int d0, int d1){
    return n;
}

__host__ __device__
int local_height_mrstar(int p, int m, int n, int d0, int d1){
    return (m/d1 + ((grid_col_rank(p, d0, d1) < m%d1)?1:0));
}

__host__ __device__
int local_width_mrstar(int p, int m, int n, int d0, int d1){
    return n;
}

__host__ __device__
int local_height_vcstar(int p, int m, int n, int d0, int d1){
    int H = d0*d1;
    return (m/H + ((p < m%H)?1:0));
}

__host__ __device__
int local_width_vcstar(int p, int m, int n, int d0, int d1){
    return n;
}

__host__ __device__
int local_height_vrstar(int p, int m, int n, int d0, int d1){
    int H = d0*d1;
    int i = p%d0;
    int j = p/d0;
    int r = i*d1 + j;
    return (m/H + ((r < m%H)?1:0));
}

__host__ __device__
int local_width_vrstar(int p, int m, int n, int d0, int d1){
    return n;
}

__host__ __device__
int local_to_global_row_index_mcmr(int p,
        int k,
        int m,
        int n,
        int grid_height,
        int grid_width
        )
{
   int local_height = local_height_mcmr(p, m, n, grid_height, grid_width);
   return (grid_row_rank(p, grid_height, grid_width) + grid_height*(k % local_height));
}

__host__ __device__
int local_to_global_col_index_mcmr(int p,
        int k,
        int m,
        int n,
        int grid_height,
        int grid_width
        )
{
   int local_height = local_height_mcmr(p, m, n, grid_height, grid_width);
   return (grid_col_rank(p, grid_height, grid_width) + grid_width*(k/local_height));
}

//
// [VR,*] TRANSPOSE TO [*,VR]
//
__host__ __device__
int my_vr_row_rank(int p, int d0, int d1){
    int i = p % d0;
    int j = p / d0;

    return (j + i*d1);
}

__host__ __device__
int my_vr_col_rank(int p, int d0, int d1){
    return -1;
}
__host__ __device__
int vcstar_to_vrstar_to_pid(int p, int d0, int d1){
    int i = p / d1;
    int j = p % d1;

    return (i + j*d0);
}

__host__ __device__
int vcstar_to_vrstar_from_pid(int p, int d0, int d1){
    int i = p % d0;
    int j = p / d0;

    return (j + i*d1);
}

__device__
void WaitOnAllSet(int const volatile* const watched_mem, int const value, int const num_inds)
{
    bool done = false;
    while (!done)
    {
        done = true;
        for (int i = 0; i < num_inds; ++i)
        {
            if (watched_mem[i] != value)
            {
                done = false;
                break;
            }
        }
    }
}

__device__
void _WaitOnAll(int me, int const volatile* const watched_mem, int const sync_counter,
               int const* const inds, int const num_inds)
{
int myidx = blockIdx.x;

    bool done = false;
    while (!done)
    {
        done = true;
        for (int i = 0; i < num_inds; ++i)
        {
            if (watched_mem[inds[i]] != sync_counter)
            {
//if(myidx == 0 ) printf("(%d) sync_counter=%d inds[%d]=%d watched_mem[inds[%d]]=%d\n", me, sync_counter, i, inds[i], i, watched_mem[inds[i]]);
                done = false;
                break;
            }
        }
    }
}

__device__
void WaitOnAll(int const volatile* const watched_mem, int const sync_counter,
               int const* const inds, int const num_inds)
{
    bool done = false;
    while (!done)
    {
        done = true;
        for (int i = 0; i < num_inds; ++i)
        {
            if (watched_mem[inds[i]] != sync_counter)
            {
                done = false;
                break;
            }
        }
    }
}

__device__
void WaitOnSingle(int const volatile* watched_mem, int const value)
{
    bool done = false;
    while (!done)
    {
        done = true;
        if (*watched_mem != value)
        {
                done = false;
                break;
        }
    }
}



__device__
void A__NotifyAll(int* workspace, int sync_counter,
               int const* pes, int npes,
               int my_rank)
{
    int const me = pes[my_rank]; // "global rank"
    int* target = workspace + me;

    // Flag myself as done
    *target = sync_counter;

    // Broadcast my value
    for (int i = 1; i <= npes; ++i)
    {
        int const id = (my_rank + i) % npes;
        int const& pe = pes[id];

        // This should be nvshmem_int_p, but this doesn't work over
        // IB. :/

//printf("(%d) sending  %d to pes[%d] (me=%d) pe=%d\n", my_rank, sync_counter, id, me, pe);
        nvshmem_int_put(target, target, 1, pe);
    }
    nvshmem_fence();
}
__device__
void A__WaitOnAll(int me, int* workspace, int sync_counter,
               int* inds, int npes)
{
    bool done = false;
    while (!done)
    {
        done = true;
        for (int i = 0; i < npes; ++i)
        {
            if (workspace[i] != sync_counter)
            {
                done = false;
                break;
            }
	    else{
//if(myidx == 0 ) printf("(%d) sync_counter=%d inds[%d]=%d workspace[inds[%d]]=%d\n", me, *sync_counter, i, inds[i], i, workspace[inds[i]]);
	    }
        }
    }
}



__device__
void _NotifyAll(int * const workspace, int const value,
               int const* pes, int const npes,
               int const my_rank)
{
    int const me = pes[my_rank]; // "global rank"
    int *const target = workspace + me;

    // Flag myself as done
    *target = value;

    int val = (int) value;
    // Broadcast my value
    for (int i = 1; i <= npes; ++i)
    {
        int const id = (my_rank + i) % npes;
        int const& pe = pes[id];

        // This should be nvshmem_int_p, but this doesn't work over
        // IB. :/

        nvshmem_int_put(target, target, 1, pe);
    }
}

__device__
void _NotifyRow(int * const workspace, int const value,
		int grid_height, int grid_width,
                int const* pes, int const npes,
                int const my_rank)
{
    int const me = pes[my_rank]; // "global rank"
    int *const target = workspace + me;

    // Flag myself as done
    *target = value;

    for (int i = 1; i <= npes; ++i)
    {
        int const id = (my_rank + i) % npes;
        int const& pe = pes[id];

        // This should be nvshmem_int_p, but this doesn't work over
        // IB. :/

        nvshmem_int_put(target, target, 1, pe);
    }
}

__device__
void NotifyAll(int * const workspace, int const sync_counter,
               int const* pes, int const npes,
               int const my_rank)
{
    int const me = pes[my_rank]; // "global rank"
    int *const target = workspace + me;

    // Flag myself as done
    *target = sync_counter;

    // Broadcast my sync_counter
    for (int i = 1; i < npes; ++i)
    {
        int const id = (my_rank + i) % npes;
        int const& pe = pes[id];

        // This should be nvshmem_int_p, but this doesn't work over
        // IB. :/
        nvshmem_int_put(target, target, 1, pe);
    }
}

__device__
void WaitOnSelective(int * const watched_mem, int const value,
	       int const* ranks_set, int const set_len, 
               int const* pes)
{
    bool done = false;
    while (!done)
    {
        done = true;
        for (int i = 0; i < set_len; ++i)
        {
	    int loc = pes[ranks_set[i]];
            if (watched_mem[loc] != value)
            {
                done = false;
                break;
            }
        }
    }
}


__device__
void NotifyOne (int* row_wspace, int const value,
	       int const target_col, int const my_row_rank, int my_col_rank,
               int const* pes, int const npes, int const grid_height, int const grid_width)
{
    int *const target = row_wspace + my_col_rank;
    int send_val = value;

    // Broadcast my value
    int peid = my_col_rank*grid_height+my_row_rank;
    int const& pe = pes[peid];

    // This should be nvshmem_int_p, but this doesn't work over
    // IB. :/
    //nvshmem_int_put(target, target, 1, pe);
    nvshmem_int_put(target, &send_val, 1, pe);
}

/*
__device__
void NotifyOne (int* row_wspace, int const value,
	       int const target_pe,
               int const* pes, int const npes)
{

    int const& pe = pes[target_pe];
    int *const target = row_wspace + pe;

    // This should be nvshmem_int_p, but this doesn't work over
    // IB. :/
    nvshmem_int_put(target, target, 1, pe);
}
*/

__device__
void NotifySelective(int * const workspace, int const value,
	       int const* ranks_set, int const set_len, int const my_rank_in_set,
               int const* pes, int const npes,
               int const my_rank)
{
    int const me = pes[my_rank]; // "global rank"
    int *const target = workspace + me;

    // Flag myself as done
    *target = value;

    // Broadcast my value
    for (int i = 1; i < set_len; ++i)
    {
        int const id = (my_rank_in_set + i) % set_len;
        int const& pe = pes[ranks_set[id]];

        // This should be nvshmem_int_p, but this doesn't work over
        // IB. :/
        nvshmem_int_put(
            target, target, 1, pe);
    }
}

__device__
int FindLocation(int const* const list, int const list_size, int const value)
{
    for (int i = 0; i < list_size; ++i)
        if (list[i] == value)
            return i;
    return -1;
}

template <typename T>
__device__
T const* PutNonLocal(T const* sbuf, T* rbuf,
                     int const total_size,
                     int const* const pes, int const npes, int const my_rank)
{
    for (int i = 1; i < npes; ++i)
    {
        auto const offset = (my_rank + i) % npes;
        auto const peer = pes[offset];
        nvshmemx_int_put_nbi_block(
            rbuf, sbuf + offset*total_size, total_size, peer);
    }
    return sbuf + my_rank * total_size;
}


template <typename T>
__device__
T const* PutNonLocalv(T const* sbuf, T* rbuf,
    		     int const* sdispls, int const* target_offset_counts,
                     int const* pes, int const npes, int const my_pe_rank)
{

    for (int i = 1; i <= npes; ++i)
    {

        int current = (my_pe_rank + i) % npes;
	int send_size = target_offset_counts[2*current+1];
	if(send_size > 0){
           int target_offset = target_offset_counts[2*current];
           int peer = pes[current];
           nvshmemx_float_put_nbi_block(rbuf + target_offset, sbuf + sdispls[current], send_size, peer);


           //nvshmemx_float_put_nbi_block(rbuf + rdispls[my_pe_rank], sbuf + sdispls[current], send_size, peer);
        //nvshmemx_float_put_nbi_block(rbuf + rdispls[current], sbuf + sdispls[current], send_size, peer);
	}
    }
    return sbuf + sdispls[my_pe_rank];
}

template <typename T>
__device__
void local_addition(T* dest_buf, T const* src_buf, int const len)
{
    auto const& my_idx = threadIdx.x;
    unsigned long const stride = blockDim.x;

    int t = my_idx;
    while(t < len){
        dest_buf[t] += src_buf[t];
        t += stride;
    }
    __syncthreads();
}

// 'my_pe_rank' is the PE rank of me 
template <typename T>
__device__
T const* PutGathervNonLocal(int const my_pe_rank, int const grid_height, int const grid_width,
		T const* sbuf, T* rbuf,
                int const my_displs, int const my_count,
                int const* pes, int const npes)
{
    int my_col_rank = my_pe_rank/grid_height;
    int my_row_rank = my_pe_rank%grid_height;

    for (int i = 1; i < grid_height; ++i)
    {
        auto const offset = (my_row_rank + i) % grid_height;
	auto const target_pe = my_col_rank*grid_height + offset;
        auto const peer = pes[target_pe];
        nvshmemx_int_put_nbi_block(rbuf+my_displs, sbuf, my_count, peer);
    }
    return rbuf + my_displs;
}


template <typename T>
__global__
void interleave_starvr_to_starmr_kernel(int const m, int const n, int const grid_height, int const grid_width,
	T* dest_buf, T const* src_buf, int const* dev_star_mr_buffer_displs, int const size)
{
    auto const& my_idx = threadIdx.x;
    unsigned long const stride = blockDim.x;

    int t = my_idx;
    while(t < size){
        int from_pe = (t/n)%grid_height;
        int from_block = t/(n*grid_height);
        int from_loc = t%n;
        int l = dev_star_mr_buffer_displs[from_pe] + from_block*n + from_loc;

        dest_buf[t] = src_buf[l];

        t += stride;
    }
    __syncthreads();
}

template<typename T>
__global__
void lower_data_put_kernel(
    int const m, int const n, int const my_pe_rank, 
    int const grid_height, int const grid_width,
    T* __restrict__ sbuf, T* __restrict__ rbuf, int const local_buf_len,
    int const my_row_rank, int my_col_rank,
    int const low_pow, int const offset,
    int const* pe_set, int const pe_set_len,
    int const* pes, int const npes, int* row_wspace)
{
    if(my_col_rank >= low_pow){
	int j = my_col_rank - offset; // target col rank
	int target = j*grid_height+my_row_rank;
  	int peer = pes[target];
        // send my buffer to row proc: pes[PE(my_row_rank-offset)]
        nvshmemx_int_put_nbi_block(rbuf, sbuf, local_buf_len, peer);

        // Order them before the sync puts
        if(threadIdx.x == 0)
           nvshmem_fence();

        // Ensure all threads in the block are up-to-date.
        __syncthreads();

        // Now that all the data has been put, we can flag to the receiver
        // PEs that the transfer is done. These are blocked after the
        // fence so there's only one fence.
        if (threadIdx.x == 0)
	    NotifyOne(row_wspace, 1, j,  my_row_rank, my_col_rank, pes, npes, grid_height, grid_width);
        __syncthreads();
    }
    else if(low_pow-offset <= my_col_rank && my_col_rank < low_pow){
        WaitOnAllSet(row_wspace+(low_pow-offset), 1, offset);
        __syncthreads();

        // perform local reduction
	local_addition(sbuf, rbuf, local_buf_len);
        __syncthreads();
    }
}

// 'me' is my PE number in given (sub)communicator.
// Need to translate this to actual PE number when communication is required
template <typename T>
__global__
void pack_mcmr_to_vc_star(int m, int n, int me, int grid_height, int grid_width, int* send_displs, T* data_buffer, T* send_buffer){

    int myidx = threadIdx.x;
    int stride = blockDim.x;

    int grid_size = grid_height*grid_width;
    int local_height = local_height_mcmr(me, m, n, grid_height, grid_width);
    int local_width = local_width_mcmr(me, m, n, grid_height, grid_width);
    int data_len = local_height*local_width;

    int k = myidx;
    while(k < data_len){
        int I = local_to_global_row_index_mcmr(me, k, m, n, grid_height, grid_width);
        int J = local_to_global_col_index_mcmr(me, k, m, n, grid_height, grid_width);
        int target = I%grid_size;
        int i = I/grid_size;
        int j = J/grid_width;
        int t = j*local_height_vcstar(target, m, n, grid_height, grid_width) + i;

        send_buffer[send_displs[target] + t] = data_buffer[k];
	k += stride;
    }
    __syncthreads();
}

// Unpack data in SDO recv_buffer to another SDO local_buffer
// 'me' is my PE number in given (sub)communicator.
template <typename T>
__global__
void unpack_mcmr_to_vc_star(int m, int n, int me, int grid_height, int grid_width, int* recv_displs, T* local_buffer, T* recv_buffer){
    int myidx = threadIdx.x;
    int stride = blockDim.x;

    int grid_size = grid_height*grid_width;
    int recv_buffer_size = recv_displs[grid_size];
    int lh = local_height_vcstar(me, m, n, grid_height, grid_width);
    int j = myidx;
    int q = 0;

    while(j < recv_buffer_size){
        while(!(recv_displs[q] <= j && j < recv_displs[q+1]))
	    q++;
        int start_idx = lh*grid_col_rank(q, grid_height, grid_width);
        int k = j-recv_displs[q];
        int a = k/lh;
        int t = start_idx + a*lh*grid_width + (k%lh);
        local_buffer[t] = recv_buffer[j];
 

	j += stride;
    }
    __syncthreads();
}


template <typename T>
__global__
void pack_mcmr_to_mrstar(int m, int n, int me, int grid_height, int grid_width, 
T* dev_local_buffer, T* dev_send_buffer){
    int myidx = threadIdx.x;
    int stride = blockDim.x;
    int total_block_size =  local_height_mcmr(me, m, n, grid_height, grid_width)* local_width_mcmr(me, m, n, grid_height, grid_width);

    // Identify the column index that I am in
    int my_col_rank = grid_col_rank(me, grid_height, grid_width);
    int my_row_rank = grid_row_rank(me, grid_height, grid_width);
    int j = myidx;

    while(j < total_block_size){
	int block_id = j/local_width_mcmr(me, m, n, grid_height, grid_width);
	int block_offset = j%local_width_mcmr(me, m, n, grid_height, grid_width);
	int t = block_offset*local_height_mcmr(me, m, n, grid_height, grid_width)+block_id;
/*
	int jj = j * local_height_mcmr(me, m, n, grid_height, grid_width);
	int row_id = jj/local_width_mcmr(me, m, n, grid_height, grid_width);
	int row_offset = jj%local_width_mcmr(me, m, n, grid_height, grid_width);
	int jprime = row_offset + row_id;
*/
	dev_send_buffer[j] = dev_local_buffer[t];
//if(me == 0) printf("j=%d jprime=%d\n", j, t);
	j += stride;
    }

    __syncthreads();

}


// Allgather-scatter version 
template <typename T>
__global__
void ver2_gather_unpack_mcmr_to_mrstar (
    int my_pe_rank, int m, int n, int grid_height, int grid_width,
    T* dev_send_buf, T* dev_recv_buf, 
    T* dev_target_buffer, int total_gather,
    int send_size, int my_displs, int* const pes, int const npes, int* dev_sync_counter, int volatile* const workspace)
{
    int const myidx = threadIdx.x;


    int const num_parts = grid_height;
    int const threads_per_part = blockDim.x;
    int part_id; // = 0; //myidx/threads_per_part;
    int const my_col_rank =  grid_col_rank(my_pe_rank, grid_height, grid_width);
    int const my_row_rank =  grid_row_rank(my_pe_rank, grid_height, grid_width);

    int const stride = threads_per_part;
    int const H = local_height_mrstar(my_pe_rank, n, m, grid_height, grid_width); 

    if(grid_height*grid_width > MAX_PES)
	printf("ERROR: Too many PEs to handle...\n");
    __shared__ int displacements[MAX_PES];
    
    if(myidx == 0){
       displacements[0] = 0;
       for(int i=1; i<=grid_height; i++){
    	   int pe =  my_col_rank*grid_height + (i-1);
	   displacements[i]=displacements[i-1] + local_height_mcmr(pe, m, n, grid_height, grid_width)*local_width_mcmr(pe, m, n, grid_height, grid_width);
        }
    }
    __syncthreads();


    int offset;
    int peer;
    int part_begin;
    int part_size;
    int j;
    int current_row_rank=my_row_rank;
    int current_pe=my_pe_rank;
    int prev_row_rank;
    int prev_pe;
    int pe_zero =  pes[my_col_rank*grid_height];
    for (int i = 1;i<grid_height; i++)
    {
    //    nvshmem_float_put((float*) dev_recv_buf+my_displs, dev_send_buf, send_size, pe_zero);
        nvshmemx_float_put_nbi_block(dev_recv_buf+my_displs, dev_send_buf, send_size, pe_zero);
    }
    __syncthreads();

    if(myidx == 0) nvshmem_fence();
    __syncthreads();

    for (int i = 1;i<grid_height; i++)
    {
        nvshmemx_int_put_nbi_block((int*) workspace+my_row_rank, dev_sync_counter, 1, pe_zero);
    }
    __syncthreads();

    if(my_row_rank == 0){
       for (int i = 1;i<grid_height; i++)
       {
           peer = pes[i + my_col_rank*grid_height];

           nvshmemx_float_put_nbi_block(dev_recv_buf, dev_recv_buf, total_gather, peer);
       }
       __syncthreads();
    }
    if(myidx == 0) nvshmem_fence();
    __syncthreads();
    if(my_row_rank == 0){
       for (int i = 1;i<grid_height; i++)
       {
           peer = pes[i + my_col_rank*grid_height];
           nvshmemx_int_put_nbi_block((int*) workspace+i, dev_sync_counter, 1, peer);
       }
       __syncthreads();
    }


    for (int i = 0;i<grid_height; i++)
    {
        int current_row_rank = i;
        part_begin = displacements[current_row_rank];
        part_size = displacements[current_row_rank+1] - displacements[current_row_rank];
    	j = myidx;
	part_id = current_row_rank;
        while(j < part_size){
	    int bid = j/H;
	    int boffset = j%H;
	    int jj = part_begin + j;
	    int t = H*(part_id + bid*num_parts) + boffset;

	    dev_target_buffer[t] = dev_recv_buf[jj];
	    j += stride;
        }
     }
     __syncthreads();
}

template <typename T>
__global__
void ver1_gather_unpack_mcmr_to_mrstar (
    int my_pe_rank, int m, int n, int grid_height, int grid_width,
    T* dev_send_buf, T* dev_recv_buf, 
    T* dev_target_buffer, int total_gather,
    int send_size, int my_displs, int* const pes, int const npes, int* dev_sync_counter, int volatile* const workspace)
{
    int const myidx = threadIdx.x;


    int const num_parts = grid_height;
    int const threads_per_part = blockDim.x;
    int part_id; // = 0; //myidx/threads_per_part;
    int const my_col_rank =  grid_col_rank(my_pe_rank, grid_height, grid_width);
    int const my_row_rank =  grid_row_rank(my_pe_rank, grid_height, grid_width);

    int const stride = threads_per_part;
    int const H = local_height_mrstar(my_pe_rank, n, m, grid_height, grid_width); 

    if(grid_height*grid_width > MAX_PES)
	printf("ERROR: Too many PEs to handle...\n");
    __shared__ int displacements[MAX_PES];
    
    if(myidx == 0){
       displacements[0] = 0;
       for(int i=1; i<=grid_height; i++){
    	   int pe =  my_col_rank*grid_height + (i-1);
	   displacements[i]=displacements[i-1] + local_height_mcmr(pe, m, n, grid_height, grid_width)*local_width_mcmr(pe, m, n, grid_height, grid_width);
        }
    }
    __syncthreads();


    int offset;
    int peer;
    int part_begin;
    int part_size;
    int j;
    int current_row_rank=my_row_rank;
    int current_pe=my_pe_rank;
    int prev_row_rank;
    int prev_pe;
    for (int i = 1;i<grid_height; i++)
    {
        int next_row_rank = (my_row_rank + i) % grid_height;
        int next_pe = next_row_rank + my_col_rank*grid_height;
        peer = pes[next_pe];

        nvshmemx_float_put_nbi_block(dev_recv_buf+my_displs, dev_send_buf, send_size, peer);
    }
/*
if(myidx == 0) nvshmem_fence();
__syncthreads();
    for (int i = 1;i<grid_height; i++)
    {
        int next_row_rank = (my_row_rank + i) % grid_height;
        int next_pe = next_row_rank + my_col_rank*grid_height;
        peer = pes[next_pe];
        nvshmem_int_put((int*) workspace+my_row_rank, dev_sync_counter, 1, peer);
    }
    __syncthreads();
*/

if(myidx == 0) nvshmem_quiet();
__syncthreads();
    for (int i = 0;i<grid_height; i++)
    {
        int current_row_rank = i;
        part_begin = displacements[current_row_rank];
        part_size = displacements[current_row_rank+1] - displacements[current_row_rank];
    	j = myidx;
	part_id = current_row_rank;
        while(j < part_size){
	    int bid = j/H;
	    int boffset = j%H;
	    int jj = part_begin + j;
	    int t = H*(part_id + bid*num_parts) + boffset;

	    dev_target_buffer[t] = dev_recv_buf[jj];
	    j += stride;
        }
     }
     __syncthreads();
}

template <typename T>
__global__
void ver0_1_gather_unpack_mcmr_to_mrstar (
    int my_pe_rank, int m, int n, int grid_height, int grid_width,
    T* dev_send_buf, T* dev_recv_buf, 
    T* dev_target_buffer, int total_gather,
    int send_size, int my_displs, int* const pes, int const npes, long* dev_sync_counter, long volatile* const workspace)
{
    int const myidx = threadIdx.x;


    int const num_parts = grid_height;
    int const threads_per_part = blockDim.x;
    int part_id; // = 0; //myidx/threads_per_part;
    int const my_col_rank =  grid_col_rank(my_pe_rank, grid_height, grid_width);
    int const my_row_rank =  grid_row_rank(my_pe_rank, grid_height, grid_width);

    int const stride = threads_per_part;
    int const H = local_height_mrstar(my_pe_rank, n, m, grid_height, grid_width); 

    if(grid_height*grid_width > MAX_PES)
	printf("ERROR: Too many PEs to handle...\n");
    __shared__ int displacements[MAX_PES];
    
    if(myidx == 0){
       displacements[0] = 0;
       for(int i=1; i<=grid_height; i++){
    	   int pe =  my_col_rank*grid_height + (i-1);
	   displacements[i]=displacements[i-1] + local_height_mcmr(pe, m, n, grid_height, grid_width)*local_width_mcmr(pe, m, n, grid_height, grid_width);
        }
    }
    __syncthreads();


    int offset;
    int peer;
    int part_begin;
    int part_size;
    int j;
    int current_row_rank=my_row_rank;
    int current_pe=my_pe_rank;
    int prev_row_rank;
    int prev_pe;
    for (int i = 1;i<grid_height; i++)
    {
        int next_row_rank = (my_row_rank + i) % grid_height;
        int next_pe = next_row_rank + my_col_rank*grid_height;
        peer = pes[next_pe];

        nvshmemx_float_put_nbi_block(dev_recv_buf+my_displs, dev_send_buf, send_size, peer);
        nvshmem_fence();
            
        nvshmemx_long_put_nbi_block((long*) workspace+my_row_rank, dev_sync_counter, 1, peer);
        //nvshmemx_int_put_nbi_block((int*) workspace+my_row_rank, dev_sync_counter, 1, peer);

//	nvshmem_quiet();

        part_begin = displacements[current_row_rank];
        part_size = displacements[current_row_rank+1] - displacements[current_row_rank];
    	j = myidx;
	part_id = current_row_rank;
        while(j < part_size){
	    int bid = j/H;
	    int boffset = j%H;
	    int jj = part_begin + j;
	    int t = H*(part_id + bid*num_parts) + boffset;

	    dev_target_buffer[t] = dev_recv_buf[jj];
	    j += stride;
        }
        __syncthreads();

#if 1
	// Check for the next partition to merge: i.e., next_row, and update
	prev_row_rank = (my_row_rank-i+grid_height) % grid_height;
	if(myidx == 0){
	   bool flag=true;
	   while(flag){
	      if(*(workspace+prev_row_rank) == *dev_sync_counter){
		   flag = false;
	      }
	   }
	}
	__syncthreads();
#else
	prev_row_rank = (my_row_rank-i+grid_height) % grid_height;
    	prev_pe = pe_from_rank_coord(prev_row_rank, my_col_rank, grid_height, grid_width);
        nvshmem_wait_until(workspace+prev_row_rank, NVSHMEM_CMP_EQ, *dev_sync_counter);
#endif

        current_row_rank = prev_row_rank;
    	current_pe = pe_from_rank_coord(current_row_rank, my_col_rank, grid_height, grid_width);
    }
//	nvshmem_quiet();

    part_begin = displacements[current_row_rank];
    part_size = displacements[current_row_rank+1] - displacements[current_row_rank];
    //part_size = local_height_mcmr(current_pe, m, n, grid_height, grid_width)*local_width_mcmr(current_pe, m, n, grid_height, grid_width);
    j = myidx;
    part_id = current_row_rank;
    while(j < part_size){
        int bid = j/H;
	int boffset = j%H;
	int jj = part_begin + j;
	int t = H*(part_id + bid*num_parts) + boffset;
//printf("jj=%d t=%d\n", jj, t);
	dev_target_buffer[t] = dev_recv_buf[jj];
	j += stride;
     }
     __syncthreads();
}

template <typename T>
__global__
void ver0_gather_unpack_mcmr_to_mrstar (
    int my_pe_rank, int m, int n, int grid_height, int grid_width,
    T* dev_send_buf, T* dev_recv_buf, 
    T* dev_target_buffer, int total_gather,
    int send_size, int my_displs, int* const pes, int const npes, int* dev_sync_counter, int volatile* const workspace)
{
    int const myidx = threadIdx.x;


    int const num_parts = grid_height;
    int const threads_per_part = blockDim.x;
    int part_id; // = 0; //myidx/threads_per_part;
    int const my_col_rank =  grid_col_rank(my_pe_rank, grid_height, grid_width);
    int const my_row_rank =  grid_row_rank(my_pe_rank, grid_height, grid_width);

    int const stride = threads_per_part;
    int const H = local_height_mrstar(my_pe_rank, n, m, grid_height, grid_width); 

    if(grid_height*grid_width > MAX_PES)
	printf("ERROR: Too many PEs to handle...\n");
    __shared__ int displacements[MAX_PES];
    
    if(myidx == 0){
       displacements[0] = 0;
       for(int i=1; i<=grid_height; i++){
    	   int pe =  my_col_rank*grid_height + (i-1);
	   displacements[i]=displacements[i-1] + local_height_mcmr(pe, m, n, grid_height, grid_width)*local_width_mcmr(pe, m, n, grid_height, grid_width);
        }
    }
    __syncthreads();


    int offset;
    int peer;
    int part_begin;
    int part_size;
    int j;
    int current_row_rank=my_row_rank;
    int current_pe=my_pe_rank;
    int prev_row_rank;
    int prev_pe;
    for (int i = 1;i<grid_height; i++)
    {
        int next_row_rank = (my_row_rank + i) % grid_height;
        int next_pe = next_row_rank + my_col_rank*grid_height;
        peer = pes[next_pe];

        nvshmemx_float_put_nbi_block(dev_recv_buf+my_displs, dev_send_buf, send_size, peer);
        nvshmem_fence();
            
        nvshmemx_int_put_nbi_block((int*) workspace+my_row_rank, dev_sync_counter, 1, peer);

//	nvshmem_quiet();

        part_begin = displacements[current_row_rank];
        part_size = displacements[current_row_rank+1] - displacements[current_row_rank];
    	j = myidx;
	part_id = current_row_rank;
        while(j < part_size){
	    int bid = j/H;
	    int boffset = j%H;
	    int jj = part_begin + j;
	    int t = H*(part_id + bid*num_parts) + boffset;

	    dev_target_buffer[t] = dev_recv_buf[jj];
	    j += stride;
        }
        __syncthreads();

#if 1
	// Check for the next partition to merge: i.e., next_row, and update
	prev_row_rank = (my_row_rank-i+grid_height) % grid_height;
	if(myidx == 0){
	   bool flag=true;
	   while(flag){
	      if(*(workspace+prev_row_rank) == *dev_sync_counter){
		   flag = false;
	      }
	   }
	}
	__syncthreads();
#else
	prev_row_rank = (my_row_rank-i+grid_height) % grid_height;
    	prev_pe = pe_from_rank_coord(prev_row_rank, my_col_rank, grid_height, grid_width);
        //nvshmem_wait_until(workspace+prev_row_rank, NVSHMEM_CMP_EQ, *dev_sync_counter);
#endif

        current_row_rank = prev_row_rank;
    	current_pe = pe_from_rank_coord(current_row_rank, my_col_rank, grid_height, grid_width);
    }
//	nvshmem_quiet();

    part_begin = displacements[current_row_rank];
    part_size = displacements[current_row_rank+1] - displacements[current_row_rank];
    //part_size = local_height_mcmr(current_pe, m, n, grid_height, grid_width)*local_width_mcmr(current_pe, m, n, grid_height, grid_width);
    j = myidx;
    part_id = current_row_rank;
    while(j < part_size){
        int bid = j/H;
	int boffset = j%H;
	int jj = part_begin + j;
	int t = H*(part_id + bid*num_parts) + boffset;
//printf("jj=%d t=%d\n", jj, t);
	dev_target_buffer[t] = dev_recv_buf[jj];
	j += stride;
     }
     __syncthreads();
}

template <typename T>
__global__
void gather_unpack_mcmr_to_mrstar (
    int my_pe_rank, int m, int n, int grid_height, int grid_width,
    T* dev_send_buf, T* dev_recv_buf, 
    T* dev_target_buffer, int total_gather,
    int send_size, int my_displs, int* const pes, int const npes, int* dev_sync_counter, int volatile* const workspace)
{
    int const myidx = threadIdx.x;


    int const num_parts = grid_height;
    int const threads_per_part = blockDim.x;
    int part_id; // = 0; //myidx/threads_per_part;
    int const my_col_rank =  grid_col_rank(my_pe_rank, grid_height, grid_width);
    int const my_row_rank =  grid_row_rank(my_pe_rank, grid_height, grid_width);

    int const stride = threads_per_part;
    int const H = local_height_mrstar(my_pe_rank, n, m, grid_height, grid_width); 

    if(grid_height*grid_width > MAX_PES)
	printf("ERROR: Too many PEs to handle...\n");
    __shared__ int displacements[MAX_PES];
    
    if(myidx == 0){
       displacements[0] = 0;
       for(int i=1; i<=grid_height; i++){
    	   int pe =  my_col_rank*grid_height + (i-1);
	   displacements[i]=displacements[i-1] + local_height_mcmr(pe, m, n, grid_height, grid_width)*local_width_mcmr(pe, m, n, grid_height, grid_width);
        }
    }
    __syncthreads();


    int offset;
    int peer;
    int part_begin;
    int part_size;
    int j;
    int current_row_rank=my_row_rank;
    int current_pe=my_pe_rank;
    int prev_row_rank;
    int prev_pe;
    for (int i = 1;i<grid_height; i++)
    {
        int next_row_rank = (my_row_rank + i) % grid_height;
        int next_pe = next_row_rank + my_col_rank*grid_height;
        peer = pes[next_pe];

        nvshmemx_float_put_nbi_block(dev_recv_buf+my_displs, dev_send_buf, send_size, peer);
        nvshmem_fence();
            
        nvshmem_int_put((int*) workspace+my_row_rank, dev_sync_counter, 1, peer);

//	nvshmem_quiet();

        part_begin = displacements[current_row_rank];
        part_size = displacements[current_row_rank+1] - displacements[current_row_rank];
    	j = myidx;
	part_id = current_row_rank;
        while(j < part_size){
	    int bid = j/H;
	    int boffset = j%H;
	    int jj = part_begin + j;
	    int t = H*(part_id + bid*num_parts) + boffset;

	    dev_target_buffer[t] = dev_recv_buf[jj];
	    j += stride;
        }
        __syncthreads();

#if 1
	// Check for the next partition to merge: i.e., next_row, and update
	prev_row_rank = (my_row_rank-i+grid_height) % grid_height;
	if(myidx == 0){
	   bool flag=true;
	   while(flag){
	      if(*(workspace+prev_row_rank) == *dev_sync_counter){
		   flag = false;
	      }
	   }
	}
	__syncthreads();
#else
	prev_row_rank = (my_row_rank-i+grid_height) % grid_height;
    	prev_pe = pe_from_rank_coord(prev_row_rank, my_col_rank, grid_height, grid_width);
        //nvshmem_wait_until(workspace+prev_row_rank, NVSHMEM_CMP_EQ, *dev_sync_counter);
#endif

        current_row_rank = prev_row_rank;
    	current_pe = pe_from_rank_coord(current_row_rank, my_col_rank, grid_height, grid_width);
    }
//	nvshmem_quiet();

    part_begin = displacements[current_row_rank];
    part_size = displacements[current_row_rank+1] - displacements[current_row_rank];
    //part_size = local_height_mcmr(current_pe, m, n, grid_height, grid_width)*local_width_mcmr(current_pe, m, n, grid_height, grid_width);
    j = myidx;
    part_id = current_row_rank;
    while(j < part_size){
        int bid = j/H;
	int boffset = j%H;
	int jj = part_begin + j;
	int t = H*(part_id + bid*num_parts) + boffset;
//printf("jj=%d t=%d\n", jj, t);
	dev_target_buffer[t] = dev_recv_buf[jj];
	j += stride;
     }
     __syncthreads();
}

__global__
void empty_kernel(){
}


template <typename T>
__global__
void allgather_unpack_mcmr_to_mrstar (
    int my_pe_rank, int m, int n, int grid_height, int grid_width,
    T* dev_send_buf, T* dev_recv_buf,
    T* dev_target_buffer, int total_gather,
    int send_size, int my_displs, int* const pes, int const npes, int* dev_sync_counter, int volatile* const workspace)
{
    int const myidx = threadIdx.x;

    int const num_parts = grid_height;
    int const threads_per_part = blockDim.x;
    int part_id; // = 0; //myidx/threads_per_part;
    int const my_col_rank =  grid_col_rank(my_pe_rank, grid_height, grid_width);
    int const my_row_rank =  grid_row_rank(my_pe_rank, grid_height, grid_width);

    int const stride = threads_per_part;
    int const H = local_height_mrstar(my_pe_rank, n, m, grid_height, grid_width);


    if(grid_height*grid_width > MAX_PES)
        printf("ERROR: Too many PEs to handle...\n");
    __shared__ int displacements[MAX_PES];

    if(myidx == 0){
       displacements[0] = 0;
       for(int i=1; i<=grid_height; i++){
           int pe =  my_col_rank*grid_height + (i-1);
           displacements[i]=displacements[i-1] + local_height_mcmr(pe, m, n, grid_height, grid_width)*local_width_mcmr(pe, m, n, grid_height, grid_width);
        }
    }
    __syncthreads();

    int offset;
    int peer;
    int part_begin;
    int part_size;
    int j;
    int current_row_rank=my_row_rank;
    int current_pe=my_pe_rank;
    int prev_row_rank;
    int prev_pe;
    for (int i = 1;i<grid_height; i++)
    {
        int next_row_rank = (my_row_rank + i) % grid_height;
        int next_pe = next_row_rank + my_col_rank*grid_height;
        peer = pes[next_pe];

//printf("(%d) next_row=%d next_pe=%d peer=%d send_size=%d\n", my_pe_rank,  next_row, next_pe, peer, send_size);

        nvshmemx_float_put_nbi_block(dev_recv_buf+my_displs, dev_send_buf, send_size, peer);
    }
    nvshmem_fence();

    // Send notification to my column comrades
    for (int i = 1;i<grid_height; i++)
    {
        int next_row_rank = (my_row_rank + i) % grid_height;
        int next_pe = next_row_rank + my_col_rank*grid_height;
        peer = pes[next_pe];

        nvshmem_int_put((int*) workspace+my_row_rank, dev_sync_counter, 1, peer);
    }

    nvshmem_quiet();
#if 1
    // Local merge to target_buffer
    part_begin = displacements[my_row_rank];
    part_size = displacements[my_row_rank+1] - displacements[my_row_rank];
    j = myidx;
    part_id = my_row_rank;
    while(j < part_size){
        int bid = j/H;
        int boffset = j%H;
        int jj = part_begin + j;
        int t = H*(part_id + bid*num_parts) + boffset;

        dev_target_buffer[t] = dev_send_buf[jj];
        j += stride;
    }
    __syncthreads();
#else
    // Local copy of send_buffer to recv_buffer
    j = myidx;
    while(j < part_size){
        dev_recv_buffer[my_displs+j] = dev_send_buf[j];
        j += stride;
    }
    __syncthreads();
#endif

    // check the arrival of notifications and merge those available
    for (int i = 1;i<grid_height; i++)
    {

        current_row_rank = (my_row_rank+i) % grid_height;
        if(myidx == 0){
           bool flag=true;
           while(flag){
              if(*(workspace+current_row_rank) == *dev_sync_counter){
                   flag = false;
              }
           }
        }
        __syncthreads();

        part_begin = displacements[current_row_rank];
        part_size = displacements[current_row_rank+1] - displacements[current_row_rank];

        j = myidx;
        part_id = current_row_rank;
        while(j < part_size){
            int bid = j/H;
            int boffset = j%H;
            int jj = part_begin + j;
            int t = H*(part_id + bid*num_parts) + boffset;
//printf("jj=%d t=%d\n", jj, t);
            dev_target_buffer[t] = dev_recv_buf[jj];
            j += stride;
         }
         __syncthreads();
    }
}

// Here, m and n represent the size of source matrix .
// But after unpacking the results will be in n by m matrix (i.e., transposed)
// Make sure not to be confused by this orientation change
template <typename T>
__global__
void unpack_mcmr_to_mrstar(int m, int n, int me, int grid_height, int grid_width, 
T* dev_target_buffer, T* recv_buffer, int recv_buffer_size){
    int const myidx = threadIdx.x;
    int const num_parts = grid_height;
    int const threads_per_part = blockDim.x/num_parts;
    int const part_id = myidx/threads_per_part;
    int const part_offset = myidx%threads_per_part;
    int const my_col_rank  =  grid_col_rank(me, grid_height, grid_width);

    int acc=0;
    int my_part_begin;
    int part_size;
    for(int r=0; r<grid_height; r++){
	int pe =  my_col_rank*grid_height + r;

        part_size = local_height_mcmr(pe, m, n, grid_height, grid_width)* local_width_mcmr(pe, m, n, grid_height, grid_width);
	if(r == part_id){
	   my_part_begin = acc;
	   break;
	}
	acc += part_size;
    }

    int const stride = threads_per_part;
    int const H = local_height_mrstar(me, n, m, grid_height, grid_width); 

    int j = part_offset;
    while(j < part_size){
	int bid = j/H;
	int boffset = j%H;
	int jj = my_part_begin + j;
	int t = H*(part_id + bid*num_parts) + boffset;

	dev_target_buffer[t] = recv_buffer[jj];
	j += stride;
    }
    __syncthreads();

}

template <typename T>
__global__
void unpack_mcmr_to_mcstar(int m, int n, int me, int grid_height, int grid_width, T* dev_target_buffer, T* recv_buffer, int recv_buffer_size){
    int const myidx = threadIdx.x;
    int const num_parts = grid_width;
    int const threads_per_part = blockDim.x/num_parts;
    int const part_id = myidx/threads_per_part;
    int const part_offset = myidx%threads_per_part;
    int const my_row_rank  =  grid_row_rank(me, grid_height, grid_width);

    int part_size;
    int my_part_begin;
    int acc = 0;
    for(int c=0; c<grid_width; c++){
	int pe = c*grid_height + my_row_rank;
	part_size = local_height_mcmr(pe, m, n, grid_height, grid_width)*local_width_mcmr(pe, m, n, grid_height, grid_width);
	if(c == part_id){
	   my_part_begin = acc;
	   break;
	}
	acc += part_size;
    }

    int const stride = threads_per_part;
    int const H = local_height_mcstar(me, m, n, grid_height, grid_width);
 
    int j = part_offset;
    while(j < part_size){
	int bid = j/H;
	int boffset = j%H;
	int jj = my_part_begin + j;
	int t = H*(part_id+bid*num_parts) + boffset;
	
	dev_target_buffer[t] = recv_buffer[jj];

	j += stride;
    }

/*
    // Identify the column index that I am in
    int row = grid_row_rank(me, grid_height, grid_width);
    int j = myidx;

    // Find the column index for initial location j
    int col_index = 0;
    int running_acc = 0;
    int i, from_pe;
    int block_id;
    int offset;

    int const stride = 1;
    while(j < recv_buffer_size){
    	for(i=col_index; i<grid_width; i++){
	    from_pe = i*grid_height + row;
	    int p_block = local_height_mcmr(from_pe, m, n, grid_height, grid_width)* local_width_mcmr(from_pe, m, n, grid_height, grid_width);
 	    if(j < running_acc+p_block){
	       col_index = i;
	       break;
	    }
	    running_acc += p_block;
        }
        if(i == grid_width){
           printf("In unpacking from [MC,MR] to [MC,*]: size mismatch\n");
        }

	// At this point, j is in column id col_index (and pe p)
	// Find the id of block in received buffer from pe 'p'
	block_id = (j-running_acc) / local_height_mcmr(me, m, n, grid_height, grid_width);
	offset = (j-running_acc) % local_height_mcmr(me, m, n, grid_height, grid_width);

	// Now compute target index t;
	int t = (col_index + grid_width*block_id)*local_height_mcmr(me, m, n, grid_height, grid_width) + offset;

//if(me ==0) printf("(%d) j=%d block_id=%d offset=%d t=%d\n", myidx, j, block_id, offset, t);
	dev_target_buffer[t] = recv_buffer[j];
        j += stride;
    }
*/
    __syncthreads();
}

template <typename T>
__global__
void Allgatherv_put_col_kernel(
    int const my_pe_rank, int m, int n, int const grid_height, int const grid_width,
    T* __restrict__ sbuf, T* __restrict__ rbuf,
    int const my_displs, int send_size,
    int const* pes, int const npes, int const sync_counter, int volatile* workspace)
{
#if 1
    int myidx = threadIdx.x;

    // There will only ever be a single block with this kernel. 
    // This greatly simplifies things.

    // Does column-wise allgatherv
    // my_data_loc points the location where local copes start

    int my_col_rank = my_pe_rank/grid_height;
    int my_row_rank = my_pe_rank%grid_height;

    for (int i = 1; i <=grid_height; ++i)
    {
        auto const offset = (my_row_rank + i) % grid_height;
        auto const target_pe = offset + my_col_rank*grid_height;
        auto const peer = pes[target_pe];

//printf("(%d) offset=%d target_pe=%d peer=%d send_size=%d\n", my_pe_rank,  offset, target_pe, peer, send_size);

        nvshmemx_float_put_nbi_block(rbuf+my_displs, sbuf, send_size, peer);
    }
#else
    int myidx = threadIdx.x;
    int stride = blockDim.x;
    int block_size = local_height_mcmr(my_pe_rank, m, n, grid_height, grid_width);

    int my_col_rank = my_pe_rank/grid_height;
    int my_row_rank = my_pe_rank%grid_height;

    int j;
    for (int i = 1; i <=grid_height; ++i)
    {
        auto const offset = (my_row_rank + i) % grid_height;
        auto const target_pe = offset + my_col_rank*grid_height;
        auto const peer = pes[target_pe];

//printf("(%d) offset=%d target_pe=%d peer=%d send_size=%d\n", my_pe_rank,  offset, target_pe, peer, send_size);

	int ii, jj, t;
	int disp;
    	j = myidx;
	while(j < send_size){

	  disp = j % block_size;
	  jj = j / block_size;
 	  ii = disp*grid_height + my_row_rank; 
 	  t = ii*n + jj;
          nvshmem_float_put(rbuf+t, sbuf+j, 1, peer);
	  j += stride;
 	}
    }
#endif

#if 0
    // Order them before the sync puts
    if (threadIdx.x == 0)
        nvshmem_quiet();
    __syncthreads();

//printf("(%d) After fence \n", my_pe_rank);


    // Now that all the data has been put, we can flag to the receiver
    // PEs that the transfer is done. These are blocked after the
    // fence so there's only one fence.

#if 1
    if (threadIdx.x == 0){
        int const me = pes[my_pe_rank]; // "global rank"
        int *const target = workspace + me;

        // Flag myself as done
        *target = sync_counter;

        for (int i = 1; i <= npes; ++i)
        {
            int const id = (my_pe_rank + i) % npes;
            int const pe = pes[id];

            // This should be nvshmem_int_p, but this doesn't work over
            // IB. :/

//if(myidx == 0) printf("(%d) sending  %d to pes[%d] (me=%d) pe=%d\n", my_pe_rank, sync_counter, id, me, pe);
            nvshmem_int_put(target, target, 1, pe);

            //nvshmem_int_put(workspace + me, sync_counter, 1, pe);
            //nvshmemx_int_put_nbi_block(target, target, 1, pe);

        }
    }
 //   if (threadIdx.x == 0){
 //       nvshmem_quiet();
 //   }
    nvshmem_quiet();
    __syncthreads();

    if (threadIdx.x == 0){
      bool done = false;
      while (!done)
      {
        done = true;
        for (int i = 0; i < npes; ++i)
        {
            //if (watched_mem[inds[i]] != *sync_counter)
            if (workspace[i] != sync_counter)
            {
if(myidx == 0) printf("waiting on (%d) npes=%d sync_counter=%d workspace[%d]=%d\n", my_pe_rank, npes, sync_counter, i, workspace[i]);
                done = false;
                break;
            }
            else{
//if(my_pe_rank == 0)
//printf("[%d] sync_counter=%d workspace[%d]=%d\n", my_pe_rank, *sync_counter, i, workspace[i]);
            }
        }
      }
    }
    __syncthreads();
#endif
#endif

}

template <typename T>
__global__
void Allgatherv_put_row_kernel(
    int const my_pe_rank, int const grid_height, int const grid_width,
    T* __restrict__ sbuf, T* __restrict__ rbuf,
    int const my_displs, int send_size,
    int const* pes, int const npes, int const sync_counter, int volatile* workspace)
{

    // There will only ever be a single block with this kernel. 
    // This greatly simplifies things.

    // Does row-wise allgatherv

    // my_data_loc points the location where local copes start

    int my_col_rank = my_pe_rank/grid_height;
    int my_row_rank = my_pe_rank%grid_height;

    for (int i = 1; i <=grid_width; ++i)
    {
        auto const offset = (my_col_rank + i) % grid_width;
        auto const target_pe = offset*grid_height + my_row_rank;
        //auto const target_pe = my_col_rank*grid_height + offset;
        auto const peer = pes[target_pe];

//printf("(%d) offset=%d target_pe=%d peer=%d send_size=%d\n", my_pe_rank,  offset, target_pe, peer, send_size);

        nvshmemx_float_put_nbi_block(rbuf+my_displs, sbuf, send_size, peer);
    }

#if 0
    // Order them before the sync puts
    if (threadIdx.x == 0)
        nvshmem_quiet();
    __syncthreads();

//printf("(%d) After fence \n", my_pe_rank);


    // Now that all the data has been put, we can flag to the receiver
    // PEs that the transfer is done. These are blocked after the
    // fence so there's only one fence.
    //if (threadIdx.x == 0)
//    A__NotifyAll(workspace, sync_counter, pes, npes, my_pe_rank);

#if 1
    if (threadIdx.x == 0){
        int const me = pes[my_pe_rank]; // "global rank"
        int *const target = workspace + me;

        // Flag myself as done
        *target = sync_counter;

        for (int i = 1; i <= npes; ++i)
        {
            int const id = (my_pe_rank + i) % npes;
            int const pe = pes[id];

            // This should be nvshmem_int_p, but this doesn't work over
            // IB. :/

//if(myidx == 0) printf("(%d) sending  %d to pes[%d] (me=%d) pe=%d\n", my_pe_rank, sync_counter, id, me, pe);
            nvshmem_int_put(target, target, 1, pe);

            //nvshmem_int_put(workspace + me, sync_counter, 1, pe);
            //nvshmemx_int_put_nbi_block(target, target, 1, pe);

        }
    }
 //   if (threadIdx.x == 0){
 //       nvshmem_quiet();
 //   }
    nvshmem_quiet();
    __syncthreads();

    if (threadIdx.x == 0){
    bool done = false;
    while (!done)
    {
        done = true;
        for (int i = 0; i < npes; ++i)
        {
            //if (watched_mem[inds[i]] != *sync_counter)
            if (workspace[i] != sync_counter)
            {
//if(myidx == 0) printf("waiting on (%d) npes=%d sync_counter=%d workspace[%d]=%d\n", my_pe_rank, npes, sync_counter, i, workspace[i]);
                done = false;
                break;
            }
            else{
//if(my_pe_rank == 0)
//printf("[%d] sync_counter=%d workspace[%d]=%d\n", my_pe_rank, *sync_counter, i, workspace[i]);
            }
        }
    }
    }
    __syncthreads();
#endif
#endif

}

// Here 'my_pe_rank' is my pe rank, not MPI rank, and needs to be mapped to actual
// PE via pes before communication.
template <typename T>
__global__
void Allgatherv_put_kernel(
    int const my_pe_rank, int const grid_height, int const grid_width,
    T* __restrict__ sbuf, T* __restrict__ rbuf,
    int const displacement, int const size,
    int const* pes, int const npes, int const sync_counter, int* workspace)
{
    NotifyAll((int*) workspace, sync_counter, pes, npes, my_pe_rank);
    // There will only ever be a single block with this kernel. 
    // This greatly simplifies things.

    // Reset the workspace.
    //for (int ii = 0; ii < npes; ++ii)
    //    workspace[pes[ii]] = 0;

    // my_data_loc points the location where local copes start
    T const* my_data_loc = PutGathervNonLocal(my_pe_rank, grid_height, grid_width, sbuf, rbuf, displacement, size, pes, npes);

    // Order them before the sync puts
    if (threadIdx.x == 0)
        nvshmem_fence();

    // Ensure all threads in the block are up-to-date.
    __syncthreads();

    // Now that all the data has been put, we can flag to the receiver
    // PEs that the transfer is done. These are blocked after the
    // fence so there's only one fence.
    if (threadIdx.x == 0)
        NotifyAll(workspace, sync_counter+1, pes, npes, my_pe_rank);

    // Do the local data copy in a blocked manner.
    auto const& my_idx = threadIdx.x;
    unsigned long const stride = blockDim.x;

    int t = my_idx;
    while (t < size){
	my_data_loc[t] = sbuf[t];
	t += stride;
    }

    // Wait for everything to finish up
    if (threadIdx.x == 0)
        WaitOnAll(workspace, sync_counter+1, pes, npes);

    // This shouldn't be necessary.
    __syncthreads();
}

template<typename T>
__global__
void recursive_halving_put_kernel(
    int const m, int const n, int const my_pe_rank, 
    int const grid_height, int const grid_width,
    T* __restrict__ sbuf, T* __restrict__ rbuf, int const local_buf_len,
    int const my_row_rank, int my_col_rank,
    int const low_pow, int const offset,
    int const* pe_set, int const pe_set_len,
    int const* pes, int const npes, int* row_wspace)
{
    // Do recursive halving
    int current_len = low_pow;
    while(current_len > 1){
       int next = current_len/2;
       if (my_col_rank >= next){// I am a sender
           // send my local buffer to (my_col_rank-next)
           int target_col = my_col_rank - next; // target col rank
           int target = target_col*grid_height+my_row_rank;
           int peer = pes[target];

           // send my buffer to row proc: pes[PE(my_col_rank-next)]
	   nvshmemx_int_put_nbi_block(rbuf, sbuf, local_buf_len, peer);
       
	   // Order them before the sync puts
           if(threadIdx.x == 0)
              nvshmem_fence();
       
	   // Ensure all threads in the block are up-to-date.
           __syncthreads();

           // Now that all the data has been put, we can flag to the receiver
           // PEs that the transfer is done. 
	   // We only need to update only one element in workspace, which is the target PE.
           // fence so there's only one fence.
           if (threadIdx.x == 0)
               NotifyOne(row_wspace, next, target_col, my_row_rank, my_col_rank, pes, npes, grid_height, grid_width);
           __syncthreads();
       }
       else{// I am a receiver
           // Wait for the arrival of message from my partner at the lower end (my_row_rank+next)
	   WaitOnAllSet(row_wspace, next, grid_width);
           __syncthreads();

           // compute reduction
	   local_addition(sbuf, rbuf, local_buf_len);
           __syncthreads();
       }

       // We may need inter-PE synchronization (or maybe not)
       current_len = next;
    }
}

template<typename T>
__global__
void  finalize_gemm_kernel(int const my_row_rank, int const my_col_rank, int const grid_height, int const grid_width, T* local_buffer, T* dev_reduction_buf, int const local_buffer_length, T* dev_final_buf, int const final_buf_length, int const* dev_location_find_displs, int const* dev_scatter_counter, int const* dev_scatter_displs, int const chunk_unit, int const* pes, int npes, int* row_wspace)
{
    if(my_col_rank == 0){// I am the root of the row
        int my_idx = threadIdx.x;
        int stride = blockDim.x;
        // Rearrange data (final results from reduction) before distributing to row PEs

        // Compute my target PE and my initial location
        int col_assigned = -1;
        int delta_assigned = -1;
        int idx = my_idx%chunk_unit;
        for(int j=0; j<grid_width; j++){
   	    if(idx < dev_location_find_displs[j+1]){
	       col_assigned = j;
	       delta_assigned = idx-dev_location_find_displs[j];
	       break;
	    }
        }

        int t = my_idx;
        while(t < local_buffer_length){
	   int loc = dev_scatter_displs[col_assigned] + delta_assigned;
	   dev_reduction_buf[loc] = local_buffer[t];
	   delta_assigned += stride;
	   t += stride;
        }

        __syncthreads();
        // Reorganization of local data is done

        // Now distribute final 
        for(int j=1; j<grid_width; j++){
	    int target = j*grid_height+my_row_rank;
  	    int peer = pes[target];
            // send my buffer to row proc: pes[PE(my_row_rank-offset)]
            nvshmemx_int_put_nbi_block(dev_final_buf, dev_reduction_buf+dev_scatter_displs[j], dev_scatter_counter[j], peer);
        }

        // Order them before the sync puts
        if(threadIdx.x == 0)
           nvshmem_fence();

        // Ensure all threads in the block are up-to-date.
        __syncthreads();

        // Now that all the data has been put, we can flag to the receiver
        // PEs that the transfer is done. These are blocked after the
        // fence so there's only one fence.
        if (threadIdx.x == 0){
            for(int j=0; j<grid_width; j++)
	        NotifyOne(row_wspace, 1, j,  my_row_rank, my_col_rank, pes, npes, grid_height, grid_width);
        }
        __syncthreads();
    }
    else{// I am a receiver of the row 
        WaitOnSingle(row_wspace+my_col_rank, 1);
        __syncthreads();
    }

}

__global__
void Global_sync_kernel(
    int* pes, int npes, int my_pe_rank, 
    int sync_counter, int* workspace)
{
#if 1
//printf("(%d) npes=%d \n", my_pe_rank, npes); 
    int const me = pes[my_pe_rank]; // "global rank"
    int *const target = workspace + me;

    // Flag myself as done
    *target = sync_counter;

    for (int i = 1; i <= npes; ++i)
    {
        int const id = (my_pe_rank + i) % npes;
        int const pe = pes[id];

        // This should be nvshmem_int_p, but this doesn't work over
        // IB. :/

//printf("(%d) sending  %d to pes[%d] (me=%d) pe=%d\n", my_pe_rank, sync_counter, id, me, pe);
        nvshmem_int_put(target, target, 1, pe);

        //nvshmem_int_put(workspace + me, sync_counter, 1, pe);
    	//nvshmemx_int_put_nbi_block(target, target, 1, pe);

    }
    nvshmem_fence();

#if 1
    bool done = false;
    while (!done)
    {
        done = true;
        for (int i = 0; i < npes; ++i)
        {
            //if (watched_mem[inds[i]] != *sync_counter)
            if (workspace[i] != sync_counter)
            {
//if(my_pe_rank == 0)
//printf("-(%d) npes=%d sync_counter=%d workspace[%d]=%d\n", my_pe_rank, npes, sync_counter, i, workspace[i]);
                done = false;
//                break;
            }
            else{
//if(my_pe_rank == 0)
//printf("[%d] sync_counter=%d workspace[%d]=%d\n", my_pe_rank, sync_counter, i, workspace[i]);
            }
        }
    }
#endif
#else
    A__NotifyAll(workspace, sync_counter, pes, npes, my_pe_rank);
    A__WaitOnAll(my_pe_rank, workspace, sync_counter, pes, npes);
#endif

/*
    __syncthreads();
    __syncthreads();
*/

 return;
}


template <typename T>
__global__
void Alltoallv_put_kernel(
int total_recv,
    T const* __restrict__ sbuf, T* __restrict__ rbuf,
    int const* sdispls, int const* target_offset_counts,
    int const* pes, int const npes, int const my_pe_rank, 
    int const  sync_counter, int* workspace)
{
    // There will only ever be a single block with this kernel. This
    // greatly simplifies things.

    // Reset the workspace.
    //for (int ii = 0; ii < npes; ++ii)
    //   workspace[pes[ii]] = 0;

    T const* my_data = PutNonLocalv(sbuf, rbuf, sdispls, target_offset_counts, pes, npes, my_pe_rank);
    __syncthreads();

    // Order them before the sync puts
    //if (threadIdx.x == 0)
    //    nvshmem_fence();

    // Ensure all threads in the block are up-to-date.
    __syncthreads();

    // Order them before the sync puts
    //if (threadIdx.x == 0)
        nvshmem_fence();

    // Ensure all threads in the block are up-to-date.
    __syncthreads();

    // Now that all the data has been put, we can flag to the receiver
    // PEs that the transfer is done. These are blocked after the
    // fence so there's only one fence.
    // (threadIdx.x == 0)
        _NotifyAll(workspace, sync_counter, pes, npes, my_pe_rank);
    __syncthreads();
        _WaitOnAll(my_pe_rank, workspace, sync_counter, pes, npes);
    __syncthreads();

#if 0
    // Do the local data copy in a blocked manner.
    int my_idx = threadIdx.x;
    auto * rbeg = rbuf + target_offset_counts[2*my_pe_rank];
    auto * rend = rbeg + target_offset_counts[2*my_pe_rank+1];
    //auto * rend = rbuf + rdispls[my_pe_rank+1];
    int stride = blockDim.x;
    for (; rbeg < rend; rbeg += stride, my_data += stride)
    {
        //auto const local_size = min(rend-rbeg, stride);
        if (my_idx < target_offset_counts[2*my_pe_rank+1])
            rbeg[my_idx] = my_data[my_idx];
    }
    // This shouldn't be necessary.
    __syncthreads();

#endif
    // Wait for everything to finish up
    //if (threadIdx.x == 0)
        //WaitOnAll(workspace, sync_counter, pes, npes);
    __syncthreads();

}

template<typename T>
__global__
void transpose_vrstar_to_starvr_kernel(int const my_pe_rank, T* local_buffer, int const local_size, T* dev_tmp_buffer, int const m, int const n, int const grid_size, int const vr_row_rank, int const H, int const W, int const* pes, int const npes, int const sync_counter, int* const workspace)
{


    int myidx = threadIdx.x;
    int stride = blockDim.x;
    int t;

    if (threadIdx.x == 0)
        nvshmem_fence();
    __syncthreads();

    if (threadIdx.x == 0)
        NotifyAll(workspace, sync_counter, pes, npes, my_pe_rank);

    t = myidx;
    while(t < local_size){
        int i = t%H;
        int j = t/H;
        int l = i*W+j; // linear index of (j, i) in WxH transposed matrix 
        dev_tmp_buffer[l] = local_buffer[t];

	t += stride;
    }
    __syncthreads();

    t = myidx;
    while(t < local_size){
        local_buffer[t] = dev_tmp_buffer[t];
	t += stride;
    }
    __syncthreads();

    // Wait for everything to finish up
    if (threadIdx.x == 0)
        WaitOnAll(workspace, 1, pes, npes);

    // This shouldn't be necessary.
    __syncthreads();
}


// NOTE: All the PE numbers used in this function are those in given subcommunicator.
// Hence, it is required to translate these PE numbers to corresponding actual numbers
// when communication is needed.
template <typename T>
__global__
void Sendrecv_put_kernel(T const* __restrict__  local_buf, int local_size, int to_pe, 
	T* __restrict__ dev_recv_buf, int from_pe, 
	int const my_pe_rank, int const* pes, int const npes, int const sync_counter, int* const workspace)
{

    // Reset the workspace.
    //if (threadIdx.x == 0)
    //    for (int ii = 0; ii < npes; ++ii)
    //        workspace[pes[ii]] = 0;
    //__syncthreads();


    int dest_pe = pes[to_pe];
    nvshmemx_float_put_nbi_block(dev_recv_buf, local_buf, local_size, dest_pe);
    //if (threadIdx.x == 0)
        nvshmem_fence();

    // Ensure all threads in the block are up-to-date.
    __syncthreads();

    // Now that all the data has been put, we can flag to the receiver
    // PEs that the transfer is done. These are blocked after the
    // fence so there's only one fence.
    //if (threadIdx.x == 0)
        _NotifyAll(workspace, sync_counter, pes, npes, my_pe_rank);
        //_NotifyAll(workspace, sync_counter, pes, npes, my_pe_rank);
    __syncthreads();
/*
    if (threadIdx.x == 0)
        nvshmem_quiet();
    __syncthreads();
*/

    // Wait for everything to finish up
    //if (threadIdx.x == 0)
        _WaitOnAll(my_pe_rank, workspace, sync_counter, pes, npes);

    // This shouldn't be necessary.
    __syncthreads();
}

template <typename T>
__global__
void Alltoall_put_kernel_boring(
    T const* __restrict__ sbuf, T* __restrict__ rbuf, int const size,
    int const* pes, int const npes, int const me, int const sync_counter, int* const workspace)
{
    // There will only ever be a single block with this kernel. This
    // greatly simplifies things.

    // Reset the workspace.
    //for (int ii = 0; ii < npes; ++ii)
    //    workspace[pes[ii]] = 0;

    // Compute my rank in the "pes" "communicator"
    auto my_rank = me; // FindLocation(pes, npes, me);

    // Find my offset
    rbuf += my_rank*size;

    // Do the nonlocal puts. This function will issue puts to each
    // rank except "my_rank". It will round-robin starting with
    // "(my_rank + 1) % npes".

    T const* my_data = PutNonLocal(sbuf, rbuf, size, pes, npes, my_rank);

    // Order them before the sync puts
    if (threadIdx.x == 0)
        nvshmem_fence();

    // Ensure all threads in the block are up-to-date.
    __syncthreads();

    // Now that all the data has been put, we can flag to the receiver
    // PEs that the transfer is done. These are blocked after the
    // fence so there's only one fence.
    if (threadIdx.x == 0)
        NotifyAll(workspace, sync_counter, pes, npes, my_rank);

    // Do the local data copy in a blocked manner.
    auto const& my_idx = threadIdx.x;
    auto const* rend = rbuf + size;
    unsigned long const stride = blockDim.x;
    for (; rbuf < rend; rbuf += stride, my_data += stride)
    {
        auto const local_size = min(rend - rbuf, stride);
        if (my_idx < local_size)
            rbuf[my_idx] = my_data[my_idx];
    }

    // Wait for everything to finish up
    if (threadIdx.x == 0)
        WaitOnAll(workspace, 1, pes, npes);

    // This shouldn't be necessary.
    __syncthreads();
}
//}// namespace <anon>

namespace hydrogen
{

void counts_mcmr_to_mrstar(int grid_height, int grid_width, int myrank, int m, int n, int* total_gather, int* my_displs)
{
    *total_gather = 0;
    int grid_size = grid_height*grid_width;
    for(int j=0; j<grid_size; j++){
        if(j == myrank)
           *my_displs = *total_gather;
        if(grid_col_rank(myrank, grid_height, grid_width) == grid_col_rank(j, grid_height, grid_width)){
           *total_gather += local_height_mcmr(j, m, n, grid_height, grid_width) * local_width_mcmr(j, m, n, grid_height, grid_width);
        }
    }
}

void counts_mcmr_to_mcstar(
	int grid_height, int grid_width, int myrank, int m, int n, int* total_gather, int* my_displs)
{
    *total_gather = 0;
    int grid_size = grid_height*grid_width;
    int pos=0;
    for(int j=0; j<grid_size; j++){
        if(j == myrank)
	   *my_displs = *total_gather;
	if(grid_row_rank(myrank, grid_height, grid_width) == grid_row_rank(j, grid_height, grid_width)){
           *total_gather += local_height_mcmr(j, m, n, grid_height, grid_width) * local_width_mcmr(j, m, n, grid_height, grid_width);
	}
    }
}

void counts_mcmr_to_vc_star(int grid_height, int grid_width, int myrank, int m, int n,
	std::vector<int>& send_counts, std::vector<int>& send_displs, int* total_send, 
	std::vector<int>& recv_counts, std::vector<int>& recv_displs, int* total_recv,
        std::vector<int>& offset_counts){

    int grid_size = grid_height*grid_width;
    for(int j=0; j<grid_width; j++){
        int p = grid_row_rank(myrank, grid_height, grid_width)+j*grid_height;
        send_counts[p] = local_height_vcstar(p, m, n, grid_height, grid_width) * local_width_mcmr(myrank, m, n, grid_height, grid_width);
    }

    *total_send = 0;
    send_displs[0]  = 0;
    for(int p=1; p<=grid_size; p++){
        send_displs[p] = send_displs[p-1] + send_counts[p-1];
        *total_send += send_counts[p-1];
    }

    for(int j=0; j<grid_width; j++){
        int p = grid_row_rank(myrank, grid_height, grid_width)+j*grid_height;
        recv_counts[p] = local_height_vcstar(myrank, m, n, grid_height, grid_width) * local_width_mcmr(p, m, n, grid_height, grid_width);
    }

    *total_recv =0;
    recv_displs[0]  = 0;
    for(int p=1; p<=grid_size; p++){
        recv_displs[p] = recv_displs[p-1] + recv_counts[p-1];
        *total_recv += recv_counts[p-1];
    }

    int my_row_rank = grid_row_rank(myrank, grid_height, grid_width);
    int index=0;
    for(int target_pe=0; target_pe<grid_size; target_pe++){
	if(grid_row_rank(target_pe, grid_height, grid_width) == grid_row_rank(myrank, grid_height, grid_width)){
	   int offset = 0;
           for(int from_pe=0; from_pe<myrank; from_pe++){
	       if(grid_row_rank(target_pe, grid_height, grid_width) == grid_row_rank(from_pe, grid_height, grid_width)){
                   offset += local_height_vcstar(target_pe, m, n, grid_height, grid_width) * local_width_mcmr(from_pe, m, n, grid_height, grid_width);
	       }
	   }
	   offset_counts[index]=offset;
	   offset_counts[index+1]=local_height_vcstar(target_pe, m, n, grid_height, grid_width) * local_width_mcmr(myrank, m, n, grid_height, grid_width);
        }
	index += 2;
    }
}


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
        int* sync_space, cudaStream_t const stream)
{
    int grid_size = grid_height*grid_width;
    std::vector<int> send_counts(grid_size, 0);
    std::vector<int> send_displs(grid_size+1, 0);
    int total_send;
    int max_send;
    std::vector<int> recv_counts(grid_size, 0);
    std::vector<int> recv_displs(grid_size+1, 0);
    int total_recv;
    int max_recv;

    // Compute the maximum size for send and recv operation. Need this because
    // these buffers are SDOs of nvshmem (OpenSHMEM constraint)
    max_send = (int) (ceil(((double)m)/grid_height)*ceil(((double)n)/grid_width));
    max_recv = (int) (ceil(((double)m)/grid_size)*n);

    int num_threads;

    std::vector<int> offset_counts (grid_size*2, 0);

    counts_mcmr_to_vc_star(grid_height, grid_width, me, m, n,
  		send_counts, send_displs, &total_send,
		recv_counts, recv_displs, &total_recv,
		offset_counts);

    std::vector<T> send_buffer(total_send);
    std::vector<T> recv_buffer(total_recv);
    num_threads = min(total_send, 1024);
    
    int* dev_send_displs = (int*) nvshmem_malloc(send_displs.size()*sizeof(int));
    if(dev_send_displs == NULL){
       throw std::runtime_error("error allocating SDO");
    }
    if(dev_send_displs == NULL){
       throw std::runtime_error("In from [MC,MR] to [VC,*]: error allocating SDO");
    }
    CHECK_CUDA(cudaMemcpy((void*) dev_send_displs, (void const*) send_displs.data(), send_displs.size()*sizeof(int), cudaMemcpyHostToDevice));

    T* dev_send_buf = (T*) nvshmem_malloc(max_send*sizeof(T));
    if(dev_send_buf == NULL){
       throw std::runtime_error("In from [MC,MR] to [VC,*]: error allocating SDO");
    }
    CHECK_CUDA(cudaMemcpy((void*) dev_send_buf, (void const*) send_buffer.data(), total_send*sizeof(T), cudaMemcpyHostToDevice));
    //CHECK_CUDA(cudaMemcpyAsync((void*) dev_send_buf, (void const*) send_buffer.data(), total_send*sizeof(T), cudaMemcpyHostToDevice, stream));

    int* dev_target_offset_counts = (int*) nvshmem_malloc(offset_counts.size()*sizeof(int));
    if(dev_target_offset_counts == NULL){
       throw std::runtime_error("In from [MC,MR] to [VC,*]: error allocating SDO");
    }
    CHECK_CUDA(cudaMemcpy((void*) dev_target_offset_counts, (void const*) offset_counts.data(), offset_counts.size()*sizeof(int), cudaMemcpyHostToDevice));

    T* dev_recv_buf = (T*) nvshmem_malloc(max_recv*sizeof(T));
    if(dev_recv_buf == NULL){
       throw std::runtime_error("In from [MC,MR] to [VC,*]: error allocating SDO");
    }

    pack_mcmr_to_vc_star<<<1,num_threads,0,stream>>> (m, n, me, grid_height, grid_width, dev_send_displs, local_buffer, dev_send_buf);

    int const sync_counter=1;
    Alltoallv_put(total_recv, me, dev_send_buf, dev_recv_buf, dev_send_displs, dev_target_offset_counts, dev_pes, npes, sync_counter, sync_space, stream); 

#if 0
    // Need to realloc local buffer because the size of the local buffer is changed as
    // the distribution is changed from [MC,MR] to [VC,*]
    // This number is actually the same as the new recv size
    // tried to use nvshmem_realloc(local_buffer, max_recv*sizeof(T)); initially, but the API is not implemented.Bahh
    //nvshmem_free(local_buffer);
    //local_buffer = (T*) nvshmem_malloc(max_recv*sizeof(T));

    int* dev_recv_displs = (int*) nvshmem_malloc(grid_size+1);
    //CHECK_CUDA(cudaMemcpyAsync((void*) dev_recv_displs, (void const*) recv_displs.data(), (grid_size+1)*sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpy((void*) dev_recv_displs, (void const*) recv_displs.data(), (grid_size+1)*sizeof(int), cudaMemcpyHostToDevice));


    num_threads = 1024;
    //num_threads = min(max_recv, 1024);
    unpack_mcmr_to_vc_star<<<1,num_threads,0,stream>>> (m, n, me, grid_height, grid_width, dev_recv_displs, local_buffer, dev_recv_buf);


sprintf(line, "___debug.%04d", me);
fp_debug = fopen(line, "w");

T* B1_VC_STAR_mem_buffer = (T*) malloc(max_recv*sizeof(T));
cudaMemcpy(B1_VC_STAR_mem_buffer, local_buffer, max_recv*sizeof(T), cudaMemcpyDeviceToHost);
fprintf(fp_debug, "Buffer of B_VC_STAR...\n");
for(int l=0; l<total_recv; l++){
       fprintf(fp_debug, "%f ", B1_VC_STAR_mem_buffer[l]);
}
fprintf(fp_debug, "\n");
fclose(fp_debug);

    nvshmem_free(dev_recv_displs);
    nvshmem_free(dev_recv_buf);
    nvshmem_free(dev_target_offset_counts);
    nvshmem_free(dev_send_buf);
    nvshmem_free(dev_send_displs);
#endif
}

void convert_mcmr_to_vcstar_float(int m, int n, int me, int grid_height, int grid_width, 
	float* local_buffer, 
        int const* dev_pes, 
	int const npes,
        int* sync_space, cudaStream_t const stream)
{
  convert_mcmr_to_vcstar<float> (m, n, me, grid_height, grid_width, 
	local_buffer, 
	1024, // error: don't need
        dev_pes, 
	npes,
        sync_space, 
	stream);
}

// Convert a Matrix (of size m x n) in [VC,*] format to the one in [VR,*] format
// 'my_pe_rank' is my nvshmem pe, PE number in given (sub)communicator, not MPI rank; 
// It is expected that my PE number is computed prior 
void convert_vcstar_to_vrstar_float(int m, int n, int my_pe_rank,
        int grid_height, int grid_width,
        float* local_buffer,
        int const* pes, int const npes,
        int* const sync_space, cudaStream_t const stream)
{
  convert_vcstar_to_vrstar<float>(m, n, my_pe_rank,
        grid_height, grid_width,
        local_buffer,
        pes, npes,
        sync_space, stream);
}

template <typename T>
void Alltoall_put(T const* sbuf, T* rbuf, int const size,
                  int const* pes, int const npes,
                  int* const sync_space, cudaStream_t const stream)
{
    // No sense using more threads than data elements, I should think...
    int const num_threads = std::min(size, 1024);
    int const me = nvshmem_my_pe();
    Alltoall_put_kernel_boring<<<1,num_threads,0,stream>>>(
        sbuf, rbuf, size, pes, npes, me, sync_space);
}

// Here, 'me' is my PE number in given (sub)communicator
template <typename T>
void Alltoallv_put(
int total_recv,
int me,
		  T const* sbuf, T* rbuf, 
		  int const* sdispls, int const* target_offset_counts,
                  int const* pes, int const npes,
                  int const sync_counter, int* sync_space, 
		  cudaStream_t const stream)
{

    // No sense using more threads than data elements, I should think...
    //int const num_threads = std::min(total_recv, 1024);
    int const num_threads = 1024;

#if 0
    void* args[] = { &total_recv,
        &sbuf, &rbuf, &sdispls, &target_offset_counts,
	&pes, &npes, &me, &sync_counter, &sync_space};
    nvshmemx_collective_launch(
                (void const*)Alltoallv_put_kernel,
                1, num_threads, args, 0, stream);
#endif

    Alltoallv_put_kernel<<<1,num_threads,0,stream>>>( total_recv,
        sbuf, rbuf, sdispls, target_offset_counts,
	pes, npes, me, sync_counter, sync_space);

//    nvshmemx_barrier_all_on_stream(stream);
}

template <typename T>
void Sendrecv_put(T const* local_buf, int const local_size, int const to_pe,
		  T* dev_recv_buf, int const from_pe,
		  //T* dev_recv_buf, int const recv_size, int const from_pe,
                  int const my_pe_rank, int const* pes, int const npes,
                  int const sync_counter, int* const sync_space, cudaStream_t const stream)
{
    // No sense using more threads than data elements, I should think...
    int const num_threads = std::min(local_size, 1024);
    Sendrecv_put_kernel<<<1,num_threads,0,stream>>>(
        local_buf, local_size, to_pe, dev_recv_buf, from_pe, my_pe_rank, pes, npes, sync_counter, sync_space);
}


// Convert a Matrix (of size m x n) in [VC,*] format to the one in [VR,*] format
// 'my_pe_rank' is my nvshmem pe, PE number in given (sub)communicator, not MPI rank; 
// It is expected that my PE number is computed prior 
template <typename T>
void convert_vcstar_to_vrstar(int m, int n, int my_pe_rank, 
	int grid_height, int grid_width,
	T* local_buffer, 
        int const* pes, int const npes,
        int* const sync_space, cudaStream_t const stream)
  {
    // Calculate the PEs to send to and receive from me
    int to_pe = vcstar_to_vrstar_to_pid(my_pe_rank, grid_height, grid_width);
    int from_pe = vcstar_to_vrstar_from_pid(my_pe_rank, grid_height, grid_width);
    int local_size =  local_height_vcstar(my_pe_rank, m, n, grid_height, grid_width) * local_width_vcstar(my_pe_rank, m, n, grid_height, grid_width);

    int my_send_size_vcstar_to_vrstar = local_size;
    int max_recv_size_vcstar_to_vrstar = (int) ceil(((double)m)/grid_height) * local_width_vcstar(from_pe, m, n, grid_height, grid_width);
    int local_recv_size_vcstar_to_vrstar = local_height_vcstar(from_pe, m, n, grid_height, grid_width) * local_width_vcstar(from_pe, m, n, grid_height, grid_width);
    T* dev_recv_buffer = (T*) nvshmem_malloc(max_recv_size_vcstar_to_vrstar*sizeof(T));
    if(dev_recv_buffer == NULL){
       throw std::runtime_error("In from [VC,*] to [VR,*]: error allocating SDO");
    }

    //Sendrecv_put(dev_local_buffer, local_size, to_pe,
    Sendrecv_put(local_buffer, local_size, to_pe,
		 dev_recv_buffer, from_pe,
		 //dev_recv_buffer, local_recv_size_vcstar_to_vrstar, from_pe,
		 my_pe_rank, pes, npes, 2, sync_space, stream);

    // At this point, dev_recv_buffer has exchanged data, which is needed to copied back to the local_buffer
    // No copying needed if to_pe == from_pe
    if(to_pe != from_pe)
      {// send/recv only when it is needed
	if(local_size != local_recv_size_vcstar_to_vrstar)
	{// nvshmem_realloc is need to adjust local buffer for new data
	    //nvshmem_realloc(local_buffer, new_recv_size_vcstar_to_vrstar*sizeof(T));
	    CHECK_CUDA(cudaFree((void*) local_buffer));
    	    CHECK_CUDA(cudaMalloc((void**) &local_buffer, (size_t) (local_recv_size_vcstar_to_vrstar*sizeof(T))));
	}
    	CHECK_CUDA(cudaMemcpy((void*) local_buffer, (void const*) dev_recv_buffer,
				   local_recv_size_vcstar_to_vrstar*sizeof(T), cudaMemcpyDeviceToDevice));
      }
    
 //   nvshmem_free(dev_local_buffer);
    nvshmem_free(dev_recv_buffer);
  }


// Convert a Matrix (of size m x n) in [VR,*] format to the one in [*,VR] format; Basically, this is done by transposing
// given [VR,*] matrix into [*,VR] matrix; Hence, the dimension of inputt matrix is changed from (m x n) to (n x m) after the
// conversion.
// 'my_pe_rank' is my nvshmem pe, not MPI rank; it is expected that my PE number is computed prior 
template<typename T>
void convert_vrstar_to_starvr(T* local_buffer, int const local_size, int const m, int const n, int const grid_height, int const grid_width, int const my_pe_rank, int const* pes, int const npes, int* const sync_space, cudaStream_t const stream)

{   
    int grid_size = grid_height*grid_width;
    int vr_row_rank = my_vr_row_rank(my_pe_rank, grid_height, grid_width);
    int H = m/grid_size + ((vr_row_rank < m%grid_size)?1:0);
    int W= n;
    if(local_size != H*W){
       throw std::runtime_error("In transpose from [VR,*] to [*,VR]: local size mismatch"); 
    }

    T* dev_tmp_buffer;
    CHECK_CUDA(cudaMalloc((void**) &dev_tmp_buffer, (size_t) (local_size*sizeof(T))));
    int const num_threads = std::min(local_size, 1024);
    transpose_vrstar_to_starvr_kernel<<<1,num_threads,0,stream>>>(my_pe_rank, local_buffer, local_size, dev_tmp_buffer,
m, n, grid_height*grid_width, vr_row_rank, H, W, pes, npes, sync_space);
    CHECK_CUDA(cudaFree((void*) dev_tmp_buffer));

}

/*
// NOTE: 'my_pe_rank' is my PE number in (sub)communicator. Need to translate this to actual PE via pes
void counts_mcmr_to_vc_star(
	int my_pe_rank,
	int k, int n, int grid_height, int grid_width,
	std::vector<int>& mr_msg_cnts, std::vector<int>& star_mr_buffer_displs,
	int *sum
)
{
    int grid_size = grid_height*grid_width;

    int mycrank = grid_col_rank(my_pe_rank, grid_height, grid_width);
    *sum = 0;

    for(int i=0; i<grid_height; i++){
        int l = mycrank*grid_height + i;
        int count_i = n * (k/grid_size + ((my_vr_row_rank(l, grid_height, grid_width) < k%grid_width)?1:0));
        mr_msg_cnts[i] = count_i;
        *sum += count_i;
    }
    for(int p=1; p<=grid_height; p++)
        star_mr_buffer_displs[p] = star_mr_buffer_displs[p-1]+mr_msg_cnts[p-1];
}
*/

// mr_msg_cnt and star_mr_buffer_displs are vectors of grid_height and grid_height+1 elements, respectively.
void counts_starvr_to_starmr(int const my_pe_rank, int const m, int const n, int const grid_height, int const grid_width,
	std::vector<int>& mr_msg_cnts, std::vector<int>& star_mr_buffer_displs, int* sum)
{
    int mycrank = grid_col_rank(my_pe_rank, grid_height, grid_width);
    int grid_size = grid_height*grid_width;
    *sum = 0;

    for(int i=0; i<grid_height; i++){
        int l = mycrank*grid_height + i;
        int count_i = n * (m/grid_size + ((my_vr_row_rank(l, grid_height, grid_width) < m%grid_width)?1:0));
        mr_msg_cnts[i] = count_i;
        *sum += count_i;
    }

    for(int i=1; i<=grid_height; i++)
        star_mr_buffer_displs[i] = star_mr_buffer_displs[i-1]+mr_msg_cnts[i-1];
}


template <typename T>
void Allgatherv_put_col(int const my_pe_rank, int m, int n, 
	int const grid_height, int const grid_width,
        T* dev_send_buf, T* dev_recv_buf, int send_size,
        int const my_displ,
        int const* pes, int const npes,
	int sync_counter,
        int* const sync_space, cudaStream_t const stream)
{
    // No sense using more threads than data elements, I should think...
    //int const num_threads = std::min(send_size, 1024);
    int const num_threads = min(send_size, 1024);
    Allgatherv_put_col_kernel<<<1,num_threads,0,stream>>>(
        my_pe_rank, m, n, grid_height, grid_width,
        dev_send_buf, dev_recv_buf, 
        my_displ, send_size,
        pes, npes, sync_counter, sync_space);

    nvshmemx_quiet_on_stream(stream);
}

template <typename T>
void Allgatherv_put_row(int const my_pe_rank, int const grid_height, int const grid_width,
        T* dev_send_buf, T* dev_recv_buf, int send_size,
        int const my_displ,
        int const* pes, int const npes,
	int sync_counter,
        int* const sync_space, cudaStream_t const stream)
{
    // No sense using more threads than data elements, I should think...
    //int const num_threads = std::min(send_size, 1024);
    int const num_threads = min(send_size, 1024);
    Allgatherv_put_row_kernel<<<1,num_threads,0,stream>>>(
        my_pe_rank, grid_height, grid_width,
        dev_send_buf, dev_recv_buf, 
        my_displ, send_size,
        pes, npes, sync_counter, sync_space);
    nvshmemx_quiet_on_stream(stream);
}

// Convert a Matrix (of size m x n) in [*,VR] format to the one in [*,MR] format.
// 'my_pe_rank' is my PE number in given communicator, which MUSTT be mapped to actual
// PE number for communication
template <typename T>
void Allgatherv_put(int const my_pe_rank, int const grid_height, int const grid_width,
	T* dev_send_buf, T* dev_recv_buf,
	int const offset, int const size,  
        int const* pes, int const npes,
        int* const sync_space, cudaStream_t const stream)
{
    // No sense using more threads than data elements, I should think...
    int const num_threads = std::min(size, 1024);
    Allgatherv_put_kernel<<<1,num_threads,0,stream>>>(
	my_pe_rank, grid_height, grid_width,
	dev_send_buf, dev_recv_buf,
	offset, size,
        pes, npes, sync_space);
}

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
        	int* const sync_space, cudaStream_t const stream)
{
    std::vector<int> mr_msg_cnts(grid_height);
    std::vector<int> star_mr_buffer_displs(grid_height+1, 0);
    int sum;

    counts_starvr_to_starmr(my_pe_rank, m, n, grid_height, grid_width, mr_msg_cnts, star_mr_buffer_displs, &sum);

    T* dev_recv_buf = (T*) nvshmem_malloc(sum*sizeof(T));
    if(dev_recv_buf == NULL){
       throw std::runtime_error("error allocating SDO");
    }
    T* dev_star_mr_buffer_displs;
    CHECK_CUDA(cudaMalloc((void**) &dev_star_mr_buffer_displs, (size_t) ((grid_height+1)*sizeof(T))));

    int rrank = grid_row_rank(my_pe_rank, grid_height, grid_width);
    Allgatherv_put(my_pe_rank, grid_height, grid_width,
		   local_buffer, dev_recv_buf,
		   star_mr_buffer_displs[rrank], mr_msg_cnts[rrank],
                   dev_pes, npes, sync_space, stream);
    nvshmem_realloc(local_buffer, sum*sizeof(T));

    int const num_threads = std::min(sum, 1024);
    interleave_starvr_to_starmr_kernel<<<1,num_threads,0,stream>>> (m, n, grid_height, grid_width,
         local_buffer, dev_recv_buf, dev_star_mr_buffer_displs, sum);

    nvshmem_free(dev_recv_buf);
    CHECK_CUDA(cudaFree((void*) dev_star_mr_buffer_displs));

}


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
        int* const sync_space, cudaStream_t const stream)
{
    // Reduce:
    // pre-send and pre-recv
    // recursive halving and computing
    // rearranging final reduction results at root
    // Scatter: results to the members on grid row

    // dev_reduction_buf will hold the intermediate reduction data received from my partener during reduction process and
    int local_reduction_buffer_length = n*local_height_mcmr(my_pe_rank, m, n, grid_height, grid_width);
    T* dev_reduction_buf = (T*) nvshmem_malloc(local_reduction_buffer_length*sizeof(T));
    if(dev_reduction_buf == NULL){
       throw std::runtime_error("error allocating SDO");
    }

    // dev_final_buf will hold the final GEMM results after reduce-scatter
    int final_buf_length = local_height_mcmr(my_pe_rank, m, n, grid_height, grid_width)*local_width_mcmr(my_pe_rank, m, n, grid_height, grid_width);
    T* dev_final_buf = (T*) nvshmem_malloc(final_buf_length*sizeof(T));
    if(dev_final_buf == NULL){
       throw std::runtime_error("error allocating SDO");
    }

    int row_width = grid_width;
    int my_row_rank = grid_row_rank(my_pe_rank, grid_height, grid_width);
    int my_col_rank = grid_col_rank(my_pe_rank, grid_height, grid_width);
  
    int low_pow = (int) pow(2.0, floor(log2((double)row_width)));
    int offset = row_width - low_pow;

    std::vector<int> row_pes (grid_width);
    for(int i=0; i<grid_width; i++)
	row_pes[i] = my_row_rank + i*grid_height;
    T* dev_row_pes;
    CHECK_CUDA(cudaMalloc((void**) &dev_row_pes, (size_t) (grid_width*sizeof(T))));
    CHECK_CUDA(cudaMemcpyAsync((void*) dev_row_pes, (void const*) row_pes.data(), grid_width*sizeof(T), cudaMemcpyHostToDevice, stream));
    
    int* dev_row_workspace;
    CHECK_CUDA(cudaMalloc((void**) &dev_row_workspace, (size_t) (grid_width*sizeof(int))));
    CHECK_CUDA(cudaMemset((void*) dev_row_workspace, 0, (size_t)(grid_width*sizeof(int))));

    int const num_threads = std::min(local_reduction_buffer_length, 1024);
    lower_data_put_kernel<<<1,num_threads,0,stream>>> (m, n, my_pe_rank, grid_height, grid_width,
    	local_buffer, dev_reduction_buf, local_reduction_buffer_length, my_row_rank, my_col_rank,
    	low_pow, offset, dev_row_pes, grid_width, dev_pes, npes, dev_row_workspace);
    
    // Clear out workspace by cudaMemset

    // nvshmem_barrier_on_stream() needed for global synchronization??? 

    CHECK_CUDA(cudaMemset((void*) dev_row_workspace, 0, (size_t)(grid_width*sizeof(int))));
    if(my_col_rank < low_pow){
	recursive_halving_put_kernel(m, n, my_pe_rank, grid_height, grid_width,
    	local_buffer, dev_reduction_buf, local_reduction_buffer_length,
    	my_row_rank, my_col_rank,
    	low_pow, offset,
    	dev_row_pes, grid_width,
    	dev_pes, npes, dev_row_workspace);
    }

    // global synchronization. Is it needed here? I think so

    // At this point, final reduction results are in 'local_buffer' on the root PE (which is row rank 0 in the grid row);
    // Hence, we need to distribute data in local_buffer in a cyclic fashion.
    // We will reuse the reduction buffer to rearrange final reduction results before performing scatter operation.
    //
    // Scatter:
    // create a new data buffer for scatter
    // rearrange the reduction result in the order of receiving buffer

    
    std::vector<int> location_find_counter(grid_width, 0);
    std::vector<int> location_find_displs((grid_width+1), 0);
    for(int i=0; i<grid_width; i++){
 	int p_rank = row_pes[i];
	location_find_counter[i] = local_height_mcmr(p_rank, m, n, grid_height, grid_width);
    }
    int chunk_unit=0;
    for(int i=1; i<=grid_width; i++){
	location_find_displs[i] = location_find_displs[i-1]+location_find_counter[i-1];
	chunk_unit += location_find_counter[i-1];
    }
    int* dev_location_find_displs;
    CHECK_CUDA(cudaMalloc((void**) &dev_location_find_displs, (size_t) ((grid_width+1)*sizeof(int))));
    CHECK_CUDA(cudaMemcpyAsync((void*) dev_location_find_displs, (void const*) location_find_displs.data(), (grid_width+1)*sizeof(int), cudaMemcpyHostToDevice, stream));

    std::vector<int> scatter_counter(grid_width, 0);
    std::vector<int> scatter_displs(grid_width+1, 0);
    for(int j=0; j<grid_width; j++){
 	int col_rank = row_pes[j];
        scatter_counter[col_rank] = local_height_mcmr(col_rank, m, n, grid_height, grid_width)*local_width_mcmr(col_rank, m, n, grid_height, grid_width);
    }
    for(int j=1; j<=grid_width; j++){
    	scatter_displs[j] = scatter_displs[j-1] + scatter_counter[j-1];
    }

    int* dev_scatter_counter;
    int* dev_scatter_displs;
    CHECK_CUDA(cudaMalloc((void**) &dev_scatter_counter, (size_t) grid_width*sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &dev_scatter_displs, (size_t) ((grid_width+1)*sizeof(int))));
    CHECK_CUDA(cudaMemcpyAsync((void*) dev_scatter_displs, (void const*) scatter_displs.data(), (grid_width+1)*sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync((void*) dev_scatter_counter, (void const*) scatter_counter.data(), grid_width*sizeof(int), cudaMemcpyHostToDevice, stream));

    CHECK_CUDA(cudaMemset((void*) dev_row_workspace, 0, (size_t)(grid_width*sizeof(int))));
    num_threads = (1024/chunk_unit)*chunk_unit;
    finalize_gemm_kernel<<<1,num_threads,0,stream>>> (my_row_rank, my_col_rank, grid_height, grid_width, local_buffer, dev_reduction_buf, local_buffer_length, dev_final_buf, final_buf_length,
dev_location_find_displs, dev_scatter_counter, dev_scatter_displs, chunk_unit, dev_pes, npes, dev_row_workspace);

    // Do we need global synchronization between PEs?
    nvshmem_free(dev_reduction_buf);
    //nvshmem_free(dev_final_buf);
    CHECK_CUDA(cudaFree((void*) dev_row_pes));
    CHECK_CUDA(cudaFree((void*) dev_row_workspace));
    CHECK_CUDA(cudaFree((void*) dev_location_find_displs));
    CHECK_CUDA(cudaFree((void*) dev_scatter_counter));
    CHECK_CUDA(cudaFree((void*) dev_scatter_displs));
}

template<typename T>
void mcmr_to_vcstar(MPI_Comm, int, int, int, int, T*, T*, cudaStream_t)
{
  throw std::runtime_error("Function not implemented\n");
}

void mcmr_to_vcstar(MPI_Comm mpi_comm,
	int m,
	int n,
	int grid_height,
	int grid_width,
	float* dev_Matrix_buffer,
	float* dev_target_buffer,
	cudaStream_t stream)
{
	int grid_size = grid_height*grid_width;
        int my_pe_rank;
        std::vector<int> pes;
        int xnpes;
 	setup_pes(mpi_comm, &my_pe_rank, pes, &xnpes);

	int mcmr_local_height  = local_height_mcmr(my_pe_rank, m, n, grid_height, grid_width);
	int mcmr_local_width   = local_width_mcmr(my_pe_rank, m, n, grid_height, grid_width);

    	std::vector<int> send_counts(grid_size, 0);
    	std::vector<int> send_displs(grid_size+1, 0);
    	int total_send;
    	int max_send;
    	std::vector<int> recv_counts(grid_size, 0);
    	std::vector<int> recv_displs(grid_size+1, 0);
    	int total_recv;
    	int max_recv;

    	// Compute the maximum size for send and recv operation. Need this because
    	// these buffers are SDOs of nvshmem (OpenSHMEM constraint)
    	max_send = (int) (ceil(((double)m)/grid_height)*ceil(((double)n)/grid_width));
    	max_recv = (int) (ceil(((double)m)/grid_size)*n);

    	int num_threads;

    	std::vector<int> offset_counts (grid_size*2, 0);

    	counts_mcmr_to_vc_star(grid_height, grid_width, my_pe_rank, m, n,
                send_counts, send_displs, &total_send,
                recv_counts, recv_displs, &total_recv,
                offset_counts);
    	num_threads = min(total_send, 1024);

    	int* dev_send_displs = (int*) nvshmem_malloc(send_displs.size()*sizeof(int));
        if(dev_send_displs == NULL){
           throw std::runtime_error("error allocating SDO");
        }
    	CHECK_CUDA(cudaMemcpy((void*) dev_send_displs, (void const*) send_displs.data(), send_displs.size()*sizeof(int), cudaMemcpyHostToDevice));

    	float* dev_send_buf = (float*) nvshmem_malloc(max_send*sizeof(float));
        if(dev_send_buf == NULL){
           throw std::runtime_error("error allocating SDO");
        }
    	int* dev_target_offset_counts = (int*) nvshmem_malloc(offset_counts.size()*sizeof(int));
        if(dev_target_offset_counts == NULL){
           throw std::runtime_error("error allocating SDO");
        }
    	CHECK_CUDA(cudaMemcpy((void*) dev_target_offset_counts, (void const*) offset_counts.data(), offset_counts.size()*sizeof(int), cudaMemcpyHostToDevice));

    	float* dev_recv_buf = (float*) nvshmem_malloc(max_recv*sizeof(float));
        if(dev_recv_buf == NULL){
           throw std::runtime_error("error allocating SDO");
        }

    	int to_pe = vcstar_to_vrstar_to_pid(my_pe_rank, grid_height, grid_width);
    	int from_pe = vcstar_to_vrstar_from_pid(my_pe_rank, grid_height, grid_width);
    	int local_recv_size_vcstar_to_vrstar = local_height_vcstar(from_pe, m, n, grid_height, grid_width) * local_width_vcstar(from_pe, m, n, grid_height, grid_width);

        int dev_local_buffer_size = max(max_recv, mcmr_local_height*mcmr_local_width);
        dev_local_buffer_size = max(dev_local_buffer_size, max_send);
        float* dev_local_buffer = (float*) nvshmem_malloc(dev_local_buffer_size*sizeof(float));
        if(dev_local_buffer == NULL){
           throw std::runtime_error("error allocating SDO");
        }
        CHECK_CUDA(cudaMemcpy((void*) dev_local_buffer, (void const*) dev_Matrix_buffer, 
                mcmr_local_height*mcmr_local_width*sizeof(float), cudaMemcpyDeviceToDevice));

    	pack_mcmr_to_vc_star<<<1,num_threads,0, stream>>> (m, n, my_pe_rank, grid_height, grid_width, dev_send_displs, dev_local_buffer, dev_send_buf);

        int workspace_size = grid_size;
        int* common_workspace = (int*) nvshmem_malloc(workspace_size*sizeof(int));
        if(common_workspace == NULL){
           throw std::runtime_error("error allocating SDO");
        }
        CHECK_CUDA(cudaMemset(common_workspace, 0, workspace_size*sizeof(int)));

        int* dev_pes = (int*) nvshmem_malloc(xnpes*sizeof(int));
        if(dev_pes == NULL){
           throw std::runtime_error("error allocating SDO");
        }
        CHECK_CUDA(cudaMemcpy((void*) dev_pes, (void const*) pes.data(), xnpes*sizeof(int), cudaMemcpyHostToDevice));

    	int sync_counter=1;
    	Alltoallv_put(total_recv, my_pe_rank, dev_send_buf, dev_recv_buf, dev_send_displs, dev_target_offset_counts, dev_pes, xnpes, sync_counter, common_workspace, stream);

    	int* dev_recv_displs = (int*) nvshmem_malloc(grid_size+1);
        if(dev_recv_displs == NULL){
           throw std::runtime_error("error allocating SDO");
        }

    	CHECK_CUDA(cudaMemcpy((void*) dev_recv_displs, (void const*) recv_displs.data(), (grid_size+1)*sizeof(int), cudaMemcpyHostToDevice));

    	num_threads = min(max_recv, 1024);
    	unpack_mcmr_to_vc_star<<<1,num_threads,0,stream>>> (m, n, my_pe_rank, grid_height, grid_width, dev_recv_displs, dev_local_buffer, dev_recv_buf);

#if 0
//printf("(%d) after unpack_mcmr_to_vc_star \n", my_pe_rank);
char line[132];
FILE* fp_debug;
sprintf(line, "___vcstar.%04d", my_pe_rank);
fp_debug = fopen(line, "w");

std::vector<float> VC_STAR_mem_buffer(max_recv);
cudaMemcpy(VC_STAR_mem_buffer.data(), dev_local_buffer, max_recv*sizeof(float), cudaMemcpyDeviceToHost);
fprintf(fp_debug, "Buffer of B_VC_STAR...\n");
for(int l=0; l<total_recv; l++){
       fprintf(fp_debug, "%f ", VC_STAR_mem_buffer[l]);
}
fprintf(fp_debug, "\n");
fclose(fp_debug);
#endif

/*
sprintf(line, "memsizes.%04d", my_pe_rank);
fp_debug = fopen(line, "w");

fprintf(fp_debug, "dev_send_displs: %d\n", send_displs.size());
fprintf(fp_debug, "dev_send_buf:    %d\n", max_send);
fprintf(fp_debug, "dev_target_offset_counts: %d\n", offset_counts.size());
fprintf(fp_debug, "dev_recv_buf: %d\n", max_recv);
fprintf(fp_debug, "dev_local_buffer: %d\n", dev_local_buffer_size);
fprintf(fp_debug, "common_workspace: %d\n", workspace_size);
fprintf(fp_debug, "dev_pes: %d\n", xnpes);
fprintf(fp_debug, "dev_recv_displs: %d\n", grid_size+1);
fclose(fp_debug);
*/

	cudaMemcpy(dev_target_buffer, dev_local_buffer, local_recv_size_vcstar_to_vrstar*sizeof(float), cudaMemcpyDeviceToDevice);

	MPI_Barrier(mpi_comm);

        //sync_counter++;
 	//Global_sync (my_pe_rank, dev_pes, xnpes, sync_counter, common_workspace, 0);
	nvshmem_free(dev_recv_displs);
	nvshmem_free(dev_send_displs);
	nvshmem_free(dev_send_buf);
	nvshmem_free(dev_target_offset_counts);
	nvshmem_free(dev_recv_buf);
	nvshmem_free(dev_local_buffer);
	nvshmem_free(common_workspace);
	nvshmem_free(dev_pes);
}

template<typename T>
//void NVSHMEM_mcmr_to_mcstar_setup(MPI_Comm, int, int, int, int, int*, int*, int**, int** , T**, T**)
void NVSHMEM_mcmr_to_mcstar_setup(MPI_Comm, int, int, int, int, int*, int*, int**, int** , T**, T**)
{
  throw std::runtime_error("Function not implemented\n");
}
void NVSHMEM_mcmr_to_mcstar_setup(
	MPI_Comm mpi_comm,
	int m, 
	int n, 
	int grid_height, 
	int grid_width, 
        int* my_pe_rank,
        int* xnpes,
        int** common_workspace,
        int** dev_pes,
        float** dev_recv_buf,
        float** dev_send_buf)
{
    int grid_size = grid_height*grid_width;
    std::vector<int> pes;
    setup_pes(mpi_comm, my_pe_rank, pes, xnpes);

    // Compute the maximum size for send and recv operation from MCMR form. 
    // Need this because
    // these buffers are SDOs of nvshmem (OpenSHMEM constraint)
    int max_send = (int) (ceil(((double)m)/grid_height)*ceil(((double)n)/grid_width));

    int workspace_size = grid_size;

    MPI_Barrier(mpi_comm);
    *common_workspace = (int*) nvshmem_malloc(workspace_size *sizeof(int));
    if(common_workspace == NULL){
       throw std::runtime_error("error allocating SDO");
    }
    *dev_pes = (int*) nvshmem_malloc(*xnpes*sizeof(int));
    if(dev_pes == NULL){
       throw std::runtime_error("error allocating SDO");
    }
    CHECK_CUDA(cudaMemset(*common_workspace, 0, workspace_size*sizeof(int)));
    CHECK_CUDA(cudaMemcpy((void*) *dev_pes, (void const*) pes.data(), *xnpes*sizeof(int), cudaMemcpyHostToDevice));
    *dev_recv_buf = (float*) nvshmem_malloc(grid_width*max_send*sizeof(float));
    if(dev_recv_buf == NULL){
       throw std::runtime_error("error allocating SDO");
    }
    *dev_send_buf = (float*) nvshmem_malloc(max_send*sizeof(float));
    if(dev_send_buf == NULL){
       throw std::runtime_error("error allocating SDO");
    }
    MPI_Barrier(mpi_comm);
}

template<typename T>
void NVSHMEM_mcmr_to_mcstar_cleanup(MPI_Comm, int*, int* , T*, T*)
{
  throw std::runtime_error("Function not implemented\n");
}
void NVSHMEM_mcmr_to_mcstar_cleanup(MPI_Comm mpi_comm,
        int* common_workspace,
        int* dev_pes,
        float* dev_recv_buf,
        float* dev_send_buf)
{
    MPI_Barrier(mpi_comm);
    nvshmem_free(common_workspace);
    nvshmem_free(dev_pes);
    nvshmem_free(dev_recv_buf);
    nvshmem_free(dev_send_buf);
    MPI_Barrier(mpi_comm);
}

template<typename T>
//void NVSHMEM_mcmr_to_mrstar_setup(MPI_Comm, int, int, int, int, int*, int*, int**, int** , T**, T**)
void NVSHMEM_mcmr_to_mrstar_setup(MPI_Comm, int, int, int, int, int*, int*, long**, int**, long**, T**, T**)
{
  throw std::runtime_error("Function not implemented\n");
}
void NVSHMEM_mcmr_to_mrstar_setup(
	MPI_Comm mpi_comm,
	int m, 
	int n, 
	int grid_height, 
	int grid_width, 
        int* my_pe_rank,
        int* xnpes,
        long** common_workspace,
        int** dev_pes,
        long** dev_sync_counter,
        float** dev_recv_buf,
        float** dev_send_buf)
{
    int grid_size = grid_height*grid_width;
    std::vector<int> pes;
    setup_pes(mpi_comm, my_pe_rank, pes, xnpes);

    // Compute the maximum size for send and recv operation from MCMR form. 
    // Need this because
    // these buffers are SDOs of nvshmem (OpenSHMEM constraint)
    int max_send = (int) (ceil(((double)m)/grid_height)*ceil(((double)n)/grid_width));

    int workspace_size = grid_size;

    MPI_Barrier(mpi_comm);
    *common_workspace = (long*) nvshmem_malloc(workspace_size *sizeof(long));
    //*common_workspace = (int*) nvshmem_malloc(workspace_size *sizeof(int));
    if(*common_workspace == NULL){
       throw std::runtime_error("error allocating SDO");
    }
    *dev_pes = (int*) nvshmem_malloc(*xnpes*sizeof(int));
    if(*dev_pes == NULL){
       throw std::runtime_error("error allocating SDO");
    }
    *dev_sync_counter = (long*) nvshmem_malloc(sizeof(long));
    if(*dev_sync_counter == NULL){
       throw std::runtime_error("error allocating SDO");
    }
    //CHECK_CUDA(cudaMemset(*common_workspace, 0, workspace_size*sizeof(int)));
    CHECK_CUDA(cudaMemset(*common_workspace, 0, workspace_size*sizeof(long)));
    CHECK_CUDA(cudaMemset(*dev_sync_counter, 0, sizeof(int)));
    CHECK_CUDA(cudaMemcpy((void*) *dev_pes, (void const*) pes.data(), *xnpes*sizeof(int), cudaMemcpyHostToDevice));
    *dev_recv_buf = (float*) nvshmem_malloc(grid_height*max_send*sizeof(float));
    if(*dev_recv_buf == NULL){
       throw std::runtime_error("error allocating SDO");
    }
    *dev_send_buf = (float*) nvshmem_malloc(max_send*sizeof(float));
    if(*dev_send_buf == NULL){
       throw std::runtime_error("error allocating SDO");
    }
    MPI_Barrier(mpi_comm);
}

template<typename T>
void NVSHMEM_mcmr_to_mrstar_cleanup(MPI_Comm, long*, int*, long*, T*, T*)
{
  throw std::runtime_error("Function not implemented\n");
}
void NVSHMEM_mcmr_to_mrstar_cleanup(MPI_Comm mpi_comm,
        long* common_workspace,
        int* dev_pes,
        long* dev_sync_counter,
        float* dev_recv_buf,
        float* dev_send_buf)
{
    MPI_Barrier(mpi_comm);
    nvshmem_free(common_workspace);
    nvshmem_free(dev_pes);
    nvshmem_free(dev_sync_counter);
    nvshmem_free(dev_recv_buf);
    nvshmem_free(dev_send_buf);
    MPI_Barrier(mpi_comm);
}

//void mcmr_to_mrstar(MPI_Comm, float*, cudaEvent_t, cudaEvent_t, int, int, int, int, int, int, int*, int*, int, T*, T*, T*, T*, cudaStream_t stream)
template<typename T>
void mcmr_to_mrstar(FILE*, MPI_Comm, float*, cudaEvent_t, cudaEvent_t, long, int, int, int, int, int, long*, int*, long*, int, T*, T*, T*, T*, cudaStream_t stream)
{
  throw std::runtime_error("Function not implemented\n");
}

#include <sys/time.h>

//void mcmr_to_mrstar(MPI_Comm mpi_comm,
void mcmr_to_mrstar(FILE *fp_debug,
	MPI_Comm mpi_comm,
	float *run_timer,
	cudaEvent_t kernel_start,
	cudaEvent_t kernel_stop,
	long sync_counter,
	//int sync_counter,
	int m, 
	int n, 
	int grid_height, 
	int grid_width, 
	int my_pe_rank,
	long* common_workspace,
	//int* common_workspace,
	int* dev_pes,
	long* dev_sync_counter,
	//int* dev_sync_counter,
	int  xnpes,
	float* dev_send_buf,
	float* dev_recv_buf,
	float* dev_local_buffer, 
        float* dev_target_buffer,
	cudaStream_t stream)
{

    int grid_size = grid_height*grid_width;

    int my_displs;
    int total_gather;
    int num_threads;
    counts_mcmr_to_mrstar(grid_height, grid_width, my_pe_rank, m, n, &total_gather, &my_displs);

    int mcmr_local_height  = local_height_mcmr(my_pe_rank, m, n, grid_height, grid_width);
    int mcmr_local_width   = local_width_mcmr(my_pe_rank, m, n, grid_height, grid_width);
    int local_buffer_size = mcmr_local_height*mcmr_local_width;

// IMPORTANT: Don't change following routines as they are known to output correct results
// From here ==============================================
    MPI_Barrier(mpi_comm);
    num_threads = min(local_buffer_size, 1024);
    pack_mcmr_to_mrstar<<<1,num_threads,0,stream>>> (m, n, my_pe_rank, grid_height, grid_width, dev_local_buffer, dev_send_buf);
    MPI_Barrier(mpi_comm);

    std::vector<int> pes(grid_size);
    CHECK_CUDA(cudaMemcpy((void*) pes.data(), (void const*) dev_pes, grid_size*sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy((void*) (dev_recv_buf+my_displs), (void const*) dev_send_buf, 
local_buffer_size*sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy((void*) dev_sync_counter, (void const*) &sync_counter, sizeof(long), cudaMemcpyHostToDevice));
    //CHECK_CUDA(cudaMemcpy((void*) dev_sync_counter, (void const*) &sync_counter, sizeof(int), cudaMemcpyHostToDevice));
    MPI_Barrier(mpi_comm);

    float cudaTime;
    cudaEventRecord(kernel_start, stream  );
// Upto here ==============================================

#if 0
// Put only experimental routines here.
    num_threads = min(local_buffer_size, 1024);
//    empty_kernel<<<1,num_threads,0,stream>>> ();
    cudaEventRecord(kernel_start, stream  );
    allgather_unpack_mcmr_to_mrstar<<<1,num_threads,0,stream>>> (
        my_pe_rank, m, n, grid_height, grid_width,
        dev_send_buf, dev_recv_buf,
        dev_target_buffer, total_gather,
        local_buffer_size, my_displs, dev_pes, xnpes, dev_sync_counter, common_workspace);
    cudaEventRecord(kernel_stop, stream  );
    cudaEventSynchronize(kernel_stop);                     \
    cudaEventElapsedTime(&cudaTime, kernel_start, kernel_stop);   \
printf("after allgather_unpack_mcmr_to_mrstar: %f\n", cudaTime);
    *run_timer += cudaTime;

#else
// IMPORTANT: Following routines are known to produce correct reults
// Don't change anything 
#if 1
    num_threads = min(total_gather, 1024);
    if(num_threads < grid_height){
  	throw std::runtime_error("Number of threads too small for given grid\n");
    }

    MPI_Barrier(mpi_comm);
    cudaEventRecord(kernel_start, stream  );
    ver0_1_gather_unpack_mcmr_to_mrstar<<<1,num_threads,0,stream>>> (
	my_pe_rank, m, n, grid_height, grid_width, 
    	dev_send_buf, dev_recv_buf, 
	dev_target_buffer, total_gather,
	local_buffer_size, my_displs, dev_pes, xnpes, dev_sync_counter, common_workspace);
    cudaEventRecord(kernel_stop, stream  );
    cudaEventSynchronize(kernel_stop);                     \
    cudaEventElapsedTime(&cudaTime, kernel_start, kernel_stop);   \
    *run_timer += cudaTime;
    MPI_Barrier(mpi_comm);

fprintf(fp_debug, "after gather_unpack_mcmr_to_mrstar: %f\n", cudaTime);
#else
#if 1
    int my_col_rank = my_pe_rank/grid_height;
    int my_row_rank = my_pe_rank%grid_height;

    for (int i = 1; i <=grid_height; ++i)
    {
        auto const offset = (my_row_rank + i) % grid_height;
        auto const target_pe = offset + my_col_rank*grid_height;
        auto const peer = pes[target_pe];

        nvshmemx_float_put_on_stream (dev_recv_buf+my_displs, dev_send_buf, local_buffer_size, peer, stream);
    }
    nvshmemx_quiet_on_stream(stream);
#else
    Allgatherv_put_col(my_pe_rank, m, n, grid_height, grid_width, dev_send_buf, dev_recv_buf, local_buffer_size, my_displs, dev_pes, xnpes, sync_counter, common_workspace, stream); 
#endif
    cudaEventRecord(kernel_stop, stream  );
    cudaEventSynchronize(kernel_stop);                     \
    cudaEventElapsedTime(&cudaTime, kernel_start, kernel_stop);   \
    *run_timer += cudaTime;
    MPI_Barrier(mpi_comm);
fprintf(fp_debug, ">> after gather_nvshmemx_float_put_on_stream: %f\n", cudaTime);

    num_threads = min(total_gather, 1024);
    if(num_threads < grid_height){
  	throw std::runtime_error("Number of threads too small for given grid\n");
    }
    MPI_Barrier(mpi_comm);
    cudaEventRecord(kernel_start, stream  );
    unpack_mcmr_to_mrstar<<<1,num_threads,0,stream>>> (m, n, my_pe_rank, grid_height, grid_width, dev_target_buffer, dev_recv_buf, total_gather);
    cudaEventRecord(kernel_stop, stream  );
    cudaEventSynchronize(kernel_stop);                     \
    cudaEventElapsedTime(&cudaTime, kernel_start, kernel_stop);   \
    *run_timer += cudaTime;
    MPI_Barrier(mpi_comm);
fprintf(fp_debug, ">> after unpack_mcmr_to_mrstar: %f\n", cudaTime);
#endif
#endif
}

template<typename T>
void mcmr_to_mcstar(MPI_Comm, float*, cudaEvent_t, cudaEvent_t, int, int, int, int, int, int, int*, int*, int, T*, T*, T*, T*, cudaStream_t stream)
{
  throw std::runtime_error("Function not implemented\n");
}

#include <sys/time.h>

//void mcmr_to_mcstar(MPI_Comm mpi_comm,
void mcmr_to_mcstar(
	MPI_Comm mpi_comm,
	float *run_timer,
	cudaEvent_t kernel_start,
	cudaEvent_t kernel_stop,
	int sync_counter,
	int m, 
	int n, 
	int grid_height, 
	int grid_width, 
	int my_pe_rank,
	int* common_workspace,
	int* dev_pes,
	int  xnpes,
	float* dev_send_buf,
	float* dev_recv_buf,
	float* dev_local_buffer, 
        float* dev_target_buffer,
	cudaStream_t stream)
{

    int grid_size = grid_height*grid_width;

    int my_displs;
    int total_gather;
    counts_mcmr_to_mcstar(grid_height, grid_width, my_pe_rank, m, n, &total_gather, &my_displs);

    int mcmr_local_height  = local_height_mcmr(my_pe_rank, m, n, grid_height, grid_width);
    int mcmr_local_width   = local_width_mcmr(my_pe_rank, m, n, grid_height, grid_width);
    int local_buffer_size = mcmr_local_height*mcmr_local_width;

    // Compute the maximum size for send and recv operation from MCMR form. 
    // Need this because
    // these buffers are SDOs of nvshmem (OpenSHMEM constraint)


    MPI_Barrier(mpi_comm);
    CHECK_CUDA(cudaMemcpy((void*) dev_send_buf, (void const*) dev_local_buffer, local_buffer_size*sizeof(float), cudaMemcpyDeviceToDevice));
    MPI_Barrier(mpi_comm);

    std::vector<int> pes(grid_size);
    CHECK_CUDA(cudaMemcpy((void*) pes.data(), (void const*) dev_pes, grid_size*sizeof(int), cudaMemcpyDeviceToHost));

    float cudaTime;
    cudaEventRecord(kernel_start, stream  );
#if 1
    int my_col_rank = my_pe_rank/grid_height;
    int my_row_rank = my_pe_rank%grid_height;

    for (int i = 1; i <=grid_width; ++i)
    {
        auto const offset = (my_col_rank + i) % grid_width;
        auto const target_pe = offset*grid_height + my_row_rank;
        auto const peer = pes[target_pe];

        nvshmemx_float_put_on_stream (dev_recv_buf+my_displs, dev_send_buf, local_buffer_size, peer, stream);
    }
    nvshmemx_quiet_on_stream(stream);

#else
    Allgatherv_put_row(my_pe_rank, grid_height, grid_width, dev_send_buf, dev_recv_buf, local_buffer_size, my_displs, dev_pes, xnpes, sync_counter, common_workspace, stream); 
#endif
    cudaEventRecord(kernel_stop, stream  );
    cudaEventSynchronize(kernel_stop);                     \
    cudaEventElapsedTime(&cudaTime, kernel_start, kernel_stop);   \
    *run_timer += cudaTime;

    int num_threads = min(total_gather, 1024);
    cudaEventRecord(kernel_start, stream  );
    unpack_mcmr_to_mcstar<<<1,num_threads,0,stream>>> (m, n, my_pe_rank, grid_height, grid_width, dev_target_buffer, dev_recv_buf, total_gather);
    cudaEventRecord(kernel_stop, stream  );
    cudaEventSynchronize(kernel_stop);                     \
    cudaEventElapsedTime(&cudaTime, kernel_start, kernel_stop);   \
    *run_timer += cudaTime;
}


template<typename T>
void mcmr_to_vrstar(MPI_Comm, int, int, int, int, T*, T*, cudaStream_t)
{
  throw std::runtime_error("Function not implemented\n");
}

void mcmr_to_vrstar(MPI_Comm mpi_comm,
	int m,
	int n,
	int grid_height,
	int grid_width,
	float* dev_B_buffer,
	float* dev_target_buffer,
	cudaStream_t stream)
{
  int grid_size = grid_height*grid_width;
  int my_pe_rank;
  std::vector<int> pes;
  int xnpes;
  setup_pes(mpi_comm, &my_pe_rank, pes, &xnpes);

  int mcmr_local_height  = local_height_mcmr(my_pe_rank, m, n, grid_height, grid_width);
  int mcmr_local_width   = local_width_mcmr(my_pe_rank, m, n, grid_height, grid_width);

  std::vector<int> send_counts(grid_size, 0);
  std::vector<int> send_displs(grid_size+1, 0);
  int total_send;
  int max_send;
  std::vector<int> recv_counts(grid_size, 0);
  std::vector<int> recv_displs(grid_size+1, 0);
  int total_recv;
  int max_recv;

  // Compute the maximum size for send and recv operation. Need this because
  // these buffers are SDOs of nvshmem (OpenSHMEM constraint)
  max_send = (int) (ceil(((double)m)/grid_height)*ceil(((double)n)/grid_width));
  max_recv = (int) (ceil(((double)m)/grid_size)*n);

  std::vector<int> offset_counts (grid_size*2, 0);

  counts_mcmr_to_vc_star(grid_height, grid_width, my_pe_rank, m, n,
                send_counts, send_displs, &total_send,
                recv_counts, recv_displs, &total_recv,
                offset_counts);
  int num_threads = min(total_send, 1024);

  int* dev_send_displs = (int*) nvshmem_malloc(send_displs.size()*sizeof(int));
  if(dev_send_displs == NULL){
     throw std::runtime_error("error allocating SDO");
  }
  float* dev_send_buf = (float*) nvshmem_malloc(max_send*sizeof(float));
  if(dev_send_buf == NULL){
     throw std::runtime_error("error allocating SDO");
  }
  int* dev_target_offset_counts = (int*) nvshmem_malloc(offset_counts.size()*sizeof(int));
  if(dev_target_offset_counts == NULL){
     throw std::runtime_error("error allocating SDO");
  }
  float* dev_recv_buf = (float*) nvshmem_malloc(max_recv*sizeof(float));
  if(dev_recv_buf == NULL){
     throw std::runtime_error("error allocating SDO");
  }
  CHECK_CUDA(cudaMemcpy((void*) dev_target_offset_counts, (void const*) offset_counts.data(), offset_counts.size()*sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy((void*) dev_send_displs, (void const*) send_displs.data(), send_displs.size()*sizeof(int), cudaMemcpyHostToDevice));

  int to_pe = vcstar_to_vrstar_to_pid(my_pe_rank, grid_height, grid_width);
  int from_pe = vcstar_to_vrstar_from_pid(my_pe_rank, grid_height, grid_width);
  int local_recv_size_vcstar_to_vrstar = local_height_vcstar(from_pe, m, n, grid_height, grid_width) * local_width_vcstar(from_pe, m, n, grid_height, grid_width);

  int dev_local_buffer_size = max(max_recv, mcmr_local_height*mcmr_local_width);
  dev_local_buffer_size = max(dev_local_buffer_size, max_send);
  dev_local_buffer_size = max(dev_local_buffer_size, local_height_vrstar(my_pe_rank, m, n, grid_height, grid_width)*local_width_vrstar(my_pe_rank, m, n, grid_height, grid_width));
  dev_local_buffer_size = max(dev_local_buffer_size, local_recv_size_vcstar_to_vrstar);
  float* dev_local_buffer = (float*) nvshmem_malloc(dev_local_buffer_size*sizeof(float));
  if(dev_local_buffer == NULL){
     throw std::runtime_error("error allocating SDO");
  }

  CHECK_CUDA(cudaMemcpy((void*) dev_local_buffer, (void const*) dev_B_buffer, 
  mcmr_local_height*mcmr_local_width*sizeof(float), cudaMemcpyDeviceToDevice));

  pack_mcmr_to_vc_star<<<1,num_threads,0, stream>>> (m, n, my_pe_rank, grid_height, grid_width, dev_send_displs, dev_local_buffer, dev_send_buf);

  int workspace_size = grid_size;
  MPI_Barrier(mpi_comm);
  int* common_workspace = (int*) nvshmem_malloc(workspace_size *sizeof(int));
  if(common_workspace == NULL){
     throw std::runtime_error("error allocating SDO");
  }
  int* dev_pes = (int*) nvshmem_malloc(xnpes*sizeof(int));
  if(dev_pes == NULL){
     throw std::runtime_error("error allocating SDO");
  }
  CHECK_CUDA(cudaMemset(common_workspace, 0, workspace_size*sizeof(int)));
  CHECK_CUDA(cudaMemcpy((void*) dev_pes, (void const*) pes.data(), xnpes*sizeof(int), cudaMemcpyHostToDevice));
  MPI_Barrier(mpi_comm);

  int sync_counter=1;
  Alltoallv_put(total_recv, my_pe_rank, dev_send_buf, dev_recv_buf, dev_send_displs, dev_target_offset_counts, dev_pes, xnpes, sync_counter, common_workspace, stream);

  int* dev_recv_displs = (int*) nvshmem_malloc(grid_size+1);
  if(dev_recv_displs == NULL){
     throw std::runtime_error("error allocating SDO");
  }

  CHECK_CUDA(cudaMemcpy((void*) dev_recv_displs, (void const*) recv_displs.data(), (grid_size+1)*sizeof(int), cudaMemcpyHostToDevice));

  num_threads = min(max_recv, 1024);
  unpack_mcmr_to_vc_star<<<1,num_threads,0,stream>>> (m, n, my_pe_rank, grid_height, grid_width, dev_recv_displs, dev_local_buffer, dev_recv_buf);

/*
//printf("(%d) after unpack_mcmr_to_vc_star \n", my_pe_rank);
char line[132];
FILE* fp_debug;
sprintf(line, "___vcstar.%04d", my_pe_rank);
fp_debug = fopen(line, "w");

std::vector<float> B1_VC_STAR_mem_buffer(max_recv);
cudaMemcpy(B1_VC_STAR_mem_buffer.data(), dev_local_buffer, max_recv*sizeof(float), cudaMemcpyDeviceToHost);
fprintf(fp_debug, "Buffer of B_VC_STAR...\n");
for(int l=0; l<total_recv; l++){
       fprintf(fp_debug, "%f ", B1_VC_STAR_mem_buffer[l]);
}
fprintf(fp_debug, "\n");
fclose(fp_debug);

sprintf(line, "to_from_pe.%04d", my_pe_rank);
fp_debug = fopen(line, "w");
fprintf(fp_debug, "to_pe=%d, from_pe=%d\n", to_pe, from_pe);
fclose(fp_debug);
*/

  // Calculate the PEs to send to and receive from me
  int local_size =  local_height_vcstar(my_pe_rank, m, n, grid_height, grid_width) * local_width_vcstar(my_pe_rank, m, n, grid_height, grid_width);

  int my_send_size_vcstar_to_vrstar = local_size;
  int max_recv_size_vcstar_to_vrstar = (int) ceil(((double)m)/grid_height) * local_width_vcstar(from_pe, m, n, grid_height, grid_width);
  float* dev_vrstar_recv_buffer = (float*) nvshmem_malloc(max_recv_size_vcstar_to_vrstar*sizeof(float));
  if(dev_vrstar_recv_buffer == NULL){
     throw std::runtime_error("error allocating SDO");
  }

  sync_counter++;
  Sendrecv_put(dev_local_buffer, local_size, to_pe,
              dev_vrstar_recv_buffer, from_pe,
              my_pe_rank, dev_pes, xnpes, sync_counter, common_workspace, stream);

//printf("(%d) after Sendrecv_put \n", my_pe_rank);
  // At this point, dev_recv_buffer has exchanged data, which is needed to copied back to the local_buffer
  // No copying needed if to_pe == from_pe
  if(!(my_pe_rank == to_pe && my_pe_rank == from_pe))
  //if(to_pe != from_pe)
  {// send/recv only when it is needed
        if(local_size != local_recv_size_vcstar_to_vrstar)
        {// nvshmem_realloc is need to adjust local buffer for new data
         //nvshmem_realloc(local_buffer, new_recv_size_vcstar_to_vrstar*sizeof(T));

	 //   dev_local_buffer = (float*) nvshmem_malloc(local_recv_size_vcstar_to_vrstar*sizeof(float));
        }
        CHECK_CUDA(cudaMemcpy((void*) dev_local_buffer, (void const*) dev_vrstar_recv_buffer,
                        local_recv_size_vcstar_to_vrstar*sizeof(float), cudaMemcpyDeviceToDevice));
  }


/*
sprintf(line, "vrstar.%04d", my_pe_rank);
fp_debug = fopen(line, "w");
std::vector<float> B1_VR_STAR_mem_buffer(local_recv_size_vcstar_to_vrstar);
cudaMemcpy(B1_VR_STAR_mem_buffer.data(), dev_local_buffer, local_recv_size_vcstar_to_vrstar*sizeof(float), cudaMemcpyDeviceToHost);
fprintf(fp_debug, "Buffer of B_VR_STAR...\n");
for(int l=0; l<local_recv_size_vcstar_to_vrstar; l++){
       fprintf(fp_debug, "%f ", B1_VR_STAR_mem_buffer[l]);
}
fprintf(fp_debug, "\n");
fclose(fp_debug);
*/

/*
sprintf(line, "memsizes.%04d", my_pe_rank);
fp_debug = fopen(line, "w");

fprintf(fp_debug, "dev_send_displs: %d\n", send_displs.size());
fprintf(fp_debug, "dev_send_buf:    %d\n", max_send);
fprintf(fp_debug, "dev_target_offset_counts: %d\n", offset_counts.size());
fprintf(fp_debug, "dev_recv_buf: %d\n", max_recv);
fprintf(fp_debug, "dev_local_buffer: %d\n", dev_local_buffer_size);
fprintf(fp_debug, "common_workspace: %d\n", workspace_size);
fprintf(fp_debug, "dev_pes: %d\n", xnpes);
fprintf(fp_debug, "dev_recv_displs: %d\n", grid_size+1);
fprintf(fp_debug, "dev_vrstar_recv_buffer: %d\n", max_recv_size_vcstar_to_vrstar);
fclose(fp_debug);
*/

  cudaMemcpy(dev_target_buffer, dev_local_buffer, local_recv_size_vcstar_to_vrstar*sizeof(float), cudaMemcpyDeviceToDevice);

  MPI_Barrier(mpi_comm);

  nvshmem_free(dev_vrstar_recv_buffer);
  nvshmem_free(dev_recv_displs);
  nvshmem_free(dev_send_displs);
  nvshmem_free(dev_send_buf);
  nvshmem_free(dev_target_offset_counts);
  nvshmem_free(dev_recv_buf);
  nvshmem_free(dev_local_buffer);
  nvshmem_free(common_workspace);
  nvshmem_free(dev_pes);

}

template void mcmr_to_vrstar<int>(MPI_Comm, int, int, int, int, int*, int*, cudaStream_t);
template void mcmr_to_vrstar<double>(MPI_Comm, int, int, int, int, double*, double*, cudaStream_t);
template void mcmr_to_vrstar<__half>(MPI_Comm, int, int, int, int, __half*, __half*, cudaStream_t);
template void mcmr_to_vrstar<El::Complex<float>>(MPI_Comm, int, int, int, int, El::Complex<float>*, El::Complex<float>*, cudaStream_t);
template void mcmr_to_vrstar<El::Complex<double>>(MPI_Comm, int, int, int, int, El::Complex<double>*, El::Complex<double>*, cudaStream_t);

/*
template void mcmr_to_mcstar<int>(MPI_Comm, int, int, int, int, int*, int*, cudaStream_t);
template void mcmr_to_mcstar<double>(MPI_Comm, int, int, int, int, double*, double*, cudaStream_t);
template void mcmr_to_mcstar<__half>(MPI_Comm, int, int, int, int, __half*, __half*, cudaStream_t);
template void mcmr_to_mcstar<El::Complex<float>>(MPI_Comm, int, int, int, int, El::Complex<float>*, El::Complex<float>*, cudaStream_t);
template void mcmr_to_mcstar<El::Complex<double>>(MPI_Comm, int, int, int, int, El::Complex<double>*, El::Complex<double>*, cudaStream_t);
*/
template void mcmr_to_mcstar<int>(MPI_Comm, float*, cudaEvent_t, cudaEvent_t, int, int, int, int, int, int, int*, int*, int, int*, int*, int*, int*, cudaStream_t stream);
template void mcmr_to_mcstar<double>(MPI_Comm, float*, cudaEvent_t, cudaEvent_t,  int, int, int, int, int, int, int*, int*, int, double*, double*, double*, double*, cudaStream_t stream);
template void mcmr_to_mcstar<__half>(MPI_Comm, float*, cudaEvent_t, cudaEvent_t,  int, int, int, int, int, int, int*, int*, int, __half*, __half*, __half*, __half*, cudaStream_t stream);
template void mcmr_to_mcstar<El::Complex<float>>(MPI_Comm, float*, cudaEvent_t, cudaEvent_t,  int, int, int, int, int, int, int*, int*, int, El::Complex<float>*, El::Complex<float>*, El::Complex<float>*, El::Complex<float>*, cudaStream_t stream);
template void mcmr_to_mcstar<El::Complex<double>>(MPI_Comm, float*, cudaEvent_t, cudaEvent_t,  int, int, int, int, int, int, int*, int*, int, El::Complex<double>*, El::Complex<double>*, El::Complex<double>*, El::Complex<double>*, cudaStream_t stream);

template void mcmr_to_vcstar<int>(MPI_Comm, int, int, int, int, int*, int*, cudaStream_t);
template void mcmr_to_vcstar<double>(MPI_Comm, int, int, int, int, double*, double*, cudaStream_t);
template void mcmr_to_vcstar<__half>(MPI_Comm, int, int, int, int, __half*, __half*, cudaStream_t);
template void mcmr_to_vcstar<El::Complex<float>>(MPI_Comm, int, int, int, int, El::Complex<float>*, El::Complex<float>*, cudaStream_t);
template void mcmr_to_vcstar<El::Complex<double>>(MPI_Comm, int, int, int, int, El::Complex<double>*, El::Complex<double>*, cudaStream_t);

/*
template void NVSHMEM_mcmr_to_mcstar_setup<int>(MPI_Comm, int, int, int, int, int*, int*, int**, int** , int**, int**);
template void NVSHMEM_mcmr_to_mcstar_setup<double>(MPI_Comm, int, int, int, int, int*, int*, int**, int** , double**, double**);
template void NVSHMEM_mcmr_to_mcstar_setup<__half>(MPI_Comm, int, int, int, int, int*, int*, int**, int** , __half**, __half**);
template void NVSHMEM_mcmr_to_mcstar_setup<El::Complex<float>>(MPI_Comm, int, int, int, int, int*, int*, int**, int** , El::Complex<float>**, El::Complex<float>**);
template void NVSHMEM_mcmr_to_mcstar_setup<El::Complex<double>>(MPI_Comm, int, int, int, int, int*, int*, int**, int** , El::Complex<double>**, El::Complex<double>**);
*/
template void NVSHMEM_mcmr_to_mcstar_setup<int>(MPI_Comm, int, int, int, int, int*, int*, int**, int** , int**, int**);
template void NVSHMEM_mcmr_to_mcstar_setup<double>(MPI_Comm, int, int, int, int, int*, int*, int**, int** , double**, double**);
template void NVSHMEM_mcmr_to_mcstar_setup<__half>(MPI_Comm, int, int, int, int, int*, int*, int**, int** , __half**, __half**);
template void NVSHMEM_mcmr_to_mcstar_setup<El::Complex<float>>(MPI_Comm, int, int, int, int, int*, int*, int**, int** , El::Complex<float>**, El::Complex<float>**);
template void NVSHMEM_mcmr_to_mcstar_setup<El::Complex<double>>(MPI_Comm, int, int, int, int, int*, int*, int**, int** , El::Complex<double>**, El::Complex<double>**);

template void NVSHMEM_mcmr_to_mcstar_cleanup<int>(MPI_Comm, int*, int* , int*, int*);
template void NVSHMEM_mcmr_to_mcstar_cleanup<double>(MPI_Comm, int*, int* , double*, double*);
template void NVSHMEM_mcmr_to_mcstar_cleanup<__half>(MPI_Comm, int*, int* , __half*, __half*);
template void NVSHMEM_mcmr_to_mcstar_cleanup<El::Complex<float>>(MPI_Comm, int*, int* , El::Complex<float>*, El::Complex<float>*);
template void NVSHMEM_mcmr_to_mcstar_cleanup<El::Complex<double>>(MPI_Comm, int*, int* , El::Complex<double>*, El::Complex<double>*);


template void mcmr_to_mrstar<int>(FILE *fp_debug, MPI_Comm, float*, cudaEvent_t, cudaEvent_t, long, int, int, int, int, int, long*, int*, long*, int, int*, int*, int*, int*, cudaStream_t stream);
template void mcmr_to_mrstar<double>(FILE *fp_debug, MPI_Comm, float*, cudaEvent_t, cudaEvent_t, long, int, int, int, int, int, long*, int*, long*, int, double*, double*, double*, double*, cudaStream_t stream);
template void mcmr_to_mrstar<__half>(FILE *fp_debug, MPI_Comm, float*, cudaEvent_t, cudaEvent_t, long, int, int, int, int, int, long*, int*, long*, int, __half*, __half*, __half*, __half*, cudaStream_t stream);
template void mcmr_to_mrstar<El::Complex<float>>(FILE *fp_debug, MPI_Comm, float*, cudaEvent_t, cudaEvent_t, long, int, int, int, int, int, long*, int*, long*, int, El::Complex<float>*, El::Complex<float>*, El::Complex<float>*, El::Complex<float>*, cudaStream_t stream);
template void mcmr_to_mrstar<El::Complex<double>>(FILE *fp_debug, MPI_Comm, float*, cudaEvent_t, cudaEvent_t, long, int, int, int, int, int, long*, int*, long*, int, El::Complex<double>*, El::Complex<double>*, El::Complex<double>*, El::Complex<double>*, cudaStream_t stream);

/*
template void mcmr_to_mrstar<int>(FILE *fp_debug, MPI_Comm, float*, cudaEvent_t, cudaEvent_t, int, int, int, int, int, int, int*, int*, int*, int, int*, int*, int*, int*, cudaStream_t stream);
template void mcmr_to_mrstar<double>(FILE *fp_debug, MPI_Comm, float*, cudaEvent_t, cudaEvent_t, int, int, int, int, int, int, int*, int*, int*, int, double*, double*, double*, double*, cudaStream_t stream);
template void mcmr_to_mrstar<__half>(FILE *fp_debug, MPI_Comm, float*, cudaEvent_t, cudaEvent_t, int, int, int, int, int, int, int*, int*, int*, int, __half*, __half*, __half*, __half*, cudaStream_t stream);
template void mcmr_to_mrstar<El::Complex<float>>(FILE *fp_debug, MPI_Comm, float*, cudaEvent_t, cudaEvent_t, int, int, int, int, int, int, int*, int*, int*, int, El::Complex<float>*, El::Complex<float>*, El::Complex<float>*, El::Complex<float>*, cudaStream_t stream);
template void mcmr_to_mrstar<El::Complex<double>>(FILE *fp_debug, MPI_Comm, float*, cudaEvent_t, cudaEvent_t, int, int, int, int, int, int, int*, int*, int*, int, El::Complex<double>*, El::Complex<double>*, El::Complex<double>*, El::Complex<double>*, cudaStream_t stream);
*/
/*
template void mcmr_to_mrstar<int>(MPI_Comm, float*, cudaEvent_t, cudaEvent_t, int, int, int, int, int, int, int*, int*, int, int*, int*, int*, int*, cudaStream_t stream);
template void mcmr_to_mrstar<double>(MPI_Comm, float*, cudaEvent_t, cudaEvent_t, int, int, int, int, int, int, int*, int*, int, double*, double*, double*, double*, cudaStream_t stream);
template void mcmr_to_mrstar<__half>(MPI_Comm, float*, cudaEvent_t, cudaEvent_t, int, int, int, int, int, int, int*, int*, int, __half*, __half*, __half*, __half*, cudaStream_t stream);
template void mcmr_to_mrstar<El::Complex<float>>(MPI_Comm, float*, cudaEvent_t, cudaEvent_t, int, int, int, int, int, int, int*, int*, int, El::Complex<float>*, El::Complex<float>*, El::Complex<float>*, El::Complex<float>*, cudaStream_t stream);
template void mcmr_to_mrstar<El::Complex<double>>(MPI_Comm, float*, cudaEvent_t, cudaEvent_t, int, int, int, int, int, int, int*, int*, int, El::Complex<double>*, El::Complex<double>*, El::Complex<double>*, El::Complex<double>*, cudaStream_t stream);
*/


/*
template void NVSHMEM_mcmr_to_mrstar_setup<int>(MPI_Comm, int, int, int, int, int*, int*, int**, int**, int**, int**, int**);
template void NVSHMEM_mcmr_to_mrstar_setup<double>(MPI_Comm, int, int, int, int, int*, int*, int**, int**, int**, double**, double**);
template void NVSHMEM_mcmr_to_mrstar_setup<__half>(MPI_Comm, int, int, int, int, int*, int*, int**, int**, int**, __half**, __half**);
template void NVSHMEM_mcmr_to_mrstar_setup<El::Complex<float>>(MPI_Comm, int, int, int, int, int*, int*, int**, int**, int**, El::Complex<float>**, El::Complex<float>**);
template void NVSHMEM_mcmr_to_mrstar_setup<El::Complex<double>>(MPI_Comm, int, int, int, int, int*, int*, int**, int**, int**,  El::Complex<double>**, El::Complex<double>**);
*/
template void NVSHMEM_mcmr_to_mrstar_setup<int>(MPI_Comm, int, int, int, int, int*, int*, long**, int**, long**, int**, int**);
template void NVSHMEM_mcmr_to_mrstar_setup<double>(MPI_Comm, int, int, int, int, int*, int*, long**, int**, long**, double**, double**);
template void NVSHMEM_mcmr_to_mrstar_setup<__half>(MPI_Comm, int, int, int, int, int*, int*, long**, int**, long**, __half**, __half**);
template void NVSHMEM_mcmr_to_mrstar_setup<El::Complex<float>>(MPI_Comm, int, int, int, int, int*, int*, long**, int**, long**, El::Complex<float>**, El::Complex<float>**);
template void NVSHMEM_mcmr_to_mrstar_setup<El::Complex<double>>(MPI_Comm, int, int, int, int, int*, int*, long**, int**, long**,  El::Complex<double>**, El::Complex<double>**);

/*
template void NVSHMEM_mcmr_to_mrstar_cleanup<int>(MPI_Comm, int*, int*, int*,  int*, int*);
template void NVSHMEM_mcmr_to_mrstar_cleanup<double>(MPI_Comm, int*, int*, int*,  double*, double*);
template void NVSHMEM_mcmr_to_mrstar_cleanup<__half>(MPI_Comm, int*, int*, int*,  __half*, __half*);
template void NVSHMEM_mcmr_to_mrstar_cleanup<El::Complex<float>>(MPI_Comm, int*, int*, int*,  El::Complex<float>*, El::Complex<float>*);
template void NVSHMEM_mcmr_to_mrstar_cleanup<El::Complex<double>>(MPI_Comm, int*, int*, int*,  El::Complex<double>*, El::Complex<double>*);
*/
template void NVSHMEM_mcmr_to_mrstar_cleanup<int>(MPI_Comm, long*, int*, long*,  int*, int*);
template void NVSHMEM_mcmr_to_mrstar_cleanup<double>(MPI_Comm, long*, int*, long*,  double*, double*);
template void NVSHMEM_mcmr_to_mrstar_cleanup<__half>(MPI_Comm, long*, int*, long*,  __half*, __half*);
template void NVSHMEM_mcmr_to_mrstar_cleanup<El::Complex<float>>(MPI_Comm, long*, int*, long*,  El::Complex<float>*, El::Complex<float>*);
template void NVSHMEM_mcmr_to_mrstar_cleanup<El::Complex<double>>(MPI_Comm, long*, int*, long*,  El::Complex<double>*, El::Complex<double>*);

}// namespace hydrogen


