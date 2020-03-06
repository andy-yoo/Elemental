#include <hydrogen/blas/nvshmem_gemm/NVSHMEM_Gemm.hpp>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>


namespace hydrogen
{

size_t GroupSize(MPI_Group group) noexcept
{
    int size = -1;
    MPI_Group_size(group, &size);
    return static_cast<size_t>(size);
}

MPI_Group GetGroup(MPI_Comm comm) noexcept
{
    MPI_Group group = MPI_GROUP_NULL;
    MPI_Comm_group(comm, &group);
    return group;
}


/** @brief Translate all ranks in src_comm to corresponding ranks in
 *         tgt_comm.
 *
 *  Invalid if src_comm's group is not a subset of tgt_comm's group.
 */
std::vector<int> Translate(MPI_Comm src_comm, MPI_Comm tgt_comm)
{
    MPI_Group const group1 = GetGroup(src_comm), group2 = GetGroup(tgt_comm);
    size_t const size1 = GroupSize(group1), size2 = GroupSize(group2);

    if (size1 > size2)
        throw std::runtime_error(
            "Translate: Source group must be no larger than target group.");

    // Both should have the same size.
    std::vector<int> src_ranks(size1), tgt_ranks(size1, -1);

    // We want to translate all ranks.
    std::iota(begin(src_ranks), end(src_ranks), 0);

    MPI_Group_translate_ranks(group1, size1, src_ranks.data(),
                              group2, tgt_ranks.data());

    return tgt_ranks;
}

/** @brief Translate ranks in the comm to their SHMEM PEs. */
std::vector<int> GetPEs(MPI_Comm comm)
{
    return Translate(comm, MPI_COMM_WORLD);
}

// Given an MPI rank (value), returns my PE number (i.e. PE rank) in given subcommunicator
int FindLocation(int const* const list, int const list_size, int const value)
{
    for (int i = 0; i < list_size; ++i)
        if (list[i] == value)
            return i;
    return -1;
}

// NOTE: The key information returned from this function is 'pes'.
// How to interpret? The indices to this vector is the rank of PEs and
// the values corresponding to the indices are actual PE number.
// It is expected that initially PE number and MPI rank of a process is identical 
// This is probably the most efficient way to map a PE number in a subcommunicator to actual PE number.
void prepare_for_conversion(MPI_Comm comm, 
    int *my_pe_rank,
    int **dev_pes, // PE mapping information in device
    std::vector<int>& pes,
    int *npes)
{
    int myrank;
    MPI_Comm_rank(comm, &myrank);
    pes = GetPEs(comm);
    *npes = pes.size(); 
    if (std::find(begin(pes), end(pes), nvshmem_my_pe()) == end(pes))
        throw std::logic_error( "My pe not found in comm. This seems bad.");
    CHECK_CUDA(cudaMalloc((void**) dev_pes, (size_t) ((*npes)*sizeof(int))));

    CHECK_CUDA(cudaMemcpy((void*) *dev_pes, (void const*) pes.data(), (*npes)*sizeof(int), cudaMemcpyHostToDevice));
    *my_pe_rank = FindLocation(pes.data(), *npes, myrank);

}

}
