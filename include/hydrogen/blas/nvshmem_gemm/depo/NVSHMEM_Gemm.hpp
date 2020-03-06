#include <hydrogen/blas/nvshmem_gemm/DataRedistribution.hpp>
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

size_t GroupSize(MPI_Group group) noexcept;

MPI_Group GetGroup(MPI_Comm comm) noexcept;

/** @brief Translate all ranks in src_comm to corresponding ranks in
 *         tgt_comm.
 *
 *  Invalid if src_comm's group is not a subset of tgt_comm's group.
 */
std::vector<int> Translate(MPI_Comm src_comm, MPI_Comm tgt_comm);

/** @brief Translate ranks in the comm to their SHMEM PEs. */
std::vector<int> GetPEs(MPI_Comm comm);

// Given an MPI rank (value), returns my PE number (i.e. PE rank) in given subcommunicator
int FindLocation(int const* const list, int const list_size, int const value);

// NOTE: The key information returned from this function is 'pes'.
// How to interpret? The indices to this vector is the rank of PEs and
// the values corresponding to the indices are actual PE number.
// It is expected that initially PE number and MPI rank of a process is identical 
// This is probably the most efficient way to map a PE number in a subcommunicator to actual PE number.
void prepare_for_conversion(MPI_Comm comm, 
    int *my_pe_rank,
    int **dev_pes, // PE mapping information in device
    std::vector<int>& pes,
    int *npes);

}

