/*
  Copyright (c) 2009-2016, Jack Poulson
  All rights reserved.

  This file is part of Elemental and is under the BSD 2-Clause License,
  which can be found in the LICENSE file in the root directory, or at
  http://opensource.org/licenses/BSD-2-Clause
*/
#define WARMUP_ITER	0

#include <El.hpp>
using namespace El;

#include <hydrogen/blas/nvshmem_gemm/DataRedistribution.hpp>
#include <hydrogen/blas/nvshmem_gemm/NVSHMEM_Gemm.hpp>

#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

template <typename T> 
void _test_placeholder (T h){
printf("%f\n", h);
}


FILE *fp_debug;

template<typename T, Device D>
void TestAssociativity
(Orientation orientA, Orientation orientB,
 T alpha,
 DistMatrix<T,MC,MR,ELEMENT,D> const& A,
 DistMatrix<T,MC,MR,ELEMENT,D> const & B,
 T beta,
 DistMatrix<T,MC,MR,ELEMENT,D> const& COrig,
 DistMatrix<T,MC,MR,ELEMENT,D> const& CFinal,
 bool print)
{
    EL_DEBUG_ONLY(CallStackEntry cse("TestAssociativity"))

    // Test (alpha op(A) op(B) + beta C) X = alpha op(A) (op(B) X) + beta C X
    const Int numRHS = 100;
    const Int n = COrig.Width();
    const Grid& g = A.Grid();
    DistMatrix<T,MC,MR,ELEMENT,D> X(g), Y(g), Z(g);
    Uniform(X, n, numRHS, TypeTraits<T>::Zero(), TypeTraits<Base<T>>::One());
    Gemm(orientB, NORMAL, TypeTraits<T>::One(), B, X, Z);
    Gemm(orientA, NORMAL, alpha, A, Z, Y);
    Gemm(NORMAL, NORMAL, beta, COrig, X, TypeTraits<T>::One(), Y);
    const Base<T> YFrobNorm = FrobeniusNorm(Y);
    if (print)
        Print(Y, "Y := alpha op(A) op(B) + beta C");
    T one = TypeTraits<T>::One();
    T neg_one = -one;
    Gemm(NORMAL, NORMAL, neg_one, CFinal, X, one, Y);
    const Base<T> EFrobNorm = FrobeniusNorm(Y);
    if (print)
        Print(Y, "E");
    OutputFromRoot
        (g.Comm(), "|| E ||_F / || Y ||_F = ",
         EFrobNorm, "/", YFrobNorm, "=", EFrobNorm/YFrobNorm);
}

#ifdef HYDROGEN_HAVE_CUDA
#define START_CUDA_TIMER                                  \
    if (D == Device::GPU)                                 \
        cudaEventRecord(start, GPUManager::Stream());

#define STOP_CUDA_TIMER                                 \
    if (D == Device::GPU)                               \
    {                                                   \
        cudaEventRecord(stop, GPUManager::Stream());    \
        cudaEventSynchronize(stop);                     \
        cudaEventElapsedTime(&cudaTime, start, stop);   \
    }

#define SUMMARIZE_CUDA_TIMER                                            \
    if (D == Device::GPU)                                               \
    {                                                                   \
        runTime = cudaTime * 1e-3;                                      \
        realGFlops = 2.*double(m)*double(n)*double(k)/(1.e9*runTime);   \
        gFlops = (IsComplex<T>::value ? 4*realGFlops : realGFlops);     \
        OutputFromRoot(g.Comm(),"Finished in ",runTime,                 \
                     " seconds (",gFlops," GFlop/s)");                  \
    }

#else
#define START_CUDA_TIMER do {} while (false)
#define STOP_CUDA_TIMER do {} while (false)
#define SUMMARIZE_CUDA_TIMER do {} while (false)
#endif

template<typename T, Device D>
void WarmupGemm
(Orientation orientA,
 Orientation orientB,
 Int m, Int n, Int k,
 T alpha, T beta,
 const Grid& g,
 bool print, bool correctness,
 Int colAlignA=0, Int rowAlignA=0,
 Int colAlignB=0, Int rowAlignB=0,
 Int colAlignC=0, Int rowAlignC=0)
{

    DistMatrix<T,MC,MR,ELEMENT,D> A(g), B(g), COrig(g), C(g);

    A.Align(colAlignA, rowAlignA);
    B.Align(colAlignB, rowAlignB);
    C.Align(colAlignC, rowAlignC);

    if (orientA == NORMAL)
        Uniform(A, m, k, TypeTraits<T>::Zero(), TypeTraits<Base<T>>::One());
    else
        Uniform(A, k, m, TypeTraits<T>::Zero(), TypeTraits<Base<T>>::One());
    if (orientB == NORMAL)
        Uniform(B, k, n, TypeTraits<T>::Zero(), TypeTraits<Base<T>>::One());
    else
        Uniform(B, n, k, TypeTraits<T>::Zero(), TypeTraits<Base<T>>::One());
    Uniform(COrig, m, n, TypeTraits<T>::Zero(), TypeTraits<Base<T>>::One());
    C = COrig;

    // Test the variant of Gemm that keeps A stationary
    mpi::Barrier(g.Comm());
    Gemm(orientA, orientB, alpha, A, B, beta, C, GEMM_SUMMA_A);
    mpi::Barrier(g.Comm());

    // Test the variant of Gemm that keeps B stationary
    mpi::Barrier(g.Comm());
    Gemm(orientA, orientB, alpha, A, B, beta, C, GEMM_SUMMA_B);
    mpi::Barrier(g.Comm());


    // Test the variant of Gemm that keeps C stationary
    mpi::Barrier(g.Comm());
    Gemm(orientA, orientB, alpha, A, B, beta, C, GEMM_SUMMA_C);
    mpi::Barrier(g.Comm());

    if (orientA == NORMAL && orientB == NORMAL)
    {
        // Test the variant of Gemm for panel-panel dot products
        mpi::Barrier(g.Comm());
        Gemm(NORMAL, NORMAL, alpha, A, B, beta, C, GEMM_SUMMA_DOT);
        mpi::Barrier(g.Comm());
    }
}

int grid_row_rank(int p, int d0, int d1){
    return p%d0;
}
int grid_col_rank(int p, int d0, int d1){
    return p/d0;
}

int local_height_mcmr(int p, int m, int n, int d0, int d1){
    return (m/d0 + ((grid_row_rank(p, d0, d1) < m%d0)?1:0));
}

int local_width_mcmr(int p, int m, int n, int d0, int d1){
    return (n/d1 + ((grid_col_rank(p, d0, d1) < n%d1)?1:0));
}


int local_height_vcstar(int p, int m, int n, int d0, int d1){
    int H = d0*d1;
    return (m/H + ((p < m%H)?1:0));
}

int local_width_vcstar(int p, int m, int n, int d0, int d1){
    return n;
}

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

int mcmr_to_vcstar(
    int i, int j, 
    int my_row_rank,
    int my_col_rank,
    int myrank,
    int grid_height,
    int grid_width,
    int grid_size)
{
  return(i % grid_size);
}

void msg_counts(
    std::vector<int>& counts,
    int m, //height,
    int n, //width,
    int my_row_rank,
    int my_col_rank,
    int p, //myrank,
    int d0, //grid_height,
    int d1, //grid_width,
    int g //grid_size
)
{
  int r_p = p/d0;
  int u = n%d1;
  for(int q=0; q<g; q++){
     if(p%d0 != q%d0 || p == q)
	continue;
     int r_q = q/d0;

     int tau_pq = (n/d1) + (r_q < std::min(u, r_p)) + (r_p<r_q && r_q<u);
     int sigma_pq = m/g + ((p<(m%g)?1:0));
   
     counts[q] = tau_pq*sigma_pq;
  }
}

int total_msg_buffer_size(
    int m, //height,
    int n, //width,
    int my_row_rank,
    int my_col_rank,
    int p, //myrank,
    int d0, //grid_height,
    int d1, //grid_width,
    int g //grid_size
)
{
  int r = p/d0;
  int u = n%d1;
  int alpha = std::min(r, u) + ((u-r-1 <= 0)?0:u-r-1);
  int beta_p = (n/d1)*d1 + alpha;
  int theta_p = m/g + ((p < m%g)?1:0);

  return(theta_p*beta_p);
}

int total_data_buffer_size(
    int height,
    int my_row_rank,
    int my_col_rank,
    int myrank,
    int grid_height,
    int grid_width,
    int grid_size)
{
  return((height-myrank <= 0)?0:(int) ceil((height-myrank)/((float)grid_size)));
}


void local_to_global_vcstar(int k,
	int local_height,
	int local_width,
	int grid_height,
	int grid_width,
	int my_row_rank,
	int my_col_rank,
 	int *I,
	int *J
        )
{
   int grid_size = grid_height*grid_width;
   int i = k % local_height;
   int j = (k/local_height);

   *I = my_row_rank+grid_size*i;
   *J = j;
}

int vcstar_global_coord_to_local_index(int p, 
        int I, int J,
        int m,
        int n,
        int d0,
        int d1
        )
{
   int grid_size = d0*d1;
   //int local_height_vcstar = m/grid_size + ((m%grid_size > p)?1:0);
   //int local_width_vcstar = n;
   int loc_height_vcstar = local_height_vcstar(p, m, n, d0, d1);
   int loc_width_vcstar = local_width_vcstar(p, m, n, d0, d1);
   int i = I/grid_size;
   int j = J;
   if(i < loc_height_vcstar && j < loc_width_vcstar)
        return j*loc_height_vcstar + i;
   else
        return -1;
}

int global_to_local_vcstar(int I, int J,
	int m,
	int n,
	int myrank,
	int d0,
	int d1,
	int my_row_rank,
	int my_col_rank
        )
{
   int grid_size = d0*d1;
   int local_height_vcstar = m/grid_size + ((m%grid_size > myrank)?1:0);
   int local_width_vcstar = n;
   int i = I/grid_size;
   int j = J;
   if(i < local_height_vcstar && j < local_width_vcstar)
	return j*local_height_vcstar + i;
   else 
	return -1;
}

int global_to_local_mcmr(int I, int J,
	int local_height,
	int local_width,
	int d0,
	int d1,
	int my_row_rank,
	int my_col_rank
        )
{
   int i = I/d0;
   int j = J/d1;
   if(i < local_height && j < local_width)
	return j*d0 + i;
   else 
	return -1;
}

void local_to_global_mcmr(int k,
	int local_height,
	int local_width,
	int grid_height,
	int grid_width,
	int my_row_rank,
	int my_col_rank,
 	int *I,
	int *J
        )
{
   int i = k % local_height;
   int j = (k/local_height);
   *I = my_row_rank+grid_height*i;
   *J = my_col_rank+grid_width*j;
}

int row_rank(int p, int d0, int d1){
    return p%d0;
}
int col_rank(int p, int d0, int d1){
    return p/d0;
}

int vcstar_to_vrstar_to_pid(int p, int d0, int d1){
    int i = p / d1;
    int j = p % d1;

    return (i + j*d0);
}
int vcstar_to_vrstar_from_pid(int p, int d0, int d1){
    int i = p % d0;
    int j = p / d0;

    return (j + i*d1);
}
int my_vr_row_rank(int p, int d0, int d1){
    int i = p % d0;
    int j = p / d0;

    return (j + i*d1);
}
int my_vr_col_rank(int p, int d0, int d1){
    return -1;
}

template<typename T, Device D>
void TestMatrix_1
(Orientation orientA,
 Orientation orientB,
 Int m, Int n, Int k,
 T alpha, T beta,
 const Grid& g,
 bool print, bool correctness,
 Int colAlignA=0, Int rowAlignA=0,
 Int colAlignB=0, Int rowAlignB=0,
 Int colAlignC=0, Int rowAlignC=0)
{
    int my_row_rank = g.Row();
    int my_col_rank = g.Col();
    int myrank = g.Rank();
    int grid_height = g.Height();
    int grid_width = g.Width();
    int grid_size = g.Size();

    double runTime, realGFlops, gFlops;
    DistMatrix<T,MC,MR,ELEMENT,D> A(g), B(g), COrig(g), C(g);

    A.Align(colAlignA, rowAlignA);
    B.Align(colAlignB, rowAlignB);
    C.Align(colAlignC, rowAlignC);

    if (orientA == NORMAL)
        Uniform(A, m, k, TypeTraits<T>::Zero(), TypeTraits<Base<T>>::One());
    else
        Uniform(A, k, m, TypeTraits<T>::Zero(), TypeTraits<Base<T>>::One());
    if (orientB == NORMAL)
        Uniform(B, k, n, TypeTraits<T>::Zero(), TypeTraits<Base<T>>::One());
    else
        Uniform(B, n, k, TypeTraits<T>::Zero(), TypeTraits<Base<T>>::One());
    Uniform(COrig, m, n, TypeTraits<T>::Zero(), TypeTraits<Base<T>>::Zero());

    Print(A, "A");
    Print(B, "B");

    mpi::Comm const& comm__ = g.Comm();

    char buff[132];
    sprintf(buff, "debug.%04d", myrank);
    fp_debug = fopen(buff, "w");
    int local_height = B.LocalHeight();
    int local_width = B.LocalWidth();

    fprintf(fp_debug, "my_row_rank = %d\n", my_row_rank);
    fprintf(fp_debug, "my_col_rank = %d\n", my_col_rank);
    fprintf(fp_debug, "myrank = %d\n", myrank);
    fprintf(fp_debug, "grid_height = %d\n", grid_height);
    fprintf(fp_debug, "grid_width = %d\n", grid_width);
    fprintf(fp_debug, "rank: %d\tlocal_height=%d local_width=%d\n", myrank, local_height, local_width);

    int ldim_b = B.LDim();

    C = COrig;
    DistMatrix<T,VR,STAR,ELEMENT,D> B_VR_STAR(g);
    DistMatrix<T,STAR,MR,ELEMENT,D> BTrans_STAR_MR(g);
    DistMatrix<T,MC,STAR,ELEMENT,D> D_MC_STAR(g);

    B_VR_STAR.AlignWith(A);
    BTrans_STAR_MR.AlignWith(A);
    D_MC_STAR.AlignWith(A);

    B_VR_STAR = B;;
//    BTrans_STAR_MR.Resize(m, n);
//    D_MC_STAR.Resize(m, n);

    Transpose(B_VR_STAR, BTrans_STAR_MR);
    LocalGemm(NORMAL, TRANSPOSE, alpha, A, BTrans_STAR_MR, D_MC_STAR);
    AxpyContract(TypeTraits<T>::One(), D_MC_STAR, C);

    Print(C, "C");
    //printf("c00=%f\n",-0.23058826 * -0.28910846+ 0.39080966 * 0.53984106 + 0.90079689 * 0.56753957 + -0.056507707 * -0.30540627 + -0.87230974 * 0.63214338 + -0.49558109 * -0.26125503 + -0.22641164 * -0.1046024);

    cudaDeviceSynchronize();
    mpi::Barrier(g.Comm());
    fclose(fp_debug);
    return;
}




template<typename T, Device D>
void TestMatrix
(int my_pe_rank,
 int npes, 
 cudaStream_t stream,

 Orientation orientA,
 Orientation orientB,
 int m, int n, int k,
 T alpha, T beta,
 const Grid& g,
 bool print, bool correctness,
 Int colAlignA=0, Int rowAlignA=0,
 Int colAlignB=0, Int rowAlignB=0,
 Int colAlignC=0, Int rowAlignC=0)
{
    int my_row_rank = g.Row();
    int my_col_rank = g.Col();
    int myrank = g.Rank();
    int grid_height = g.Height();
    int grid_width = g.Width();
    int grid_size = g.Size();

    double runTime, realGFlops, gFlops;
    DistMatrix<T,MC,MR,ELEMENT,D> A(g), B(g), COrig(g), C(g);

    A.Align(colAlignA, rowAlignA);
    B.Align(colAlignB, rowAlignB);
    C.Align(colAlignC, rowAlignC);

    if (orientA == NORMAL)
        Uniform(A, m, k, TypeTraits<T>::Zero(), TypeTraits<Base<T>>::One());
    else
        Uniform(A, k, m, TypeTraits<T>::Zero(), TypeTraits<Base<T>>::One());
    if (orientB == NORMAL)
        Uniform(B, k, n, TypeTraits<T>::Zero(), TypeTraits<Base<T>>::One());
    else
        Uniform(B, n, k, TypeTraits<T>::Zero(), TypeTraits<Base<T>>::One());
    Uniform(COrig, m, n, TypeTraits<T>::Zero(), TypeTraits<Base<T>>::Zero());
    C = COrig;

    Print(A, "A");
    Print(B, "B");

    //printf("C[0,0]=%f\n", -0.23058826 * -0.28910846+ 0.39080966 * 0.53984106+ 0.90079689 * 0.56753957+ -0.056507707* -0.30540627+ -0.87230974 * 0.63214338 + -0.49558109 * -0.26125503 + -0.22641164* -0.1046024);



    char buff[132];
    sprintf(buff, "debug.%04d", myrank);
    fp_debug = fopen(buff, "w");
#if 0
    int local_height = B.LocalHeight();
    int local_width = B.LocalWidth();
#else
    int local_height = local_height_mcmr(myrank, k, n, grid_height, grid_width);
    int local_width = local_width_mcmr(myrank, k, n, grid_height, grid_width);
#endif

    fprintf(fp_debug, "my_row_rank = %d\n", my_row_rank);
    fprintf(fp_debug, "my_col_rank = %d\n", my_col_rank);
    fprintf(fp_debug, "myrank = %d\n", myrank);
    fprintf(fp_debug, "grid_height = %d\n", grid_height);
    fprintf(fp_debug, "grid_width = %d\n", grid_width);
    fprintf(fp_debug, "rank: %d\tlocal_height=%d local_width=%d\n", myrank, local_height, local_width);
    //printf( "rank: %d\tlocal_height=%d local_width=%d\n", myrank, local_height, local_width);

    int ldim_b = B.LDim();
    Matrix<T, D>& local_mat = B.Matrix();
    fprintf(fp_debug, "rank: %d\tLocal_Height=%d Local_Width=%d\n", myrank, local_mat.Height(), local_mat.Width());
    //printf( "rank: %d\tLocal_Height=%d Local_Width=%d\n", myrank, local_mat.Height(), local_mat.Width());
    for(int i=0; i<local_height; i++){
          for(int j=0; j<local_width; j++)
	     fprintf(fp_debug, "%f[%d,%d]:to (%d) ", local_mat.Get(i,j), 
		my_row_rank+grid_height*i, my_col_rank+grid_width*j,
		mcmr_to_vcstar(my_row_rank+grid_height*i, my_col_rank+grid_width*j,
    			my_row_rank,
    			my_col_rank,
    			myrank,
    			grid_height,
    			grid_width,
    			grid_size));
          fprintf(fp_debug, "\n");
    }
    fprintf(fp_debug, "Needs %d entries in DATA buffer\n", n*total_data_buffer_size(k, my_row_rank, my_col_rank, myrank, grid_height, grid_width, grid_size));
    fprintf(fp_debug, "Needs %d entries in MSG buffer\n", total_msg_buffer_size(k, n, my_row_rank, my_col_rank, myrank, grid_height, grid_width, grid_size));
    std::vector<int> msg_cnts(grid_size, 0);
    fprintf(fp_debug, "MSG counts\n");
    msg_counts(msg_cnts, k, n, my_row_rank, my_col_rank, myrank, grid_height, grid_width, grid_size);
    for(int q=0; q<msg_cnts.size(); q++)
       fprintf(fp_debug, "q:%d %d \n", q, msg_cnts[q]);

#if 1
    //Matrix<T, D>& local_mat = B.Matrix();
    auto kernel_B_buffer = local_mat.Buffer();
    auto B_buffer = local_mat.Buffer();
    T* mem_buffer = (T*) malloc(local_height*local_width*sizeof(T));
    cudaMemcpy(mem_buffer, B_buffer, local_height*local_width*sizeof(T), cudaMemcpyDeviceToHost);
    fprintf(fp_debug, "Buffer of B...\n");
    for(int j=0; j<local_height*local_width; j++){
	int I = local_to_global_row_index_mcmr(myrank, j, k, n, grid_height, grid_width);
	int J = local_to_global_col_index_mcmr(myrank, j, k, n, grid_height, grid_width);
  	
      	fprintf(fp_debug, "%f[%d,%d] ", mem_buffer[j], I, J);
    }
    fprintf(fp_debug, "\n");
#endif

//
// [MC,MR] TO [VC,*]
//
    std::vector<int> send_counts(grid_size, 0);
    std::vector<int> send_displs(grid_size, 0);
    std::vector<int> recv_counts(grid_size, 0);
    std::vector<int> recv_displs(grid_size+1, 0);
#if 0
    for(int j=0; j<local_height*local_width; j++){
	int I = local_to_global_row_index_mcmr(myrank, j, k, n, grid_height, grid_width);

	send_counts[I%grid_size]++;
    }
#else
    for(int j=0; j<grid_width; j++){
	int p = grid_row_rank(myrank, grid_height, grid_width)+j*grid_height;
	send_counts[p] = local_height_vcstar(p, k, n, grid_height, grid_width) * local_width_mcmr(myrank, k, n, grid_height, grid_width);
    }
#endif
    for(int p=0; p<grid_size; p++){
	fprintf(fp_debug, "To %d: will send %d elements\n", p, send_counts[p]);
    }
    int total_send = send_counts[0];
    send_displs[0]  = 0;
    for(int p=1; p<grid_size; p++){
	send_displs[p] = send_displs[p-1] + send_counts[p-1];
	total_send += send_counts[p];
    }
    for(int p=0; p<grid_size; p++){
	fprintf(fp_debug, "To %d: position = %d\n", p, send_displs[p]);
    }
    fprintf(fp_debug, "Will send %d in total\n", total_send);

#if 0
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
#else
    for(int j=0; j<grid_width; j++){
	int p = grid_row_rank(myrank, grid_height, grid_width)+j*grid_height;
	recv_counts[p] = local_height_vcstar(myrank, k, n, grid_height, grid_width) * local_width_mcmr(p, k, n, grid_height, grid_width);
    }
#endif
    for(int p=0; p<grid_size; p++){
	fprintf(fp_debug, "From %d: will receive %d elements\n", p, recv_counts[p]);
    }

    int total_recv =0;
    for(int p=0; p<grid_size; p++)
	total_recv += recv_counts[p];
    recv_displs[0]  = 0;
    for(int p=1; p<=grid_size; p++){
	recv_displs[p] = recv_displs[p-1] + recv_counts[p-1];
    }
    for(int p=0; p<grid_size; p++){
	fprintf(fp_debug, "From %d: position = %d\n", p, recv_displs[p]);
    }
    fprintf(fp_debug, "Will receive %d in total\n", total_recv);

    fprintf(fp_debug, "max_send=%d max_recv=%d\n", 
	(int) (ceil(((double)k)/grid_height)*ceil(((double)n)/grid_width)), (int) (ceil(((double)k)/grid_size)*n));

    std::vector<int> send_pos(grid_size, 0);
    std::vector<T> send_buffer(total_send);
    for(int l=0; l<local_height*local_width; l++){
	int I = local_to_global_row_index_mcmr(myrank, l, k, n, grid_height, grid_width);
	int J = local_to_global_col_index_mcmr(myrank, l, k, n, grid_height, grid_width);
	int target = I%grid_size;
	int i = I/grid_size;
	int j = J/grid_width;
        int t = j*local_height_vcstar(target, k, n, grid_height, grid_width) + i;

	//send_buffer[send_displs[target] + send_pos[target]] = mem_buffer[k];
	send_buffer[send_displs[target] + t] = mem_buffer[l];
    }
    fprintf(fp_debug, "========SEND============\n");
    for(int k=0; k<total_send; k++)
	fprintf(fp_debug, "%d : %f\n", k, send_buffer[k]);

    std::vector<T> recv_buffer(total_recv);
    MPI_Alltoallv(send_buffer.data(),  send_counts.data(),
    	send_displs.data(), MPI_FLOAT,
    	recv_buffer.data(), recv_counts.data(),
    	recv_displs.data(), MPI_FLOAT, MPI_COMM_WORLD);
    fprintf(fp_debug, "========RECV============\n");
    for(int k=0; k<total_recv; k++)
	fprintf(fp_debug, "%d : %f\n", k, recv_buffer[k]);

    std::vector<T> local_buffer(total_recv);
#if 0
    for(int q=0; q<grid_size; q++){
	int I = myrank;
	int J = grid_col_rank(q, grid_height, grid_width);
	for(int k=recv_displs[q]; k<recv_displs[q+1]; k++){
	    int t = vcstar_global_coord_to_local_index(myrank, I, J, k, n, grid_height, grid_width);
	    fprintf(fp_debug, "k=%d I=%d J=%d t=%d\n", k, I, J, t);

	    local_buffer[t] = recv_buffer[k];
	    I = I + grid_size;
	    if( I >=  m){
		I = myrank;
		J = J + grid_width;
	   }
       	}
    }
#else
    int lh = local_height_vcstar(myrank, k, n, grid_height, grid_width);
    for(int q=0; q<grid_size; q++){
	int start_idx = lh*grid_col_rank(q, grid_height, grid_width);
	for(int j=recv_displs[q]; j<recv_displs[q+1]; j++){
	    int c = j-recv_displs[q];
	    int a = c/lh;
	    int t = start_idx + a*lh*grid_width + (c%lh);
	    fprintf(fp_debug, "j=%d c=%d t=%d\n", j, c, t);
	    local_buffer[t] = recv_buffer[j];
       	}
    }
#endif
    fprintf(fp_debug, "[vc, star]\n");
    for(int l=0; l<local_buffer.size(); l++)
      	fprintf(fp_debug, "%f ", local_buffer[l]);
    fprintf(fp_debug, "\n");

    // Using new kernels/nvshmem
    MPI_Comm mpi_comm;
    mpi::Comm const& comm__ = g.Comm();
    mpi_comm = comm__.GetMPIComm();

    int myperank;
    int* dev_pes;
    std::vector<int> pes;
    int xnpes;
    prepare_for_conversion(mpi_comm, &myperank, &dev_pes, pes, &xnpes);
    fprintf(fp_debug, "my_pe_rank=%d npes=%d\n", myperank, xnpes);
    for(int i=0; i<pes.size(); i++)
	fprintf(fp_debug, "i=%d rank=%d\n", i, pes[i]);

    T* dev_local_buffer = (T*) nvshmem_malloc(total_recv*sizeof(T));
    CHECK_CUDA(cudaMemcpyAsync((void*) dev_local_buffer, (void const*) kernel_B_buffer,
        (local_height*local_width)*sizeof(T), cudaMemcpyDeviceToDevice, stream));

    int workspace_size = 32, dev_pes_size = 32;
    int* common_workspace;
    common_workspace = (int*) nvshmem_malloc(workspace_size*sizeof(int));
    CHECK_CUDA(cudaMemsetAsync( common_workspace, 0, workspace_size*sizeof(int), stream));

    convert_mcmr_to_vcstar_float (m, n, myperank, grid_height, grid_width,
        dev_local_buffer, 
        dev_pes, 
        xnpes,
        common_workspace, stream);

    convert_vcstar_to_vrstar_float(m, n, my_pe_rank, grid_height, grid_width,
        dev_local_buffer, 
        dev_pes, xnpes,
        common_workspace, stream);

    int to_pe = vcstar_to_vrstar_to_pid(my_pe_rank, grid_height, grid_width);
    int from_pe = vcstar_to_vrstar_from_pid(my_pe_rank, grid_height, grid_width);
    int local_recv_size_vcstar_to_vrstar = local_height_vcstar(from_pe, m, n, grid_height, grid_width) * local_width_vcstar(from_pe, m, n, grid_height, grid_width);
    std::vector<T> host_local_buffer(local_recv_size_vcstar_to_vrstar);
    CHECK_CUDA(cudaMemcpyAsync((void*) host_local_buffer.data(), (void const*) dev_local_buffer,
                local_recv_size_vcstar_to_vrstar*sizeof(T), cudaMemcpyDeviceToHost, stream));
fprintf(fp_debug, "At the end of convert_vcstar_to_vrstar: [vr,*]\n");
for(int j=0; j<host_local_buffer.size(); j++)
fprintf(fp_debug, "%f ", host_local_buffer[j]);
fprintf(fp_debug, "\n");



    DistMatrix<T,VC,STAR,ELEMENT,D> B_VC_STAR(g);

    B_VC_STAR = B;
    local_height = B_VC_STAR.LocalHeight();
    local_width = B_VC_STAR.LocalWidth();
    local_mat = B_VC_STAR.Matrix();
    fprintf(fp_debug, "[VC,*]\n");
    fprintf(fp_debug, "rank: %d\tLocal_Height_B_VC_STAR=%d Local_Width_B_VC_STAR=%d\n", myrank, local_height, local_width);
    for(int i=0; i<local_height; i++){
          for(int j=0; j<local_width; j++)
	     fprintf(fp_debug, "%f[%d,%d] ", local_mat.Get(i,j), my_row_rank+grid_size*i, j);
          fprintf(fp_debug, "\n");
    }
    auto B_VC_STAR_buffer = B_VC_STAR.Buffer();
    T* B_VC_STAR_mem_buffer = (T*) malloc(local_height*local_width*sizeof(T));
    cudaMemcpy(B_VC_STAR_mem_buffer, B_VC_STAR_buffer, local_height*local_width*sizeof(T), cudaMemcpyDeviceToHost);
    fprintf(fp_debug, "Buffer of B_VC_STAR...\n");
    for(int l=0; l<local_height*local_width; l++){
	int i = l % local_height;
	int j = (l/local_height);
      	fprintf(fp_debug, "%f ", B_VC_STAR_mem_buffer[l]);
    }
    fprintf(fp_debug, "\n");


//
// [VC,*] TO [VR,*]
//
    int to_pid = vcstar_to_vrstar_to_pid(myrank, grid_height, grid_width);
    int from_pid = vcstar_to_vrstar_from_pid(myrank, grid_height, grid_width);
    fprintf(fp_debug, "to_pid=%d from_pid=%d total_recv=%d\n", to_pid, from_pid, total_recv);

    int my_send_size_vcstar_to_vrstar =
local_height_vcstar(myrank, k, n, grid_height, grid_width) * local_width_vcstar(myrank, k, n, grid_height, grid_width);
    int new_recv_size_vcstar_to_vrstar =
local_height_vcstar(from_pid, k, n, grid_height, grid_width) * local_width_vcstar(from_pid, k, n, grid_height, grid_width);

    std::vector<T> rbuffer(new_recv_size_vcstar_to_vrstar);
    MPI_Status status;
    MPI_Sendrecv(local_buffer.data(), my_send_size_vcstar_to_vrstar, MPI_FLOAT,
                to_pid, 0,
                rbuffer.data(), new_recv_size_vcstar_to_vrstar, MPI_FLOAT,
                from_pid, 0,
                MPI_COMM_WORLD, &status);
    fprintf(fp_debug, "[vr, *]\n");
    for(int i=0; i<total_recv; i++)
	fprintf(fp_debug, "%f ", rbuffer[i]);
    fprintf(fp_debug, "\n");
    

#if 1
    DistMatrix<T,VR,STAR,ELEMENT,D> B_VR_STAR(g);
    B_VR_STAR = B_VC_STAR;;
    local_height = B_VR_STAR.LocalHeight();
    local_width = B_VR_STAR.LocalWidth();
    local_mat = B_VR_STAR.Matrix();
    fprintf(fp_debug, "[VR,*]\n");
    fprintf(fp_debug, "rank: %d\tLocal_Height_B_VR_STAR=%d Local_Width_B_VR_STAR=%d\n", myrank, local_height, local_width);
   int conv_row_rank = grid_width*my_row_rank + my_col_rank;
   for(int i=0; i<local_height; i++){
          for(int j=0; j<local_width; j++)
	     fprintf(fp_debug, "%f[%d,%d] ", local_mat.Get(i,j),
		 conv_row_rank+grid_size*i, j);
          fprintf(fp_debug, "\n");
    }
    auto B_VR_STAR_buffer = B_VR_STAR.Buffer();
    T* B_VR_STAR_mem_buffer = (T*) malloc(local_height*local_width*sizeof(T));
    cudaMemcpy(B_VR_STAR_mem_buffer, B_VR_STAR_buffer, local_height*local_width*sizeof(T), cudaMemcpyDeviceToHost);
    fprintf(fp_debug, "Buffer of B_VR_STAR...\n");
    for(int k=0; k<local_height*local_width; k++){
	int i = k % local_height;
	int j = (k/local_height);
        fprintf(fp_debug, "%f[%d,%d] ", B_VR_STAR_mem_buffer[k],
		 conv_row_rank+grid_size*i, j);
    }
    fprintf(fp_debug, "\n");
#endif


//
// [VR,*] TRANSPOSE TO [*,VR]
//
    std::vector<T> vr_star_buffer(rbuffer);
    std::vector<T> star_vr_buffer(vr_star_buffer.size());
    int vr_row_rank = my_vr_row_rank(myrank, grid_height, grid_width);
    int H = m/grid_size + ((vr_row_rank < m%grid_size)?1:0);
    int W = n;
#if 0
    int l = 0;
    for(int i=0; i<H; i++)
       for(int j=0; j<W; j++){
	  int m = j*H+i;
	  star_vr_buffer[l] = vr_star_buffer[m];
	  l++;
	}
#else
    for(int t=0; t<H*W; t++){
       	int i = t%H;
     	int j = t/H;
	int l = i*W+j; // linear index of (j, i) in WxH transposed matrix 

       	int i_p = l%H;
     	int j_p = l/H;
	int l_p = i_p*W+j_p; // linear index of (j, i) in WxH transposed matrix 
	star_vr_buffer[l] = vr_star_buffer[t];

	fprintf(fp_debug, "t=%d l=%d (i, j)=(%d, %d) (ip, jp)=(%d, %d)\n", t, l, i, j, i_p, j_p);
    }
#endif
    fprintf(fp_debug, "[*,vr]\n");
    for(int i=0; i<star_vr_buffer.size(); i++)
	fprintf(fp_debug, "%f ", star_vr_buffer[i]);
    fprintf(fp_debug, "\n");

    DistMatrix<T,STAR,VR,ELEMENT,D> BTrans_STAR_VR(g);
    Transpose(B_VR_STAR, BTrans_STAR_VR);

    auto BTrans_STAR_VR_buffer = BTrans_STAR_VR.Buffer();
    T* BTrans_STAR_VR_mem_buffer = (T*) malloc(local_height*local_width*sizeof(T));
    cudaMemcpy(BTrans_STAR_VR_mem_buffer, BTrans_STAR_VR_buffer, local_height*local_width*sizeof(T), cudaMemcpyDeviceToHost);
    fprintf(fp_debug, "Buffer of BTrans_STAR_VR...\n");
    for(int k=0; k<local_height*local_width; k++){
	int i = k % local_height;
	int j = (k/local_height);
        fprintf(fp_debug, "%f ", BTrans_STAR_VR_mem_buffer[k]);
    }
    fprintf(fp_debug, "\n");


//
// [*,VR] TO [*,MR]
//
    std::vector<T> star_mr_buffer;
    std::vector<int> mr_msg_cnts(grid_height);
    int mycrank = col_rank(myrank, grid_height, grid_width);
    int sum = 0;

    for(int i=0; i<grid_height; i++){
	int l = mycrank*grid_height + i;
	int count_i = n * (k/grid_size + ((my_vr_row_rank(l, grid_height, grid_width) < k%grid_width)?1:0));
    fprintf(fp_debug, "l=%d my_vr_row_rank=%d\n", l, my_vr_row_rank(l, grid_height, grid_width));
	mr_msg_cnts[i] = count_i;
	sum += count_i;
    }

    fprintf(fp_debug, "mr_counts\n");
    for(int i=0; i<mr_msg_cnts.size(); i++)
	fprintf(fp_debug, "%d: %d\n", mycrank*grid_height + i, mr_msg_cnts[i]);
    fprintf(fp_debug, "sum = %d\n", sum);


    star_mr_buffer.resize(sum);
    std::vector<int> star_mr_buffer_displs(grid_height+1, 0);
    for(int i=1; i<=grid_height; i++)
	star_mr_buffer_displs[i] = star_mr_buffer_displs[i-1]+mr_msg_cnts[i-1];

    int color = myrank/grid_height;
    int key = myrank%grid_height;
    int tsum = 0;
    int xcnt = 0;
    MPI_Comm col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, key, &col_comm);
    
    MPI_Allgatherv(star_vr_buffer.data(), star_vr_buffer.size(), MPI_FLOAT,
                   star_mr_buffer.data(), mr_msg_cnts.data(),
		   star_mr_buffer_displs.data(), MPI_FLOAT, col_comm);

#if 0
    std::vector<T> star_mr_buffer_interleaved; //(sum);
    while(tsum < sum){
        for(int i=0; i<grid_height; i++){
	    int l = star_mr_buffer_displs[i]+xcnt*n;
	    if(l+n <= star_mr_buffer_displs[i+1]){
		for(int j=0; j<n; j++){
		    star_mr_buffer_interleaved.push_back(star_mr_buffer[l+j]);
		}
		tsum += n;
	    }
	}
	xcnt++;
    }
#else
    std::vector<T> star_mr_buffer_interleaved(sum);
    int t = 0;
    while(t < sum){
	int from_pe = (t/n)%grid_height;
	int from_block = t/(n*grid_height);
  	int from_loc = t%n;
	int l = star_mr_buffer_displs[from_pe] + from_block*n + from_loc;

    fprintf(fp_debug, "t=%d l=%d from_pe=%d from_block=%d from_loc=%d l=%d\n", t, l, from_pe, from_block, from_loc, l);
	star_mr_buffer_interleaved[t] = star_mr_buffer[l];
	
	t++;
    }
#endif
    fprintf(fp_debug, "[*, mr] buffer\n");
    for(int i=0; i<sum; i++)
	fprintf(fp_debug, "%f ", star_mr_buffer_interleaved[i]);
    fprintf(fp_debug, "\n");

    DistMatrix<T,STAR,MR,ELEMENT,D> BTrans_STAR_MR(g);
    //Transpose(B_VR_STAR, BTrans_STAR_MR);
    BTrans_STAR_MR = BTrans_STAR_VR;
    local_height = BTrans_STAR_MR.LocalHeight();
    local_width = BTrans_STAR_MR.LocalWidth();
    local_mat = BTrans_STAR_MR.Matrix();
    fprintf(fp_debug, "local_height=%d local_width=%d\n", local_height, local_width);
    fprintf(fp_debug, "[*,MR] (Transpose)\n");
    for(int i=0; i<local_height; i++){
          for(int j=0; j<local_width; j++)
	     fprintf(fp_debug, "%f[%d,%d] ", local_mat.Get(i,j),
		 conv_row_rank+grid_size*i, j);
          fprintf(fp_debug, "\n");
    }
    auto BTrans_STAR_MR_buffer = BTrans_STAR_MR.Buffer();
    T* BTrans_STAR_MR_mem_buffer = (T*) malloc(local_height*local_width*sizeof(T));
    cudaMemcpy(BTrans_STAR_MR_mem_buffer, BTrans_STAR_MR_buffer, local_height*local_width*sizeof(T), cudaMemcpyDeviceToHost);
    fprintf(fp_debug, "Buffer of BTrans_STAR_MR...\n");
    for(int l=0; l<local_height*local_width; l++){
        fprintf(fp_debug, "%f ", BTrans_STAR_MR_mem_buffer[l]);
    }
    fprintf(fp_debug, "\n");


    local_height = A.LocalHeight();
    local_width = A.LocalWidth();
    local_mat = A.Matrix();
fprintf(fp_debug, "A: Height=%d Width=%d\n", A.Height(), A.Width());
    fprintf(fp_debug, "A: \n");
    for(int i=0; i<local_height; i++){
          for(int j=0; j<local_width; j++)
	     fprintf(fp_debug, "%f ", local_mat.Get(i,j));
          fprintf(fp_debug, "\n");
    }


//
// LocalGemm
//
    DistMatrix<T,MC,STAR,ELEMENT,D> D1_MC_STAR(g);
    DistMatrix<T,MC,MR,ELEMENT,D> C_MC_MR(g);

    LocalGemm(NORMAL, TRANSPOSE, (T)1.0, A, BTrans_STAR_MR, D1_MC_STAR);

    local_height = D1_MC_STAR.LocalHeight();
    local_width = D1_MC_STAR.LocalWidth();
    local_mat = D1_MC_STAR.Matrix();
fprintf(fp_debug, "D1_MC_STAR: LocalHeight=%d LocalWidth=%d\n", local_height, local_width);
    fprintf(fp_debug, "D1[MC,*] \n");
    for(int i=0; i<local_height; i++){
          for(int j=0; j<local_width; j++)
	     fprintf(fp_debug, "%f ", local_mat.Get(i,j));
          fprintf(fp_debug, "\n");
    }

    auto D1_MC_STAR_buffer = D1_MC_STAR.Buffer();
    T* D1_MC_STAR_mem_buffer = (T*) malloc(local_height*local_width*sizeof(T));
    cudaMemcpy(D1_MC_STAR_mem_buffer, D1_MC_STAR_buffer, local_height*local_width*sizeof(T), cudaMemcpyDeviceToHost);
    fprintf(fp_debug, "Buffer of D1_MC_STAR...\n");
    for(int l=0; l<local_height*local_width; l++){
	int i = l % local_height;
	int j = (l/local_height);
        fprintf(fp_debug, "%f ", D1_MC_STAR_mem_buffer[l]);
    }
    fprintf(fp_debug, "\n");

//
// Axpy
//

    color = myrank%grid_height;
    key = myrank;
    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, key, &row_comm);
    int my_row_comm_rank;
    MPI_Comm_rank(row_comm, &my_row_comm_rank);


    int rrank = row_rank(myrank, grid_height, grid_width);
    int crank = col_rank(myrank, grid_height, grid_width);
    int col_length = m/grid_height + ((rrank < m%grid_height)?1:0);
    std::vector<int> row_lengths(grid_width, 0);
    std::vector<int> row_displs(grid_width+1, 0);
    int xsum = 0;
    for(int i=0; i<grid_width; i++){
	row_lengths[i] = col_length * (n/grid_width + ((i < n%grid_width)?1:0));
	xsum += row_lengths[i];
    }
    row_displs[0] = 0;
    for(int i=1; i<=grid_width; i++){
	row_displs[i] = row_displs[i-1] + row_lengths[i-1];
    }

    std::vector<T> mcstar_send_buffer(xsum);
    xcnt = 0;
    while(xcnt < xsum/col_length){
	int i = xcnt/grid_width;
	int j = xcnt%grid_width;

	for(int l=0; l<col_length; l++){
	    mcstar_send_buffer[row_displs[j]+i*col_length+l] =
		D1_MC_STAR_mem_buffer[xcnt*col_length+l];
	}
	xcnt++;
    }	

    std::vector<T> mcstar_recv_buffer(row_lengths[my_row_comm_rank], (T)0.0);
    MPI_Reduce_scatter(mcstar_send_buffer.data(), mcstar_recv_buffer.data(), row_lengths.data(), MPI_FLOAT, MPI_SUM, row_comm);
    fprintf(fp_debug, "reduce_scatter: mcstar\n");
    for(int i=0; i<mcstar_recv_buffer.size(); i++)
	fprintf(fp_debug, "%f ", mcstar_recv_buffer[i]);
    fprintf(fp_debug, "\n");

    //C.Resize(m, n);
    C_MC_MR = COrig;
    local_height = C_MC_MR.LocalHeight();
    local_width = C_MC_MR.LocalWidth();
fprintf(fp_debug, "C: LOcalHeight=%d LOcalWidth=%d\n", local_height, local_width);
    AxpyContract(TypeTraits<T>::One(), D1_MC_STAR, C_MC_MR);

    local_mat = C_MC_MR.Matrix();
    fprintf(fp_debug, "C[MC,MR] \n");
    for(int i=0; i<local_height; i++){
          for(int j=0; j<local_width; j++)
	     fprintf(fp_debug, "%f ", local_mat.Get(i,j));
          fprintf(fp_debug, "\n");
    }
    auto C_MC_MR_buffer = C_MC_MR.Buffer();
    T* C_MC_MR_mem_buffer = (T*) malloc(local_height*local_width*sizeof(T));
    cudaMemcpy(C_MC_MR_mem_buffer, C_MC_MR_buffer, local_height*local_width*sizeof(T), cudaMemcpyDeviceToHost);
    fprintf(fp_debug, "Buffer of C_MC_MR...\n");
    for(int l=0; l<local_height*local_width; l++){
	int i = l % local_height;
	int j = (l/local_height);
        fprintf(fp_debug, "%f ", C_MC_MR_mem_buffer[l]);
    }
    fprintf(fp_debug, "\n");

    Print(C_MC_MR, "C_MC_MR");
    cudaDeviceSynchronize();
    mpi::Barrier(g.Comm());
    fclose(fp_debug);
  
    return;
}

template<typename T, Device D>
void TestGemm_2
(Orientation orientA,
 Orientation orientB,
 Int m, Int n, Int k,
 T alpha, T beta,
 const Grid& g,
 bool print, bool correctness,
 Int colAlignA=0, Int rowAlignA=0,
 Int colAlignB=0, Int rowAlignB=0,
 Int colAlignC=0, Int rowAlignC=0)
{
    OutputFromRoot(g.Comm(),"Testing with ",TypeName<T>());
    PushIndent();

    double runTime, realGFlops, gFlops;
    DistMatrix<T,MC,MR,ELEMENT,D> A(g), B(g), COrig(g), C(g);

    A.Align(colAlignA, rowAlignA);
    B.Align(colAlignB, rowAlignB);
    C.Align(colAlignC, rowAlignC);

    if (orientA == NORMAL)
        Uniform(A, m, k, TypeTraits<T>::Zero(), TypeTraits<Base<T>>::One());
    else
        Uniform(A, k, m, TypeTraits<T>::Zero(), TypeTraits<Base<T>>::One());
    if (orientB == NORMAL)
        Uniform(B, k, n, TypeTraits<T>::Zero(), TypeTraits<Base<T>>::One());
    else
        Uniform(B, n, k, TypeTraits<T>::Zero(), TypeTraits<Base<T>>::One());
    Uniform(COrig, m, n, TypeTraits<T>::Zero(), TypeTraits<Base<T>>::Zero());
    //Uniform(COrig, m, n, TypeTraits<T>::Zero(), TypeTraits<Base<T>>::One());
    if (print)
    {
        Print(A, "A");
        Print(B, "B");
        Print(COrig, "COrig");
    }

    C = COrig;
    OutputFromRoot(g.Comm(),"Stationary A algorithm:");
    mpi::Barrier(g.Comm());
    //Gemm(orientA, orientB, alpha, A, B, beta, C, GEMM_SUMMA_A);

    mpi::Barrier(g.Comm());

    if (print)
        Print(C, BuildString("C := ",alpha," A B + ",beta," C"));

    const Int bsize = Blocksize();
    const Grid& g_ = A.Grid();

/*
    DistMatrixReadProxy<T,T,MC,MR,ELEMENT,D> AProx(A);
    DistMatrixReadProxy<T,T,MC,MR,ELEMENT,D> BProx(B);
    DistMatrixReadWriteProxy<T,T,MC,MR,ELEMENT,D> CProx(C);
    auto& A_ = AProx.GetLocked();
    auto& B_ = BProx.GetLocked();
    auto& C_ = CProx.Get();
*/
    auto& A_ = A;
    auto& B_ = B;
    auto& C_ = C;
    Print(C_, "C_");

    // Temporary distributions
    DistMatrix<T,VR,STAR,ELEMENT,D> B1_VR_STAR(g);
    DistMatrix<T,STAR,MR,ELEMENT,D> B1Trans_STAR_MR(g);
    DistMatrix<T,MC,STAR,ELEMENT,D> D1_MC_STAR(g);

    B1_VR_STAR.AlignWith(A_);
    B1Trans_STAR_MR.AlignWith(A_);
    D1_MC_STAR.AlignWith(A_);

#if 0
    for(Int k=0; k<n; k+=bsize)
    {
        const Int nb = Min(bsize,n-k);
        auto B1 = B_(ALL, IR(k,k+nb));
        auto C1 = C_(ALL, IR(k,k+nb));

        // D1[MC,*] := alpha A[MC,MR] B1[MR,*]
        B1_VR_STAR = B1;
        Transpose(B1_VR_STAR, B1Trans_STAR_MR);
        LocalGemm(NORMAL, TRANSPOSE, alpha, A_, B1Trans_STAR_MR, D1_MC_STAR);
        
        // C1[MC,MR] += scattered result of D1[MC,*] summed over grid rows
        AxpyContract(TypeTraits<T>::One(), D1_MC_STAR, C1);
     }
#endif
     B1_VR_STAR = B_;
     Transpose(B1_VR_STAR, B1Trans_STAR_MR);
     LocalGemm(NORMAL, TRANSPOSE, alpha, A_, B1Trans_STAR_MR, D1_MC_STAR);
     
     AxpyContract(TypeTraits<T>::One(), D1_MC_STAR, C_);

     Print(A_, "A_");
     Print(B_, "B_");
     Print(C_, "C_");
}

template<typename T, Device D>
void TestGemm
(Orientation orientA,
 Orientation orientB,
 Int m, Int n, Int k,
 T alpha, T beta,
 const Grid& g,
 bool print, bool correctness,
 Int colAlignA=0, Int rowAlignA=0,
 Int colAlignB=0, Int rowAlignB=0,
 Int colAlignC=0, Int rowAlignC=0)
{
    OutputFromRoot(g.Comm(),"Testing with ",TypeName<T>());
    PushIndent();

    double runTime, realGFlops, gFlops;
    DistMatrix<T,MC,MR,ELEMENT,D> A(g), B(g), COrig(g), C(g);

    A.Align(colAlignA, rowAlignA);
    B.Align(colAlignB, rowAlignB);
    C.Align(colAlignC, rowAlignC);

    if (orientA == NORMAL)
        Uniform(A, m, k, TypeTraits<T>::Zero(), TypeTraits<Base<T>>::One());
    else
        Uniform(A, k, m, TypeTraits<T>::Zero(), TypeTraits<Base<T>>::One());
    if (orientB == NORMAL)
        Uniform(B, k, n, TypeTraits<T>::Zero(), TypeTraits<Base<T>>::One());
    else
        Uniform(B, n, k, TypeTraits<T>::Zero(), TypeTraits<Base<T>>::One());
    Uniform(COrig, m, n, TypeTraits<T>::Zero(), TypeTraits<Base<T>>::One());
    if (print)
    {
        Print(A, "A");
        Print(B, "B");
        Print(COrig, "COrig");
    }

    Timer timer;
#ifdef HYDROGEN_HAVE_CUDA
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float cudaTime;
#endif

    // Test the variant of Gemm that keeps A stationary
    // Warming up
    for(int w=0; w<WARMUP_ITER; w++){
       C = COrig;
       mpi::Barrier(g.Comm());
       Gemm(orientA, orientB, alpha, A, B, beta, C, GEMM_SUMMA_A);
       mpi::Barrier(g.Comm());
    }

    C = COrig;
    OutputFromRoot(g.Comm(),"Stationary A algorithm:");
    PushIndent();
    mpi::Barrier(g.Comm());
    timer.Start();
    START_CUDA_TIMER;
    Gemm(orientA, orientB, alpha, A, B, beta, C, GEMM_SUMMA_A);
    STOP_CUDA_TIMER;

    mpi::Barrier(g.Comm());
    runTime = timer.Stop();
    realGFlops = 2.*double(m)*double(n)*double(k)/(1.e9*runTime);
    gFlops = (IsComplex<T>::value ? 4*realGFlops : realGFlops);
    if (D == Device::CPU)
      OutputFromRoot
          (g.Comm(),"Finished in ",runTime," seconds (",gFlops," GFlop/s)");
    SUMMARIZE_CUDA_TIMER;

    if (print)
        Print(C, BuildString("C := ",alpha," A B + ",beta," C"));
    if (correctness)
        TestAssociativity(orientA, orientB, alpha, A, B, beta, COrig, C, print);
    PopIndent();

    // Test the variant of Gemm that keeps B stationary
    // Warming up
    for(int w=0; w<WARMUP_ITER; w++){
       C = COrig;
       mpi::Barrier(g.Comm());
       Gemm(orientA, orientB, alpha, A, B, beta, C, GEMM_SUMMA_B);
       mpi::Barrier(g.Comm());
    }
    C = COrig;
    OutputFromRoot(g.Comm(),"Stationary B Algorithm:");
    PushIndent();
    mpi::Barrier(g.Comm());
    timer.Start();
    START_CUDA_TIMER;
    Gemm(orientA, orientB, alpha, A, B, beta, C, GEMM_SUMMA_B);
    STOP_CUDA_TIMER;

    mpi::Barrier(g.Comm());
    runTime = timer.Stop();
    realGFlops = 2.*double(m)*double(n)*double(k)/(1.e9*runTime);
    gFlops = (IsComplex<T>::value ? 4*realGFlops : realGFlops);

    if (D == Device::CPU)
      OutputFromRoot
          (g.Comm(),"Finished in ",runTime," seconds (",gFlops," GFlop/s)");
    SUMMARIZE_CUDA_TIMER;

    if (print)
        Print(C, BuildString("C := ",alpha," A B + ",beta," C"));
    if (correctness)
        TestAssociativity(orientA, orientB, alpha, A, B, beta, COrig, C, print);
    PopIndent();

    // Test the variant of Gemm that keeps C stationary
    // Warming up
    for(int w=0; w<WARMUP_ITER; w++){
       C = COrig;
       mpi::Barrier(g.Comm());
       Gemm(orientA, orientB, alpha, A, B, beta, C, GEMM_SUMMA_C);
       mpi::Barrier(g.Comm());
    }
    C = COrig;
    OutputFromRoot(g.Comm(),"Stationary C Algorithm:");
    PushIndent();
    mpi::Barrier(g.Comm());
    timer.Start();
    START_CUDA_TIMER;
    Gemm(orientA, orientB, alpha, A, B, beta, C, GEMM_SUMMA_C);
    STOP_CUDA_TIMER;

    mpi::Barrier(g.Comm());
    runTime = timer.Stop();
    realGFlops = 2.*double(m)*double(n)*double(k)/(1.e9*runTime);
    gFlops = (IsComplex<T>::value ? 4*realGFlops : realGFlops);
    if (D == Device::CPU)
        OutputFromRoot
            (g.Comm(),"Finished in ",runTime," seconds (",gFlops," GFlop/s)");
    SUMMARIZE_CUDA_TIMER;
    if (print)
        Print(C, BuildString("C := ",alpha," A B + ",beta," C"));
    if (correctness)
        TestAssociativity
            (orientA, orientB, alpha, A, B, beta, COrig, C, print);
    PopIndent();

    if (orientA == NORMAL && orientB == NORMAL)
    {
        // Test the variant of Gemm for panel-panel dot products
        // Warming up
        for(int w=0; w<WARMUP_ITER; w++){
           C = COrig;
           mpi::Barrier(g.Comm());
           Gemm(NORMAL, NORMAL, alpha, A, B, beta, C, GEMM_SUMMA_DOT);
           mpi::Barrier(g.Comm());
        }
        OutputFromRoot(g.Comm(),"Dot Product Algorithm:");
        PushIndent();
        C = COrig;
        mpi::Barrier(g.Comm());
        timer.Start();
        START_CUDA_TIMER;
        Gemm(NORMAL, NORMAL, alpha, A, B, beta, C, GEMM_SUMMA_DOT);
        STOP_CUDA_TIMER;

        mpi::Barrier(g.Comm());
        runTime = timer.Stop();
        realGFlops = 2.*double(m)*double(n)*double(k)/(1.e9*runTime);
        gFlops = (IsComplex<T>::value ? 4*realGFlops : realGFlops);
        if (D == Device::CPU)
            OutputFromRoot
                (g.Comm(),"Finished in ",runTime," seconds (",gFlops,
                 " GFlop/s)");
        SUMMARIZE_CUDA_TIMER;

        if (print)
            Print(C, BuildString("C := ",alpha," A B + ",beta," C"));
        if (correctness)
            TestAssociativity
                (orientA, orientB, alpha, A, B, beta, COrig, C, print);
        PopIndent();
    }
    PopIndent();
#ifdef HYDROGEN_HAVE_CUDA
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif
}

int
main(int argc, char* argv[])
{

    Environment env(argc, argv);
    mpi::Comm comm = mpi::NewWorldComm();

//    MPI_Init(&argc, &argv);

#if 1

//    try
//    {
        const bool colMajor = Input("--colMajor","column-major ordering?",true);
        int gridHeight = Input("--gridHeight","height of process grid",0);
        const char transA = Input("--transA","orientation of A: N/T/C",'N');
        const char transB = Input("--transB","orientation of B: N/T/C",'N');
        const Int m = Input("--m","height of result",100);
        const Int n = Input("--n","width of result",100);
        const Int k = Input("--k","inner dimension",100);
        const Int nb = Input("--nb","algorithmic blocksize",96);
        const bool print = Input("--print","print matrices?",false);
        const bool correctness = Input("--correctness","correctness?",true);
        const Int colAlignA = Input("--colAlignA","column align of A",0);
        const Int colAlignB = Input("--colAlignB","column align of B",0);
        const Int colAlignC = Input("--colAlignC","column align of C",0);
        const Int rowAlignA = Input("--rowAlignA","row align of A",0);
        const Int rowAlignB = Input("--rowAlignB","row align of B",0);
        const Int rowAlignC = Input("--rowAlignC","row align of C",0);
        const bool testCPU = El::Input("--testCPU", "test CPU gemm?", true);
        const bool testGPU = El::Input("--testGPU", "test GPU gemm?", false);

        ProcessInput();
        PrintInputReport();

        if (gridHeight == 0)
            gridHeight = Grid::DefaultHeight(mpi::Size(comm));
        const GridOrder order = (colMajor ? COLUMN_MAJOR : ROW_MAJOR);
        const Grid g(std::move(comm), gridHeight, order);
        const Orientation orientA = CharToOrientation(transA);
        const Orientation orientB = CharToOrientation(transB);
        SetBlocksize(nb);

        ComplainIfDebug();
        OutputFromRoot(g.Comm(),"Will test Gemm",transA,transB);
#endif

//    mpi::Comm const& comm__ = g.Comm();
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    nvshmemx_init_attr_t attr;
    int mype, npes;
    int mype_node;

    attr.mpi_comm = &mpi_comm;
    nvshmemx_init_attr (NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    mype = nvshmemx_my_pe(NVSHMEMX_TEAM_WORLD);
    npes = nvshmemx_n_pes(NVSHMEMX_TEAM_WORLD);
    mype_node = nvshmemx_my_pe(NVSHMEMX_TEAM_NODE);

    int dev_count;
    cudaGetDeviceCount(&dev_count);
    cudaSetDevice(mype_node%dev_count);

    cudaStream_t stream;
    CHECK_CUDA( cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

#if 1
        //TestGemm<float, Device::GPU>
        TestMatrix<float, Device::GPU>
                (mype, npes, stream,
		 orientA, orientB,
                 m, n, k,
                 float(1.f), float(0.f),
                 g,
                 print, correctness,
                 colAlignA, rowAlignA,
                 colAlignB, rowAlignB,
                 colAlignC, rowAlignC);
        //}
    //catch(exception& e) { ReportException(e); }
#endif
    cudaStreamDestroy(stream);

    nvshmem_finalize();
    return 0;
}
