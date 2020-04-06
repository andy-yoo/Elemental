/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>

#include <sys/time.h>
#include <hydrogen/blas/nvshmem_gemm/DataRedistribution.hpp>
#include <hydrogen/blas/nvshmem_gemm/NVSHMEM_Gemm.hpp>


namespace El {
namespace gemm {

// Cannon's algorithm
template<typename T>
void Cannon_NN
(T alpha,
  const AbstractDistMatrix<T>& APre,
  const AbstractDistMatrix<T>& BPre,
        AbstractDistMatrix<T>& CPre)
{
    EL_DEBUG_CSE

    if (APre.GetLocalDevice() != Device::CPU)
        LogicError("Cannon_NN not implemented for device!");

    const Grid& g = APre.Grid();
    if (g.Height() != g.Width())
        LogicError("Process grid must be square for Cannon's");

    // Force A, B, and C to be in [MC,MR] distributions aligned with C
    DistMatrixReadWriteProxy<T,T,MC,MR> CProx(CPre);
    auto& C = CProx.Get();

    ElementalProxyCtrl ctrlA, ctrlB;
    ctrlA.colConstrain = true; ctrlA.colAlign = C.ColAlign();
    ctrlB.rowConstrain = true; ctrlB.rowAlign = C.RowAlign();

    DistMatrixReadProxy<T,T,MC,MR> AProx(APre, ctrlA);
    DistMatrixReadProxy<T,T,MC,MR> BProx(BPre, ctrlB);
    auto& A = AProx.GetLocked();
    auto& B = BProx.GetLocked();

    const Int row = g.Row();
    const Int col = g.Col();
    const Int pSqrt = g.Height();
    mpi::Comm const& rowComm = g.RowComm();
    mpi::Comm const& colComm = g.ColComm();
    if (A.Width() % pSqrt != 0)
        LogicError("For now, width(A) must be integer multiple of sqrt(p)");

    // Load the initial A and B packages (may want to transpose B...)
    const Int localHeightA = A.LocalHeight();
    const Int localHeightB = B.LocalHeight();
    const Int localWidthA = A.LocalWidth();
    const Int localWidthB = B.LocalWidth();
    Matrix<T> pkgA(localHeightA,localWidthA,localHeightA),
              pkgB(localHeightB,localWidthB,localHeightB);
    for(Int jLoc=0; jLoc<localWidthA; ++jLoc)
        MemCopy
        (pkgA.Buffer(0,jLoc), A.LockedBuffer(0,jLoc), localHeightA);
    for(Int jLoc=0; jLoc<localWidthB; ++jLoc)
        MemCopy
        (pkgB.Buffer(0,jLoc), B.LockedBuffer(0,jLoc), localHeightB);

    // Perform the initial circular shifts so that our A and B packages align
    const Int rowShiftA = A.RowShift();
    const Int colShiftB = B.ColShift();
    const Int leftInitA  = Mod(col-colShiftB,pSqrt);
    const Int rightInitA = Mod(col+colShiftB,pSqrt);
    const Int aboveInitB = Mod(row-rowShiftA,pSqrt);
    const Int belowInitB = Mod(row+rowShiftA,pSqrt);
    const Int pkgSizeA = localHeightA*localWidthA;
    const Int pkgSizeB = localHeightB*localWidthB;
    mpi::SendRecv(pkgA.Buffer(), pkgSizeA, leftInitA, rightInitA, rowComm,
                  SyncInfo<Device::CPU>{});
    mpi::SendRecv(pkgB.Buffer(), pkgSizeB, aboveInitB, belowInitB, colComm,
                  SyncInfo<Device::CPU>{});

    // Now begin the data flow
    const Int aboveRow = Mod(row-1,pSqrt);
    const Int belowRow = Mod(row+1,pSqrt);
    const Int leftCol  = Mod(col-1,pSqrt);
    const Int rightCol = Mod(col+1,pSqrt);
    for(Int q=0; q<pSqrt; ++q)
    {
        Gemm(NORMAL, NORMAL, alpha, pkgA, pkgB, TypeTraits<T>::One(), C.Matrix());
        if (q != pSqrt-1)
        {
            mpi::SendRecv(
                pkgA.Buffer(), pkgSizeA, leftCol, rightCol, rowComm,
                SyncInfo<Device::CPU>{});
            mpi::SendRecv(
                pkgB.Buffer(), pkgSizeB, aboveRow, belowRow, colComm,
                SyncInfo<Device::CPU>{});
        }
    }
}

// Normal Normal Gemm that avoids communicating the matrix A
template <Device D, typename T, typename=EnableIf<IsDeviceValidType<T,D>>>
void SUMMA_NNA_impl
(T alpha,
 AbstractDistMatrix<T> const& APre,
 AbstractDistMatrix<T> const& BPre,
 AbstractDistMatrix<T>& CPre)
{
    EL_DEBUG_CSE
    AUTO_PROFILE_REGION(
        "SUMMA.NNA",
        SyncInfoFromMatrix(
            static_cast<Matrix<T,D> const&>(CPre.LockedMatrix())));

    const Int n = CPre.Width();
    const Int bsize = Blocksize();
    const Grid& g = APre.Grid();

    DistMatrixReadProxy<T,T,MC,MR,ELEMENT,D> AProx(APre);
    DistMatrixReadProxy<T,T,MC,MR,ELEMENT,D> BProx(BPre);
    DistMatrixReadWriteProxy<T,T,MC,MR,ELEMENT,D> CProx(CPre);
    auto& A = AProx.GetLocked();
    auto& B = BProx.GetLocked();
    auto& C = CProx.Get();

    cudaStream_t stream;
    if (BPre.GetLocalDevice() == El::Device::GPU)
       stream = static_cast<El::Matrix<T, El::Device::GPU> const&>(BPre.LockedMatrix()).Stream();

    // Temporary distributions
    DistMatrix<T,VR,STAR,ELEMENT,D> B1_VR_STAR(g);
    DistMatrix<T,STAR,MR,ELEMENT,D> B1Trans_STAR_MR(g);
    DistMatrix<T,MC,STAR,ELEMENT,D> D1_MC_STAR(g);

    DistMatrix<T,VC,STAR,ELEMENT,D> B1_VC_STAR(g);

    B1_VR_STAR.AlignWith(A);
    B1Trans_STAR_MR.AlignWith(A);
    D1_MC_STAR.AlignWith(A);

    B1_VC_STAR.AlignWith(A);

    int my_row_rank = g.Row();
    int my_col_rank = g.Col();
    int myrank = g.Rank();
    int grid_height = g.Height();
    int grid_width = g.Width();
    int grid_size = g.Size();

    for(Int k=0; k<n; k+=bsize)
    {
        const Int nb = Min(bsize,n-k);
        auto B1 = B(ALL, IR(k,k+nb));
        auto C1 = C(ALL, IR(k,k+nb));

        // D1[MC,*] := alpha A[MC,MR] B1[MR,*]

            
    	    
//#ifdef HYDROGEN_HAVE_NVSHMEM_GEMM
#if 0
	int B1_Height = B1.Height();
	int B1_Width = B1.Width();
	int B1_LocalHeight = B1.LocalHeight();
	int B1_LocalWidth = B1.LocalWidth();
	int B1_LDim= B1.LDim();

 	int m_p = C1.Height();
	int n_p = C1.Width();
	int k_p = nb;
    
        MPI_Comm mpi_comm;
        mpi::Comm const& comm__ = g.Comm();
        mpi_comm = comm__.GetMPIComm();

        Matrix<T, D>& local_mat = B1.Matrix();
        auto B1_buffer = local_mat.Buffer();

	B1_VR_STAR.Resize(B1_Height, B1_Width);
	
	// Now do B1_VR_STAR = B1 using B1_buffer
        Matrix<T, D>& local_vr_star_mat = B1_VR_STAR.Matrix();
        auto B1_VR_STAR_buffer = local_vr_star_mat.Buffer();

	mcmr_to_vrstar (mpi_comm,
        	B1_Height, B1_Width,
        	grid_height,
        	grid_width,
        	B1_buffer,
		B1_VR_STAR_buffer,
		stream);
#else
        B1_VR_STAR = B1;
#endif

        Transpose(B1_VR_STAR, B1Trans_STAR_MR);
        LocalGemm(NORMAL, TRANSPOSE, alpha, A, B1Trans_STAR_MR, D1_MC_STAR);

        // C1[MC,MR] += scattered result of D1[MC,*] summed over grid rows
        AxpyContract(TypeTraits<T>::One(), D1_MC_STAR, C1);
    }
}

template <Device D, typename T,
          typename=DisableIf<IsDeviceValidType<T,D>>, typename=void>
void SUMMA_NNA_impl(T alpha,
                    AbstractDistMatrix<T> const& APre,
                    AbstractDistMatrix<T> const& BPre,
                    AbstractDistMatrix<T>& CPre)
{
    LogicError("SUMMA_NNA_impl type-device combo not supported.");
}

template <typename T>
void SUMMA_NNA
(T alpha,
 AbstractDistMatrix<T> const& APre,
 AbstractDistMatrix<T> const& BPre,
 AbstractDistMatrix<T>& CPre)
{
    EL_DEBUG_CSE

    switch (CPre.GetLocalDevice())
    {
    case Device::CPU:
        SUMMA_NNA_impl<Device::CPU>(alpha, APre, BPre, CPre);
        break;
#ifdef HYDROGEN_HAVE_CUDA
    case Device::GPU:
        SUMMA_NNA_impl<Device::GPU>(alpha, APre, BPre, CPre);
        break;
#endif // HYDROGEN_HAVE_CUDA
    default:
        LogicError("SUMMA_NNA: Bad device.");
    }
}

// Normal Normal Gemm that avoids communicating the matrix B
template <Device D, typename T, typename=EnableIf<IsDeviceValidType<T,D>>>
void SUMMA_NNB_impl
(T alpha,
  const AbstractDistMatrix<T>& APre,
  const AbstractDistMatrix<T>& BPre,
        AbstractDistMatrix<T>& CPre)
{
    EL_DEBUG_CSE
    AUTO_PROFILE_REGION(
        "SUMMA.NNB",
        SyncInfoFromMatrix(
            static_cast<Matrix<T,D> const&>(CPre.LockedMatrix())));

    const Int m = CPre.Height();
    const Int bsize = Blocksize();
    const Grid& g = APre.Grid();

    DistMatrixReadProxy<T,T,MC,MR,ELEMENT,D> AProx(APre);
    DistMatrixReadProxy<T,T,MC,MR,ELEMENT,D> BProx(BPre);
    DistMatrixReadWriteProxy<T,T,MC,MR,ELEMENT,D> CProx(CPre);
    auto& A = AProx.GetLocked();
    auto& B = BProx.GetLocked();
    auto& C = CProx.Get();

    // Temporary distributions
    DistMatrix<T,STAR,MC,ELEMENT,D> A1_STAR_MC(g);
    DistMatrix<T,MR,STAR,ELEMENT,D> D1Trans_MR_STAR(g);

    A1_STAR_MC.AlignWith(B);
    D1Trans_MR_STAR.AlignWith(B);

    for(Int k=0; k<m; k+=bsize)
    {
        const Int nb = Min(bsize,m-k);
        auto A1 = A(IR(k,k+nb), ALL);
        auto C1 = C(IR(k,k+nb), ALL);

        // D1^T[MR,* ] := alpha B^T[MR,MC] A1^T[MC,* ]
        A1_STAR_MC = A1;
        LocalGemm(
            TRANSPOSE, TRANSPOSE, alpha, B, A1_STAR_MC, D1Trans_MR_STAR);

        TransposeAxpyContract(TypeTraits<T>::One(), D1Trans_MR_STAR, C1);
    }
}

template <Device D, typename T,
          typename=DisableIf<IsDeviceValidType<T,D>>, typename=void>
void SUMMA_NNB_impl(T alpha,
                    AbstractDistMatrix<T> const& APre,
                    AbstractDistMatrix<T> const& BPre,
                    AbstractDistMatrix<T>& CPre)
{
    LogicError("SUMMA_NNB_impl type-device combo not supported.");
}

template <typename T>
void SUMMA_NNB
(T alpha,
 AbstractDistMatrix<T> const& APre,
 AbstractDistMatrix<T> const& BPre,
 AbstractDistMatrix<T>& CPre)
{
    EL_DEBUG_CSE

    switch (CPre.GetLocalDevice())
    {
    case Device::CPU:
        SUMMA_NNB_impl<Device::CPU>(alpha, APre, BPre, CPre);
        break;
#ifdef HYDROGEN_HAVE_CUDA
    case Device::GPU:
        SUMMA_NNB_impl<Device::GPU>(alpha, APre, BPre, CPre);
        break;
#endif // HYDROGEN_HAVE_CUDA
    default:
        LogicError("SUMMA_NNB: Bad device.");
    }
}

// Normal Normal Gemm that avoids communicating the matrix C
template <Device D, typename T, typename=EnableIf<IsDeviceValidType<T,D>>>
void SUMMA_NNC_impl(T alpha,
                    AbstractDistMatrix<T> const& APre,
                    AbstractDistMatrix<T> const& BPre,
                    AbstractDistMatrix<T>& CPre)
{
    EL_DEBUG_CSE
    AUTO_PROFILE_REGION(
        "SUMMA.NNC",
        SyncInfoFromMatrix(
            static_cast<Matrix<T,D> const&>(CPre.LockedMatrix())));
    const Int sumDim = APre.Width();
    const Int bsize = Blocksize();
    const Grid& g = APre.Grid();
    const Int BsumDim = BPre.Width();

    DistMatrixReadProxy<T,T,MC,MR,ELEMENT,D> AProx(APre);
    DistMatrixReadProxy<T,T,MC,MR,ELEMENT,D> BProx(BPre);
    DistMatrixReadWriteProxy<T,T,MC,MR,ELEMENT,D> CProx(CPre);
    auto& A = AProx.GetLocked();
    auto& B = BProx.GetLocked();
    auto& C = CProx.Get();

    // Temporary distributions
    DistMatrix<T,MC,STAR,ELEMENT,D> A1_MC_STAR(g);
    DistMatrix<T,MR,STAR,ELEMENT,D> B1Trans_MR_STAR(g);

    A1_MC_STAR.AlignWith(C);
    B1Trans_MR_STAR.AlignWith(C);


    for(Int k=0; k<sumDim; k+=bsize)
    {
        const Int nb = Min(bsize,sumDim-k);
        auto A1 = A(ALL,        IR(k,k+nb));
        auto B1 = B(IR(k,k+nb), ALL       );

        A1_MC_STAR = A1;

        Transpose(B1, B1Trans_MR_STAR);
        LocalGemm
        (NORMAL, TRANSPOSE, alpha, A1_MC_STAR, B1Trans_MR_STAR, TypeTraits<T>::One(), C);
     }
}


template <Device D, typename T, typename=EnableIf<IsDeviceValidType<T,D>>>
void SUMMA_NNC_impl(float *run_timer,
    		    cudaEvent_t kernel_start, 
		    cudaEvent_t kernel_stop,
		    T alpha,
                    AbstractDistMatrix<T> const& APre,
                    AbstractDistMatrix<T> const& BPre,
                    AbstractDistMatrix<T>& CPre)
{

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    EL_DEBUG_CSE
    AUTO_PROFILE_REGION(
        "SUMMA.NNC",
        SyncInfoFromMatrix(
            static_cast<Matrix<T,D> const&>(CPre.LockedMatrix())));
    const Int sumDim = APre.Width();
    const Int bsize = Blocksize();
    const Grid& g = APre.Grid();
    const Int BsumDim = BPre.Width();

    DistMatrixReadProxy<T,T,MC,MR,ELEMENT,D> AProx(APre);
    DistMatrixReadProxy<T,T,MC,MR,ELEMENT,D> BProx(BPre);
    DistMatrixReadWriteProxy<T,T,MC,MR,ELEMENT,D> CProx(CPre);
    auto& A = AProx.GetLocked();
    auto& B = BProx.GetLocked();
    auto& C = CProx.Get();


    cudaStream_t stream;
    if (APre.GetLocalDevice() == El::Device::GPU)
       stream = static_cast<El::Matrix<T, El::Device::GPU> const&>(APre.LockedMatrix()).Stream();

    // Temporary distributions
    DistMatrix<T,MC,STAR,ELEMENT,D> A1_MC_STAR(g);
    DistMatrix<T,MR,STAR,ELEMENT,D> B1Trans_MR_STAR(g);
    DistMatrix<T,MC,MR,ELEMENT,D> B1_MCMR(g);
    DistMatrix<T,MC,MR,ELEMENT,D> A1_MCMR(g);

    A1_MC_STAR.AlignWith(C);
    B1Trans_MR_STAR.AlignWith(C);
    B1_MCMR.AlignWith(C);


//#ifdef HYDROGEN_HAVE_NVSHMEM_GEMM

    char* env_string;
    int alg; // 0=Vanilla, 1=NVSHMEM
    env_string = getenv ("EXP_ALG");
    if (env_string !=NULL)
      alg = atoi(env_string);

    float cudaTime;
if(alg ==  0){
char line[132];
FILE *fp_debug;
    MPI_Comm mpi_comm;
    mpi::Comm const& comm__ = g.Comm();
    mpi_comm = comm__.GetMPIComm();

int mpi_myrank;
MPI_Comm_rank(mpi_comm, &mpi_myrank);
sprintf(line, "vanilla.%04d", mpi_myrank);
FILE* fp_probe = fopen(line, "a");

    for(Int k=0; k<sumDim; k+=bsize)
    {
        const Int nb = Min(bsize,sumDim-k);
        auto A1 = A(ALL,        IR(k,k+nb));
        auto B1 = B(IR(k,k+nb), ALL       );

    cudaEventRecord(start, stream  );
        A1_MC_STAR = A1;
    cudaEventRecord(stop, stream  );
    cudaEventSynchronize(stop);                     \
    cudaEventElapsedTime(&cudaTime, start, stop);   \
    fprintf(fp_probe, "Alpha1: %f\n", cudaTime);
    *run_timer += cudaTime;

    cudaEventRecord(start, stream  );
        Transpose(B1, B1Trans_MR_STAR);
    cudaEventRecord(stop, stream  );
    cudaEventSynchronize(stop);                     \
    cudaEventElapsedTime(&cudaTime, start, stop);   \
    fprintf(fp_probe, "Alpha2: %f\n", cudaTime);
    *run_timer += cudaTime;

    cudaEventRecord(start, stream  );
        LocalGemm
        (NORMAL, TRANSPOSE, alpha, A1_MC_STAR, B1Trans_MR_STAR, TypeTraits<T>::One(), C);
    cudaEventRecord(kernel_stop, stream  );
    cudaEventSynchronize(kernel_stop);                     \
    cudaEventElapsedTime(&cudaTime, start, kernel_stop);   \
    fprintf(fp_probe, "Alpha3: %f\n", cudaTime);
    *run_timer += cudaTime;
    }

fclose(fp_probe);
}
else{


    int my_row_rank = g.Row();
    int my_col_rank = g.Col();
    int myrank = g.Rank();
    int grid_height = g.Height();
    int grid_width = g.Width();
    int grid_size = g.Size();

    MPI_Comm mpi_comm;
    mpi::Comm const& comm__ = g.Comm();
    mpi_comm = comm__.GetMPIComm();

    int my_pe_rank;
    int xnpes;
    int* common_workspace;
    int* dev_pes;
    T* dev_recv_buf;
    T* dev_send_buf;
    int A_width = (sumDim < bsize)?sumDim:bsize;

    int B_my_pe_rank;
    int B_xnpes;
    int* B_common_workspace;
    int* B_dev_pes;
    int* B_dev_sync_counter;
    T* B_dev_recv_buf;
    T* B_dev_send_buf;
    int B_width = (BsumDim < bsize)?BsumDim:bsize;

char line[132];
FILE *fp_debug;

int mpi_myrank;
MPI_Comm_rank(mpi_comm, &mpi_myrank);
sprintf(line, "__elapsed.%04d", mpi_myrank);
fp_debug = fopen(line, "w");

sprintf(line, "nvshmem.%04d", mpi_myrank);
FILE* fp_probe = fopen(line, "a");

    int sync_counter = 0;
    NVSHMEM_mcmr_to_mcstar_setup(
	mpi_comm,
        A.Height(), 
	A_width,
        grid_height,
        grid_width,
        &my_pe_rank,
        &xnpes,
        &common_workspace,
        &dev_pes,
        &dev_recv_buf,
        &dev_send_buf);

    NVSHMEM_mcmr_to_mrstar_setup(
	mpi_comm,
        B.Height(), 
	B_width,
        grid_height,
        grid_width,
        &B_my_pe_rank,
        &B_xnpes,
        &B_common_workspace,
        &B_dev_pes,
        &B_dev_sync_counter,
        &B_dev_recv_buf,
        &B_dev_send_buf);


    for(Int k=0; k<sumDim; k+=bsize)
    {
        const Int nb = Min(bsize,sumDim-k);
        auto A1 = A(ALL,        IR(k,k+nb));
        auto B1 = B(IR(k,k+nb), ALL       );

/*
        Copy (A1, A1_MCMR);
	Matrix<T, D>& local_a1mcmr_mat = A1_MCMR.Matrix();
        auto A1_MCMR_buffer = local_a1mcmr_mat.Buffer();
	std::vector<T> host_a1mcmr_buffer(A1.LocalHeight()*A1.LocalWidth());
	CHECK_CUDA(cudaMemcpy(host_a1mcmr_buffer.data(), A1_MCMR_buffer, A1.LocalHeight()*A1.LocalWidth()*sizeof(T), cudaMemcpyDeviceToHost));
	//std::vector<T> host_a1mcmr_buffer(A1_MCMR.LocalHeight()*A1_MCMR.LocalWidth());
	//CHECK_CUDA(cudaMemcpy(host_a1mcmr_buffer.data(), A1_MCMR_buffer, A1_MCMR.LocalHeight()*A1_MCMR.LocalWidth()*sizeof(T), cudaMemcpyDeviceToHost));
    fprintf(fp_debug, "A1[mc,mr]   ...\n");
    for(int j=0; j<host_a1mcmr_buffer.size(); j++){
        fprintf(fp_debug, "%f ", (T) host_a1mcmr_buffer[j]);
    }
    fprintf(fp_debug, "\n");
*/
/*
fprintf(fp_debug, "== A1.height=%d A1.width=%d\n", A1.Height(), A1.Width());
fprintf(fp_debug, "== A1.Localheight=%d A1.Localwidth=%d\n", A1.LocalHeight(), A1.LocalWidth());
fprintf(fp_debug, "== B1.height=%d B1.width=%d\n", B1.Height(), B1.Width());
fprintf(fp_debug, "== B1.Localheight=%d B1.Localwidth=%d\n", B1.LocalHeight(), B1.LocalWidth());

std::vector<T> host_b1_buffer(B1.LocalHeight()*B1.LocalWidth());
CHECK_CUDA(cudaMemcpy(host_b1_buffer.data(), B1_buffer, B1.LocalHeight()*B1.LocalWidth()*sizeof(T), cudaMemcpyDeviceToHost));
fprintf(fp_debug, "B1[mc,mr] \n");
for(int j=0; j<host_b1_buffer.size(); j++){
        fprintf(fp_debug, "%f ", (T) host_b1_buffer[j] );
}
fprintf(fp_debug, "\n");
fprintf(fp_debug, "B1[I,J]\n");
for(int i=0; i<B1.Height(); i++){
   for(int j=0; j<B1.Width(); j++)
        fprintf(fp_debug, "%f ", (T) B1.Get(i, j));
   fprintf(fp_debug, "\n");
}
*/

        // C[MC,MR] += alpha A1[MC,*] (B1^T[MR,*])^T
        //           = alpha A1[MC,*] B1[*,MR]
	int A1_Height = A1.Height();
	int A1_Width = A1.Width();
	int A1_LocalHeight = A1.LocalHeight();
	int A1_LocalWidth = A1.LocalWidth();
	int A1_LDim= A1.LDim();

        Matrix<T, D>& local_mat = A1.Matrix();
        auto A1_buffer = local_mat.Buffer();


/*
std::vector<T> host_a1_buffer(A1.LocalHeight()*A1.LocalWidth());
CHECK_CUDA(cudaMemcpy(host_a1_buffer.data(), A1_buffer, A1.LocalHeight()*A1.LocalWidth()*sizeof(T), cudaMemcpyDeviceToHost));
for(int j=0; j<host_a1_buffer.size(); j++){
        fprintf(fp_debug, "%f ", (T) host_a1_buffer[j] );
}
fprintf(fp_debug, "\n");
*/

	A1_MC_STAR.Resize(A1_Height, A1_Width);
	
	// Now do A1_MC_STAR = A1; using A1_buffer
        Matrix<T, D>& local_mc_star_mat = A1_MC_STAR.Matrix();
        auto A1_MC_STAR_buffer = local_mc_star_mat.Buffer();

#if 1
    cudaEventRecord(start, stream  );
	mcmr_to_mcstar(mpi_comm,
		run_timer,
		kernel_start,
		kernel_stop,
		sync_counter,
        	A1_Height, A1_Width,
        	grid_height,
        	grid_width,
        my_pe_rank,
        common_workspace,
        dev_pes,
        xnpes,
        dev_send_buf,
        dev_recv_buf,
        	A1_buffer,
        	//A1_MCMR_buffer,
		A1_MC_STAR_buffer,
        stream);
    cudaEventRecord(kernel_stop, stream  );
    cudaEventSynchronize(kernel_stop);                     \
    cudaEventElapsedTime(&cudaTime, start, kernel_stop);   \
    fprintf(fp_probe, "Beta1: %f\n", cudaTime);
#else
        A1_MC_STAR = A1;
#endif

#if 1
	int B1_Height = B1.Height();
	int B1_Width = B1.Width();
	int B1_LocalHeight = B1.LocalHeight();
	int B1_LocalWidth = B1.LocalWidth();
	int B1_LDim= B1.LDim();

        Copy (B1, B1_MCMR);
	Matrix<T, D>& local_b1mcmr_mat = B1_MCMR.Matrix();
        auto B1_MCMR_buffer = local_b1mcmr_mat.Buffer();

	B1Trans_MR_STAR.Resize(B1_Width,B1_Height);
	
        Matrix<T, D>& local_mr_star_mat = B1Trans_MR_STAR.Matrix();
        auto B1Trans_MR_STAR_buffer = local_mr_star_mat.Buffer();

        MPI_Barrier(mpi_comm);
	sync_counter++;
        MPI_Barrier(mpi_comm);

    cudaEventRecord(start, stream  );
	mcmr_to_mrstar(fp_debug, 
		mpi_comm,
		run_timer,
		kernel_start,
		kernel_stop,
		sync_counter,
        	B1_Height, B1_Width,
        	grid_height,
        	grid_width,
        B_my_pe_rank,
        B_common_workspace,
        B_dev_pes,
	B_dev_sync_counter,
        B_xnpes,
        B_dev_send_buf,
        B_dev_recv_buf,
		B1_MCMR_buffer,
		B1Trans_MR_STAR_buffer,
        stream);
    cudaEventRecord(kernel_stop, stream  );
    cudaEventSynchronize(kernel_stop);                     \
    cudaEventElapsedTime(&cudaTime, start, kernel_stop);   \
    fprintf(fp_probe, "Beta2: %f\n", cudaTime);

        MPI_Barrier(mpi_comm);
#else
        Transpose(B1, B1Trans_MR_STAR);
#endif


#if 1
    float cudaTime;
    cudaEventRecord(kernel_start, stream  );

        LocalGemm
        (NORMAL, TRANSPOSE, alpha, A1_MC_STAR, B1Trans_MR_STAR, TypeTraits<T>::One(), C);
    cudaEventRecord(kernel_stop, stream  );
    cudaEventSynchronize(kernel_stop);                     \
    cudaEventElapsedTime(&cudaTime, kernel_start, kernel_stop);   \
    *run_timer += cudaTime;
    fprintf(fp_probe, "Beta3: %f\n", cudaTime);
#endif

    }
fclose(fp_probe);
    MPI_Barrier(mpi_comm);
    NVSHMEM_mcmr_to_mcstar_cleanup(mpi_comm, common_workspace, dev_pes, dev_recv_buf, dev_send_buf);
    NVSHMEM_mcmr_to_mrstar_cleanup(mpi_comm, B_common_workspace, B_dev_pes, B_dev_sync_counter, B_dev_recv_buf, B_dev_send_buf);
    MPI_Barrier(mpi_comm);
}
}
template <Device D, typename T,
          typename=DisableIf<IsDeviceValidType<T,D>>, typename=void>
void SUMMA_NNC_impl(T alpha,
               AbstractDistMatrix<T> const& APre,
               AbstractDistMatrix<T> const& BPre,
               AbstractDistMatrix<T>& CPre)
{
    LogicError("SUMMA_NNC_impl type-device combo not supported.");
}
template <Device D, typename T,
          typename=DisableIf<IsDeviceValidType<T,D>>, typename=void>
void SUMMA_NNC_impl(float *run_timer,
	       cudaEvent_t kernel_start,
	       cudaEvent_t kernel_stop,
	       T alpha,
               AbstractDistMatrix<T> const& APre,
               AbstractDistMatrix<T> const& BPre,
               AbstractDistMatrix<T>& CPre)
{
    LogicError("SUMMA_NNC_impl type-device combo not supported.");
}

template<typename T>
void SUMMA_NNC
(T alpha,
  const AbstractDistMatrix<T>& APre,
  const AbstractDistMatrix<T>& BPre,
        AbstractDistMatrix<T>& CPre)
{
    EL_DEBUG_CSE

    switch (CPre.GetLocalDevice())
    {
    case Device::CPU:
        SUMMA_NNC_impl<Device::CPU>(alpha, APre, BPre, CPre);
        break;
#ifdef HYDROGEN_HAVE_CUDA
    case Device::GPU:
        SUMMA_NNC_impl<Device::GPU>(alpha, APre, BPre, CPre);
        break;
#endif // HYDROGEN_HAVE_CUDA
    default:
        LogicError("SUMMA_NNC: Bad device.");
    }
}

template<typename T>
void SUMMA_NNC
(float *run_timer,
 cudaEvent_t kernel_start,
 cudaEvent_t kernel_stop,
  T alpha,
  const AbstractDistMatrix<T>& APre,
  const AbstractDistMatrix<T>& BPre,
        AbstractDistMatrix<T>& CPre)
{
    EL_DEBUG_CSE

    switch (CPre.GetLocalDevice())
    {
    case Device::CPU:
        SUMMA_NNC_impl<Device::CPU>(alpha, APre, BPre, CPre);
        break;
#ifdef HYDROGEN_HAVE_CUDA
    case Device::GPU:
        SUMMA_NNC_impl<Device::GPU>(run_timer, kernel_start, kernel_stop, alpha, APre, BPre, CPre);
        break;
#endif // HYDROGEN_HAVE_CUDA
    default:
        LogicError("SUMMA_NNC: Bad device.");
    }
}


// Normal Normal Gemm for panel-panel dot products
//
// Use summations of local multiplications from a 1D distribution of A and B
// to update blockSize x blockSize submatrices of C
//
template <Device D, typename T, typename=EnableIf<IsDeviceValidType<T,D>>>
void SUMMA_NNDot_impl
(T alpha,
  const AbstractDistMatrix<T>& APre,
  const AbstractDistMatrix<T>& BPre,
        AbstractDistMatrix<T>& CPre,
  Int blockSize)
{
    EL_DEBUG_CSE
    AUTO_PROFILE_REGION(
        "SUMMA.NNDot",
        SyncInfoFromMatrix(
            static_cast<Matrix<T,D> const&>(CPre.LockedMatrix())));

    const Int m = CPre.Height();
    const Int n = CPre.Width();
    const Grid& g = APre.Grid();

    DistMatrixReadProxy<T,T,STAR,VC,ELEMENT,D> AProx(APre);
    auto& A = AProx.GetLocked();

    ElementalProxyCtrl BCtrl;
    BCtrl.colConstrain = true;
    BCtrl.colAlign = A.RowAlign();
    DistMatrixReadProxy<T,T,VC,STAR,ELEMENT,D> BProx(BPre, BCtrl);
    auto& B = BProx.GetLocked();

    DistMatrixReadWriteProxy<T,T,MC,MR,ELEMENT,D> CProx(CPre);
    auto& C = CProx.Get();

    DistMatrix<T,STAR,STAR,ELEMENT,D> C11_STAR_STAR(g);
    for(Int kOuter=0; kOuter<m; kOuter+=blockSize)
    {
        const Int nbOuter = Min(blockSize,m-kOuter);
        const Range<Int> indOuter(kOuter, kOuter+nbOuter);

        auto A1 = A(indOuter, ALL);

        for(Int kInner=0; kInner<n; kInner+=blockSize)
        {
            const Int nbInner = Min(blockSize,n-kInner);
            const Range<Int> indInner(kInner, kInner+nbInner);

            auto B1  = B(ALL,      indInner);
            auto C11 = C(indOuter, indInner);

            LocalGemm(NORMAL, NORMAL, alpha, A1, B1, C11_STAR_STAR);
            AxpyContract(TypeTraits<T>::One(), C11_STAR_STAR, C11);
        }
    }
}

template <Device D, typename T,
          typename=DisableIf<IsDeviceValidType<T,D>>,typename=void>
void SUMMA_NNDot_impl
(T alpha,
  const AbstractDistMatrix<T>& APre,
  const AbstractDistMatrix<T>& BPre,
        AbstractDistMatrix<T>& CPre,
  Int blockSize)
{
    LogicError("SUMMA_NNDot_impl type-device combo not supported.");
}

template <typename T>
void SUMMA_NNDot
(T alpha,
  const AbstractDistMatrix<T>& APre,
  const AbstractDistMatrix<T>& BPre,
        AbstractDistMatrix<T>& CPre,
  Int blockSize=2000)
{
    EL_DEBUG_CSE

    switch (CPre.GetLocalDevice())
    {
    case Device::CPU:
        SUMMA_NNDot_impl<Device::CPU>(alpha, APre, BPre, CPre, blockSize);
        break;
#ifdef HYDROGEN_HAVE_CUDA
    case Device::GPU:
        SUMMA_NNDot_impl<Device::GPU>(alpha, APre, BPre, CPre, blockSize);
        break;
#endif // HYDROGEN_HAVE_CUDA
    default:
        LogicError("SUMMA_NNDot: Bad device.");
    }
}

template<typename T>
void SUMMA_NN
(T alpha,
  const AbstractDistMatrix<T>& A,
  const AbstractDistMatrix<T>& B,
        AbstractDistMatrix<T>& C,
  GemmAlgorithm alg=GEMM_DEFAULT)
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      AssertSameGrids(A, B, C);
      if (A.Height() != C.Height() ||
          B.Width() != C.Width() ||
          A.Width() != B.Height())
          LogicError
          ("Nonconformal matrices:\n",
           DimsString(A,"A"),"\n",
           DimsString(B,"B"),"\n",
           DimsString(C,"C"));
   )

    const Int m = C.Height();
    const Int n = C.Width();
    const Int sumDim = A.Width();
    const double weightTowardsC = 2.;
    const double weightAwayFromDot = 10.;

    // TODO(poulson): Make this tunable
    const Int blockSizeDot = 2000;

    switch(alg)
    {
    case GEMM_DEFAULT:
        if (weightAwayFromDot*m <= sumDim && weightAwayFromDot*n <= sumDim)
        {
            // FIXME (trb 03/27/18): There's a correctness issue with
            // this method. This exception is for your own safety.
            SUMMA_NNDot(alpha, A, B, C, blockSizeDot);
        }
        else if (m <= n && weightTowardsC*m <= sumDim)
            SUMMA_NNB(alpha, A, B, C);
        else if (n <= m && weightTowardsC*n <= sumDim)
            SUMMA_NNA(alpha, A, B, C);
        else
            SUMMA_NNC(alpha, A, B, C);
        break;
    case GEMM_SUMMA_A:   SUMMA_NNA(alpha, A, B, C); break;
    case GEMM_SUMMA_B:   SUMMA_NNB(alpha, A, B, C); break;
    case GEMM_SUMMA_C:   SUMMA_NNC(alpha, A, B, C); break;
    case GEMM_SUMMA_DOT: SUMMA_NNDot(alpha, A, B, C, blockSizeDot); break;
    default: LogicError("Unsupported Gemm option");
    }
}

template<typename T>
void SUMMA_NN
(float *run_timer,
  cudaEvent_t kernel_start,
  cudaEvent_t kernel_stop,
  T alpha,
  const AbstractDistMatrix<T>& A,
  const AbstractDistMatrix<T>& B,
        AbstractDistMatrix<T>& C,
  GemmAlgorithm alg=GEMM_DEFAULT)
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      AssertSameGrids(A, B, C);
      if (A.Height() != C.Height() ||
          B.Width() != C.Width() ||
          A.Width() != B.Height())
          LogicError
          ("Nonconformal matrices:\n",
           DimsString(A,"A"),"\n",
           DimsString(B,"B"),"\n",
           DimsString(C,"C"));
   )

    const Int m = C.Height();
    const Int n = C.Width();
    const Int sumDim = A.Width();
    const double weightTowardsC = 2.;
    const double weightAwayFromDot = 10.;

    // TODO(poulson): Make this tunable
    const Int blockSizeDot = 2000;

    switch(alg)
    {
    case GEMM_DEFAULT:
        if (weightAwayFromDot*m <= sumDim && weightAwayFromDot*n <= sumDim)
        {
            // FIXME (trb 03/27/18): There's a correctness issue with
            // this method. This exception is for your own safety.
            SUMMA_NNDot(alpha, A, B, C, blockSizeDot);
        }
        else if (m <= n && weightTowardsC*m <= sumDim)
            SUMMA_NNB(alpha, A, B, C);
        else if (n <= m && weightTowardsC*n <= sumDim)
            SUMMA_NNA(alpha, A, B, C);
        else
            SUMMA_NNC(alpha, A, B, C);
        break;
    case GEMM_SUMMA_A:   SUMMA_NNA(alpha, A, B, C); break;
    case GEMM_SUMMA_B:   SUMMA_NNB(alpha, A, B, C); break;
    case GEMM_SUMMA_C:   SUMMA_NNC(run_timer, kernel_start, kernel_stop, alpha, A, B, C); break;
    case GEMM_SUMMA_DOT: SUMMA_NNDot(alpha, A, B, C, blockSizeDot); break;
    default: LogicError("Unsupported Gemm option");
    }
}

} // namespace gemm
} // namespace El
