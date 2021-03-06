/*
  Copyright (c) 2009-2016, Jack Poulson
  All rights reserved.
  This file is part of Elemental and is under the BSD 2-Clause License,
  which can be found in the LICENSE file in the root directory, or at
  http://opensource.org/licenses/BSD-2-Clause
*/
#include <sys/time.h>

#include <El.hpp>
using namespace El;


#include <hydrogen/blas/nvshmem_gemm/DataRedistribution.hpp>
#include <hydrogen/blas/nvshmem_gemm/NVSHMEM_Gemm.hpp>

#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

//#define NUM_ITERS	500
//#define NUM_RUNS	1000

struct timeval s, e;
double elapsed;

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
                     " seconds (",gFlops," GFlop/s)",			\
		     " Num. Iterations is ", NUM_RUNS);                  \
    }

#else
#define START_CUDA_TIMER do {} while (false)
#define STOP_CUDA_TIMER do {} while (false)
#define SUMMARIZE_CUDA_TIMER do {} while (false)
#endif

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
 Int colAlignC=0, Int rowAlignC=0,
 char alg='A')
{
    int NUM_ITERS = 0;
    int NUM_RUNS = 1;
    char* env_string;
    env_string = getenv ("EXP_NUM_ITERS");
    if (env_string !=NULL)
      NUM_ITERS = atoi(env_string);
    env_string = getenv ("EXP_NUM_RUNS");
    if (env_string !=NULL)
      NUM_RUNS = atoi(env_string);

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

    cudaEvent_t kernel_start, kernel_stop;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);
    float run_timer = 0.0;

/*
    cudaEvent_t ovhead_start, ovhead_stop;
    cudaEventCreate(&ovhead_start);
    cudaEventCreate(&ovhead_stop);
    cudaStream_t stream;
    if (A.GetLocalDevice() == El::Device::GPU)
       stream = static_cast<El::Matrix<T, El::Device::GPU> const&>(A.LockedMatrix()).Stream();
    kernel_launch_overhead (stream, ovhead_start, ovhead_stop);
*/

#endif

switch (alg){
  case 'A':
    // Test the variant of Gemm that keeps A stationary
    C = COrig;
    OutputFromRoot(g.Comm(),"Stationary A algorithm:");
    PushIndent();

    mpi::Barrier(g.Comm());
    for(int repeat=0; repeat<NUM_ITERS; repeat++)
        Gemm(orientA, orientB, alpha, A, B, beta, C, GEMM_SUMMA_A);
    for(int repeat=0; repeat<NUM_ITERS; repeat++)
        Gemm(orientA, orientB, alpha, A, B, beta, C, GEMM_SUMMA_A);
    mpi::Barrier(g.Comm());

    timer.Start();
    START_CUDA_TIMER;
    for(int repeat=0; repeat<NUM_RUNS; repeat++)
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
    break;

  case 'C':

    // Test the variant of Gemm that keeps C stationary
    C = COrig;
    OutputFromRoot(g.Comm(),"Stationary C Algorithm:");
    PushIndent();

    run_timer = 0.0;
    mpi::Barrier(g.Comm());
    for(int repeat=0; repeat<NUM_RUNS; repeat++){
        Gemm(orientA, orientB, alpha, A, B, beta, C, GEMM_SUMMA_C);
        Gemm(orientA, orientB, alpha, A, B, beta, C, GEMM_SUMMA_C);
        timer.Start();
        START_CUDA_TIMER;
        Gemm(&run_timer, kernel_start, kernel_stop,
		orientA, orientB, alpha, A, B, beta, C, GEMM_SUMMA_C);
        STOP_CUDA_TIMER;
    }
    mpi::Barrier(g.Comm());
    runTime = timer.Stop();

    printf("run_timer=%f\n", run_timer/1000.0);

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
    break;

  case 'B':
    // Test the variant of Gemm that keeps B stationary
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
    break;
  default:
    if (orientA == NORMAL && orientB == NORMAL)
    {
        // Test the variant of Gemm for panel-panel dot products
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
	const char alg = El::Input("--alg", "SUMMA Algorithm", 'A');


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

#ifdef HYDROGEN_HAVE_CUDA
        if (testGPU)
        {
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


            TestGemm<float,Device::GPU>
                (orientA, orientB,
                 m, n, k,
//                 float(3), float(4),
                 float(1.f), float(0.f),
                 g,
                 print, correctness,
                 colAlignA, rowAlignA,
                 colAlignB, rowAlignB,
                 colAlignC, rowAlignC,
		 alg);
        }
#else
        (void)testGPU;
#endif
        if (testCPU)
        {
        }
        //}
    //catch(exception& e) { ReportException(e); }

    return 0;
}
