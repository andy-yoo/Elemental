if (HYDROGEN_HAVE_GPU)
  add_subdirectory(gpu)

  add_subdirectory(nvshmem_gemm)
  if(HYDROGEN_HAVE_CUTLASS)
    add_subdirectory(cutlass_blas)
  endif() 

  set(CUDA_SOURCES "${CUDA_SOURCES}" PARENT_SCOPE)
endif ()
