set_full_path(THIS_DIR_CUDA_SOURCES
  cutlassBLAS.cu
  )

# Propagate the files up the tree
set(CUDA_SOURCES "${CUDA_SOURCES}" "${THIS_DIR_CUDA_SOURCES}" PARENT_SCOPE)
