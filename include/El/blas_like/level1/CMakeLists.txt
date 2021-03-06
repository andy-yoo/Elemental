# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  AllReduce.hpp
  Axpy.hpp
  AxpyContract.hpp
  AxpyTrapezoid.hpp
  Broadcast.hpp
  Concatenate.hpp
  Conjugate.hpp
  ConjugateDiagonal.hpp
  ConjugateSubmatrix.hpp
  Contract.hpp
  Copy.hpp
  DiagonalScale.hpp
  DiagonalScaleTrapezoid.hpp
  DiagonalSolve.hpp
  Dot.hpp
  EntrywiseFill.hpp
  EntrywiseMap.hpp
  Fill.hpp
  FillDiagonal.hpp
  GetDiagonal.hpp
  GetMappedDiagonal.hpp
  GetSubmatrix.hpp
  Givens.hpp
  Hadamard.hpp
  ImagPart.hpp
  IndexDependentFill.hpp
  IndexDependentMap.hpp
  Kronecker.hpp
  MakeDiagonalReal.hpp
  MakeReal.hpp
  MakeSubmatrixReal.hpp
  MakeSymmetric.hpp
  MakeTrapezoidal.hpp
  Nrm2.hpp
  QuasiDiagonalScale.hpp
  QuasiDiagonalSolve.hpp
  RealPart.hpp
  Recv.hpp
  Reshape.hpp
  Rotate.hpp
  Round.hpp
  SafeScale.hpp
  Scale.hpp
  ScaleTrapezoid.hpp
  Send.hpp
  SendRecv.hpp
  SetDiagonal.hpp
  SetSubmatrix.hpp
  Shift.hpp
  ShiftDiagonal.hpp
  Transpose.hpp
  TransposeAxpy.hpp
  TransposeAxpyContract.hpp
  TransposeContract.hpp
  UpdateDiagonal.hpp
  UpdateMappedDiagonal.hpp
  UpdateSubmatrix.hpp
  Zero.hpp
  decl.hpp
  impl.hpp
  )

# Add the subdirectories
add_subdirectory(Axpy)
add_subdirectory(Copy)
add_subdirectory(Transpose)

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
