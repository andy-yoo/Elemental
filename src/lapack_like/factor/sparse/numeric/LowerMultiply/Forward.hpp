/*
   Copyright (c) 2009-2012, Jack Poulson, Lexing Ying, and 
   The University of Texas at Austin.
   All rights reserved.

   Copyright (c) 2013, Jack Poulson, Lexing Ying, and Stanford University.
   All rights reserved.

   Copyright (c) 2013-2014, Jack Poulson and 
   The Georgia Institute of Technology.
   All rights reserved.

   Copyright (c) 2014-2015, Jack Poulson and Stanford University.
   All rights reserved.
   
   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef EL_SPARSEDIRECT_NUMERIC_LOWERMULTIPLY_FORWARD_HPP
#define EL_SPARSEDIRECT_NUMERIC_LOWERMULTIPLY_FORWARD_HPP

#include "./FrontForward.hpp"

namespace El {

template<typename F> 
inline void LowerForwardMultiply
( const SymmNodeInfo& info, 
  const SymmFront<F>& front, MatrixNode<F>& X )
{
    DEBUG_ONLY(CallStackEntry cse("LowerForwardMultiply"))

    const Int numChildren = info.children.size();
    for( Int c=0; c<numChildren; ++c )
        LowerForwardMultiply
        ( *info.children[c], *front.children[c], *X.children[c] );

    // Set up a workspace
    // TODO: Only set up a workspace if there is not a parent 
    //       (or a duplicate's parent)
    auto& W = X.work;
    const Int numRHS = X.matrix.Width();
    W.Resize( front.L.Height(), numRHS );
    Matrix<F> WT, WB;
    PartitionDown( W, WT, WB, info.size );
    WT = X.matrix;
    Zero( WB );

    // Multiply against this front
    FrontLowerForwardMultiply( front, W );

    // Update using the children (if they exist)
    for( Int c=0; c<numChildren; ++c )
    {
        auto& childW = X.children[c]->work;
        const Int childSize = info.children[c]->size;
        const Int childHeight = childW.Height();
        const Int childUSize = childHeight-childSize;

        auto childU = childW( IR(childSize,childHeight), IR(0,numRHS) );
        for( Int iChild=0; iChild<childUSize; ++iChild )
        {
            const Int iFront = info.childRelInds[c][iChild]; 
            for( Int j=0; j<numRHS; ++j )
                W.Update( iFront, j, childU.Get(iChild,j) );
        }
        childW.Empty();
    }

    // Store this node's portion of the result
    X.matrix = WT;
}

template<typename F>
inline void LowerForwardMultiply
( const DistSymmNodeInfo& info,
  const DistSymmFront<F>& front, DistMultiVecNode<F>& X )
{
    DEBUG_ONLY(CallStackEntry cse("LowerForwardMultiply"))

    const bool frontIs1D = FrontIs1D( front.type );
    const Grid& grid = ( frontIs1D ? front.L1D.Grid() : front.L2D.Grid() );
    if( front.duplicate != nullptr )
    {
        LowerForwardMultiply( *info.duplicate, *front.duplicate, *X.duplicate );

        const auto& W = X.duplicate->work;
        X.work.LockedAttach( W.Height(), W.Width(), grid, 0, 0, W );
        return;
    }

    const auto& childInfo = *info.child;
    const auto& childFront = *front.child;
    if( childFront.type != front.type )
        LogicError("Incompatible front type mixture");
    LowerForwardMultiply( childInfo, childFront, *X.child );

    // Set up a workspace
    // TODO: Only set up a workspace if there is a parent
    const Int numRHS = X.matrix.Width();
    const Int frontHeight =
      ( frontIs1D ? front.L1D.Height() : front.L2D.Height() );
    auto& W = X.work;
    W.SetGrid( grid );
    W.Resize( frontHeight, numRHS );
    DistMatrix<F,VC,STAR> WT(grid), WB(grid);
    PartitionDown( W, WT, WB, info.size );
    WT = X.matrix;
    Zero( WB );
    mpi::Comm comm = W.DistComm();
    const int commSize = mpi::Size( comm );

    FrontLowerForwardMultiply( front, W );

    // Compute the metadata for transmitting child updates
    X.ComputeCommMeta( info );
    auto& childW = X.child->work;
    auto childU = childW( IR(childInfo.size,childW.Height()), IR(0,numRHS) );
    vector<int> sendSizes(commSize), recvSizes(commSize);
    for( int q=0; q<commSize; ++q )
    {
        sendSizes[q] = X.commMeta.numChildSendInds[q]*numRHS;
        recvSizes[q] = X.commMeta.childRecvInds[q].size()*numRHS;
    }
    DEBUG_ONLY(VerifySendsAndRecvs( sendSizes, recvSizes, comm ))
    vector<int> sendOffs, recvOffs;
    const int sendBufSize = Scan( sendSizes, sendOffs );
    const int recvBufSize = Scan( recvSizes, recvOffs );

    // Pack our child's update
    vector<F> sendBuf( sendBufSize );
    const Int myChild = ( childInfo.onLeft ? 0 : 1 );
    auto packOffs = sendOffs;
    const Int localHeight = childU.LocalHeight();
    for( Int iChildLoc=0; iChildLoc<localHeight; ++iChildLoc )
    {
        const Int iChild = childU.GlobalRow(iChildLoc);
        const Int q = W.RowOwner( info.childRelInds[myChild][iChild] );
        for( Int j=0; j<numRHS; ++j )
            sendBuf[packOffs[q]++] = childU.GetLocal(iChildLoc,j);
    }
    SwapClear( packOffs );
    childW.Empty();
    if( X.child->duplicate != nullptr )
        X.child->duplicate->work.Empty();

    // AllToAll to send and receive the child updates
    vector<F> recvBuf( recvBufSize );
    SparseAllToAll
    ( sendBuf, sendSizes, sendOffs,
      recvBuf, recvSizes, recvOffs, comm );
    SwapClear( sendBuf );
    SwapClear( sendSizes );
    SwapClear( sendOffs );

    // Unpack the child updates
    for( int q=0; q<commSize; ++q )
    {
        const F* recvVals = &recvBuf[recvOffs[q]];
        const auto& recvInds = X.commMeta.childRecvInds[q];
        for( unsigned k=0; k<recvInds.size(); ++k )
            blas::Axpy
            ( numRHS, F(1),
              &recvVals[k*numRHS],     1, 
              W.Buffer(recvInds[k],0), W.LDim() );
    }
    SwapClear( recvBuf );
    SwapClear( recvSizes );
    SwapClear( recvOffs );

    // Unpack the workspace
    X.matrix = WT;
}

template<typename F>
inline void LowerForwardMultiply
( const DistSymmNodeInfo& info,
  const DistSymmFront<F>& front, DistMatrixNode<F>& X )
{
    DEBUG_ONLY(CallStackEntry cse("DistLowerForwardMultiply"))
    const Grid& grid = front.L2D.Grid();
    if( front.duplicate != nullptr )
    {
        LowerForwardMultiply( *info.duplicate, *front.duplicate, *X.duplicate );
        const auto& work = X.duplicate->work;
        X.work.LockedAttach( grid, work );
        return;
    }

    // Set up a workspace
    // TODO: Only set up a workspace if there is a parent
    const Int numRHS = X.matrix.Width();
    const Int frontHeight = front.L2D.Height();
    auto& W = X.work;
    W.SetGrid( grid );
    W.Align( 0, 0 );
    W.Resize( frontHeight, numRHS );
    DistMatrix<F> WT(grid), WB(grid);
    PartitionDown( W, WT, WB, info.size );
    WT = X.matrix;
    Zero( WB );
    mpi::Comm comm = W.DistComm();
    const int commSize = mpi::Size( comm );

    FrontLowerForwardMultiply( front, W );

    // Compute the metadata
    X.ComputeCommMeta( info );
    const auto& childInfo = *info.child;
    auto& childW = X.child->work;
    auto childU = childW( IR(childInfo.size,childW.Height()), IR(0,numRHS) );
    vector<int> sendSizes(commSize), recvSizes(commSize);
    for( int q=0; q<commSize; ++q )
    {
        sendSizes[q] = X.commMeta.numChildSendInds[q];
        recvSizes[q] = X.commMeta.childRecvInds[q].size()/2;
    }
    DEBUG_ONLY(VerifySendsAndRecvs( sendSizes, recvSizes, comm ))
    vector<int> sendOffs, recvOffs;
    const int sendBufSize = Scan( sendSizes, sendOffs );
    const int recvBufSize = Scan( recvSizes, recvOffs );

    // Pack send data
    vector<F> sendBuf( sendBufSize );
    const Int myChild = ( childInfo.onLeft ? 0 : 1 );
    auto packOffs = sendOffs;
    const Int localWidth = childU.LocalWidth();
    const Int localHeight = childU.LocalHeight();
    for( Int iChildLoc=0; iChildLoc<localHeight; ++iChildLoc )
    {
        const Int iChild = childU.GlobalRow(iChildLoc);
        const Int iParent = info.childRelInds[myChild][iChild];
        for( Int jChildLoc=0; jChildLoc<localWidth; ++jChildLoc )
        {
            const Int j = childU.GlobalCol(jChildLoc);
            const int q = W.Owner( iParent, j );
            sendBuf[packOffs[q]++] = childU.GetLocal(iChildLoc,jChildLoc);
        }
    }
    SwapClear( packOffs );
    childW.Empty();
    if( X.child->duplicate != nullptr )
        X.child->duplicate->work.Empty();

    // AllToAll to send and receive the child updates
    vector<F> recvBuf( recvBufSize );
    SparseAllToAll
    ( sendBuf, sendSizes, sendOffs,
      recvBuf, recvSizes, recvOffs, comm );
    SwapClear( sendBuf );
    SwapClear( sendSizes );
    SwapClear( sendOffs );

    // Unpack the child updates (with an Axpy)
    for( int q=0; q<commSize; ++q )
    {
        const auto& recvInds = X.commMeta.childRecvInds[q];
        for( unsigned k=0; k<recvInds.size()/2; ++k )
        {
            const Int iLoc = recvInds[2*k+0];
            const Int jLoc = recvInds[2*k+1];
            W.UpdateLocal( iLoc, jLoc, recvBuf[recvOffs[q]+k] );
        }
    }
    SwapClear( recvBuf );
    SwapClear( recvSizes );
    SwapClear( recvOffs );

    // Store this node's portion of the result
    X.matrix = WT;
}

} // namespace El

#endif // ifndef EL_SPARSEDIRECT_NUMERIC_LOWERMULTIPLY_FORWARD_HPP