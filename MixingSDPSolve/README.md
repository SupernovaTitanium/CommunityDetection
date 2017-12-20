# MixingSDPSolve

To Compile: (in matlab)
	
	mex MixMaxCutDense.cpp 
	mex MixMaxCut.cpp
	mex -largeArrayDims MixMaxCutSparse.cpp
	mex -largeArrayDims MixMaxCutComposite.cpp
	mex -largeArrayDims MixMaxCutSparseAAT.cpp

Usage of MixMaxCutDense (in matlab):

	z = MixMaxCutDense( C, sdp_rank, iter );
	
	(find max_{z \in \{0,1\}} tr(z'Cz))



Usage of MixMaxCut (in matlab):

	z = MixMaxCut( A, sdp_rank, iter );

	(find max_{z \in \{0,1\}} tr(z'AA'z)))



Usage of MixMaxCutSparse (in matlab):
	
	Usage:function z = MixMaxCutSparse(C_sparse, sdp_rank, iter)

	(find max_{z \in \{0,1\} } < C_sparse , zz'> )



Usage of MixMaxCutComposite (in matlab):
	
	z = MixMaxCutComposite(C_sparse, b, L, sdp_rank, iter)

	(find max_{z \in \{0,1\} } < C_sparse + b*LL' , zz'> )



Usage of MixMaxCutSparseAAT (in matlab):

	z = MixMaxCutSparseAAT( A_sp, sdp_rank, iter );

	(  find max_{z \in \{0,1\} } tr(z' A_sp A_sp' z))  )
