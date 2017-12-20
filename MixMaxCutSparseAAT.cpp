#include<iostream>
#include<vector>
#include<string>
#include<cmath>
#include <omp.h>
#include <algorithm>
#include "IQPSolve.h"
#include "SparseAATFunc.h"
#include "mex.h"

using namespace std;


void usage()
{
	mexErrMsgTxt("Usage:function z = MixMaxCutSparseAAT(A_sparse, sdp_rank, iter)");
}


void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[]){
	if (nrhs!=3)
		usage();
	int N = mxGetM(prhs[0]);
	int D = mxGetN(prhs[0]);
	
	double* A_val = mxGetPr(prhs[0]);
	size_t* A_ir =  mxGetIr(prhs[0]);
	size_t* A_jc =  mxGetJc(prhs[0]);
	
	int SDP_K = (int) *(mxGetPr(prhs[1]));
	int iter = (int) *(mxGetPr(prhs[2]));
	//V,alpha,objGCD
	plhs[0] = mxCreateDoubleMatrix(N,1,mxREAL);
	
	SparseMat A;
	A.resize(N);
	for(int j=0;j<D;j++){
		size_t start = A_jc[j];
		size_t end = A_jc[j+1];
		for(int r=start;r<end;r++){
				size_t i = A_ir[r];
				double v = A_val[r];
				A[i].push_back(make_pair(j,v));
		}
	}
	
	ExtensibleFunction* fun = new SparseAATFunc(N, D, SDP_K, A);
	IQPSolve* solver = new IQPSolve();
	solver->setIter(iter);
	Vector z;
	solver->solve(fun, z);
	delete fun;
	delete solver;
	
	double* z_out = mxGetPr(plhs[0]);
	for(int i=0;i<N;i++)
		z_out[i] = z[i];
}

