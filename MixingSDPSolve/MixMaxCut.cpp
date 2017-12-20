#include<iostream>
#include<vector>
#include<string>
#include<cmath>
#include <omp.h>
#include <algorithm>
#include "IQPSolve.h"
#include "LowRankFunc.h"
#include "mex.h"

using namespace std;


void usage()
{
	mexErrMsgTxt("Usage:function z = MixMaxCut(A, sdp_rank, sdp_iter)\n" 
							 "\tSolve max_{ z in{0,1}^N } tr(z'AA'z) with Mixing SDP solver with rounding.");
}


void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[]){
	double* Ain;
	int N,D;
	if (nrhs!=3)
		usage();
	Ain = mxGetPr(prhs[0]);
	N= mxGetM(prhs[0]);
	D = mxGetN(prhs[0]);
	
	int SDP_K = (int) *(mxGetPr(prhs[1]));
	int iter = (int) *(mxGetPr(prhs[2]));
	//V,alpha,objGCD
	plhs[0] = mxCreateDoubleMatrix(N,1,mxREAL);
	//coordinate_solver(R,Z,max_iter,lambda,mxGetPr(plhs[0]),mxGetPr(plhs[1]),mxGetPr(plhs[2]),n,d,k);  
	
	Matrix A;
	A.resize(N);
	for(int i=0;i<N;i++){
		A[i].resize(D);
		for(int j=0;j<D;j++){
			A[i][j] = Ain[ j*N + i ] ;
		}
	}
	
	ExtensibleFunction* fun = new LowRankFunc(N, D, SDP_K, A);
	IQPSolve* solve = new IQPSolve();
	solve->setIter(iter);
	Vector z;
	solve->solve(fun, z);
	delete fun;
	delete solve;
	
	double* z_out = mxGetPr(plhs[0]);
	for(int i=0;i<N;i++)
		z_out[i] = z[i];
}

