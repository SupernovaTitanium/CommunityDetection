#include<iostream>
#include<vector>
#include<string>
#include<cmath>
#include <omp.h>
#include <algorithm>
#include "IQPSolve.h"
#include "SimpleFun.h"
#include "mex.h"

using namespace std;


void usage()
{
	mexErrMsgTxt("Usage:function z = MixMaxCutDense(C, sdp_rank, sdp_iter)\n" 
							 "\tSolve max_{ z in{0,1}^N } tr(z'Cz) with Mixing SDP solver and randomized rounding.");
}


void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[]){
	double* Cin;
	int N,D;
	if (nrhs!=3)
		usage();
	Cin = mxGetPr(prhs[0]);
	N= mxGetM(prhs[0]);
	int N2 = mxGetN(prhs[0]);
	if( N != N2 ){
			usage(); //print and exit
	}
	
	int SDP_K = (int) *(mxGetPr(prhs[1]));
	int iter = (int) *(mxGetPr(prhs[2]));

	plhs[0] = mxCreateDoubleMatrix(N,1,mxREAL);
	
	Matrix C;
	C.resize(N);
	for(int i=0;i<N;i++){
		C[i].resize(N);
		for(int j=0;j<N;j++){
			C[i][j] = Cin[ j*N + i ] ;
		}
	}
	
	ExtensibleFunction* fun = new SimpleFun(N, SDP_K, C);
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

