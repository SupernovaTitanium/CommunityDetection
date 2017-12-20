#include<iostream>
#include<vector>
#include<string>
#include<cmath>
#include <omp.h>
#include <algorithm>
#include "IQPSolve.h"
#include "SparseFunc.h"
#include "LowRankFunc.h"
#include "CompositeFunc.h"
#include "mex.h"

using namespace std;


void usage()
{
	mexErrMsgTxt("Usage:function z = MixMaxCutComposite(C_sparse, b, L, sdp_rank, iter)");
}


void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[]){
	
	if (nrhs!=5){
		usage();
	}

	int N = mxGetM(prhs[0]);
	int N2 = mxGetN(prhs[0]);
	int N3 = mxGetM(prhs[2]);
	int D = mxGetN(prhs[2]);
	if( N2 != N || N3 != N  ){
			mexErrMsgTxt("Argument 1 & 3 must be N*N and N*D matrices respectively.");
	}
	
	//Construct Sparse Matrix S
	SparseMat S;
	double* S_val = mxGetPr(prhs[0]);
	size_t* S_ir =  mxGetIr(prhs[0]);
	size_t* S_jc =  mxGetJc(prhs[0]);
	S.resize(N);
	for(int j=0;j<N;j++){
		size_t start = S_jc[j];
		size_t end = S_jc[j+1];
		for(int r=start;r<end;r++){
				size_t i = S_ir[r];
				double v = S_val[r];
				S[i].push_back(make_pair(j,v));
		}
	}
	
	double b = (double) *(mxGetPr(prhs[1]));
	//Construct Lowrank A
	Matrix A;
	double* Ain = mxGetPr(prhs[2]);
	A.resize(N);
	for(int i=0;i<N;i++){
		A[i].resize(D);
		for(int j=0;j<D;j++){
			A[i][j] = Ain[ j*N + i ] ;
		}
	}
	
	int SDP_K = (int) *(mxGetPr(prhs[3]));
	int iter = (int) *(mxGetPr(prhs[4]));
	
	//Output: z
	plhs[0] = mxCreateDoubleMatrix(N,1,mxREAL);
	
	//cerr << "N=" << N << ", D=" << D << ", K=" << SDP_K << ", b=" << b << ", A[0,0]=" << A[0][0] << endl;
	//Construct Composite Function: S + b*AA'
	ExtensibleFunction* S_fun = new SparseFunc(N, SDP_K, S);
	ExtensibleFunction* AAT_fun = new LowRankFunc( N, D, SDP_K, A );
	ExtensibleFunction* fun = new CompositeFunc( N, SDP_K, 1.0, S_fun, b, AAT_fun );
	
	IQPSolve* solver = new IQPSolve();
	solver->setIter(iter);
	Vector z;
	solver->solve(fun, z);
	delete S_fun;
	delete AAT_fun;
	delete fun;
	delete solver;

	double* z_out = mxGetPr(plhs[0]);
	for(int i=0;i<N;i++)
		z_out[i] = z[i];
}

