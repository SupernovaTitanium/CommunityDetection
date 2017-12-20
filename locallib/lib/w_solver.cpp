//use as w solver at each iteration
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdlib.h>  
#include <vector>
#include <algorithm>
#include <math.h> 
#include <set>
#include <omp.h>
#include <unistd.h>
#include "mex.h"
#include <stdio.h>
#include <lbfgs.h>
#include <matrix.h>

using namespace std;
//use compressed columned format 
struct sp_matrix{
	int* row_ptr;
	int* col_ptr;
	int nnz;
	int n_size;
	int m_size;
	double* data;
};
struct sp_matrix_Z{
	sp_matrix R;
	double* Z;
	double  lambda;
	int K;
};

static lbfgsfloatval_t evaluate(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
    )
{
    int i;
    lbfgsfloatval_t fx = 0.0;
    //first compute ZW O(K)*(nnz)
    //then calculate Z'L(ZW)+lambda(W)
    sp_matrix lZW;
    sp_matrix_Z const_para = *(*sp_matrix_Z)instance;
    lZW.nnz = const_para.R.nnz;
    lZW.n_size = const_para.R.n_size;
    lZW.m_size = const_para.R.m_size;
    lZW.col_ptr = const_para.R.col_ptr;
    lZW.row_ptr = const_para.R.row_ptr;

    double lambda = const_para.lambda;
    for(int i=0;i<lZW.n_size;i++){
    	for(int j=lZW.col_ptr[i];j<lZW.col_ptr[i+1]-1;j++){
    			//i,j is the index now
    		double temp=0;
    		for(int k=0;k<const_para.K;k++){
    			temp+=const_para.Z[k*lZW.n_size+i]*x[lZW.row_ptr[j]*const_para.K+k];
    		}
    		lZW.data[j] = (-const_para.R.data[j]*exp(-temp)+1.0-const_para.R.data[j])/(1+exp(-temp));
    		fx += -const_para.R.data[j]*log(1.0/(1.0+exp(lZW.data[j])))-(1-const_para.R.data[j])*log(1.0-(1.0/(1.0+exp(lZW.data[j]))));
    	}
    }
  	for(int i=0;i<const_para.R.n_size;i++){
  		for(int j=0;j<const_para.K;j++){
  			double temp=0;
  			for(int k=lZW.col_ptr[i];k<lZW.col_ptr[i+1]-1;k++){
  				temp+= const_para.Z[j*n+lZW.row_ptr[k]]*lZW.data[k];
  			}
  			g[j+i*const_para.K]=temp+lambda*x[j+i*const_para.K];
  			fx += 0.5*lambda*x[j+i*const_para.K]*x[j+i*const_para.K];
  		}
  	}
    return fx;
}
static int progress(
    void *instance,
    const lbfgsfloatval_t *x,
    const lbfgsfloatval_t *g,
    const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm,
    const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step,
    int n,
    int k,
    int ls
    )
{
    printf("Iteration %d:\n", k);
    printf("  fx = %f\n", fx);
    return 0;
}

//only handle sparse now
void lbfgs_mex_interface(sp_matrix R,double* Z,double* W0,double lambda,double* W_opt,int N,int K){
    int i, ret = 0;
    lbfgsfloatval_t fx;
    lbfgsfloatval_t *x = lbfgs_malloc(N*K);
    lbfgs_parameter_t param;
    sp_matrix_Z* RZ;
    if (x == NULL) {
        printf("ERROR: Failed to allocate a memory block for variables.\n");
        return 1;
    }
    /* Initialize the variables as W0 */
    for (i = 0;i < K*N;i++) {
       x[i] = *(W0+i);
    }
    RZ->R = R;
    RZ->Z = Z;
    RZ->lambda = lambda;
    RZ->K = K;
    /* Initialize the parameters for the L-BFGS optimization. */
    lbfgs_parameter_init(&param);
    ret = lbfgs(N*K, x, &fx, evaluate, progress, RZ, &param);
    /* Report the result. */
    printf("L-BFGS optimization terminated with status code = %d\n", ret);
    lbfgs_free(x);
}

void usage()
{
	mexErrMsgTxt("Usage:function [W_opt] = w_solver(R_sp,Z,W0,lambda)");
}
void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[]){
	
	//  (Warning!)  will save  mwSize as Int 
	double lambda;
	double* W0;
	double* Z;
	int N,K;
	int nnz,m_size,n_size;
	size_t* col_ptr,*row_ptr;
	double* data;
	if (nrhs!=4)
		usage();
	Z = mxGetPr(prhs[1]);
	N = mxGetM(prhs[1]);
	K = mxGetPr(prhs[1]);
	W0 = mxGetPr(prhs[2]);
	lambda = mxGetScalar(prhs[3]);
	//handling sparse matrix 
	data = mxGetPr(prhs[0]);
	col_ptr = mxGetJc(prhs[0]);
	row_ptr = mxGetIr(prhs[0]);
	m_size = mxGetM(prhs[0]);
	n_size = mxGetN(prhs[0]);
	nnz = col_ptr[n_size];
	//unit test
	for(int i=0;i<nnz;i++){
		printf("%lf\n",data[i]);
	}
	for(int i=0;i<nnz;i++){
		printf("%u\n",row_ptr[i]);
	}
	for(int i=0;i<n_size+1;i++){
		printf("%u\n",col_ptr[i]);
	}
	printf("(%d,%d,%d)\n",m_size,n_size,nnz);
	sp_matrix R_sp;
	R_sp.nnz = nnz;
	R_sp.row_ptr = row_ptr;
	R_sp.col_ptr = col_ptr;
	R_sp.data = data;
	R_sp.m_size = m_size;
	R_sp.n_size = n_size;
	plhs[0]=mxCreateDoubleMatrix(K,N,mxREAL);
	lbfgs_mex_interface(R_sp,Z,W0,lambda,mxGetPr(plhs[0]),N,K);
}