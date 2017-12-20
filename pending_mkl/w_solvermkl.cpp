#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdlib.h>  
#include <vector>
#include <algorithm>
#include <math.h> 
#include <set>
#include <omp.h>
#include <string.h>
#include <unistd.h>
#include "mex.h"
#include <lbfgs.h>
#include <matrix.h>
#include <time.h>
#include "sparsetype.h"
#include <omp.h>
#include "mkl.h"
#include "mkl_cblas.h"
#include "mkl_types.h"
#define log_adjust 1e-200
#define exp_max 300
#define exp_min -300
#define core 120
#define parelleln 3000
using namespace std;
static double safe_logistic(double x){
    if( x > exp_max){      
      return 1.0;
    }
    else if(x < exp_min){
      return 0.0;
    }
    else
      return (1.0/(1.0+exp(-x)));
}
static double safe_log(double x){
    if(x < log_adjust) {
        printf("log(0) occurs\n");
        return log(log_adjust);
      }
    return log(x);
}
 static lbfgsfloatval_t evaluate_dense(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
    )
{   

    lbfgsfloatval_t fx = 0.0;
    sp_matrix lZW0;   
    sp_matrix lZW1;  
    sp_matrix_Z* const_para = (sp_matrix_Z*)instance;
    lZW0.nnz = const_para->R0.nnz;
    lZW0.n_size = const_para->R0.n_size;
    lZW0.m_size = const_para->R0.m_size;
    lZW0.col_ptr = const_para->R0.col_ptr;
    lZW0.row_ptr = const_para->R0.row_ptr;
    lZW0.data = (double*) malloc(sizeof(double)*lZW0.nnz);
    lZW1.nnz = const_para->R1.nnz;
    lZW1.n_size = const_para->R1.n_size;
    lZW1.m_size = const_para->R1.m_size;
    lZW1.col_ptr = const_para->R1.col_ptr;
    lZW1.row_ptr = const_para->R1.row_ptr;
    lZW1.data = (double*) malloc(sizeof(double)*lZW1.nnz);
    double lambda = const_para->lambda;
    // #pragma omp parallel for
    for(int i=0;i<lZW0.n_size;i++){
      for(int j=lZW0.col_ptr[i];j<lZW0.col_ptr[i+1];j++){       
        double temp=0;
        for(int k=0;k<const_para->K;k++){
          //temp+=const_para->Z[k*lZW0.n_size+lZW0.row_ptr[j]]*x[i*const_para->K+k];
          temp+=const_para->Zt[lZW0.row_ptr[j]*const_para->K+k]*x[i*const_para->K+k];
        }
        lZW0.data[j] = safe_logistic(temp);
        
        fx += -safe_log(1.0-safe_logistic(temp));
      }
    }
    // #pragma omp parallel for
    for(int i=0;i<lZW1.n_size;i++){
      for(int j=lZW1.col_ptr[i];j<lZW1.col_ptr[i+1];j++){
        double temp=0;
        for(int k=0;k<const_para->K;k++){
          //temp+=const_para->Z[k*lZW1.n_size+lZW1.row_ptr[j]]*x[i*const_para->K+k];
          temp+=const_para->Zt[lZW1.row_ptr[j]*const_para->K+k]*x[i*const_para->K+k];
        }
        lZW1.data[j] = -safe_logistic(-temp);
        
        fx += -safe_log(safe_logistic(temp));
      }
    }
    // #pragma omp parallel for
    for(int i=0;i<const_para->N;i++){
      for(int j=0;j<const_para->K;j++){
        double temp=0;
        for(int k=lZW0.col_ptr[i];k<lZW0.col_ptr[i+1];k++){
          temp+= const_para->Z[j*const_para->N+lZW0.row_ptr[k]]*lZW0.data[k];
        }
        for(int k=lZW1.col_ptr[i];k<lZW1.col_ptr[i+1];k++){
          temp+= const_para->Z[j*const_para->N+lZW1.row_ptr[k]]*lZW1.data[k];
        }
        g[j+i*const_para->K]=temp+lambda*x[j+i*const_para->K];
        fx += 0.5*lambda*x[j+i*const_para->K]*x[j+i*const_para->K];
      }
    }
    return fx;
}
static lbfgsfloatval_t evaluate_sparse(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
    )
{   

    lbfgsfloatval_t fx = 0.0;
    sp_matrix lZW0;   
    sp_matrix lZW1;  
    sp_matrix_Zs* const_para = (sp_matrix_Zs*)instance;
    lZW0.nnz = const_para->R0.nnz;
    lZW0.n_size = const_para->R0.n_size;
    lZW0.m_size = const_para->R0.m_size;
    lZW0.col_ptr = const_para->R0.col_ptr;
    lZW0.row_ptr = const_para->R0.row_ptr;
    lZW0.data = (double*) malloc(sizeof(double)*lZW0.nnz);
    lZW1.nnz = const_para->R1.nnz;
    lZW1.n_size = const_para->R1.n_size;
    lZW1.m_size = const_para->R1.m_size;
    lZW1.col_ptr = const_para->R1.col_ptr;
    lZW1.row_ptr = const_para->R1.row_ptr;
    lZW1.data = (double*) malloc(sizeof(double)*lZW1.nnz);
    double lambda = const_para->lambda;   
    // #pragma omp parallel for 
    for(int i=0;i<lZW0.n_size;i++){
      for(int j=lZW0.col_ptr[i];j<lZW0.col_ptr[i+1];j++){    
        double temp=0;
        for(int k=const_para->Z.col_ptr[i];k<const_para->Z.col_ptr[i+1];k++){
          temp+=const_para->Z.data[k]*x[lZW0.row_ptr[j]*const_para->K+const_para->Z.row_ptr[k]];
        }
        lZW0.data[j] = safe_logistic(temp);
        
        fx += -safe_log(1.0-safe_logistic(temp));
      }
    }
    // #pragma omp parallel for    
    for(int i=0;i<lZW1.n_size;i++){
      for(int j=lZW1.col_ptr[i];j<lZW1.col_ptr[i+1];j++){
        double temp=0;
        for(int k=const_para->Z.col_ptr[i];k<const_para->Z.col_ptr[i+1];k++){
          temp+=const_para->Z.data[k]*x[lZW1.row_ptr[j]*const_para->K+const_para->Z.row_ptr[k]];
        }
        lZW1.data[j] = -safe_logistic(-temp);
        
        fx += -safe_log(safe_logistic(temp));
      }
    }
    // #pragma omp parallel 
    // {
		for(int i=0;i<const_para->N;i++){

	      for(int j=0;j<const_para->K;j++){

	        g[j+i*const_para->K]=lambda*x[j+i*const_para->K];
	        fx += 0.5*lambda*x[j+i*const_para->K]*x[j+i*const_para->K];
	       
	      }
	    }
	// }
    // #pragma omp parallel for
    for(int i=0;i<const_para->N;i++){
      for(int j=const_para->Z.col_ptr[i];j<const_para->Z.col_ptr[i+1];j++){
        for(int k=lZW0.col_ptr[i];k<lZW0.col_ptr[i+1];k++)
          g[lZW0.row_ptr[k]*const_para->K+const_para->Z.row_ptr[j]]+=const_para->Z.data[j]*lZW0.data[k];
     
         for(int k=lZW1.col_ptr[i];k<lZW1.col_ptr[i+1];k++)
          g[lZW1.row_ptr[k]*const_para->K+const_para->Z.row_ptr[j]]+=const_para->Z.data[j]*lZW1.data[k];
      }
    }

    return fx;

}

static lbfgsfloatval_t evaluate_sparse_p(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
    )
{   

    lbfgsfloatval_t fx = 0.0;
    sp_matrix lZW0;   
    sp_matrix lZW1;  
    sp_matrix_Zs* const_para = (sp_matrix_Zs*)instance;
    lZW0.nnz = const_para->R0.nnz;
    lZW0.n_size = const_para->R0.n_size;
    lZW0.m_size = const_para->R0.m_size;
    lZW0.col_ptr = const_para->R0.col_ptr;
    lZW0.row_ptr = const_para->R0.row_ptr;
    lZW0.data = (double*) malloc(sizeof(double)*lZW0.nnz);
    lZW1.nnz = const_para->R1.nnz;
    lZW1.n_size = const_para->R1.n_size;
    lZW1.m_size = const_para->R1.m_size;
    lZW1.col_ptr = const_para->R1.col_ptr;
    lZW1.row_ptr = const_para->R1.row_ptr;
    lZW1.data = (double*) malloc(sizeof(double)*lZW1.nnz);
    double lambda = const_para->lambda;   

   #pragma omp parallel  if(const_para->N>parelleln) num_threads(core) 
    {    		
    	#pragma omp for schedule(static) reduction( +:fx) 
	    for(int i=0;i<lZW0.n_size;i++){
	      for(int j=lZW0.col_ptr[i];j<lZW0.col_ptr[i+1];j++){    
	        double temp=0;
	        for(int k=const_para->Z.col_ptr[i];k<const_para->Z.col_ptr[i+1];k++){
	          temp+=const_para->Z.data[k]*x[lZW0.row_ptr[j]*const_para->K+const_para->Z.row_ptr[k]];
	        }
	        lZW0.data[j] = safe_logistic(temp);
	        
	        fx += -safe_log(1.0-safe_logistic(temp));
	      }
	    }
	    #pragma omp for schedule(static) reduction( +:fx)     
	    for(int i=0;i<lZW1.n_size;i++){
	      for(int j=lZW1.col_ptr[i];j<lZW1.col_ptr[i+1];j++){
	        double temp=0;
	        for(int k=const_para->Z.col_ptr[i];k<const_para->Z.col_ptr[i+1];k++){
	          temp+=const_para->Z.data[k]*x[lZW1.row_ptr[j]*const_para->K+const_para->Z.row_ptr[k]];
	        }
	        lZW1.data[j] = -safe_logistic(-temp);
	        
	        fx += -safe_log(safe_logistic(temp));
	      }
	    }
	    #pragma omp for schedule(static) reduction( +:fx)   
		for(int i=0;i<const_para->N;i++){
	
	      for(int j=0;j<const_para->K;j++){

	        g[j+i*const_para->K]=lambda*x[j+i*const_para->K];
	        fx += 0.5*lambda*x[j+i*const_para->K]*x[j+i*const_para->K];
	         // printf( "<T:%d> - (%d %d) set g[%d] fx=%f on cpu %d\n", omp_get_thread_num(), i,j,j+i*const_para->K,fx,sched_getcpu() );
	      }
	    }
	   #pragma omp for schedule(static) 
	    for(int i=0;i<const_para->N;i++){
	      for(int j=const_para->Z.col_ptr[i];j<const_para->Z.col_ptr[i+1];j++){
		        for(int k=lZW0.col_ptr[i];k<lZW0.col_ptr[i+1];k++){
		           #pragma omp atomic
		          g[lZW0.row_ptr[k]*const_para->K+const_para->Z.row_ptr[j]]+=const_para->Z.data[j]*lZW0.data[k];
				}	     
		         for(int k=lZW1.col_ptr[i];k<lZW1.col_ptr[i+1];k++){
		           #pragma omp atomic
		          g[lZW1.row_ptr[k]*const_para->K+const_para->Z.row_ptr[j]]+=const_para->Z.data[j]*lZW1.data[k];
		     	}
	    	}
		}
	}
    return fx;

}
 static lbfgsfloatval_t evaluate_dense_p(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
    )
{   

    lbfgsfloatval_t fx = 0.0;
    sp_matrix lZW0;   
    sp_matrix lZW1;  
    sp_matrix_Z* const_para = (sp_matrix_Z*)instance;
    lZW0.nnz = const_para->R0.nnz;
    lZW0.n_size = const_para->R0.n_size;
    lZW0.m_size = const_para->R0.m_size;
    lZW0.col_ptr = const_para->R0.col_ptr;
    lZW0.row_ptr = const_para->R0.row_ptr;
    lZW0.data = (double*) malloc(sizeof(double)*lZW0.nnz);
    lZW1.nnz = const_para->R1.nnz;
    lZW1.n_size = const_para->R1.n_size;
    lZW1.m_size = const_para->R1.m_size;
    lZW1.col_ptr = const_para->R1.col_ptr;
    lZW1.row_ptr = const_para->R1.row_ptr;
    lZW1.data = (double*) malloc(sizeof(double)*lZW1.nnz);
    double lambda = const_para->lambda;
      #pragma omp parallel  if(const_para->N>parelleln) num_threads(core) 
    {    		
    	#pragma omp for schedule(static) reduction( +:fx) 
	    for(int i=0;i<lZW0.n_size;i++){
	      for(int j=lZW0.col_ptr[i];j<lZW0.col_ptr[i+1];j++){       
	        double temp=0;
	        for(int k=0;k<const_para->K;k++){
	          // temp+=const_para->Z[k*lZW0.n_size+lZW0.row_ptr[j]]*x[i*const_para->K+k];
	        	temp+=const_para->Zt[lZW0.row_ptr[j]*const_para->K+k]*x[i*const_para->K+k];
	        }
	        lZW0.data[j] = safe_logistic(temp);
	        
	        fx += -safe_log(1.0-safe_logistic(temp));
	      }
	    }
    #pragma omp for schedule(static) reduction( +:fx) 
	    for(int i=0;i<lZW1.n_size;i++){
	      for(int j=lZW1.col_ptr[i];j<lZW1.col_ptr[i+1];j++){
	        double temp=0;
	        for(int k=0;k<const_para->K;k++){
	          // temp+=const_para->Z[k*lZW1.n_size+lZW1.row_ptr[j]]*x[i*const_para->K+k];
	        	temp+=const_para->Zt[lZW1.row_ptr[j]*const_para->K+k]*x[i*const_para->K+k];
	        }
	        lZW1.data[j] = -safe_logistic(-temp);
	        
	        fx += -safe_log(safe_logistic(temp));
	      }
	    }
  	#pragma omp for schedule(static) reduction( +:fx)
	    for(int i=0;i<const_para->N;i++){
	      for(int j=0;j<const_para->K;j++){
	        double temp=0;
	        for(int k=lZW0.col_ptr[i];k<lZW0.col_ptr[i+1];k++){
	           temp+= const_para->Z[j*const_para->N+lZW0.row_ptr[k]]*lZW0.data[k];
	          //temp+= const_para->Z[lZW0.row_ptr[k]*const_para->K+j]*lZW0.data[k];
	        }
	        for(int k=lZW1.col_ptr[i];k<lZW1.col_ptr[i+1];k++){
	           temp+= const_para->Z[j*const_para->N+lZW1.row_ptr[k]]*lZW1.data[k];
	          //temp+= const_para->Z[lZW1.row_ptr[k]*const_para->K+j]*lZW1.data[k];
	        }
	        g[j+i*const_para->K]=temp+lambda*x[j+i*const_para->K];
	        fx += 0.5*lambda*x[j+i*const_para->K]*x[j+i*const_para->K];
	      }
	    }
   	}
    return fx;
}
 static lbfgsfloatval_t evaluate_dense_mkl(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
    )
{   

    lbfgsfloatval_t fx = 0.0;
    spm_matrix lZW0;   
    spm_matrix lZW1;  
    spm_matrix_Z* const_para = (spm_matrix_Z*)instance;
    lZW0.nnz = const_para->R0.nnz;
    lZW0.n_size = const_para->R0.n_size;
    lZW0.m_size = const_para->R0.m_size;
    lZW0.col_ptr = const_para->R0.col_ptr;
    lZW0.row_ptr = const_para->R0.row_ptr;
    lZW0.data = (double*) malloc(sizeof(double)*lZW0.nnz);
    lZW1.nnz = const_para->R1.nnz;
    lZW1.n_size = const_para->R1.n_size;
    lZW1.m_size = const_para->R1.m_size;
    lZW1.col_ptr = const_para->R1.col_ptr;
    lZW1.row_ptr = const_para->R1.row_ptr;
    lZW1.data = (double*) malloc(sizeof(double)*lZW1.nnz);
    double lambda = const_para->lambda;
    printf("here\n");
    double* trya = (double*)malloc(sizeof(double)*10);
    for(int i=0;i<10;i++){
        trya[i]=i;
    }
    MKL_INT aa=10;
    MKL_INT bb=1;
    double qq = cblas_ddot(aa,trya,bb,trya,bb);
    printf("qq=%f\n",qq); 
    //use as scalar product
    for(int i=0;i<lZW0.n_size;i++){
      for(int j=lZW0.col_ptr[i];j<lZW0.col_ptr[i+1] ;j++){   
        printf("[%d,%d]\n",lZW0.row_ptr[j],i);
        for(int k=0;k<const_para->K;k++){
          // temp+=const_para->Z[k*lZW0.n_size+lZW0.row_ptr[j]]*x[i*const_para->K+k];
            printf("[%p,%p]=[%f,%f]\n",&(const_para->Zt[lZW0.row_ptr[j]*const_para->K+k]),&(x[i*const_para->K+k]),const_para->Zt[lZW0.row_ptr[j]*const_para->K+k],x[i*const_para->K+k]);
            printf("[%f,%f]\n",const_para->Zt[lZW0.row_ptr[j]*const_para->K+k],x[i*const_para->K+k]);
        }
         fprintf(stderr,"K = %d\n",const_para->K);
        double temp=cblas_ddot(const_para->K,&(const_para->Zt[lZW0.row_ptr[j]*const_para->K]),1,&(x[i*const_para->K]),1);
        fprintf(stderr,"temp = %f\n",temp);
        temp=0;
       for(int k=0;k<const_para->K;k++){
          // temp+=const_para->Z[k*lZW0.n_size+lZW0.row_ptr[j]]*x[i*const_para->K+k];
            temp+=const_para->Zt[lZW0.row_ptr[j]*const_para->K+k]*x[i*const_para->K+k];
        }
         printf("temp = %f\n",temp);
        lZW0.data[j] = safe_logistic(temp);        
        fx += -safe_log(1.0-safe_logistic(temp));
      }
    }


    //use as dense matrix copy



    // #pragma omp parallel for
    for(int i=0;i<lZW1.n_size;i++){
      for(int j=lZW1.col_ptr[i];j<lZW1.col_ptr[i+1];j++){
        double temp=cblas_ddot(const_para->K,const_para->Zt+lZW1.row_ptr[j]*const_para->K,1,x+i*const_para->K,1);
        lZW1.data[j] = -safe_logistic(-temp);        
        fx += -safe_log(safe_logistic(temp));
      }
    }
    char        transa, ordering;
    char        matdescra[6];
    transa = 't';
    ordering = 'c';
    matdescra[0] = 'G';
    matdescra[1] = 'l';
    matdescra[2] = 'n';
    matdescra[3] = 'c';
    double alpha=1.0;
    double beta=0.0;
    memset(g,0x00,const_para->K*const_para->N);
    size_t Nv = const_para->N;
    size_t Kv = const_para->K;
    mkl_dcsrmm(&transa, &const_para->N, &const_para->K,&const_para->N, &alpha, matdescra, lZW0.data, lZW0.row_ptr, lZW0.col_ptr,lZW0.col_ptr+1,const_para->Z,&const_para->K,&alpha,g,&const_para->K);
    mkl_dcsrmm(&transa, &const_para->N, &const_para->K,&const_para->N, &alpha, matdescra, lZW1.data, lZW1.row_ptr, lZW1.col_ptr,lZW1.col_ptr+1,const_para->Z,&const_para->K,&beta,g,&const_para->K);
    mkl_dimatcopy (ordering,transa,const_para->N,const_para->K,1.0,g,const_para->N,const_para->K);
    cblas_daxpy (const_para->K*const_para->N,lambda,x,1.0,g,1.0);
    fx=fx+0.5*lambda*cblas_ddot(const_para->K*const_para->N,x,1,x,1);
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
    printf("\n");
    return 0;
}

//only handle sparse now
 void lbfgs_mex_interface_sparseZ(sp_matrix R0,sp_matrix R1,sp_matrix Z,double* W0,double lambda,double* W_opt,double* value,int N,int K){
    int i, ret = 0;
    lbfgsfloatval_t fx;
    lbfgsfloatval_t *x = lbfgs_malloc(N*K);
    lbfgs_parameter_t param;
    sp_matrix_Zs* RZ = (sp_matrix_Zs*)malloc(sizeof(sp_matrix_Zs)*1) ;
    if (x == NULL) {
        printf("ERROR: Failed to allocate a memory block for variables.\n");
      	return;
    }

    /* Initialize the variables as W0 */
    for (i = 0;i < K*N;i++) {
       x[i] = *(W0+i);
    }
    RZ->R0 = R0;
    RZ->R1 = R1;
    RZ->Z = Z;
    RZ->lambda = lambda;
    RZ->K = K;
    RZ->N = N;
    /* Initialize the parameters for the L-BFGS optimization. */
    lbfgs_parameter_init(&param);
    param.epsilon = 1e-2;
    //printf("start lbfgs\n");
    ret = lbfgs(N*K, x, &fx, evaluate_sparse_p, progress, RZ, &param);
    /* Report the result. */
    // printf("L-BFGS optimization terminated with status code = %d\n", ret);
   
    for(int i=0;i<N*K;i++){
      W_opt[i] = x[i];
    }
    *value = fx; 
    lbfgs_free(x);
    free(RZ);
}

void lbfgs_mex_interface_denseZ(spm_matrix R0,spm_matrix R1,double* Z,double* Zt,double* W0,double lambda,double* W_opt,double* value,int N,int K){
    int i, ret = 0;
    lbfgsfloatval_t fx;
    lbfgsfloatval_t *x = lbfgs_malloc(N*K);
    lbfgs_parameter_t param;
    spm_matrix_Z* RZ = (spm_matrix_Z*)malloc(sizeof(spm_matrix_Z)*1) ;
    if (x == NULL) {
        printf("ERROR: Failed to allocate a memory block for variables.\n");
        return;
    }

    /* Initialize the variables as W0 */
    for (i = 0;i < K*N;i++) {
       x[i] = *(W0+i);
    }
    RZ->R0 = R0;
    RZ->R1 = R1;
    RZ->Z = Z;
    RZ->Zt = Zt;
    RZ->lambda = lambda;
    RZ->K = K;
    RZ->N = N;
    /* Initialize the parameters for the L-BFGS optimization. */
    lbfgs_parameter_init(&param);
    param.epsilon = 1e-2;
    //printf("start lbfgs\n");
    ret = lbfgs(N*K, x, &fx, evaluate_dense_mkl, progress, RZ, &param);
    /* Report the result. */
    // printf("L-BFGS optimization terminated with status code = %d\n", ret);    
    for(int i=0;i<N*K;i++){
      W_opt[i] = x[i];
    }
   
    *value = fx; 
    lbfgs_free(x);
    free(RZ);
}
void usage()
{
	mexErrMsgTxt("Usage:[sparse] function [W_o,value] = w_solvermkl(R0_sp',R1_sp',Z'_sparse,W0,lambda)\nUsage:[dense]  function [W_o,value] = w_solver(R0_sp,R1_sp,Z,W0,lambda,Z");
	
}
void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[]){
	

	double lambda;
	double* W0;
	double* Z;
	double* Zt;
    spm_matrix Zs;
	int N,K;
	int nnz,m_size,n_size;
	size_t* col_ptr,*row_ptr;
	double* data;
	if (nrhs < 5 || nrhs > 6)
		usage();

  //deal with function overload

  //
    double* trya = (double*)malloc(sizeof(double)*10);
    for(int i=0;i<10;i++){
        trya[i]=i;
    }
    MKL_INT aa=10;
    MKL_INT bb=1;
    printf("%d\n",sizeof(MKL_INT)); 
    double qq = cblas_ddot(aa,trya,bb,trya,bb);
    printf("qq=%f\n",qq); 
  plhs[1]=mxCreateDoubleMatrix(1,1,mxREAL);
  if (!(mxIsSparse(prhs[2]))){
        // printf( "The input Z is dense and N*K\n");


		N = mxGetN(prhs[4]);
		K = mxGetM(prhs[4]);
		W0 = mxGetPr(prhs[4]);
		lambda = mxGetScalar(prhs[5]);
	 
		data = mxGetPr(prhs[0]);
		col_ptr = mxGetJc(prhs[0]);
		row_ptr = mxGetIr(prhs[0]);
		m_size = mxGetM(prhs[0]);
		n_size = mxGetN(prhs[0]);
		nnz = col_ptr[n_size];

		sp_matrix R0_sp;
        spm_matrix R0m_sp;

		R0_sp.nnz = nnz;
		R0_sp.row_ptr = row_ptr;
		R0_sp.col_ptr = col_ptr;
		R0_sp.data = data;
		R0_sp.m_size = m_size;
		R0_sp.n_size = n_size;

        R0m_sp.nnz = nnz;
        R0m_sp.row_ptr = (MKL_INT*)malloc(sizeof(MKL_INT)*nnz);
        R0m_sp.col_ptr = (MKL_INT*)malloc(sizeof(MKL_INT)*(n_size+1));
        for(int i=0;i<nnz;i++){
            R0m_sp.row_ptr[i]=static_cast<int>(R0_sp.row_ptr[i]);
        }
        for(int i=0;i<(n_size+1);i++){
            R0m_sp.col_ptr[i]=static_cast<int>(R0_sp.col_ptr[i]);
        }
        R0m_sp.data = data;
        R0m_sp.m_size = m_size;
        R0m_sp.n_size = n_size;





		data = mxGetPr(prhs[1]);
		col_ptr = mxGetJc(prhs[1]);
		row_ptr = mxGetIr(prhs[1]);
		m_size = mxGetM(prhs[1]);
		n_size = mxGetN(prhs[1]);
		nnz = col_ptr[n_size];
		
		sp_matrix R1_sp;
        spm_matrix R1m_sp;

		R1_sp.nnz = nnz;
		R1_sp.row_ptr = row_ptr;
		R1_sp.col_ptr = col_ptr;
		R1_sp.data = data;
		R1_sp.m_size = m_size;
		R1_sp.n_size = n_size;


        R1m_sp.nnz = nnz;
        R1m_sp.row_ptr = (MKL_INT*)malloc(sizeof(MKL_INT)*nnz);
        R1m_sp.col_ptr = (MKL_INT*)malloc(sizeof(MKL_INT)*(n_size+1));
        for(int i=0;i<nnz;i++){
            R1m_sp.row_ptr[i]=static_cast<int>(R1_sp.row_ptr[i]);
        }
        for(int i=0;i<(n_size+1);i++){
            R1m_sp.col_ptr[i]=static_cast<int>(R1_sp.col_ptr[i]);
        }
        R1m_sp.data = data;
        R1m_sp.m_size = m_size;
        R1m_sp.n_size = n_size;

        Z = mxGetPr(prhs[2]);
        plhs[0]=mxCreateDoubleMatrix(K,N,mxREAL);
        Zt = mxGetPr(prhs[3]);
        lbfgs_mex_interface_denseZ(R0m_sp,R1m_sp,Z,Zt,W0,lambda,mxGetPr(plhs[0]),mxGetPr(plhs[1]),N,K);
  }else{  
    // printf( "The input Z is sparse and should be sent as transpose(same as R0,R1)\n");  
  		N = mxGetN(prhs[3]);
		K = mxGetM(prhs[3]);
		W0 = mxGetPr(prhs[3]);
		lambda = mxGetScalar(prhs[4]);
	 
		data = mxGetPr(prhs[0]);
		col_ptr = mxGetJc(prhs[0]);
		row_ptr = mxGetIr(prhs[0]);
		m_size = mxGetM(prhs[0]);
		n_size = mxGetN(prhs[0]);
		nnz = col_ptr[n_size];

		sp_matrix R0_sp;
		R0_sp.nnz = nnz;
		R0_sp.row_ptr = row_ptr;
		R0_sp.col_ptr = col_ptr;
		R0_sp.data = data;
		R0_sp.m_size = m_size;
		R0_sp.n_size = n_size;
		data = mxGetPr(prhs[1]);
		col_ptr = mxGetJc(prhs[1]);
		row_ptr = mxGetIr(prhs[1]);
		m_size = mxGetM(prhs[1]);
		n_size = mxGetN(prhs[1]);
		nnz = col_ptr[n_size];
		
		sp_matrix R1_sp;
		R1_sp.nnz = nnz;
		R1_sp.row_ptr = row_ptr;
		R1_sp.col_ptr = col_ptr;
		R1_sp.data = data;
		R1_sp.m_size = m_size;
		R1_sp.n_size = n_size;
    data = mxGetPr(prhs[2]);
    col_ptr = mxGetJc(prhs[2]);
    row_ptr = mxGetIr(prhs[2]);
    m_size = mxGetM(prhs[2]);
    n_size = mxGetN(prhs[2]);
    nnz = col_ptr[n_size];
    
    sp_matrix Zs;
    Zs.nnz = nnz;
    Zs.row_ptr = row_ptr;
    Zs.col_ptr = col_ptr;
    Zs.data = data;
    Zs.m_size = m_size;
    Zs.n_size = n_size;
    plhs[0]=mxCreateDoubleMatrix(K,N,mxREAL);
    lbfgs_mex_interface_sparseZ(R0_sp,R1_sp,Zs,W0,lambda,mxGetPr(plhs[0]),mxGetPr(plhs[1]),N,K);
  }  
}