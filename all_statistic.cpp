// Use for fast compute intersting measure. 
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

using namespace std;

int* table(double threshold,double* R_test,double* R_guess,double* R,int N){
	int *result=new int[5];
	//TP FP FN TN all
	memset(result, 0, sizeof(int) * 5);
	int link;
	for(int i=0;i<N*N;i++){
		if(R_test[i] == 0)
			continue;
	    link = 0;
	    if(R_guess[i] >= threshold)
	    	link = 1;
	    if(link == 1 && R[i]==1)
	    	result[0]+=1;
	    if(link == 1 && R[i]==0)
	    	result[1]+=1;
	    if(link == 0 && R[i]==1)
	    	result[2]+=1;
	    if(link == 0 && R[i]==0)
	    	result[3]+=1;
	}
	result[4]=result[0]+result[1]+result[2]+result[3];
	return result;

}
bool comparepair(std::pair<double,int> b1, std::pair<double,int> b2)
{
    return b1.first >= b2.first;
}
void statistic(double* R,double* R_test,double* R_guess,double* Z_true,double* threshold,int N,int K,int level,double* acc,double* rec,double* pre,double* F1,double* F1_2,double* auc)
{
  	// printf("fuck\n");	
	for(int i=0;i<level;i++){
		double level_val = threshold[i];
		int* result = table(level_val,R_test,R_guess,R,N);
		//acc = TP+TN/all  rec =TP/TP+FN   pre = TP/TP+FP  F1=   F2 = 
		//return;
		acc[i] = (double)(result[0]+result[3])/(double)result[4];
		rec[i] = (double)(result[0])/(double)(result[0]+result[2]);
		pre[i] = (double)(result[0])/(double)(result[0]+result[1]);
		F1[i] =  (double)(result[0])*2/(double)(2*result[0]+result[1]+result[2]);
		F1_2[i]= ((double)(result[0])*2/(double)(2*result[0]+result[1]+result[2])+(double)(result[3]*2)/(double)(2*result[3]+result[1]+result[2]))/2;
	}
	//computing auc;
	vector<std::pair<double, int> > scribe;
	int num_link=0,p_link=0,n_link=0;
    // retrieve the test links (true label & predict score)
    // printf("fuck\n");
	for(int i=0;i<N*N;i++){
		// printf("(%d,%d,%d)=(%d,%d,%f,%f)",N,K,i,p_link,n_link,R_guess[i],R[i]);
		// fflush(stdout);
		if(R_test[i] == 0)
			continue;
		scribe.push_back(make_pair(R_guess[i],R[i]));
		num_link++;
		p_link += R[i];
		n_link += (1-R[i]);

	}
	// printf("fuck\n");
    // sort the score & labels in descending order.
    std::stable_sort(scribe.begin(),scribe.end(), comparepair);
    double *tp = new double[num_link];
    double *fp = new double[num_link];
    int span_pos = 0;
    for(int i=0;i<num_link;i++){
    	span_pos += scribe[i].second;
    	// printf("value=%f,link=%d\n",scribe[i].first,scribe[i].second);
    	// printf("span_pos=%d\n",span_pos);
    	// printf("(%d,%d)\n",p_link,n_link);
    	if(p_link != 0)
    		tp[i] = (double)(span_pos)/(double)(p_link);
    	else
    		tp[i] = 1.0;
    	if(n_link != 0)
   			fp[i] = (double)(i-span_pos+1.0)/(double)(n_link);
    	else
    		fp[i] = 1.0;
    	// printf("(%f,%f)\n",tp[i],fp[i]);
    }
    // printf("fuck\n");
    double pauc = 0;
    for (int i=1;i<num_link;i++){
       	pauc = pauc + (fp[i]-fp[i-1])*(tp[i]+tp[i-1])/2;
    }
    // printf("%f \n",pauc);
    *auc=pauc;
}
void usage()
{
	mexErrMsgTxt("Usage:function [acc rec prec F1 F1_2 auc] = all_statistic(R,R_test,R_guess,Z_true,beta,threshold)");
}
void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[]){
	double *R,*R_test,*Z_true;
	double *R_guess,*values,*threshold;
	int N,K,level;
	double beta;
	if (nrhs!=6)
		usage();
	R = mxGetPr(prhs[0]);
	N = mxGetM(prhs[0]);
	R_test = mxGetPr(prhs[1]);
	R_guess = mxGetPr(prhs[2]);
	Z_true = mxGetPr(prhs[3]);
	values = mxGetPr(prhs[4]);
	beta = values[0];
	threshold = mxGetPr(prhs[5]);
    K = mxGetN(prhs[3]);
    level = mxGetN(prhs[5]);
    //V,alpha,objGCD
	plhs[0]=mxCreateDoubleMatrix(level,1,mxREAL);
	plhs[1]=mxCreateDoubleMatrix(level,1,mxREAL);
	plhs[2]=mxCreateDoubleMatrix(level,1,mxREAL);
	plhs[3]=mxCreateDoubleMatrix(level,1,mxREAL);
	plhs[4]=mxCreateDoubleMatrix(level,1,mxREAL);
	plhs[5]=mxCreateDoubleMatrix(1,1,mxREAL);
	statistic(R,R_test,R_guess,Z_true,threshold,N,K,level,mxGetPr(plhs[0]),mxGetPr(plhs[1]),mxGetPr(plhs[2]),mxGetPr(plhs[3]),mxGetPr(plhs[4]),mxGetPr(plhs[5]));
}