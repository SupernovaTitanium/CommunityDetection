#ifndef UTIL
#define UTIL

#include<cmath>
#include<vector>
#include<map>
#include<string>
#include<cstring>
#include<stdlib.h>
#include<fstream>
#include<iostream>
#include<algorithm>
#include<omp.h>
#include<unordered_map>
#include<time.h>
#include<tuple>
#include<cassert>
#include<limits.h>
#include <random>
using namespace std;

typedef vector<double> Vector;
typedef vector<Vector> Matrix;
typedef vector<pair<int,double> > SparseVec;
typedef vector<SparseVec> SparseMat;
typedef unordered_map<int,double> HashVec;
typedef double Float;
const int LINE_LEN = 100000000;
const int FNAME_LEN = 1000;
const int INF = INT_MAX;

#define EPS 1e-12
#define INFI 1e10

int seed = 1;
default_random_engine generator(seed);
normal_distribution<double> normal(0.0,1.0);
double randn(){
	return normal(generator);
}

void randn(Vector& v, int K){
	v.resize(K);
	for(int i=0;i<v.size();i++)
		v[i] = randn();
}

class ScoreComp{
	
	public:
	ScoreComp(Float* _score){
		score = _score;
	}
	bool operator()(const int& ind1, const int& ind2){
		return score[ind1] > score[ind2];
	}
	private:
	Float* score;
};


vector<string> split(string str, string pattern){

	vector<string> str_split;
	size_t i=0;
	size_t index=0;
	while( index != string::npos ){

		index = str.find(pattern,i);
		str_split.push_back(str.substr(i,index-i));

		i = index+1;
	}
	
	if( str_split.back()=="" )
		str_split.pop_back();

	return str_split;
}

void readMat(char* fname, int& N, int& D, Matrix& mat){
	ifstream fin(fname);
	
	fin >> N >> D;
	mat.resize(N);
	for(int i=0;i<N;i++)
		mat[i].resize(D);
	
	for(int i=0;i<N;i++)
		for(int j=0;j<D;j++){
			fin >> (mat[i][j]);
		}
}

void readSymMat(char* fname, int& N, Matrix& mat){
	ifstream fin(fname);
	
	fin >> N;
	mat.resize(N);
	for(int i=0;i<N;i++)
		mat[i].resize(N);
	
	for(int i=0;i<N;i++)
		for(int j=0;j<N;j++){
			fin >> (mat[i][j]);
		}
}

double inner_prod(double* w, SparseVec* sv){

	double sum = 0.0;
	for(SparseVec::iterator it=sv->begin(); it!=sv->end(); it++)
		sum += w[it->first]*it->second;
	return sum;
}


double norm_sq( double* v, int size ){

	double sum = 0.0;
	for(int i=0;i<size;i++){
		if( v[i] != 0.0 )
			sum += v[i]*v[i];
	}
	return sum;
}


double normalize(Vector& x, Vector& x2){
	
	double sum = 0.0;
	for(int i=0;i<x.size();i++)
		sum += x[i]*x[i];
	
	if( sum == 0.0 )
		return 0.0;
	
	sum = sqrt(sum);
	
	x2.resize( x.size() );
	for(int i=0;i<x.size();i++)
		x2[i] = x[i]/sum;
}

void mat_scale( Matrix& A, double s, Matrix& B ){
	
	B.resize(A.size());
	for(int i=0;i<B.size();i++){
		B[i].resize(A[i].size());
		for(int j=0;j<B[i].size();j++)
			B[i][j] = s*A[i][j];
	}
}

double sum(Vector& v){
	
	double s=0.0;
	for(int i=0;i<v.size();i++)
		s += v[i];
	return s;
}


void transpose( Matrix& A, int N, int D, Matrix& B){
	
	B.resize(D);
	for(int j=0;j<D;j++)
		B[j].clear();
	for(int i=0;i<N;i++){
		for(int j=0;j<D;j++){
			B[j].push_back(A[i][j]);
		}
	}
}

#endif
