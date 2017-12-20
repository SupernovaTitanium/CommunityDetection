#ifndef MAXCUTSOLVE_H
#define MAXCUTSOLVE_H

#include "util.h"
#include "SDPSolve.h"

using namespace std;

class MaxCutSolve{
	
	public:
	MaxCutSolve(){
		_sdp_solve = new MixSDPSolve();
		_T = 100;
	}
	
	void setIter(int iter){
		_sdp_solve->setIter(iter);
	}

	void solve(Function* f, Vector& x){
		
		int N, K;
		f->getDim(N, K);
		
		Matrix X;
		X.resize(N);
		for(int i=0;i<N;i++)
			X[i].resize(K);
		
		_sdp_solve->solve(f, X);
		
		//randomized rounding
		double max_obj = -1e300;
		for(int t=0;t<_T;t++){
			Vector r;
			randn( r, K );
			normalize( r, r );
			Vector x_t = sign(mat_prod( X, r, K ));
			//Vector x_t = mat_prod( X, r, K );
			double obj = objective( f, x_t );
			if( obj > max_obj ){
				max_obj = obj;
				x = x_t;
			}
		}
		cerr << "max-obj=" << max_obj << endl;
	}
	
	private:
	Vector mat_prod( Matrix& X, Vector& r, int K ){
		
		Vector v;
		v.resize(X.size());
		for(int i=0;i<X.size();i++){
			double sum = 0.0;
			for(int k=0;k<K;k++){
				sum += X[i][k]*r[k];
			}
			v[i] = sum;
		}
		
		return v;
	}
	
	Vector sign(Vector v){
		
		Vector v2;
		v2.resize(v.size());
		for(int i=0;i<v.size();i++){
			if( v[i] >= 0.0 )
				v2[i] = +1.0;
			else
				v2[i] = -1.0;
		}
		return v2;
	}
	
	double objective(Function* f, Vector& x){

		int N, K;
		f->getDim(N,K);
		f->setAllValues(0.0);
		
		for(int i=0;i<N;i++)
			f->setValue(i,0,x[i]);
		
		return f->funVal_with_constant();
	}
	
	SDPSolve* _sdp_solve;
	int _N;
	int _K;
	int _T;
};

#endif
