#ifndef SDPSOLVE_H
#define SDPSOLVE_H

#include <iostream>
#include "util.h"
#include <cassert>
#include "Function.h"
using namespace std;


/** A solver that finds
 *  
 *  	X* = argmax_{X}  tr(X'*C*X)
 *  	s.t. X_ii = 1, i=1...N.
 *
 *  given an objective function.
 */
class SDPSolve{
	
	public:
	virtual void solve(Function* f, Matrix& X) = 0;
	virtual void setIter(int iter)=0;
};

class MixSDPSolve: public SDPSolve{
	public:
	
	MixSDPSolve(){
		_iter = 10;
	}
	
	virtual void setIter(int iter){
		_iter = iter;
	}
	
	virtual void solve(Function* f, Matrix& X){
		
		int N, K;
		f->getDim(N, K);
		
		//random initialize
		X.resize(N);
		for(int i=0;i<N;i++){
			
			randn(X[i], K);
			normalize(X[i], X[i]);
			f->setValues(i, X[i].begin(), X[i].end());
		}
		//cerr << "init obj=" << f->funVal() << endl;
		//main loop
		vector<int> indexes;
		indexes.resize(N);
		for(int i=0;i<N;i++)
			indexes[i] = i;
		
		double obj=1e300, last_obj = 1e300, prec=1e300;
		Vector g;
		g.resize(K);
		Vector tmp_Xi;
		tmp_Xi.resize(K);
		int max_iter = _iter;
		int iter=0;
		while(iter<max_iter){
			
			random_shuffle(indexes.begin(), indexes.end());
			for(int r=0;r<N;r++){
				
				int i = indexes[r];
				
				f->grad(i, g);
				double norm2 = normalize(g, X[i]);
				if( norm2 == 0.0 )
					continue;
				f->setValues(i, X[i].begin(), X[i].end());
			}
			
			//if( iter % 1 == 0 )
			//	cerr << "SDPiter=" << iter << ", obj=" <<  f->funVal_with_constant()  << endl;
			
			obj = f->funVal_with_constant();
			prec = obj - last_obj;
			
			last_obj = obj;
			iter++;
		}

		cerr << "SDP precision = " << prec << endl;
	}

	private:
	int _iter;
};

#endif
