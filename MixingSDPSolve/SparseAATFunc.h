#include "Function.h"

/* A low-rank SDP function of the form:
 * 
 * 	tr(X'AA'X)
 *
 * where X:N*K, A:N*D is a sparse matrix.
 */
class SparseAATFunc:public ExtensibleFunction{
	
	public:
	SparseAATFunc(int N, int D, int K, SparseMat& A){
		
		_N = N;
		_D = D;
		_K = K;
		_A = A;
		
		_X.resize(N);
		for(int i=0;i<_N;i++){
			_X[i].resize(_K);
			for(int k=0;k<_K;k++)
				_X[i][k] = 0.0;
		}
		
		_AtX.resize(_D);
		for(int j=0;j<_D;j++){
			_AtX[j].resize(_K);
			for(int k=0;k<_K;k++)
				_AtX[j][k] = 0.0;
		}
		
		_sum_C_cache = -1e300;
	}
	
	void getDim(int& N, int& K){
		N = _N;
		K = _K;
	}
	
	void setValues(int i, Vector::iterator xi_begin, Vector::iterator xi_end){
		Vector x_new(xi_begin, xi_end);
		Vector x_old(_X[i].begin(), _X[i].end());
		
		//update AtX
		SparseVec& Ai = _A[i];
		for(SparseVec::iterator it=Ai.begin(); it!=Ai.end(); it++){
			int j = it->first;
			double Aij = it->second;
			for(int k=0;k<x_new.size();k++){
				_AtX[j][k] += Aij*(x_new[k]-x_old[k]);
			}
		}
		
		_X[i] = x_new;
	}
	
	void setValue(int i, int k, double xik){
		
		double xik_old = _X[i][k];
		
		//update AtX
		SparseVec& Ai = _A[i];
		for(SparseVec::iterator it=Ai.begin(); it!=Ai.end(); it++){
			int j = it->first;
			double Aij = it->second;
			_AtX[j][k] += Aij*(xik - xik_old);
		}
		
		_X[i][k] = xik;
	}
	
	void setAllValues(double v){
		
		Vector At_row_sum;
		At_row_sum.resize(_D, 0.0);
		for(int i=0;i<_N;i++){
			for(SparseVec::iterator it=_A[i].begin(); it!=_A[i].end(); it++){
				At_row_sum[it->first] += it->second;
			}
		}
		
		for(int j=0;j<_D;j++){
			for(int k=0;k<_K;k++){
				_AtX[j][k] = v*At_row_sum[j];
			}
		}

		for(int i=0;i<_N;i++){
			for(int k=0;k<_K;k++)
				_X[i][k] = v;
		}
	}
	
	void grad(int i, Vector& g){
		
		g.resize(_K);
		for(int k=0;k<_K;k++)
			g[k] = 0.0;

		SparseVec& Ai = _A[i];
		for(SparseVec::iterator it=Ai.begin(); it!=Ai.end(); it++){
			int j = it->first;
			double Aij = it->second;
			for(int k=0;k<_K;k++)
				g[k] += _AtX[j][k]*Aij;
		}
		for(int k=0;k<_K;k++)
			g[k] *= 2.0;
		///////////////////////////////////adding eye(N)
		/*for(int k=0;k<_K;k++){
			g[k] -= 0.2*_X[i][k];
		}*/
		/////////////////////////////////////////////
	}
	
	double funVal(){
		double sum = 0.0;
		for(int j=0;j<_D;j++)
			for(int k=0;k<_K;k++)
				sum += _AtX[j][k]*_AtX[j][k];
		//////////////////////////////////////////////////////Adding eye(N)
		/*for(int j=0;j<_D;j++){
			for(int k=0;k<_K;k++){
				sum -= 0.1*_X[j][k]*_X[j][k];
			}
		}*/
		///////////////////////////////////////////////////////////
		return sum;
	}

	void sum_by_row(Vector::iterator s_begin, Vector::iterator s_end){
		
		assert( s_begin+_N == s_end );

		//compute (A'1)
		Vector tmp;
		tmp.resize(_D,0.0);
		for(int i=0;i<_N;i++)
			for(SparseVec::iterator it=_A[i].begin(); it!=_A[i].end(); it++)
				tmp[it->first] += it->second;
		
		// AA'1 = A(A'1)
		for(int i=0;i<_N;i++){
			double sum = 0.0;
			for(SparseVec::iterator it=_A[i].begin(); it!=_A[i].end(); it++){
				sum += it->second*tmp[ it->first ];
			}
			*(s_begin+i) = sum;
		}
		//////////////////////////////////adding eye(N)
		/*for(int i=0;i<_N;i++){
			*(s_begin+i) -= 0.1;
		}*/
		////////////////////////////////////////////////
	}
	
	void Xtv(Vector& v, Vector& Xtv){
		Xtv.resize(_K);
		for(int k=0;k<_K;k++)
			Xtv[k] = 0.0;
		for(int i=0;i<_N;i++)
			for(int k=0;k<_K;k++)
				Xtv[k] += v[i]*_X[i][k];

	}
	
	virtual double sum_C(){
		
		if( _sum_C_cache == -1e300 ){
			Vector tmp;
			tmp.resize(_N);
			sum_by_row(tmp.begin(), tmp.end());
			double sum = 0.0;
			for(int i=0;i<_N;i++)
				sum += tmp[i];
			_sum_C_cache = sum;
			return sum;
		}else
			return _sum_C_cache;

	}
	
	virtual double funVal_with_constant(){
		return 0.0;
	}
	
	private:
	SparseMat _A; //N by D
	Matrix _X; //N by K
	Matrix _AtX; // D by K
	int _N;
	int _D;
	int _K;

	double _sum_C_cache;
};
