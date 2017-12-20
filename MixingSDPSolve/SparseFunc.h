#include "Function.h"

/* A Sparse SDP function of the form:
 * 
 * 	tr(X'C'X)
 *
 * where X:N*K, C:N*N is a sparse matrix.
 */
class SparseFunc:public ExtensibleFunction{
	
	public:
	SparseFunc(int N, int K, SparseMat& C){
		
		_N = N;
		_K = K;
		_C = C;
		
		_X.resize(N);
		for(int i=0;i<_N;i++){
			_X[i].resize(_K);
			for(int k=0;k<_K;k++)
				_X[i][k] = 0.0;
		}
		
		_sum_C_cache = -1e300;
	}
	
	void getDim(int& N, int& K){
		N = _N;
		K = _K;
	}
	
	void setValues(int i, Vector::iterator xi_begin, Vector::iterator xi_end){
		Vector x_new(xi_begin, xi_end);
		_X[i] = x_new;
	}
	
	void setValue(int i, int k, double xik){
		
		_X[i][k] = xik;
	}
	
	void setAllValues(double v){
		
		for(int i=0;i<_N;i++){
			for(int k=0;k<_K;k++)
				_X[i][k] = v;
		}
	}
	
	void grad(int i, Vector& g){
		
		// g_i = 2Cx_i
		g.resize(_K);
		for(int k=0;k<_K;k++)
			g[k] = 0.0;
		
		for(SparseVec::iterator it=_C[i].begin(); it!=_C[i].end(); it++){
				for(int k=0;k<_K;k++){
								g[k] += _X[it->first][k] * it->second;
				}
		}
		
		for(int k=0;k<_K;k++)
			g[k] *= 2.0;
	}
	
	double funVal(){
		double sum = 0.0;
		for(int i=0;i<_N;i++){
				for(SparseVec::iterator it=_C[i].begin(); it!=_C[i].end(); it++){
						for(int k=0;k<_K;k++)
								sum += _X[i][k] * it->second * _X[it->first][k];
				}
		}
		
		return sum;
	}

	void sum_by_row(Vector::iterator s_begin, Vector::iterator s_end){
		
		assert( s_begin+_N == s_end );

		for(int i=0;i<_N;i++){
			double sum = 0.0;
			for(SparseVec::iterator it=_C[i].begin(); it!=_C[i].end(); it++){
					sum += it->second;
			}
			*(s_begin+i) = sum;
		}
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
			double sum = 0.0;
			for(int i=0;i<_N;i++){
							for(SparseVec::iterator it=_C[i].begin(); it!=_C[i].end(); it++){
											sum += it->second;
							}
			}
			_sum_C_cache = sum;
			return sum;
		}else
			return _sum_C_cache;

	}
	
	virtual double funVal_with_constant(){
		return 0.0;
	}
	
	private:
	SparseMat _C; //N by N sparse mat
	Matrix _X; //N by K
	int _N;
	int _K;

	double _sum_C_cache;
};
