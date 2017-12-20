#include "Function.h"

/* A Sparse SDP function of the form:
 * 
 * 	tr(X'( a*M_1 + b*M_2 )'X)
 *
 * where each tr(X'M_lX) is a Extensible function.
 */
class CompositeFunc:public ExtensibleFunction{
	
	public:
	CompositeFunc(int N, int K, double a, ExtensibleFunction* M1, double b, ExtensibleFunction* M2){
		
		_N = N;
		_K = K;
		
		_a = a;
		_b = b;
		_M1 = M1;
		_M2 = M2;
		
		_sum_C_cache = -1e300;
	}
	
	void getDim(int& N, int& K){
		N = _N;
		K = _K;
	}
	
	void setValues(int i, Vector::iterator xi_begin, Vector::iterator xi_end){
		_M1->setValues(i, xi_begin, xi_end);
		_M2->setValues(i, xi_begin, xi_end);
	}
	
	void setValue(int i, int k, double xik){
		_M1->setValue(i,k,xik);
		_M2->setValue(i,k,xik);
	}
	
	void setAllValues(double v){
		
		_M1->setAllValues(v);
		_M2->setAllValues(v);
	}
	
	void grad(int i, Vector& g){
		
		// g_i = 2Cx_i
		g.resize(_K);
		
		Vector g_1,g_2;
		_M1->grad( i, g_1 );
		_M2->grad( i, g_2 );
		for(int k=0;k<_K;k++)
						g[k] = _a*g_1[k] + _b*g_2[k];
	}
	
	double funVal(){
		return _a*_M1->funVal() + _b*_M2->funVal();
	}

	void sum_by_row(Vector::iterator s_begin, Vector::iterator s_end){
		
		assert( s_begin+_N == s_end );
		
		Vector s_1, s_2;
		s_1.resize(_N);
		s_2.resize(_N);
		_M1->sum_by_row( s_1.begin(), s_1.end() );
		_M2->sum_by_row( s_2.begin(), s_2.end() );

		for(int i=0;i<_N;i++)
				*(s_begin + i) = _a*s_1[i] + _b*s_2[i];
	}
	
	void Xtv(Vector& v, Vector& Xtv){
		_M2->Xtv(v, Xtv);
	}
	
	virtual double sum_C(){
		
		if( _sum_C_cache == -1e300 ){
			
			_sum_C_cache = _a*_M1->sum_C() + _b*_M2->sum_C();
			
			return _sum_C_cache;

		}else
			return _sum_C_cache;

	}
	
	virtual double funVal_with_constant(){
		return 0.0;
	}
	
	private:
	int _N;
	int _K;
	double _a;
	ExtensibleFunction* _M1;
	double _b;
	ExtensibleFunction* _M2;

	double _sum_C_cache;
};
