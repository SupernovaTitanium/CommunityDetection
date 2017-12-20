#include "util.h"
#include "MaxCutSolve.h"

using namespace std;

class IQPSolve{
	
	public:
	IQPSolve(){
		_maxcut_solve = new MaxCutSolve();
	}
	
	void setIter(int iter){
		_maxcut_solve->setIter(iter);
	}

	/** A function decorator that, given a function f0=tr(X'CX), obtains
	 *  another function f1=tr(X2'C2X2) where
	 *  C2 = [0 b'         X2=[X0;
	 *        b C ]  and       X ]
	 */
	class IQPToMaxCutDec:public Function{
		public:
		IQPToMaxCutDec(ExtensibleFunction* fun){
			_fun = fun;
			int N,K;
			_fun->getDim(N,K);
			_b.resize(N);
			_fun->sum_by_row(_b.begin(), _b.end());
			for(int i=0;i<N;i++)
				_b[i] /= 4.0;
		}
		virtual void getDim(int& N, int& K){
			_fun->getDim(N,K);
			N = N+1;
		}
		
		virtual void setValues(int i, Vector::iterator xi_begin, Vector::iterator xi_end){
			if( i == 0 ){
				_X0 = Vector(xi_begin, xi_end);
			}else{
				_fun->setValues(i-1, xi_begin, xi_end);
			}
		}
		
		virtual void setValue(int i, int k, double xik){
			if( i==0 )
				_X0[k] = xik;
			else
				_fun->setValue(i-1,k,xik);
		}
		
		virtual void setAllValues(double v){
			for(int k=0;k<_X0.size();k++)
				_X0[k] = v;
			_fun->setAllValues(v);
		}
		
		
		virtual void grad(int i, Vector& g){
			
			if( i==0 ){
				_fun->Xtv( _b, g );
			}else{
				int N,K;
				_fun->getDim(N,K);
				_fun->grad(i-1,g);
				for(int k=0;k<K;k++){
					g[k] /= 4.0;
					g[k] += 2.0*_X0[k]*_b[i-1];
				}
			}
		}
		
		virtual double funVal(){
			
			double fval = 0.0;
			
			Vector tmp;
			_fun->Xtv(_b, tmp);
			for(int k=0;k<tmp.size();k++)
				fval += 2.0*tmp[k]*_X0[k];
			
			fval += _fun->funVal()/4.0;

			return fval;
		}

		double funVal_with_constant(){
			
			double fval = funVal();
			fval += _fun->sum_C()/4.0;
			
			return fval;
		}

		private:
		ExtensibleFunction* _fun;
		Vector _X0;
		Vector _b; //length=N
	};

	void solve(ExtensibleFunction* f, Vector& x){
		
		IQPToMaxCutDec* f2 = new IQPToMaxCutDec(f);
		
		Vector y;
		_maxcut_solve->solve(f2, y); //+1 and -1
		
		x.resize(y.size()-1);
		for(int i=1;i<y.size();i++)
			x[i-1] = y[i]*y[0];
		
		for(int i=0;i<x.size();i++)
			x[i] = (x[i]+1)/2;
		
		cerr << "IQP obj=" << objective( f, x ) << endl;
	}
	
	
	private:
	
	
	double objective(Function* f, Vector& x){

		int N, K;
		f->getDim(N,K);
		f->setAllValues(0.0);
		
		for(int i=0;i<N;i++)
			f->setValue(i,0,x[i]);
		
		return f->funVal();
	}
	
	
	MaxCutSolve* _maxcut_solve;
	
};
