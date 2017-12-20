#ifndef FUNCTION_H
#define FUNCTION_H

#include "util.h"

/** A function of the form:
 *
 *  tr(X'CX)
 *
 *  where X:N*K and C is N*N.
 */
class Function{

	public:
	
	/* Get the dimension of the problem.
	 */
	virtual void getDim(int& N, int& K) = 0;
	
	/* Set the values of X_{i,:}.
	 */
	virtual void setValues(int i, Vector::iterator xi_begin, Vector::iterator xi_end) = 0;
	
	virtual void setValue(int i, int k, double xik)=0;

	virtual void setAllValues(double v)=0;
	
	/** Given an index i, get current gradient [CX]_{i,:}.
	 */
	virtual void grad(int i, Vector& g) = 0;
	
	/* Compute the current function value tr(X'CX).
	 */ 
	virtual double funVal() = 0;

	virtual double funVal_with_constant() = 0;
};

class ExtensibleFunction: public Function{
	public:
	
	/** Compute s_i = \sum_{j} C_{ij} for all i.
	 */
	virtual void sum_by_row(Vector::iterator s_begin, Vector::iterator s_end)=0;
	
	virtual void Xtv(Vector& v, Vector& Xtv)=0;
	
	virtual double sum_C() = 0;
};

#endif
