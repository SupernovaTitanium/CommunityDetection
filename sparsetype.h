#ifndef SPARSETYPE_H
#define SPARSETYPE_H

#include <cstdio>
#include <iostream>
#include <cstdlib>
//#include "mkl_types.h"
struct sp_matrix{
	size_t* row_ptr;
	size_t* col_ptr;
	int nnz;
	int n_size;
	int m_size;
	double* data;
};
struct spm_matrix{
	size_t	* row_ptr;
	size_t	* col_ptr;
	int	 nnz;
	int	 n_size;
	int	 m_size;
	double* data;
};
struct sp_matrix_Z{
	sp_matrix R0;
	sp_matrix R1;
	double* Z;
	double* Zt;
	double  lambda;
	int K;
	int N;
};
struct spm_matrix_Z{
	spm_matrix R0;
	spm_matrix R1;
	double* Z;
	double* Zt;
	double  lambda;
	int	 K;
	int	 N;
};
struct sp_matrix_Zs{
  sp_matrix R0;
  sp_matrix R1;
  sp_matrix Z;
  double  lambda;
  int K;
  int N;
};
struct spm_matrix_Zs{
  spm_matrix R0;
  spm_matrix R1;
  spm_matrix Z;
  double  lambda;
  int	 K;
  int	 N;
};

#endif
