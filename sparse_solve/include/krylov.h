#include <gsl/gsl_spblas.h>
#include <gsl/gsl_vector.h>
int cg( gsl_spmatrix* A, gsl_vector* b, gsl_vector* x, double tol );
int gmres( gsl_spmatrix* A, gsl_vector* b, gsl_vector* x, double tol, int maxit );
