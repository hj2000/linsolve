#include <assert.h>
#include <gsl/gsl_spblas.h>
#include <gsl/gsl_vector.h>
#include <math.h>
enum gsl_precond_type
{
    ILU,
    JACOBI,
    GAUSS_SEIDEL,
    SOR,
    SSOR,
    AOR,
    RICHARDSON
};
struct gsl_precond
{
    gsl_precond_type type;
    gsl_spmatrix*    LDU;
    gsl_vector*      D;
    gsl_vector*      x_last;
};
gsl_precond* gsl_precond_alloc( gsl_precond_type type, const gsl_spmatrix* A );
void         gsl_precond_free( gsl_precond* p );
void         gsl_precondition( const gsl_precond* pred, const gsl_vector* v, gsl_vector* z );

int cg( const gsl_spmatrix* A, const gsl_vector* b, gsl_vector* x, double tol );
int pcg( gsl_precond_type type, const gsl_spmatrix* A, const gsl_vector* b, gsl_vector* x,
         double tol );
int gmres( const gsl_spmatrix* A, const gsl_vector* b, gsl_vector* x, double tol, int maxit );
int pgmres( gsl_precond_type type, const gsl_spmatrix* A, const gsl_vector* b, gsl_vector* x,
            double tol, int maxit );
