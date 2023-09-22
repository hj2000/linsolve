#include "krylov.h"
int cg( gsl_spmatrix* A, gsl_vector* b, gsl_vector* x, double tol )
{
    int         n  = b->size;
    gsl_vector* r0 = gsl_vector_alloc( n );
    gsl_blas_dcopy( b, r0 );
    gsl_spblas_dgemv( CblasNoTrans, -1.0, A, x, 1.0, r0 );

    gsl_vector* r = gsl_vector_alloc( n );
    gsl_blas_dcopy( r0, r );

    gsl_vector* p = gsl_vector_alloc( n );
    gsl_vector_set_zero( p );

    int         iter    = 0;
    gsl_vector* Ap      = gsl_vector_alloc( n );
    double      norm_r  = gsl_blas_dnrm2( r );
    double      norm_r0 = norm_r;
    double      norm_b  = gsl_blas_dnrm2( b );
    while ( norm_r / norm_b > tol )
    {
        iter++;
        double beta = norm_r / norm_r0;
        beta *= beta;
        gsl_blas_dscal( beta, p );
        gsl_blas_daxpy( 1.0, r, p );

        gsl_spblas_dgemv( CblasNoTrans, 1.0, A, p, 0.0, Ap );
        double ptap = 0.0;
        gsl_blas_ddot( p, Ap, &ptap );

        double alpha = norm_r * norm_r;
        alpha /= ptap;
        norm_r0 = norm_r;
        gsl_blas_daxpy( alpha, p, x );
        gsl_blas_daxpy( -alpha, Ap, r );
        norm_r = gsl_blas_dnrm2( r );
    }
    gsl_vector_free( r );
    gsl_vector_free( r0 );
    gsl_vector_free( p );
    gsl_vector_free( Ap );
    return iter;
}