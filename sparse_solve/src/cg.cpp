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

    int         iter = 0;
    gsl_vector* Ap   = gsl_vector_alloc( n );

    while ( gsl_blas_dnrm2( r ) > tol )
    {
        iter++;
        double beta = gsl_blas_dnrm2( r ) / gsl_blas_dnrm2( r0 );
        beta *= beta;
        gsl_blas_dcopy( r, p );
        gsl_blas_daxpy( beta, p, p );

        gsl_spblas_dgemv( CblasNoTrans, 1.0, A, p, 0.0, Ap );
        double ptap = 0.0;
        gsl_blas_ddot( p, Ap, &ptap );
        double alpha = gsl_blas_dnrm2( r );
        alpha *= alpha;
        alpha /= ptap;
        gsl_blas_dcopy( r, r0 );
        gsl_blas_daxpy( alpha, p, x );
        gsl_blas_daxpy( -alpha, Ap, r );
    }
    gsl_vector_free( r );
    gsl_vector_free( r0 );
    gsl_vector_free( p );
    gsl_vector_free( Ap );
    return iter;
}

int pcg( gsl_spmatrix* A, gsl_vector* b, gsl_vector* x, double tol )
{
    int  n  = b->size;
    auto pa = gsl_vector_alloc( n );
    for ( int i = 0; i < n; i++ )
    {
        double alpha = 1.0 / gsl_spmatrix_get( A, i, i );
        gsl_vector_set( pa, i, alpha );
    }
    gsl_vector* r0 = gsl_vector_alloc( n );
    gsl_blas_dcopy( b, r0 );
    gsl_spblas_dgemv( CblasNoTrans, -1.0, A, x, 1.0, r0 );
    gsl_vector_mul( r0, pa );

    gsl_vector* r = gsl_vector_alloc( n );
    gsl_blas_dcopy( r0, r );

    gsl_vector* p = gsl_vector_alloc( n );
    gsl_vector_set_zero( p );

    int         iter = 0;
    gsl_vector* Ap   = gsl_vector_alloc( n );

    while ( gsl_blas_dnrm2( r ) > tol )
    {
        iter++;
        double beta = gsl_blas_dnrm2( r ) / gsl_blas_dnrm2( r0 );
        beta *= beta;
        gsl_blas_dcopy( r, p );
        gsl_blas_daxpy( beta, p, p );

        gsl_spblas_dgemv( CblasNoTrans, 1.0, A, p, 0.0, Ap );
        gsl_vector_mul( Ap, pa );
        double ptap = 0.0;
        gsl_blas_ddot( p, Ap, &ptap );
        double alpha = gsl_blas_dnrm2( r );
        alpha *= alpha;
        alpha /= ptap;
        gsl_blas_dcopy( r, r0 );
        gsl_blas_daxpy( alpha, p, x );
        gsl_blas_daxpy( -alpha, Ap, r );
    }
    gsl_vector_free( r );
    gsl_vector_free( r0 );
    gsl_vector_free( p );
    gsl_vector_free( Ap );
    gsl_vector_free( pa );
    return iter;
}