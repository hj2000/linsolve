#include "krylov.h"
int cg( const gsl_spmatrix* A, const gsl_vector* b, gsl_vector* x, double tol )
{
    int         n = b->size;
    gsl_vector* r = gsl_vector_alloc( n );
    gsl_blas_dcopy( b, r );
    gsl_spblas_dgemv( CblasNoTrans, -1.0, A, x, 1.0, r );



    gsl_vector* p = gsl_vector_alloc( n );
    gsl_vector_set_zero( p );

    int         iter    = 0;
    gsl_vector* Ap      = gsl_vector_alloc( n );
    double      norm_r  = gsl_blas_dnrm2( r );
    double      norm_r0 = norm_r;
    double      norm_b  = norm_r;
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
        // printf( "%d,norm=%e\n", iter, norm_r );
    }
    gsl_vector_free( r );
    gsl_vector_free( p );
    gsl_vector_free( Ap );
    return iter;
}
int pcg( gsl_precond_type type, const gsl_spmatrix* A, const gsl_vector* b, gsl_vector* x,
         double tol )
{
    int          n    = b->size;
    gsl_precond* pred = gsl_precond_alloc( type, A );

    gsl_vector* r = gsl_vector_alloc( n );
    gsl_blas_dcopy( b, r );
    gsl_spblas_dgemv( CblasNoTrans, -1.0, A, x, 1.0, r );

    gsl_vector* p = gsl_vector_alloc( n );
    gsl_vector_set_zero( p );

    gsl_vector* z = gsl_vector_alloc( n );
    gsl_vector_set_zero( z );

    int iter = 0;

    gsl_vector* Ap      = gsl_vector_alloc( n );
    double      norm_r  = gsl_blas_dnrm2( r );
    double      norm_r0 = norm_r;
    double      norm_b  = norm_r;
    double      rTz     = 0.0;
    double      rTz0    = 0.0;
    // gsl_blas_dcopy( r, z );
    gsl_precondition( pred, r, z );
    gsl_blas_ddot( r, z, &rTz0 );

    gsl_vector_set_zero( z );

    while ( norm_r / norm_b > tol )
    {
        // gsl_blas_dcopy( r, z );
        iter++;
        gsl_precondition( pred, r, z );
        gsl_blas_ddot( r, z, &rTz );
        double mu = rTz / rTz0;
        rTz0      = rTz;
        gsl_blas_dscal( mu, p );
        gsl_blas_daxpy( 1.0, z, p );

        double eta = 0.0;
        gsl_blas_ddot( r, z, &eta );
        gsl_spblas_dgemv( CblasNoTrans, 1.0, A, p, 0.0, Ap );
        double etam = 0.0;
        gsl_blas_ddot( Ap, p, &etam );
        eta /= etam;
        gsl_blas_daxpy( eta, p, x );
        gsl_blas_daxpy( -eta, Ap, r );
        norm_r = gsl_blas_dnrm2( r );
        // printf( "%d,norm=%e\n", iter, norm_r );
    }

    gsl_vector_free( r );

    gsl_vector_free( p );
    gsl_vector_free( Ap );
    gsl_vector_free( z );
    gsl_precond_free( pred );
    return iter;
}