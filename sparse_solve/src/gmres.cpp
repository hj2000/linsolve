#include "krylov.h"
int gmres( gsl_spmatrix* A, gsl_vector* b, gsl_vector* x, double tol, int maxit )
{
    int         n = b->size;
    gsl_vector* r = gsl_vector_alloc( n );
    gsl_blas_dcopy( b, r );
    gsl_spblas_dgemv( CblasNoTrans, -1.0, A, x, 1.0, r );
    double      beta = gsl_blas_dnrm2( r );
    gsl_matrix* V    = gsl_matrix_alloc( maxit + 1, n );
    auto        v1   = gsl_matrix_row( V, 0 );
    gsl_blas_dcopy( r, &v1.vector );
    gsl_blas_dscal( 1.0 / beta, &v1.vector );
    gsl_matrix* H = gsl_matrix_alloc( maxit + 1, maxit );
    gsl_matrix_set_zero( H );
    gsl_vector* eta = gsl_vector_alloc( maxit + 1 );
    gsl_vector_set_zero( eta );
    gsl_vector_set( eta, 0, beta );
    int m = maxit;
    for ( int i = 0; i < maxit; i++ )
    {
        auto w = gsl_matrix_row( V, i + 1 );
        auto v = gsl_matrix_row( V, i );
        gsl_spblas_dgemv( CblasNoTrans, 1.0, A, &v.vector, 0.0, &w.vector );
        for ( int j = 0; j <= i; j++ )
        {
            auto   vj = gsl_matrix_row( V, j );
            double h  = 0.0;
            gsl_blas_ddot( &w.vector, &vj.vector, &h );
            gsl_matrix_set( H, j, i, h );
            gsl_blas_daxpy( -h, &vj.vector, &w.vector );
        }
        double hw = gsl_blas_dnrm2( &w.vector );
        gsl_matrix_set( H, i + 1, i, hw );
        if ( hw / beta < tol )
        {
            m = i + 1;
            break;
        }
        gsl_blas_dscal( 1.0 / hw, &w.vector );
    }
    for ( int i = 0; i < m; i++ )
    {
        double a = gsl_matrix_get( H, i, i );
        double b = gsl_matrix_get( H, i + 1, i );
        double c, s;
        gsl_blas_drotg( &a, &b, &c, &s );
        auto h1 = gsl_matrix_row( H, i );
        auto h2 = gsl_matrix_row( H, i + 1 );
        gsl_blas_drot( &h1.vector, &h2.vector, c, s );
        cblas_drot( 1, eta->data + i, 1, eta->data + i + 1, 1, c, s );
    }
    auto R  = gsl_matrix_submatrix( H, 0, 0, m, m );
    auto me = gsl_vector_subvector( eta, 0, m );
    gsl_blas_dtrsv( CblasUpper, CblasNoTrans, CblasNonUnit, &R.matrix, &me.vector );
    for ( int i = 0; i < m; i++ )
    {
        auto vi = gsl_matrix_row( V, i );
        gsl_blas_daxpy( gsl_vector_get( &me.vector, i ), &vi.vector, x );
    }
    gsl_matrix_free( H );
    gsl_matrix_free( V );
    gsl_vector_free( r );
    gsl_vector_free( eta );
    return m;
}