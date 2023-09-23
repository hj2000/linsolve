#include "krylov.h"

gsl_precond* gsl_precond_alloc( gsl_precond_type type, const gsl_spmatrix* A )
{
    gsl_precond* res = (gsl_precond*)malloc( sizeof( gsl_precond ) );
    res->type        = type;
    res->LDU         = gsl_spmatrix_compress( A, GSL_SPMATRIX_CSR );
    // gsl_spmatrix_csr()
    // res->LDU = const_cast<gsl_spmatrix*>( A );
    res->D = gsl_vector_alloc( A->size1 );
    for ( int i = 0; i < A->size1; i++ )
    {
        gsl_vector_set( res->D, i, gsl_spmatrix_get( A, i, i ) );
    }
    if ( type == JACOBI )
    {
        res->x_last = gsl_vector_alloc( A->size1 );
    }
    return res;
}
void jacobi_iterate( const gsl_spmatrix* LDU, const gsl_vector* D, const gsl_vector* b,
                     gsl_vector* x_last, gsl_vector* x, int itn )
{
    for ( int k = 0; k < itn; k++ )
    {
        gsl_blas_dcopy( x, x_last );
        gsl_blas_dcopy( b, x );
        gsl_spblas_dgemv( CblasNoTrans, -1.0, LDU, x_last, 1.0, x );
        gsl_vector_div( x, D );
        gsl_blas_daxpy( 1.0, x_last, x );
    }
}
void gauss_seidel_iterate( const gsl_spmatrix* LDU, const gsl_vector* D, const gsl_vector* b,
                           gsl_vector* x, int n )
{
    for ( int k = 0; k < n; k++ )
    {
        for ( int i = 0; i < LDU->size1; i++ )
        {
            double x_e = gsl_vector_get( b, i );
            for ( int j = LDU->p[i]; j < LDU->p[i + 1]; j++ )
            {

                int ja = LDU->i[j];
                if ( ja != i )
                {
                    x_e -= LDU->data[j] * gsl_vector_get( x, ja );
                }
            }
            x_e *= 1.0 / gsl_vector_get( D, i );
            gsl_vector_set( x, i, x_e );
        }
    }
}
void gsl_precondition( const gsl_precond* pred, const gsl_vector* v, gsl_vector* z )
{
    switch ( pred->type )
    {
    case JACOBI:
        gsl_vector_set_zero( z );
        jacobi_iterate( pred->LDU, pred->D, v, pred->x_last, z, 2 );
        break;
    case GAUSS_SEIDEL:
        gsl_vector_set_zero( z );
        gauss_seidel_iterate( pred->LDU, pred->D, v, z, 2 );
        break;
    default:
        break;
    }
}
void gsl_precond_free( gsl_precond* p )
{
    free( p );
}