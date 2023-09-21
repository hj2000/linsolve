// #include <stdio.h>
// #include "krylov.h"
// int main()
// {
//     gsl_spmatrix *A = gsl_spmatrix_alloc_nzmax(5, 5, 25, GSL_SPMATRIX_CSR);
//     int ptr[] = {0, 5, 10, 15, 20, 25};
//     int ind[] = {0, 1, 2, 3, 4,
//                  0, 1, 2, 3, 4,
//                  0, 1, 2, 3, 4,
//                  0, 1, 2, 3, 4,
//                  0, 1, 2, 3, 4};
//     double val[] = {10, 1, 2, 3, 4,
//                     1, 9, -1, 2, -3,
//                     2, -1, 7, 3, -5,
//                     3, 2, 3, 12, -1,
//                     4, -3, -5, -1, 15};
//     A->data = val;
//     A->i = ind;
//     A->p = ptr;
//     A->nz = 25;
//     gsl_vector *b = gsl_vector_alloc(5);
//     double rhs[] = {12, -27, 14, -17, 12};
//     b->data = rhs;
//     // double gauss[] = {0.62075, 0.062075, 0.12415, 0.186225, 0.2483};
//     auto x = gsl_vector_alloc(5);
//     // gsl_vector_set_basis
//     gsl_vector_set_zero(x);
//     // x->data = gauss;
//     auto it = cg(A, b, x, 1e-4);
//     gsl_vector_fprintf(stdout, x, "%f");
//     auto r = gsl_vector_alloc(5);
//     gsl_blas_dcopy(b, r);
//     gsl_spblas_dgemv(CblasNoTrans, -1.0, A, x, 1.0, r);
//     gsl_vector_fprintf(stdout, r, "%f");
//     return 0;
// }

#include "krylov.h"
#include <cmath>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_vector.h>
int main()
{
    gsl_spmatrix* A = gsl_spmatrix_alloc( 4, 4 );
    for ( int i = 0; i < 4; i++ )
    {
        if ( i > 0 )
        {
            gsl_spmatrix_set( A, i, i - 1, -1 );
        }
        if ( i < 3 )
        {
            gsl_spmatrix_set( A, i, i + 1, -1 );
        }
        gsl_spmatrix_set( A, i, i, 10 );
    }
    gsl_vector* b = gsl_vector_alloc( 4 );
    b->size       = 4;
    double _b[]   = { 1, 0, 1, 2 };
    b->data       = _b;
    gsl_vector* x = gsl_vector_alloc( 4 );
    gsl_vector_set_zero( x );
    gmres( A, b, x, 1e-10, 4 );
    gsl_vector* r = gsl_vector_alloc( 4 );
    gsl_blas_dcopy( b, r );
    gsl_spblas_dgemv( CblasNoTrans, -1.0, A, x, 1.0, r );
    gsl_vector_fprintf( stdout, r, "%e" );
}