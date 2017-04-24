/* Vector-Matrix multiplication: Y = A * X.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include "vec_mat_mult.h"

/* Write the kernel for vector-matrix multiplication using GPU global memory. */
__global__ void vec_mat_kernel_naive(float *Ad, float *Xd, float *Yd)
{
	//Multiply A and X
	
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	
	float prod = 0;

	for ( int i = 0; i < MATRIX_SIZE; ++i ) {
		
		float A_element = Ad[ MATRIX_SIZE*tx + i ];
		float X_element = Xd[ i ];
		prod += A_element * X_element;
	}

	Yd[ tx ] = prod;
}


/* Write the kernel for vector-matrix multiplication using GPU shared memory. */
__global__ void vec_mat_kernel_optimized(float *Ad, float *Xd, float *Yd)
{
	//Multiply A and X

	__shared__ float shared_X[ 16 ];
	__shared__ float shared_A[ 16 ][ 16 ];

	int row_num = blockIdx.y * blockDim.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int temp = 0;

	for ( int i = 0; i < MATRIX_SIZE; i = i + 16) {
	
		shared_A[ ty ][ tx ] = Ad[ MATRIX_SIZE * row_num + tx  + i ];
		shared_X[ tx ] = Xd[ tx + i ];

		__syncthreads();

		if ( threadIdx.x == 0 ) {
		
			for ( int k = 0; k < blockDim.x; k++ ) {
				temp += shared_A[ tx ][ k ] * shared_X[k];
			}


		}
		__syncthreads();
	}
	
	if ( threadIdx.x == 0 ){
		Yd[ row_num ] = temp;
	}
}



#endif // #ifndef _MATRIXMUL_KERNEL_H_
