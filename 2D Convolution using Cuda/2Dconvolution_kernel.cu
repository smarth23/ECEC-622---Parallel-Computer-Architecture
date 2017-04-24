
#ifndef _2DCONVOLUTION_KERNEL_H_
#define _2DCONVOLUTION_KERNEL_H_

#include <stdio.h>
#include "2Dconvolution.h"

// #define THREAD_BLOCK_SIZE 32
// #define KERNEL_SIZE 5
// #define MATRIX_SIZE 1024
__global__ void ConvolutionKernel(float *M, float *N, float *P)
{


int row=blockIdx.y*blockDim.y+threadIdx.y;
int col=blockIdx.x*blockDim.x+threadIdx.x;


	

			double sum = 0;
			// check the start and end values of m and n to prevent overrunning the 
			//  matrix edges
			unsigned int M_start = (row < 2)? 2 - row : 0;
			unsigned int M_stop = (row > (MATRIX_SIZE - 3))?
									MATRIX_SIZE - row + 2 : 5;
			unsigned int N_start = (col < 2)? 2 - col : 0;
			unsigned int N_stop = (col > (MATRIX_SIZE - 3))?
									(MATRIX_SIZE-col) + 2 : 5;
			// overlay M over N centered at element (row,col).  For each 
			//  overlapping element, multiply the two and accumulate
			for(unsigned int i = M_start; i < M_stop; ++i) 
			{
				for(unsigned int j = N_start; j < N_stop; j++) 
				{
					sum += M[i * 5 + j] * 
							N[MATRIX_SIZE*(row + i - 2) + (col+j - 2)];
				}
			}
			// store the result
			P[row*MATRIX_SIZE + col] = (float)sum;
}
#endif // #ifndef _2DCONVOLUTION_KERNEL_H_
