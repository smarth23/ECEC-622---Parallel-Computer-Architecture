/* Gaussian elimination
 *  * Author: Naga Kandasamy
 *   * Date created: 02/07/2014
 *    * Date of last update: 01/30/2017
 *     * Compile as follows: gcc -o gauss_eliminate gauss_eliminate.c compute_gold.c -lpthread -std=c99 -lm
 *      */
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "gauss_eliminate.h"

#define MIN_NUMBER 2
#define MAX_NUMBER 50
#define NUM_THREADS 32

/* Function prototypes. */
extern int compute_gold (float *, unsigned int);
Matrix allocate_matrix (int num_rows, int num_columns, int init);
void gauss_eliminate_using_pthreads (Matrix, unsigned int num_elements);
void *div_step (void*);
void *elim_step (void*);
int perform_simple_check (const Matrix);
void print_matrix (const Matrix);
float get_random_number (int, int);
int check_results (float *, float *, unsigned int, float);

struct thread_args{
  int thread_id;
  int num_elements;
  int loop;
  Matrix upper;
};

int
main (int argc, char **argv)
{
  /* Check command line arguments. */
  if (argc > 1)
    {
      printf ("Error. This program accepts no arguments. \n");
      exit (0);
    }
 /* Matrices for the program. */
  Matrix A;     // The input matrix
  Matrix U_reference;   // The upper triangular matrix computed by the reference code
  Matrix U_mt;      // The upper triangular matrix computed by the pthread code

  /* Initialize the random number generator with a seed value. */
  srand (time (NULL));

  /* Allocate memory and initialize the matrices. */
  A = allocate_matrix (MATRIX_SIZE, MATRIX_SIZE, 1);  // Allocate and populate a random square matrix
  U_reference = allocate_matrix (MATRIX_SIZE, MATRIX_SIZE, 0);  // Allocate space for the reference result
  U_mt = allocate_matrix (MATRIX_SIZE, MATRIX_SIZE, 0); // Allocate space for the multi-threaded result

  /* Copy the contents of the A matrix into the U matrices. */
  for (int i = 0; i < A.num_rows; i++)
    {
      for (int j = 0; j < A.num_rows; j++)
  {
    U_reference.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
    U_mt.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
  }
    }

printf ("Performing gaussian elimination using the reference code. \n");
  struct timeval start, stop;
  gettimeofday (&start, NULL);
  int status = compute_gold (U_reference.elements, A.num_rows);
  gettimeofday (&stop, NULL);
  printf ("CPU run time = %0.2f s. \n",
    (float) (stop.tv_sec - start.tv_sec +
       (stop.tv_usec - start.tv_usec) / (float) 1000000));

  if (status == 0)
    {
      printf("Failed to convert given matrix to upper triangular. Try again. Exiting. \n");
      exit (0);
    }
  status = perform_simple_check (U_reference);  // Check that the principal diagonal elements are 1 
  if (status == 0)
    {
      printf ("The upper triangular matrix is incorrect. Exiting. \n");
      exit (0);
    }
  printf ("Single-threaded Gaussian elimination was successful. \n");

  /* Perform the Gaussian elimination using pthreads. The resulting upper triangular matrix should be returned in U_mt */
  gettimeofday (&start, NULL);
  gauss_eliminate_using_pthreads (U_mt, U_mt.num_columns);
  gettimeofday (&stop, NULL);
  printf ("CPU run time = %0.2f s. \n",
    (float) (stop.tv_sec - start.tv_sec +
       (stop.tv_usec - start.tv_usec) / (float) 1000000));
 /* check if the pthread result is equivalent to the expected solution within a specified tolerance. */
  int size = MATRIX_SIZE * MATRIX_SIZE;
  int res = check_results (U_reference.elements, U_mt.elements, size, 0.001f);
  printf ("Test %s\n", (1 == res) ? "PASSED" : "FAILED");

  /* Free memory allocated for the matrices. */
  free (A.elements);
  free (U_reference.elements);
  free (U_mt.elements);

  return 0;
}
/* Write code to perform gaussian elimination using pthreads. */
void
gauss_eliminate_using_pthreads (Matrix U, unsigned int num_elements)
{
  int i, k, j;
  struct thread_args t_args[NUM_THREADS];
  pthread_t threads[NUM_THREADS];
  

  for(k = 0; k < num_elements; k++)
  {
   for (j = (k + 1); j < num_elements; j++)
  {     // Reduce the current row
    if (U.elements[num_elements * k + k] == 0)
      {
        printf
    ("Numerical instability detected. The principal diagonal element is zero. \n");
      }
    U.elements[num_elements * k + j] = (float) (U.elements[num_elements * k + j] / U.elements[num_elements * k + k]);  // Division step
  }
      U.elements[num_elements * k + k] = 1;  // Set the principal diagonal entry in U to be 1 

for(i = 0; i < NUM_THREADS; i++)
    {
      t_args[i].thread_id = i;
      t_args[i].upper = U;
      t_args[i].num_elements = num_elements;
      t_args[i].loop = k;
      pthread_create(&threads[i], NULL, *elim_step, (void *)&t_args[i]);
    }
    for(i = 0; i < NUM_THREADS; i++)
    {
      pthread_join(threads[i], NULL);
    }       
for(i = k + 1; i < num_elements; i++)
    {
      U.elements[num_elements * i + k] = 0;
    }
 }
}

void
*elim_step (void* thread_args)
{
int i, j, k, start, end, chunk, extra, tid, num_elements;
Matrix U;
struct thread_args *my_data;
my_data = (struct thread_args *) thread_args;
tid = my_data->thread_id;
k = my_data->loop;
U = my_data->upper;
num_elements = my_data->num_elements;
chunk = (num_elements - (k+1))/NUM_THREADS;
extra = (num_elements - (k+1))%NUM_THREADS;
start = (tid * chunk) + extra + k;
end = (tid + 1) * chunk + extra + k;

if(tid == 0)
{
  chunk = chunk + extra;
  start = start - extra;
}

for (i = (start + 1); i <= end; i++)
  {
    for (j = (k + 1); j < num_elements; j++)
      {
        U.elements[num_elements * i + j] =  U.elements[num_elements * i + j] - (U.elements[num_elements * i + k] * U.elements[num_elements * k + j]); 
 U.elements[num_elements * k + j]  ;
      }
  }

}
/* Function checks if the results generated by the single threaded and multi threaded versions match. */
int
check_results (float *A, float *B, unsigned int size, float tolerance)
{
  for (int i=0;i < size; i++)
  {
      if(fabsf(A[i] - B[i]) != 0.00)
      {
       // printf("Matrix gold element %d = %0.02f\n ",i,A[i]);
               // printf("Matrix pthread element %d = %0.02f\n ",i,B[i]);
                     }
                       }  
                         for (int i = 0; i < size; i++)
                             if (fabsf (A[i] - B[i]) > tolerance)
                                   return 0;
                                    return 1;
                                    }
Matrix
allocate_matrix (int num_rows, int num_columns, int init)
{
  Matrix M;
  M.num_columns = M.pitch = num_columns;
  M.num_rows = num_rows;
  int size = M.num_rows * M.num_columns;

  M.elements = (float *) malloc (size * sizeof (float));
  for (unsigned int i = 0; i < size; i++)
    {
      if (init == 0)
  M.elements[i] = 0;
      else
  M.elements[i] = get_random_number (MIN_NUMBER, MAX_NUMBER);
    }
  return M;
}


/* Returns a random floating-point number between the specified min and max values. */ 
float
get_random_number (int min, int max)
{
  return (float)
    floor ((double)
     (min + (max - min + 1) * ((float) rand () / (float) RAND_MAX)));
}

/* Performs a simple check on the upper triangular matrix. Checks to see if the principal diagonal elements are 1. */
int
perform_simple_check (const Matrix M)
{
  for (unsigned int i = 0; i < M.num_rows; i++)
    if ((fabs (M.elements[M.num_rows * i + i] - 1.0)) > 0.001)
      return 0;
  return 1;
}

