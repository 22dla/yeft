#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>

#define M 2048
#define N 2048
#define K 2048

int A[M][N];
int B[N][K];
int C[M][K];
int D[M][K];
 
double timer()
{
    double t;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    t = (double)tv.tv_sec * 1.0 + (double)tv.tv_usec * 1e-6;
    return t;
}
 
int main()
{
    int i, j, k;
    double t1, t2, T1, T2;
    for(i = 0; i < M; i++)
        for(j = 0; j < N; j++)
            A[i][j] = i;
    for(i = 0; i < N; i++)
        for(j = 0; j < K; j++)
            B[i][j] = j;
    t1 = timer();
    
    #pragma acc parallel loop local(i, j, k) copyin(A, B) copy(C) \
    swapin(B(dimension order:1, 2))
    for(i = 0; i < M; i++)
    {
        #pragma acc loop tile(2) annotate(tilemask(C))
        for(k = 0; k < K; k++)
        {
            for(j = 0; j < N; j++)
            {
                C[i][k] += A[i][j]*B[j][k];
            }
        }
    }
  
    t2 = timer();
    T1 = t2-t1;
    printf("matrixMul with OpenACC A[%d][%d] * B[%d][%d], use time:%.6f\n", M, N, N, K, T1);

    t1 = timer();
    
    for(i = 0; i < M; i++)
    {
        for(k = 0; k < K; k++)
        {
            for(j = 0; j < N; j++)
            {
                D[i][k] += A[i][j]*B[j][k];
            }
        }
    }
  
    t2 = timer();
    T2 = t2-t1;
    printf("matrixMul without OpenACC A[%d][%d] * B[%d][%d], use time:%.6f\n", M, N, N, K, T2);
    printf("Gain = %.6f\n", T2/T1);

    for(i = 0; i < M; i++)
    {
        for(k = 0; k < K; k++)
        {
            if(C[i][k] != D[i][k]){
                printf("Error\n");
                return 0;
            }
        }
    }
    printf("Success test\n");
 
      return 0;
  }
