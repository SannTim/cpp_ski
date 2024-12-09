#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#define  Max(a,b) ((a)>(b)?(a):(b))

long long int N;
float   maxeps = 0.1e-7;
int itmax = 100;
int i,j,k;
float eps;
float *A,  *B, *A_local, *B_local;
int start, stop, startresid, stopresid;

void relax();
void resid();
void init();
void verify(); 
int *recvcounts, *displs;
void gather_B_and_update_A(int local_rows, int rank, int size) {
    MPI_Gatherv(B_local, N*local_rows, MPI_FLOAT, B, recvcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
    // printf("%d : tut\n", rank);
    if (rank == 0)
        for(j=1; j<=N-2; j++)
            for(i=1; i<=N-2; i++)
                A[i*N + j] = B[i*N + j];
    MPI_Bcast(A, N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&eps, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
}


int main(int an, char **as)
{
    
    char** for_mpi = as + 4;
    MPI_Init(&an, &for_mpi);
    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    

	FILE * outf = fopen(as[3], "w");
    
	N = 2*2*2*2*2*2*strtoll(as[1], 0, 10) + 2;
    // printf("N = %lld\n", N);
    int local_rows = N / size;
    
    // printf("Hello from process %d out of %d, lr %d\n", rank, size, local_rows);
    if (N % size > rank){
        local_rows += 1;
    }
    // printf("Hello from process %d out of %d, lr %d\n", rank, size, local_rows);
	A = (float*)malloc(N * N * sizeof(float*));
	B = (float*)malloc(N * N * sizeof(float*));
    
    A_local = (float*)malloc(local_rows * N * sizeof(float*));
    B_local = (float*)malloc(local_rows * N * sizeof(float*));
	
    
    //заполнение необходимых данных для параллельного выполнения

    struct timeval start_time, end_time;
    if (rank == 0){ 
        gettimeofday(&start_time, NULL);
    }
        
    
    recvcounts = (int*)malloc(size*sizeof(int));
    displs = (int*)malloc(size*sizeof(int));
    displs[0] = 0;
    recvcounts[0] = local_rows*N;
    for (i = 1; i < size; i++){
        if (N%size > i)
            recvcounts[i] = (N/size + 1)*N;
        else
            recvcounts[i] = (N/size)*N;
        
        displs[i] = displs[i-1] + recvcounts[i-1];
    }
    // заполение данных о начале и конце для relax и resid
    
    if (rank == 0) 
        start = 2;
    else {
        start = rank*(N/size);
        if (N%size > rank){
            start += rank;
        }
        else {
            start += N%size;
        }
    }
        
    if (rank == size - 1)
        stop = N-3;
    else {
        if (rank < N%size)
            stop = (rank+1)*(N/size)+rank;
        else stop = (rank+1)*(N/size)+N%size-1;
    }

    startresid = rank == 0 ? start - 1: start;
    stopresid = rank == size - 1 ? stop + 1: stop;
    // printf("%d : startresid = %d, stopresid = %d \n",rank, startresid, stopresid);    
	int it;
	init();
    // MPI_Scatter(A[0], local_rows * N, MPI_FLOAT, local_A[0], local_rows * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
	for(it=1; it<=itmax; it++)
	{
		eps = 0.;
        // printf("%d : %d \n", rank , it);
		relax(size, rank, local_rows);
        resid(rank);
        gather_B_and_update_A(local_rows, rank, size);
        if (rank == 0){
            // if (it == 1){
            //     for (i = 0; i < N; i++){
			// 	for (j = 0; j < N; j++)
			// 		printf("%f", B[i*N + j]);
			// 	printf("\n");
			// }
            // }
            fprintf(outf, "it=%4i   eps=%f\n", it,eps);
        }
        // printf("%d : %d \n", rank , it);
		if (eps < maxeps) break;
	}
    if (rank == 0)
	verify(outf);
    if (rank == 0){
        gettimeofday(&end_time, NULL);
	    FILE *file = fopen(as[2], "w");
        long seconds = end_time.tv_sec-start_time.tv_sec;
        long microseconds = end_time.tv_usec-start_time.tv_usec;
        fprintf(file, "%f", seconds + microseconds*1e-6);
        fclose(file);
    }
	
	fclose(outf);
    free(A);
    free(B);
    free(A_local);
    free(B_local);
    free(recvcounts);
    free(displs);
    MPI_Finalize();
	return 0;
}

void init()
{ 
	for(j=0; j<=N-1; j++)
	for(i=0; i<=N-1; i++)
	{
		if(i==0 || i==N-1 || j==0 || j==N-1) A[i*N + j]= 0.;
		else A[i * N + j]= ( 1. + i + j ) ;
	}
} 

void relax(int size, int rank, int local_rows)
{
    // printf("%d: st=%d sp=%d\n", rank, start, stop);
    int k = 0;
    if (rank == 0)
        k = 2;
	for(j=2; j<=N-3; j++)
	for(i=start; i<=stop; i++)
	{
		B_local[(i-start+k)*N + j]=(A[(i-2) * N + j]+A[(i-1)*N + j]+A[(i+2)*N +j]+A[(i+1)*N + j]+A[i*N + j-2]+A[i*N + j-1]+A[i*N + j+2]+A[i*N + j+1])/8.;
	}
}

void resid(int rank)
{ 
    
    //находим лучий эпсилонт на узле
    float tmp = 0;
    int k = 0;
    if (rank == 0)
        k = 1;
    
	for(j=1; j<=N-2; j++)
	for(i=startresid; i<=stopresid; i++)
	{
		float e;
		e = fabs(A[i*N + j] - B_local[(i-startresid+k)*N + j]);
		tmp = Max(tmp,e);
	}
   
    //передаем его на главный узел
    MPI_Reduce(&tmp, &eps, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
}

void verify(FILE* outf)
{
	float s;
	s=0.;
	for(j=0; j<=N-1; j++)
	for(i=0; i<=N-1; i++)
	{
		s=s+A[i*N + j]*(i+1)*(j+1)/(N*N);
	}
	fprintf(outf, "  S = %f\n",s);
}
