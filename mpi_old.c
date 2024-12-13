#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <mpi-ext.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <setjmp.h>
#include <signal.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#define  Max(a,b) ((a)>(b)?(a):(b))

long long int N;
float   maxeps = 0.1e-7;
int itmax = 100;
int i,j,k;
float eps;
float *A,  *B, *B_local;
int start, stop, startresid, stopresid;
int local_rows;
int it;
jmp_buf jump_buffer;
MPI_Comm globalcomm;
MPI_Errhandler err_handler;
void relax(int size, int rank, int local_rows);
void resid(int rank);
void init();
void verify(FILE* outf);
int rank, size;
void save_checkpoint(int rank, int size, int it);
int load_checkpoint(int rank, int size, int* it);
void update_parameters(int size, int rank, long long int N, int* recvcounts, int* displs, 
                       int* start, int* stop, int* startresid, int* stopresid, float *B_local);
int recover_from_failure(int* rank, int* size, int *it);
int *recvcounts, *displs;
void error_handler_function(MPI_Comm* comm, int* errcode, ...);

void gather_B_and_update_A(int local_rows, int rank, int size) {
    MPI_Gatherv(B_local, N*local_rows, MPI_FLOAT, B, recvcounts, displs, MPI_FLOAT, 0, globalcomm);
    // printf("%d : tut\n", rank);
	//
	//if (rank == 0){
		// printf("sendcount: %lld, recvcounts: %d, displs: %d\n", N*local_rows, *recvcounts, *displs);
	//}
    if (rank == 0)
        for(j=1; j<=N-2; j++)
            for(i=1; i<=N-2; i++)
                A[i*N + j] = B[i*N + j];
    MPI_Bcast(A, N * N, MPI_FLOAT, 0, globalcomm);
    MPI_Bcast(&eps, 1, MPI_FLOAT, 0, globalcomm);
}

void printMatrix(float* A, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", A[i * N + j]);
        }
        printf("\n");
    }
}

void printArray(float* arr, int N) {
    for (int i = 0; i < N; i++) {
        printf("%f ", arr[i]);
    }
    printf("\n");
}

int main(int an, char **as)
{
    
    char** for_mpi = as + 4;

	globalcomm = MPI_COMM_WORLD;
    MPI_Init(&an, &for_mpi);
    MPI_Comm_rank(globalcomm, &rank);
    MPI_Comm_size(globalcomm, &size);
	MPI_Comm_create_errhandler(error_handler_function, &err_handler);
    MPI_Comm_set_errhandler(globalcomm, err_handler);
	// printf("Start\n");
	MPI_Barrier(globalcomm);

	
	FILE * outf = fopen(as[3], "w");
    
	N = 2*2*2*2*2*2*strtoll(as[1], 0, 10) + 2;
    // printf("N = %lld\n", N);
	
    //printf("N = %lld,  size = %d \n",N ,size  ); 
    local_rows = N / size;
	//if (rank == 0)
	//	printf("local rows: %d", local_rows);
    B_local = (float*)malloc(N * N * sizeof(float*));
    // printf("Hello from process %d out of %d, lr %d\n", rank, size, local_rows);
    if (N % size > rank){
        local_rows += 1;
    }
    // printf("Hello from process %d out of %d, lr %d\n", rank, size, local_rows);
	A = (float*)malloc(N * N * sizeof(float*));
	B = (float*)malloc(N * N * sizeof(float*));
    

	
    
    //заполнение необходимых данных для параллельного выполнения

    struct timeval start_time, end_time;
    if (rank == 0){ 
        gettimeofday(&start_time, NULL);
    }
        
    
    recvcounts = (int*)malloc(size*sizeof(int));
    displs = (int*)malloc(size*sizeof(int));
	update_parameters(size, rank, N, recvcounts, displs, &start, &stop, &startresid, &stopresid, B_local);

    // printf("%d : startresid = %d, stopresid = %d \n",rank, startresid, stopresid);    
	init();
	// printf("Init Done!\n");
	save_checkpoint(rank, size, it);
	if (setjmp(jump_buffer) == 0) {
		MPI_Barrier(globalcomm);
	}
	for (it = 1; it <= itmax; it++) {
        eps = 0.0;

        if (rank == 0) {
            fprintf(stdout, "Iteration %d\n", it);
        }

		if (rank == 4 && size==8 && it == 4){
			//printMatrix(A, N);
			raise(SIGKILL);
		}

		if (rank == 2 && size == 7 && it == 50){
			raise(SIGKILL);
		}

		

        relax(size, rank, local_rows);
        resid(rank);
        gather_B_and_update_A(local_rows, rank, size);
		if (rank == 0 && it == 75){
			// printMatrix(A, N);
		}
        if (eps < maxeps) break;

        save_checkpoint(rank, size, it);
		if (setjmp(jump_buffer) == 0) {
			MPI_Barrier(globalcomm);
		}
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
    MPI_Reduce(&tmp, &eps, 1, MPI_FLOAT, MPI_MAX, 0, globalcomm);
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


void save_checkpoint(int rank, int size, int it) {
    FILE *file = fopen("checkpoint.bin", "wb");
    if (rank == 0) {
        fwrite(A, sizeof(float), N * N, file);
        fwrite(&it, sizeof(int), 1, file);
    }
    fclose(file);
}

int load_checkpoint(int rank, int size, int* it) {
    FILE *file = fopen("checkpoint.bin", "rb");
    if (file == NULL) {
        return 0;
    }
    fread(A, sizeof(float), N * N, file);
    fread(it, sizeof(int), 1, file);
    fclose(file);
    return 1;
}

void update_parameters(int size, int rank, long long int N, int* recvcounts, int* displs, 
                       int* start, int* stop, int* startresid, int* stopresid, float *B_local) {
    // Обновление массивов recvcounts и displs
    recvcounts[0] = (N / size + (N % size > 0 ? 1 : 0)) * N;
    displs[0] = 0;

    for (int i = 1; i < size; i++) {
        if (N % size > i) {
            recvcounts[i] = (N / size + 1) * N;
        } else {
            recvcounts[i] = (N / size) * N;
        }

        displs[i] = displs[i - 1] + recvcounts[i - 1];
    }

    // Обновление start
    if (rank == 0) {
        *start = 2;
    } else {
        *start = rank * (N / size);
        if (N % size > rank) {
            *start += rank;
        } else {
            *start += N % size;
        }
    }

    // Обновление stop
    if (rank == size - 1) {
        *stop = N - 3;
    } else {
        if (rank < N % size) {
            *stop = (rank + 1) * (N / size) + rank;
        } else {
            *stop = (rank + 1) * (N / size) + N % size - 1;
        }
    }

    // Обновление startresid и stopresid
    *startresid = (rank == 0) ? *start - 1 : *start;
    *stopresid = (rank == size - 1) ? *stop + 1 : *stop;

    //printf("N = %lld,  size = %d \n",N ,size  ); 
	
	local_rows = N / size;
	if (N % size > rank){
        local_rows += 1;
    }

		
}


int recover_from_failure(int* rank, int* size, int* it) {
    MPI_Comm new_comm;
    int err = MPIX_Comm_shrink(globalcomm, &new_comm);
    if (err != MPI_SUCCESS) {
        fprintf(stderr, "Failed to shrink communicator.\n");
        return 0;
    }
	globalcomm = new_comm;
    MPI_Comm_rank(new_comm, rank);
    MPI_Comm_size(new_comm, size);

    update_parameters(*size, *rank, N, recvcounts, displs, &start, &stop, &startresid, &stopresid, B_local);

    if (!load_checkpoint(*rank, *size, it)) {
        fprintf(stderr, "Process %d: Unable to load checkpoint. Exiting...\n", *rank);
        MPI_Abort(new_comm, 1);
    }
    
    // Synchronize all processes in the new communicator
    MPI_Barrier(new_comm);
    return 1;
}


void error_handler_function(MPI_Comm* comm, int* errcode, ...) {
	printf("Rank %d: Recovering.\n", rank);
    if (!recover_from_failure(&rank, &size, &it)) {
        MPI_Abort(*comm, *errcode);
    }
	longjmp(jump_buffer, 0);
}

