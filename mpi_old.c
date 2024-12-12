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
int it;

MPI_Comm global_comm;
MPI_Errhandler err_handler;
void relax(int size, int rank, int local_rows);
void resid(int rank);
void init();
void verify(FILE* outf);
int rank, size;
void save_checkpoint(int rank, int size, int it);
int load_checkpoint(int rank, int size, int* it);
void update_parameters(int size, int rank, int N, int* recvcounts, int* displs, 
                       int* start, int* stop, int* startresid, int* stopresid);
int recover_from_failure(int* rank, int* size, int *it);
int *recvcounts, *displs;
void error_handler_function(MPI_Comm* comm, int* errcode, ...);

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

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
	MPI_Comm_create_errhandler(error_handler_function, &err_handler);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, err_handler);

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
	update_parameters(size, rank, N, recvcounts, displs, &start, &stop, &startresid, &stopresid);

    // printf("%d : startresid = %d, stopresid = %d \n",rank, startresid, stopresid);    
	init();
	save_checkpoint(rank, size, it);
	
	for (it = 1; it <= itmax; it++) {
        eps = 0.0;

        // Вставим попытку выполнить шаги и обработку ошибок
        int success = 0;
        if (rank == 0) {
            fprintf(stdout, "Iteration %d\n", it);
        }

        relax(size, rank, local_rows);
        resid(rank);
        gather_B_and_update_A(local_rows, rank, size);

        // Проверка завершения
        if (eps < maxeps) break;

        save_checkpoint(rank, size, it); // Сохраняем контрольную точку
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


void save_checkpoint(int rank, int size, int it) {
    MPI_File checkpoint;
    MPI_File_open(MPI_COMM_WORLD, "checkpoint.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &checkpoint);
    MPI_File_set_view(checkpoint, rank * N * N * sizeof(float) / size, MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);
    MPI_File_write_all(checkpoint, A + displs[rank], recvcounts[rank], MPI_FLOAT, MPI_STATUS_IGNORE);
    if (rank == 0) {
        MPI_File_write_at(checkpoint, N * N * sizeof(float), &it, 1, MPI_INT, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&checkpoint);
}

int load_checkpoint(int rank, int size, int* it) {
    MPI_File checkpoint;
    if (MPI_File_open(MPI_COMM_WORLD, "checkpoint.bin", MPI_MODE_RDONLY, MPI_INFO_NULL, &checkpoint) != MPI_SUCCESS) {
        return 0;
    }
    MPI_File_set_view(checkpoint, rank * N * N * sizeof(float) / size, MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);
    MPI_File_read_all(checkpoint, A + displs[rank], recvcounts[rank], MPI_FLOAT, MPI_STATUS_IGNORE);
    if (rank == 0) {
        MPI_File_read_at(checkpoint, N * N * sizeof(float), it, 1, MPI_INT, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&checkpoint);
    return 1;
}

void update_parameters(int size, int rank, int N, int* recvcounts, int* displs, 
                       int* start, int* stop, int* startresid, int* stopresid) {
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
}


int recover_from_failure(int* rank, int* size, int* it) {
    MPI_Comm new_comm;
	it--;
    // Attempt to shrink the communicator
    int err = MPI_Comm_shrink(MPI_COMM_WORLD, &new_comm);
    if (err != MPI_SUCCESS) {
        fprintf(stderr, "Failed to shrink communicator.\n");
        return 0;
    }

    // Update rank and size with the new communicator
    MPI_Comm_rank(new_comm, rank);
    MPI_Comm_size(new_comm, size);

    // Update parameters based on the new communicator
    update_parameters(*size, *rank, N, recvcounts, displs, &start, &stop, &startresid, &stopresid);

    // Allocate memory for local arrays based on the new distribution
    int local_rows = recvcounts[*rank] / N;
    free(A_local);
    free(B_local);
    A_local = (float*)malloc(local_rows * N * sizeof(float));
    B_local = (float*)malloc(local_rows * N * sizeof(float));

    // Load the last checkpoint to recover state
    if (!load_checkpoint(*rank, *size, it)) {
        fprintf(stderr, "Process %d: Unable to load checkpoint. Exiting...\n", *rank);
        MPI_Abort(new_comm, 1);
    }

    // Copy recovered data into local arrays
    memcpy(A_local, A + displs[*rank], recvcounts[*rank] * sizeof(float));
    memcpy(B_local, A_local, recvcounts[*rank] * sizeof(float));

    // Update the global communicator
    MPI_COMM_WORLD = new_comm;

    // Synchronize all processes in the new communicator
    MPI_Barrier(new_comm);

    return 1;
}


void error_handler_function(MPI_Comm* comm, int* errcode, ...) {
    char error_string[BUFSIZ];
    int length_of_error_string;

    MPI_Error_string(*errcode, error_string, &length_of_error_string);
    fprintf(stderr, "Error in communicator: %s\n", error_string);

    // Call recover function
    int rank, size, iteration = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (!recover_from_failure(&rank, &size, &it)) {
        MPI_Abort(*comm, *errcode);
    }
}

