#include <mpi.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <unistd.h>  // для sleep
#include <ctime>     // для генерации случайного времени
#define TIME_SLEEP_MAX 3
const int TAG_MARKER = 0;
const char* FILENAME = "critical.txt";


void enter_critical_section(int rank) {
    std::ifstream infile(FILENAME);
    
    if (infile.good()) {
        std::cerr << "Error: Process " << rank << " detected that file " << FILENAME << " already exists!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1); 
    } else {
        std::ofstream outfile(FILENAME);
        outfile << "Process " << rank << " is in the critical section." << std::endl;
        outfile.close();
        
        
        int sleep_time = rand() % TIME_SLEEP_MAX + 1; 
        std::cout << "Process " << rank << " enters critical section, sleeping for " << sleep_time << " seconds." << std::endl;
        sleep(sleep_time);  

        std::remove(FILENAME);
        std::cout << "Process " << rank << " exits critical section." << std::endl;
    }
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(time(0) + rank);// для рандома

    int marker = (rank == 0) ? 1 : 0;

    for (int i = 0; i < size; ++i) {
        // Если у процесса есть маркер, он входит в критическую секцию
        if (marker == 1) {
            enter_critical_section(rank);
            int next = (rank + 1) % size;
            std::cout << "Process " << rank << " sends marker to process " << next << std::endl;
            MPI_Send(&marker, 1, MPI_INT, next, TAG_MARKER, MPI_COMM_WORLD);
            marker = 0;  
        }

        if (marker == 0) {
            int prev = (rank == 0) ? size - 1 : rank - 1;
            MPI_Recv(&marker, 1, MPI_INT, prev, TAG_MARKER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    MPI_Finalize();
    return 0;
}

