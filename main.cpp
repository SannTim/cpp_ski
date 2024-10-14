#include <mpi.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <unistd.h>  // для sleep
#include <ctime>     // для генерации случайного времени
#include <vector>
#include <algorithm>

#define TIME_SLEEP_MAX 3
const int TAG_MARKER = 0;
const int TAG_RELEASE = 1;  
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

void print_remaining_processes(const std::vector<int>& remaining_processes) {
    std::cout << "Remaining processes: ";
    for (int p : remaining_processes) {
        std::cout << p << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
    MPI_Comm_size(MPI_COMM_WORLD, &size);  

    srand(time(0) + rank); 

    int has_marker = (rank == 0) ? 1 : 0;  

    if (rank == 0) {
        std::vector<int> remaining_processes; 
        for (int i = 1; i < size; i++) {
            remaining_processes.push_back(i);
        }

        while (!remaining_processes.empty()) {
            
            int random_index = rand() % remaining_processes.size();
            int next_process = remaining_processes[random_index];
            print_remaining_processes(remaining_processes);
            std::cout << "Process 0 is broadcasting marker to process " << next_process << std::endl;
            MPI_Bcast(&next_process, 1, MPI_INT, 0, MPI_COMM_WORLD);
            std::cout << "Bcast DONE\n";
            MPI_Recv(nullptr, 0, MPI_BYTE, next_process, TAG_RELEASE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            remaining_processes.erase(remaining_processes.begin() + random_index);
        }
    } else {
        while (true) {
            int received_marker;
            MPI_Bcast(&received_marker, 1, MPI_INT, 0, MPI_COMM_WORLD);

            if (received_marker == rank) {
                // Если процесс получил маркер, он заходит в критическую секцию
                enter_critical_section(rank);
                std::cout << "Process " << rank << " releasing marker back to process 0" << std::endl;
                MPI_Send(nullptr, 0, MPI_BYTE, 0, TAG_RELEASE, MPI_COMM_WORLD);
                break;
            }
        }
    }

    std::cout << "Process " << rank << ": DONE" << std::endl;

    MPI_Finalize();
    return 0;
}
