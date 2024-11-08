#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <unistd.h>

#define TIME_SLEEP_MAX 3
#define MAX_ENTRANCE 3

const int TAG_REQUEST = 1;
const int TAG_MARKER = 2;
const int TAG_DONE = 3;
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
    srand(time(0) + rank);

    int has_marker = (rank == 0) ? 1 : 0;
    std::vector<int> LN(size, 0);
    std::vector<int> RN(size, 0);
    std::vector<int> request_queue;

    int entrances_left = MAX_ENTRANCE;

    if (rank == 0) {
        int completed_processes = 0;
        
        while (completed_processes < size - 1) {
            if (!request_queue.empty()) {
                int next_process = request_queue.front();
                request_queue.erase(request_queue.begin());

                std::cout << "Process 0 sends marker to process " << next_process << std::endl;
                MPI_Send(&LN[0], size, MPI_INT, next_process, TAG_MARKER, MPI_COMM_WORLD);
                has_marker = 0;

                MPI_Recv(&LN[0], size, MPI_INT, next_process, TAG_MARKER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::cout << "Process 0 received marker back from process " << next_process << std::endl;
                has_marker = 1;
            }

            MPI_Status status;
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TAG_REQUEST) {
                int requester;
                MPI_Recv(&requester, 1, MPI_INT, status.MPI_SOURCE, TAG_REQUEST, MPI_COMM_WORLD, &status);
                int request_number;
                MPI_Recv(&request_number, 1, MPI_INT, requester, TAG_REQUEST, MPI_COMM_WORLD, &status);

                RN[requester] = std::max(RN[requester], request_number);
                if (has_marker && RN[requester] == LN[requester] + 1) {
                    request_queue.push_back(requester);
                }
            } else if (status.MPI_TAG == TAG_DONE) {
                int done_process;
                MPI_Recv(&done_process, 1, MPI_INT, status.MPI_SOURCE, TAG_DONE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                completed_processes++;
                std::cout << "Process 0 received completion notice from process " << done_process 
                          << " (" << completed_processes << " of " << size - 1 << " completed)" << std::endl;
            }
        }
    } else {
        while (entrances_left > 0) {
            if (!has_marker) {
                RN[rank]++;
                int request_number = RN[rank];

                MPI_Send(&rank, 1, MPI_INT, 0, TAG_REQUEST, MPI_COMM_WORLD);
                MPI_Send(&request_number, 1, MPI_INT, 0, TAG_REQUEST, MPI_COMM_WORLD);
                std::cout << "Process " << rank << " requested entry to critical section (Request #" << request_number << ")" << std::endl;
            }

            MPI_Status status;
            int received_data[size];
            MPI_Recv(&received_data, size, MPI_INT, MPI_ANY_SOURCE, TAG_MARKER, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TAG_MARKER) {
                has_marker = 1;
                for (int i = 0; i < size; ++i) {
                    LN[i] = std::max(LN[i], received_data[i]);
                }

                enter_critical_section(rank);
                
                LN[rank] = RN[rank];
                entrances_left--;
                std::cout << "Process " << rank << " has " << entrances_left << " entrances left." << std::endl;

                MPI_Send(&LN[0], size, MPI_INT, 0, TAG_MARKER, MPI_COMM_WORLD);
                has_marker = 0;

                if (entrances_left == 0) {
                    int done_message = rank;
                    MPI_Send(&done_message, 1, MPI_INT, 0, TAG_DONE, MPI_COMM_WORLD);
                    std::cout << "Process " << rank << " completed all entries to critical section." << std::endl;
                }
            }
        }
    }

    std::cout << "Process " << rank << ": DONE" << std::endl;

    MPI_Finalize();
    return 0;
}

