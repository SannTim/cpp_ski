#include <mpi.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <queue>
#include <unistd.h>  // для sleep
#include <ctime>     // для генерации случайного времени
#include <vector>

#define MAX_ENTRANCE 1

struct Request {
    int rank;  // Rank of the requesting process
    int sn;    // Sequence number of the request
};

const int TAG_REQUEST = 1;
const int TAG_MARKER = 2;
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


        std::remove(FILENAME);
        std::cout << "Process " << rank << " exits critical section." << std::endl;
    }
}

void send_request(std::vector<int>& RN, int rank, int size, std::vector<MPI_Request>& requests) {
    Request req;
    RN[rank]++;
    req.sn = RN[rank];
    req.rank = rank;

    for (int i = 0; i < size; i++) {
        if (i != rank) {
            MPI_Request request;
            MPI_Isend(&req, sizeof(Request), MPI_BYTE, i, TAG_REQUEST, MPI_COMM_WORLD, &request);
            requests.push_back(request);
        }
    }
    std::cout << "Process " << rank << " sent request asynchronously to all processes." << std::endl;
}

void process_incoming_requests(std::vector<int>& RN, std::queue<int>& que, int size) {
    MPI_Status status;
    Request incoming_req;

    int flag;
    MPI_Iprobe(MPI_ANY_SOURCE, TAG_REQUEST, MPI_COMM_WORLD, &flag, &status);
    while (flag) {
        MPI_Recv(&incoming_req, sizeof(Request), MPI_BYTE, status.MPI_SOURCE, TAG_REQUEST, MPI_COMM_WORLD, &status);

        if (incoming_req.sn > RN[incoming_req.rank]) {
            RN[incoming_req.rank] = incoming_req.sn;
        }

        // std::cout << "Received request from process " << incoming_req.rank 
		//                 << " with sequence number " << incoming_req.sn << std::endl;

        // Update the flag to check for more messages
        MPI_Iprobe(MPI_ANY_SOURCE, TAG_REQUEST, MPI_COMM_WORLD, &flag, &status);
    }
}

void receive_marker(bool& has_marker, std::queue<int>& que, std::vector<int>& LN, int size) {
    MPI_Status status;
    int queue_size;

    MPI_Recv(&queue_size, 1, MPI_INT, MPI_ANY_SOURCE, TAG_MARKER, MPI_COMM_WORLD, &status);

    std::vector<int> queue_array(queue_size);
    MPI_Recv(queue_array.data(), queue_size, MPI_INT, status.MPI_SOURCE, TAG_MARKER, MPI_COMM_WORLD, &status);
    for (auto proc : queue_array) {
        que.push(proc);
    }

    MPI_Recv(LN.data(), size, MPI_INT, status.MPI_SOURCE, TAG_MARKER, MPI_COMM_WORLD, &status);

    has_marker = true;
    std::cout << "Process received marker." << std::endl;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
	std::vector<MPI_Request> requests;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    srand(time(0) + rank);
	
    bool has_marker = (rank == 0);
    std::vector<int> LN(size, 0);   // Last processed requests
    std::vector<int> RN(size, 0);   // Request counters
    std::queue<int> que;
    int entrances_left = MAX_ENTRANCE;
	int exit_counter = 0;
    if (rank != 0) {
        send_request(RN, rank, size, requests);
    }

    while (entrances_left > 0 || has_marker) {
        process_incoming_requests(RN, que, size);
		
        if (has_marker) {
            if (entrances_left > 0) {
                enter_critical_section(rank);
                entrances_left--;
            }

            for (int i = 0; i < size; i++) {
                if (RN[i] == LN[i] + 1) {
                    que.push(i);
					LN[i] = RN[i];
                }
            }

            if (!que.empty()) {
                int next = que.front();
                que.pop();
                int queue_size = que.size();

                std::cout << "Process " << rank << " sending marker to process " << next << std::endl;

                // Send queue and LN
                MPI_Send(&queue_size, 1, MPI_INT, next, TAG_MARKER, MPI_COMM_WORLD);
                std::vector<int> queue_array(queue_size);
                for (int i = 0; i < queue_size; ++i) {
                    queue_array[i] = que.front();
                    que.pop();
                }
                MPI_Send(queue_array.data(), queue_size, MPI_INT, next, TAG_MARKER, MPI_COMM_WORLD);
                MPI_Send(LN.data(), size, MPI_INT, next, TAG_MARKER, MPI_COMM_WORLD);

                has_marker = false;
            }
			exit_counter++;
			if (exit_counter == 3) break;
        } else {
            int flag;
            MPI_Iprobe(MPI_ANY_SOURCE, TAG_MARKER, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
            if (flag) {
                receive_marker(has_marker, que, LN, size);
            }
        }
    }
	MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    std::cout << "Process " << rank << ": DONE" << std::endl;

    MPI_Finalize();
    return 0;
}

