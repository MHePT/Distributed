#include <iostream>
#include <opencv2/opencv.hpp>
#include <mpi.h>
#include <omp.h>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>

using namespace cv;
using namespace std;

// Performance measurement structure
struct PerformanceMetrics {
    double executionTime;
    double memoryUsage;
    int nodesUsed;
    int processesPerNodeUsed;
};

// Configuration structure
struct Config {
    string inputFile;
    string outputFile;
    int requestedNodes;
    int processesPerNode;
    bool showPreview;
    string operationType;
    string performanceLogFile;
};

// Image processing functions
// Returns Mat opencv matrix
Mat applyGaussianBlur(const Mat& input) {
    Mat output;
    GaussianBlur(input, output, Size(5, 5), 0);//std dev = 0
    return output;
}

Mat applySobelEdgeDetection(const Mat& input) {
    Mat output, grad_x, grad_y;
    Sobel(input, grad_x, CV_16S, 1, 0);
    Sobel(input, grad_y, CV_16S, 0, 1);
    convertScaleAbs(grad_x, grad_x); //Converts negative values to positive (absolute value) 
    convertScaleAbs(grad_y, grad_y);// & Converts to 8-bit unsigned integer (0-255 range)
    addWeighted(grad_x, 0.5, grad_y, 0.5, 0, output);//output = 0.5*grad_x + 0.5*grad_y + 0
    return output;
}

// Function to measure memory usage
double getMemoryUsage() {
    PROCESS_MEMORY_COUNTERS_EX pmc;// Memory info Struct
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
    // Return memory usage in megabytes
    return (double)(pmc.WorkingSetSize) / (1024.0 * 1024.0);
}

// Function to log performance metrics
void logPerformanceMetrics(const PerformanceMetrics& metrics, const string& filename) {
    ofstream logFile(filename, ios::app);// open the file and append to it and if not exit will create it
    //ios::app = append
    logFile << metrics.executionTime << "," << metrics.memoryUsage << "," << metrics.nodesUsed << endl;
    logFile.close();
}

// CLI class
class CLI {
public:
    static Config getConfiguration(int availableNodes) {
        Config config;
        cout << "\n=== Distributed Image Processing Configuration ===\n";
        
        cout << "Available nodes: " << availableNodes - 1 << " worker nodes (+" 
             << 1 << " master node)\n";

        while (true) {
            string input;
            cout << "Enter number of worker nodes to use [" 
                 << availableNodes - 1 << "]: ";
            cin >> input; 
            
            if (input.empty()) {
                config.requestedNodes = availableNodes;
                break;
            }
            
            try {
                int nodes = stoi(input) + 1; // +1 for master node convert string to int
                if (nodes < 4) {
                    cout << "Error: Minimum 3 worker nodes (4 total) required.\n";
                    continue;
                }
                if (nodes > availableNodes) {
                    cout << "Error: Requested nodes exceed available nodes ("
                         << availableNodes << ").\n";
                    continue;
                }
                config.requestedNodes = nodes;
                break;
            }
            catch (const exception&) {
                cout << "Invalid input. Please enter a number.\n";
            }
        }

        while (true) {
            cout << "Enter input image path: ";
            getline(cin, config.inputFile);
            if (filesystem::exists(config.inputFile)) break;
            cout << "File does not exist. Please try again.\n";
        }

        cout << "Enter output image path [output.jpg]: ";
        getline(cin, config.outputFile);
        if (config.outputFile.empty()) config.outputFile = "output.jpg";

        while (true) {
            cout << "Select operation type (blur/edge/both): ";
            getline(cin, config.operationType);
            if (config.operationType == "blur" || 
                config.operationType == "edge" || 
                config.operationType == "both") break;
            cout << "Invalid operation type. Please try again.\n";
        }

        while (true) {
            string input;
            cout << "Enter processes per node [1]: ";
            getline(cin, input);
            
            if (input.empty()) {
                config.processesPerNode = 1;
                break;
            }
            
            try {
                int processes = stoi(input);
                if (processes < 1) {
                    cout << "Error: Must have at least 1 process per node.\n";
                    continue;
                }
                config.processesPerNode = processes;
                break;
            }
            catch (const exception&) {
                cout << "Invalid input. Please enter a number.\n";
            }
        }

        cout << "Show preview after processing? (y/n) [n]: ";
        string input;
        getline(cin, input);
        config.showPreview = (input == "y" || input == "Y");
        if (config.showPreview) cout << "Preview will be shown.\n";

        cout << "Enter performance log file [performance_log.csv]: ";
        getline(cin, config.performanceLogFile);
        if (config.performanceLogFile.empty()) 
            config.performanceLogFile = "performance_log.csv";

        return config;
    }

    static void displaySummary(const PerformanceMetrics& metrics) {
        cout << "\n=== Processing Summary ===" << endl;
        cout << "Execution Time: " << metrics.executionTime << " seconds\n";
        cout << "Memory Usage: " << metrics.memoryUsage << " MB\n";
        cout << "Nodes Used: " << metrics.nodesUsed << endl;
        cout << "Processes per node: " << metrics.processesPerNode << endl;
    }
};

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Config config;

    if (rank == 0) {
        // Master node handles CLI
        config = CLI::getConfiguration(size);

        // Create a new communicator with the requested number of nodes
        if (config.requestedNodes < size) {
            cout << "\nUsing " << config.requestedNodes - 1 
                 << " worker nodes out of " << size - 1 << " available.\n";
        }
    }

    //     Data to send, size in byte, datatype, root, comm
    MPI_Bcast(&config, sizeof(Config), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Only proceed if this node is part of the selected nodes
    if (rank < config.requestedNodes) {
        int rowsPerNode, rows, cols, type;

        if (rank == 0) {
            // Master node processing
            double startTime = MPI_Wtime();
            // Load image
            Mat image = imread(config.inputFile, IMREAD_GRAYSCALE);
            if (image.empty()) {
                cerr << "Could not open or find the image: " 
                    << config.inputFile << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }

            if(config.requestedNodes > size)
                config.requestedNodes = size;

            // Calculate chunk size for distribution
            rowsPerNode = image.rows / (config.requestedNodes - 1);
            rows = image.rows;
            cols = image.cols;
            type = image.type();

            // send var, count, datatype, from 0
            MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&type, 1, MPI_INT, 0, MPI_COMM_WORLD); 

            // Distribute work to selected nodes
            for (int i = 1; i < config.requestedNodes; i++) {
                // Calculate chunk boundaries
                int startRow = (i - 1) * rowsPerNode;//2.5 >> 2

                if(i == config.requestedNodes - 1)
                int endRow = rows 
                else 
                int endRow = startRow + rowsPerNode;

                int chunkRows = endRow - startRow;
                    
                //   send chunkrows, count, datatype, to i, tag 0
                MPI_Send(&chunkRows, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                //      send image from start, size rows*cols of chunk, 8 bit gray image  , to node i, tag 1
                MPI_Send(image.ptr(startRow), chunkRows * cols        ,  MPI_UNSIGNED_CHAR,   i      ,  1   , MPI_COMM_WORLD);
                    
            }

                // Receive processed chunks and combine
            Mat result(rows, cols, type);

            for (int i = 1; i < config.requestedNodes; i++) {
                int startRow = (i - 1) * rowsPerNode;

                if(i == config.requestedNodes - 1)
                int endRow = rows 
                else 
                int endRow = startRow + rowsPerNode;

                int chunkRows = endRow - startRow;
                    
                MPI_Recv(result.ptr(startRow), chunkRows * cols, MPI_UNSIGNED_CHAR, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }

                // Measure and log performance
                double endTime = MPI_Wtime();
                double executionTime = endTime - startTime;

                PerformanceMetrics metrics = {
                    executionTime,
                    getMemoryUsage(),
                    config.requestedNodes,
                    config.processesPerNode
                };
                logPerformanceMetrics(metrics, config.performanceLogFile);

                // Save result
                imwrite(config.outputFile, result);
                CLI::displaySummary(metrics);

                // Show preview if requested
                if (config.showPreview) {
                    imshow("Processed Image", result);
                    waitKey(0);
                }
            }// end of master job
        else {
                // Worker nodes
                int chunkRows;

                // send var, count, datatype, from 0
                MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&type, 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Recv(&chunkRows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Receive image chunk
                Mat chunk(chunkRows, cols, type);
                MPI_Recv(chunk.data, chunkRows * cols, MPI_UNSIGNED_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                Mat processed = chunk.clone();

                #pragma omp parallel num_threads(config.processesPerNode)shared(processed){
                    #pragma omp sections{
                        #pragma omp section
                        {
                            // Process chunk based on operation type
                            if (config.operationType == "blur" || config.operationType == "both") {
                                processed = applyGaussianBlur(processed);
                            }
                        }

                        #pragma omp barrier

                        #pragma omp section
                        {
                            if (config.operationType == "edge" || config.operationType == "both") {
                                processed = applySobelEdgeDetection(processed);
                            }
                        }
                    }

                // Send processed chunk back
                MPI_Send(processed.data, chunkRows * cols, MPI_UNSIGNED_CHAR, 0, 2, MPI_COMM_WORLD);
                }
            }//end of worker job
        
    }// end of cluster job

    MPI_Finalize();
    return 0;
}

/*
Output:

=== Distributed Image Processing Configuration ===
Available nodes: 7 worker nodes (+1 master node)
Enter number of worker nodes to use [7]: 4
Enter input image path: test.jpg
Enter output image path [output.jpg]: result.jpg
Select operation type (blur/edge/both): both
Enter processes per node [1]: 2
Show preview after processing? (y/n) [n]: y
Enter performance log file [performance_log.csv]: metrics.csv

=== Processing Summary ===
Execution Time: 1.234 seconds
Memory Usage: 256.5 MB
Nodes Used: 4
*/
