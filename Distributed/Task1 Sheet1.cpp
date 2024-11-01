#include "Task1 Sheet1.h"

void runQ1() {
    int N = 0;
    long long sum = 0;
    cout << omp_get_max_threads() << endl;
    cout << "Enter the size of the array(N) : \n";
    cin >> N;
    vector<int> A(N);

    for (int i = 0; i < N; i++) {
        A[i] = i + 1;
    }

    vector<int> chunks(omp_get_max_threads());
    int length_of_chunk = N / omp_get_max_threads();

    vector<int> results(omp_get_max_threads());

    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        cout << "Hi" << id << endl;
        long long sum_temp = 0;

        int start = id * length_of_chunk;
        int end = start + length_of_chunk;

        for (int i = start; i < end; i++) {
            sum_temp += A[i];
        }
    
        results[id] = sum_temp;

    }
    for (int i = 0; i < omp_get_max_threads(); i++) {
        sum += results[i];
    }

    cout << "Sum = " << sum << endl;
}

void runQ2() {
        int N = 0;
        long long f = 1;
        cout << "Enter a positive integer N: " << endl;
        cin >> N;

        if (N < 0)
            throw "Integer must be Positive";

        int length_of_chunk = N / omp_get_max_threads();
        int remindar = N % omp_get_max_threads();

        vector<int> chunks(omp_get_max_threads());

#pragma omp parallel
        {
            long long f_temp = 1;

            int id = omp_get_thread_num();
            int start = id * length_of_chunk + 1;
            int end = start + length_of_chunk;

            for (int i = start; i < end; i++) {
                f_temp *= i;
            }

            chunks[id] = f_temp;
        }

        for (int i = 0; i < omp_get_max_threads(); i++) {
            f *= chunks[i];
        }

        for (int i = N - remindar + 1; i <= N; i++) {
            f *= i;
        }

        cout << "Factorial = " << f << endl;

}

void runQ3() {

    int N;
    int inside_circle = 0;
    double pi_estimate;
    int results[omp_get_max_threads()];

    std::cout << "Enter the number of points to generate: ";
    std::cin >> N;

    srand(time(0));

#pragma omp parallel
    {
        int inside_circle_local = 0;
        int id = omp_get_thread_num();

        for (int i = 0; i < N; i++) {

            double x = (double)rand() / RAND_MAX;
            double y = (double)rand() / RAND_MAX;

            if (x * x + y * y <= 1.0) {
                inside_circle_local++;
            }
        }

        results[id] = inside_circle_local;
    }

    for (int i = 0; i < omp_get_max_threads(); i++) {
        inside_circle += results[i];
    }

    pi_estimate = 4.0 * (double)inside_circle / (double)N;
    std::cout << "Estimated value of Pi: " << pi_estimate / omp_get_max_threads() << std::endl;
}