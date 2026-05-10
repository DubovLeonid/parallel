#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <sstream>

using namespace std;

string make_filename(const string& prefix, int size) {
    ostringstream ss;
    ss << prefix << size << ".txt";
    return ss.str();
}

vector<vector<double>> generate_matrix(int size, unsigned seed) {
    mt19937 gen(seed);
    uniform_real_distribution<double> dist(0.0, 100.0);

    vector<vector<double>> matrix(size, vector<double>(size));
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            matrix[i][j] = dist(gen);
    return matrix;
}

void save_matrix(const vector<vector<double>>& matrix, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: cannot open file " << filename << endl;
        return;
    }
    file << matrix.size() << " " << matrix[0].size() << endl;
    for (const auto& row : matrix) {
        for (size_t j = 0; j < row.size(); ++j) {
            file << fixed << setprecision(6) << row[j];
            if (j < row.size() - 1) file << " ";
        }
        file << endl;
    }
    file.close();
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

 
     vector<int> sizes = { 200, 400, 800, 1200, 1600, 2000 };  // полный эксперимент

    ofstream results_file;
    if (rank == 0) {
        results_file.open("mpi_results.csv", ios::app);
        ifstream check("mpi_results.csv");
        if (check.peek() == ifstream::traits_type::eof()) {
            results_file << "size,processes,time_sec,total_elements,mult_ops" << endl;
        }
        check.close();
    }

    for (int n : sizes) {
        vector<vector<double>> A, B;
        vector<double> flatA, flatB;

        if (rank == 0) {
            A = generate_matrix(n, 12345);
            B = generate_matrix(n, 67890);

            flatA.resize(n * n);
            flatB.resize(n * n);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    flatA[i * n + j] = A[i][j];
                    flatB[i * n + j] = B[i][j];
                }
            }

            // Сохраняем исходные матрицы для верификации в Python
            save_matrix(A, make_filename("matrix_A_", n));
            save_matrix(B, make_filename("matrix_B_", n));

            cout << "\n=== Size: " << n << "x" << n
                << " | Processes: " << num_procs << " ===" << endl;
        }

        int base_rows = n / num_procs;
        int remainder = n % num_procs;

        vector<int> sendcounts(num_procs);
        vector<int> displs(num_procs);

        for (int i = 0; i < num_procs; ++i) {
            int rows_for_proc = base_rows + (i < remainder ? 1 : 0);
            sendcounts[i] = rows_for_proc * n;
            displs[i] = (i == 0) ? 0 : displs[i - 1] + sendcounts[i - 1];
        }

        int my_rows = sendcounts[rank] / n;

        if (rank != 0) flatB.resize(n * n);
        MPI_Bcast(flatB.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        vector<double> my_A(my_rows * n);
        MPI_Scatterv(
            flatA.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
            my_A.data(), my_rows * n, MPI_DOUBLE,
            0, MPI_COMM_WORLD
        );

        MPI_Barrier(MPI_COMM_WORLD);
        double start_time = MPI_Wtime();

        vector<double> my_C(my_rows * n, 0.0);
        for (int i = 0; i < my_rows; ++i) {
            for (int j = 0; j < n; ++j) {
                double sum = 0.0;
                for (int k = 0; k < n; ++k) {
                    sum += my_A[i * n + k] * flatB[k * n + j];
                }
                my_C[i * n + j] = sum;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double end_time = MPI_Wtime();
        double elapsed = end_time - start_time;

        vector<double> C_flat;
        if (rank == 0) C_flat.resize(n * n);

        MPI_Gatherv(
            my_C.data(), my_rows * n, MPI_DOUBLE,
            C_flat.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
            0, MPI_COMM_WORLD
        );

        if (rank == 0) {
            vector<vector<double>> C(n, vector<double>(n));
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    C[i][j] = C_flat[i * n + j];

            long long total_elements = (long long)n * n;
            long long mult_ops = (long long)n * n * n * 2;

            cout << "  Time: " << elapsed << " sec" << endl;

            results_file << n << "," << num_procs << "," << elapsed
                << "," << total_elements << "," << mult_ops << endl;

            // Сохраняем результат для верификации в Python
            save_matrix(C, make_filename("matrix_C_mpi_", n));
        }
    }

    if (rank == 0) {
        results_file.close();
        cout << "\n=== Done! Results saved to mpi_results.csv ===" << endl;
        cout << "=== Run verify.py to check results ===" << endl;
    }

    MPI_Finalize();
    return 0;
}