#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <sstream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

using namespace std;
using namespace std::chrono;

__global__ void matrixMulNaive(const double* A, const double* B, double* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

string make_filename(const string& prefix, int size) {
    ostringstream ss;
    ss << prefix << size << ".txt";
    return ss.str();
}

vector<vector<double>> generate_matrix(int size) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(0.0, 100.0);

    vector<vector<double>> matrix(size, vector<double>(size));
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            matrix[i][j] = dist(gen);
    return matrix;
}

void save_matrix(const vector<vector<double>>& matrix, const string& filename) {
    ofstream file(filename);
    file << matrix.size() << " " << matrix[0].size() << endl;
    for (const auto& row : matrix) {
        for (size_t j = 0; j < row.size(); ++j) {
            file << fixed << setprecision(15) << row[j];
            if (j < row.size() - 1) file << " ";
        }
        file << endl;
    }
    file.close();
}

vector<double> flatten_matrix(const vector<vector<double>>& matrix) {
    int n = matrix.size();
    vector<double> flat(n * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            flat[i * n + j] = matrix[i][j];
    return flat;
}

int main() {
    setlocale(LC_ALL, "Russian");

    vector<int> sizes = { 200, 400, 800, 1200, 1600, 2000 };

    vector<pair<int, int>> block_configs = {
        {8, 8},
        {16, 16},
        {32, 16},
        {32, 32},
    };

    ofstream results_file("cuda_experiments_results.csv");
    results_file << "size,block_x,block_y,grid_x,grid_y,threads_per_block,"
        << "time_seconds,gflops" << endl;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cout << "========================================" << endl;
    cout << "ИНФОРМАЦИЯ О GPU" << endl;
    cout << "========================================" << endl;
    cout << "Устройство: " << prop.name << endl;
    cout << "Максимум потоков на блок: " << prop.maxThreadsPerBlock << endl;
    cout << "Размер warp: " << prop.warpSize << endl;
    cout << "Мультипроцессоров: " << prop.multiProcessorCount << endl;
    cout << "Макс. shared memory на блок: " << prop.sharedMemPerBlock / 1024 << " KB" << endl;
    cout << "========================================" << endl << endl;

    for (int size : sizes) {
        cout << "======================================" << endl;
        cout << "МАТРИЦЫ " << size << "x" << size << endl;
        cout << "======================================" << endl;

        auto A = generate_matrix(size);
        auto B = generate_matrix(size);

        save_matrix(A, make_filename("matrix_A_", size));
        save_matrix(B, make_filename("matrix_B_", size));

        vector<double> h_A = flatten_matrix(A);
        vector<double> h_B = flatten_matrix(B);
        vector<double> h_C_gpu(size * size);

        double* d_A, * d_B, * d_C;
        size_t bytes = size * size * sizeof(double);

        cudaMalloc(&d_A, bytes);
        cudaMalloc(&d_B, bytes);
        cudaMalloc(&d_C, bytes);

        cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

        long long operations = (long long)size * size * size * 2;

        for (const auto& config : block_configs) {
            int block_x = config.first;
            int block_y = config.second;
            int threads_per_block = block_x * block_y;

            dim3 blockDim(block_x, block_y);
            dim3 gridDim((size + blockDim.x - 1) / blockDim.x,
                (size + blockDim.y - 1) / blockDim.y);

            cout << "  Конфигурация: блоки " << block_x << "x" << block_y
                << " (" << threads_per_block << " потоков), "
                << "сетка " << gridDim.x << "x" << gridDim.y << endl;

            cudaDeviceSynchronize();
            auto start = high_resolution_clock::now();

            matrixMulNaive << <gridDim, blockDim >> > (d_A, d_B, d_C, size);

            cudaDeviceSynchronize();
            auto end = high_resolution_clock::now();
            duration<double> elapsed = end - start;

            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess) {
                cout << "    Ошибка CUDA: " << cudaGetErrorString(error) << endl;
                results_file << size << "," << block_x << "," << block_y << ","
                    << gridDim.x << "," << gridDim.y << ","
                    << threads_per_block << ",ERROR,0" << endl;
                continue;
            }

            cudaMemcpy(h_C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost);

            string gpu_filename = "matrix_C_gpu_" + to_string(size) + "_"
                + to_string(block_x) + "x" + to_string(block_y) + ".txt";
            auto C_gpu = vector<vector<double>>(size, vector<double>(size));
            for (int i = 0; i < size; ++i)
                for (int j = 0; j < size; ++j)
                    C_gpu[i][j] = h_C_gpu[i * size + j];
            save_matrix(C_gpu, gpu_filename);

            double gflops = (operations / elapsed.count()) / 1e9;

            cout << "    Время: " << fixed << setprecision(6) << elapsed.count() << " сек" << endl;
            cout << "    Производительность: " << setprecision(2) << gflops << " GFLOPS" << endl;

            results_file << size << "," << block_x << "," << block_y << ","
                << gridDim.x << "," << gridDim.y << ","
                << threads_per_block << ","
                << elapsed.count() << ","
                << gflops << endl;
        }

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        cout << endl;
    }

    results_file.close();

    cout << "========================================" << endl;
    cout << "ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ" << endl;
    cout << "Результаты сохранены в: cuda_experiments_results.csv" << endl;
    cout << "========================================" << endl;

    return 0;
}