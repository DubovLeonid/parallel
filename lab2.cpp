#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <sstream>
#include <omp.h>

using namespace std;
using namespace std::chrono;




vector<double> generate_matrix_flat(int size) {
    vector<double> matrix(size * size);

#pragma omp parallel
    {
        random_device rd;
        mt19937 gen(rd() + omp_get_thread_num());
        uniform_real_distribution<double> dist(0.0, 100.0);

#pragma omp for
        for (int idx = 0; idx < size * size; ++idx)
            matrix[idx] = dist(gen);
    }
    return matrix;
}

// Сохранение 
void save_matrix_flat(const vector<double>& matrix, int size, const string& filename) {
    ofstream file(filename);
    file << size << " " << size << endl;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            file << fixed << setprecision(6) << matrix[i * size + j];
            if (j < size - 1) file << " ";
        }
        file << endl;
    }
    file.close();
}

// Параллельное умножение static
vector<double> multiply_parallel_static_flat(const vector<double>& A,
    const vector<double>& B, int n) {
    vector<double> C(n * n, 0.0);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            int idx_a = i * n;       // начало строки i в A
            for (int k = 0; k < n; ++k)
                sum += A[idx_a + k] * B[k * n + j];
            C[i * n + j] = sum;
        }
    }
    return C;
}

// Параллельное умножение dynamic 
vector<double> multiply_parallel_dynamic_flat(const vector<double>& A,
    const vector<double>& B, int n) {
    vector<double> C(n * n, 0.0);
#pragma omp parallel for schedule(dynamic, 8)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            int idx_a = i * n;
            for (int k = 0; k < n; ++k)
                sum += A[idx_a + k] * B[k * n + j];
            C[i * n + j] = sum;
        }
    }
    return C;
}

// Оптимизированное умножение 
vector<double> multiply_parallel_optimized_flat(const vector<double>& A,
    const vector<double>& B, int n) {
    vector<double> C(n * n, 0.0);
#pragma omp parallel for schedule(guided)
    for (int i = 0; i < n; ++i) {
        int idx_c = i * n;
        int idx_a = i * n;
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
#pragma omp simd reduction(+:sum)
            for (int k = 0; k < n; ++k)
                sum += A[idx_a + k] * B[k * n + j];
            C[idx_c + j] = sum;
        }
    }
    return C;
}


typedef vector<double>(*MatrixFuncFlat)(const vector<double>&,
    const vector<double>&, int);

void run_experiment_flat(int size, const vector<int>& thread_counts, ofstream& results_file) {
    long long total_elements = (long long)size * size;
    long long operations = (long long)size * size * size * 2;

    cout << "\n" << string(60, '=') << endl;
    cout << "Размер матрицы: " << size << "x" << size << endl;
    cout << "Элементов: " << total_elements << endl;
    cout << "Операций: " << operations << endl;

    // ===== 1. Генерация матриц ОДИН раз =====
    cout << "  Генерация матриц " << endl;
    auto matA = generate_matrix_flat(size);
    auto matB = generate_matrix_flat(size);

    save_matrix_flat(matA, size, "matrix_A_" + to_string(size) + ".txt");
    save_matrix_flat(matB, size, "matrix_B_" + to_string(size) + ".txt");

    // ===== 2. Все методы для каждого количества потоков =====
    vector<pair<string, MatrixFuncFlat>> methods = {
        {"Статический", multiply_parallel_static_flat},
        {"Динамический", multiply_parallel_dynamic_flat},
        {"Оптимизированный", multiply_parallel_optimized_flat}
    };

    bool baseline_saved = false;

    for (int num_threads : thread_counts) {
        omp_set_num_threads(num_threads);
        cout << "\n  --- Потоков: " << num_threads << " ---" << endl;

        for (const auto& method : methods) {
            auto start = high_resolution_clock::now();
            auto matC = method.second(matA, matB, size);
            auto end = high_resolution_clock::now();
            duration<double> elapsed = end - start;

            cout << "    " << method.first << ": " << fixed << setprecision(4)
                << elapsed.count() << " сек" << endl;

            results_file << size << "," << num_threads << "," << method.first << ","
                << total_elements << "," << operations << ","
                << elapsed.count() << endl;

            // Сохраняем эталонную C от Статический с 1 потоком
            if (num_threads == 1 && method.first == "Статический" && !baseline_saved) {
                save_matrix_flat(matC, size, "matrix_C_" + to_string(size) + ".txt");
                baseline_saved = true;
            }
        }
    }
}

int main() {
    setlocale(LC_ALL, "Russian");

    int max_threads = omp_get_max_threads();
    cout << "Максимальное количество потоков: " << max_threads << endl;
    cout << "Количество процессоров: " << omp_get_num_procs() << endl;

    vector<int> sizes = { 200, 400, 800, 1200, 1600, 2000 };
    vector<int> thread_counts = { 1, 2, 4, 8 };

    // Убираем потоки, превышающие максимальное
    thread_counts.erase(
        remove_if(thread_counts.begin(), thread_counts.end(),
            [max_threads](int n) { return n > max_threads; }),
        thread_counts.end()
    );

    if (thread_counts.empty()) {
        thread_counts.push_back(max_threads);
    }

    cout << "Будут использованы потоки: ";
    for (int t : thread_counts) cout << t << " ";
    cout << endl;

    ofstream results_file("openmp_results.csv");
    results_file << "size,threads,method,total_elements,operations,time_seconds" << endl;

    auto total_start = high_resolution_clock::now();

    for (int size : sizes) {
        run_experiment_flat(size, thread_counts, results_file);
    }

    auto total_end = high_resolution_clock::now();
    duration<double> total_elapsed = total_end - total_start;

    cout << "\n" << string(60, '=') << endl;
    cout << "Общее время выполнения: " << total_elapsed.count() << " сек" << endl;

    results_file.close();

    cout << "\nИнформация о системе OpenMP:" << endl;
#ifdef _OPENMP
    cout << "  Версия OpenMP: " << _OPENMP << endl;
#endif
    cout << "  Максимальное количество потоков: " << omp_get_max_threads() << endl;
    cout << "  Количество процессоров: " << omp_get_num_procs() << endl;

    return 0;
}