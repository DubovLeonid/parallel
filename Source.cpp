#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <sstream>

using namespace std;
using namespace std::chrono;

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
            file << fixed << setprecision(6) << row[j];
            if (j < row.size() - 1) file << " ";
        }
        file << endl;
    }
    file.close();
}

vector<vector<double>> multiply_matrices(const vector<vector<double>>& A,
    const vector<vector<double>>& B) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k)
                sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
    return C;
}

int main() {
    setlocale(LC_ALL, "Russian");

    vector<int> sizes = { 200, 400, 800 };
    const int FINAL_SIZE = 800;

    ofstream time_file("execution_times.txt");
    time_file << "size,total_elements,operations_count,time_seconds" << endl;

    for (int size : sizes) {
        long long total_elements = (long long)size * size;
        long long operations = (long long)size * size * size * 2;

        cout << "Объем задачи: " << size << "x" << size << endl;
        cout << "  Элементов: " << total_elements << endl;
        cout << "  Операций: " << operations << endl;

        auto A = generate_matrix(size);
        auto B = generate_matrix(size);

        save_matrix(A, make_filename("matrix_A_", size));
        save_matrix(B, make_filename("matrix_B_", size));

        auto start = high_resolution_clock::now();
        auto C = multiply_matrices(A, B);
        auto end = high_resolution_clock::now();

        duration<double> elapsed = end - start;

        save_matrix(C, make_filename("matrix_C_", size));

        if (size == FINAL_SIZE)
            save_matrix(C, "result_matrix.txt");

        time_file << size << "," << total_elements << "," << operations << "," << elapsed.count() << endl;
        cout << "  Время: " << elapsed.count() << " сек" << endl;
    }

    time_file.close();
    return 0;
}