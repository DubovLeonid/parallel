import numpy as np
import matplotlib.pyplot as plt
import os
import sys


def load_cpp_matrix(filename):
    with open(filename, 'r') as f:
        rows, cols = map(int, f.readline().split())
        matrix = np.zeros((rows, cols))
        for i, line in enumerate(f):
            matrix[i, :] = np.fromstring(line, sep=' ')
    return matrix


def verify_results(sizes):
    print("\n===== ВЕРИФИКАЦИЯ РЕЗУЛЬТАТОВ =====")
    all_correct = True

    for size in sizes:
        try:
            print(f"\nОбъем задачи: {size}x{size} ({size * size} элементов)")

            A = load_cpp_matrix(f"matrix_A_{size}.txt")
            B = load_cpp_matrix(f"matrix_B_{size}.txt")
            C_cpp = load_cpp_matrix(f"matrix_C_{size}.txt")

            C_numpy = np.dot(A, B)

            max_diff = np.max(np.abs(C_cpp - C_numpy))
            is_correct = np.allclose(C_cpp, C_numpy, rtol=1e-5, atol=1e-8)

            status = "✓ ПРОЙДЕНА" if is_correct else "✗ ОШИБКА"
            print(f"Размер {size}x{size}: {status} (макс. расхождение: {max_diff:.2e})")

            if not is_correct:
                all_correct = False

        except FileNotFoundError:
            print(f"Размер {size}x{size}: Файл не найден")
            all_correct = False

    return all_correct


def plot_execution_time():
    if not os.path.exists("execution_times.txt"):
        print("Файл execution_times.txt не найден!")
        return

    data = np.loadtxt("execution_times.txt", delimiter=",", skiprows=1)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    sizes = data[:, 0].astype(int)
    times = data[:, 3]
    operations = data[:, 2].astype(int)

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times, 'b-o', linewidth=2, markersize=8, markerfacecolor='red')
    plt.xlabel("Размер матрицы (N x N)", fontsize=12)
    plt.ylabel("Время выполнения (секунды)", fontsize=12)
    plt.title("Зависимость времени умножения матриц от размера", fontsize=14)
    plt.grid(True, alpha=0.3, linestyle='--')

    for i, (size, time) in enumerate(zip(sizes, times)):
        plt.annotate(f'{time:.3f} с', (size, time), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig("execution_time_plot.png", dpi=150)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, operations, 'g-s', linewidth=2, markersize=8, markerfacecolor='orange')
    plt.xlabel("Размер матрицы (N x N)", fontsize=12)
    plt.ylabel("Количество операций", fontsize=12)
    plt.title("Зависимость количества операций от размера матрицы", fontsize=14)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ticklabel_format(style='scientific', axis='y', scilimits=(9, 9))

    for i, (size, op) in enumerate(zip(sizes, operations)):
        plt.annotate(f'{op:,}', (size, op), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=8,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

    plt.tight_layout()
    plt.savefig("operations_plot.png", dpi=150)
    plt.show()

    print("\n" + "=" * 65)
    print("ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 65)
    print(f"{'Размер':<12} {'Элементов':<12} {'Операций':<15} {'Время (с)':<12}")
    print("-" * 65)
    for i in range(len(sizes)):
        elements = int(sizes[i]) * int(sizes[i])
        print(f"{sizes[i]}x{sizes[i]:<6} {elements:<12,} {operations[i]:<15,} {times[i]:<12.4f}")
    print("=" * 65)


if __name__ == "__main__":
    sizes = [200, 400, 800]

    is_correct = verify_results(sizes)

    if is_correct:
        print("\n" + "=" * 50)
        print("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ УСПЕШНО!")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("ОБНАРУЖЕНЫ РАСХОЖДЕНИЯ!")
        print("=" * 50)
        sys.exit(1)

    print("\n===== ПОСТРОЕНИЕ ГРАФИКОВ =====")
    plot_execution_time()
    print("\nГрафики сохранены:")
    print("  - execution_time_plot.png")
    print("  - operations_plot.png")