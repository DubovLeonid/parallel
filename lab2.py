import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys


def load_cpp_matrix(filename):
    """Загрузка матрицы из файла C++"""
    with open(filename, 'r') as f:
        rows, cols = map(int, f.readline().split())
        matrix = np.zeros((rows, cols))
        for i, line in enumerate(f):
            matrix[i, :] = np.fromstring(line, sep=' ')
    return matrix


def verify_results(sizes):
    """Верификация результатов умножения через numpy"""
    print("\n" + "=" * 60)
    print("ВЕРИФИКАЦИЯ РЕЗУЛЬТАТОВ (OpenMP)")
    print("=" * 60)
    all_correct = True

    for size in sizes:
        try:
            print(f"\nПроверка размера {size}×{size}:")
            print(f"  Объём задачи: {size * size:,} элементов")

            A = load_cpp_matrix(f"matrix_A_{size}.txt")
            B = load_cpp_matrix(f"matrix_B_{size}.txt")
            C_cpp = load_cpp_matrix(f"matrix_C_{size}.txt")

            # Эталонное умножение через numpy
            C_numpy = np.dot(A, B)

            max_diff = np.max(np.abs(C_cpp - C_numpy))
            is_correct = np.allclose(C_cpp, C_numpy, rtol=1e-5, atol=1e-8)

            status = " ПРОЙДЕНА" if is_correct else "✗ ОШИБКА"
            print(f"  Результат: {status}")
            print(f"  Макс. расхождение: {max_diff:.2e}")

            if not is_correct:
                all_correct = False

        except FileNotFoundError as e:
            print(f"   Файл не найден: {e}")
            all_correct = False

    return all_correct


def load_and_plot_results(csv_file="openmp_results.csv"):
    """Загрузка CSV и построение графиков"""
    if not os.path.exists(csv_file):
        print(f"Файл {csv_file} не найден!")
        return None

    df = pd.read_csv(csv_file, encoding='windows-1251')

    print("\n" + "=" * 60)
    print("ЗАГРУЖЕННЫЕ ДАННЫЕ ИЗ CSV")
    print("=" * 60)
    print(df.to_string(index=False))

    sizes = sorted(df['size'].unique())
    threads_list = sorted(df['threads'].unique())
    methods = df['method'].unique()

    print(f"\nРазмеры матриц: {sizes}")
    print(f"Количество потоков: {threads_list}")
    print(f"Методы: {list(methods)}")

    # ===== График 1: Время от размера матрицы (static) =====
    plt.figure(figsize=(12, 7))
    static_data = df[df['method'] == 'Статический']

    for threads in threads_list:
        subset = static_data[static_data['threads'] == threads]
        if not subset.empty:
            plt.plot(subset['size'], subset['time_seconds'],
                     marker='o', linewidth=2, markersize=8,
                     label=f'{threads} поток(ов)')

    plt.xlabel("Размер матрицы (N×N)", fontsize=12)
    plt.ylabel("Время выполнения (сек)", fontsize=12)
    plt.title("OpenMP Static: зависимость времени от размера матрицы", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig("openmp_time_vs_size.png", dpi=150)
    plt.show()

    # ===== График 2: Время от числа потоков (static) =====
    plt.figure(figsize=(12, 7))

    for size in sizes:
        subset = static_data[static_data['size'] == size]
        if not subset.empty:
            plt.plot(subset['threads'], subset['time_seconds'],
                     marker='s', linewidth=2, markersize=8,
                     label=f'{size}×{size}')

    plt.xlabel("Количество потоков", fontsize=12)
    plt.ylabel("Время выполнения (сек)", fontsize=12)
    plt.title("OpenMP Static: зависимость времени от числа потоков", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(threads_list)
    plt.tight_layout()
    plt.savefig("openmp_time_vs_threads.png", dpi=150)
    plt.show()

    # ===== График 3: Сравнение методов для максимального размера =====
    max_size = max(sizes)
    plt.figure(figsize=(12, 7))

    for method in methods:
        subset = df[(df['size'] == max_size) & (df['method'] == method)]
        if not subset.empty:
            plt.plot(subset['threads'], subset['time_seconds'],
                     marker='D', linewidth=2, markersize=8, label=method)

    plt.xlabel("Количество потоков", fontsize=12)
    plt.ylabel("Время выполнения (сек)", fontsize=12)
    plt.title(f"Сравнение методов OpenMP (размер {max_size}×{max_size})", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(threads_list)
    plt.tight_layout()
    plt.savefig("openmp_methods_comparison.png", dpi=150)
    plt.show()

    # ===== График 4: Ускорение (speedup) =====
    plt.figure(figsize=(12, 7))

    for size in sizes:
        seq_time = static_data[(static_data['size'] == size) &
                               (static_data['threads'] == 1)]['time_seconds']
        if seq_time.empty:
            continue
        seq_time = seq_time.values[0]

        speedups = []
        valid_threads = []
        for threads in threads_list:
            par_time = static_data[(static_data['size'] == size) &
                                   (static_data['threads'] == threads)]['time_seconds']
            if not par_time.empty and par_time.values[0] > 0:
                speedups.append(seq_time / par_time.values[0])
                valid_threads.append(threads)

        if speedups:
            plt.plot(valid_threads, speedups, marker='^', linewidth=2,
                     markersize=8, label=f'{size}×{size}')

    plt.plot(threads_list, threads_list, 'k--', linewidth=1, alpha=0.5, label='Идеальное')

    plt.xlabel("Количество потоков", fontsize=12)
    plt.ylabel("Ускорение (Speedup)", fontsize=12)
    plt.title("OpenMP Static: ускорение относительно 1 потока", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(threads_list)
    plt.tight_layout()
    plt.savefig("openmp_speedup.png", dpi=150)
    plt.show()

    # ===== График 5: Эффективность распараллеливания =====
    plt.figure(figsize=(12, 7))

    for size in sizes:
        seq_time = static_data[(static_data['size'] == size) &
                               (static_data['threads'] == 1)]['time_seconds']
        if seq_time.empty:
            continue
        seq_time = seq_time.values[0]

        efficiencies = []
        valid_threads = []
        for threads in threads_list:
            if threads == 1:
                efficiencies.append(1.0)
                valid_threads.append(threads)
            else:
                par_time = static_data[(static_data['size'] == size) &
                                       (static_data['threads'] == threads)]['time_seconds']
                if not par_time.empty and par_time.values[0] > 0:
                    speedup = seq_time / par_time.values[0]
                    efficiencies.append(speedup / threads)
                    valid_threads.append(threads)

        if efficiencies:
            plt.plot(valid_threads, efficiencies, marker='v', linewidth=2,
                     markersize=8, label=f'{size}×{size}')

    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Идеальная (100%)')
    plt.xlabel("Количество потоков", fontsize=12)
    plt.ylabel("Эффективность", fontsize=12)
    plt.title("OpenMP Static: эффективность распараллеливания", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(threads_list)
    plt.tight_layout()
    plt.savefig("openmp_efficiency.png", dpi=150)
    plt.show()

    # ===== Сводная таблица =====
    print("\n" + "=" * 90)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ (Static)")
    print("=" * 90)
    print(f"{'Размер':<10} {'Потоков':<10} {'Время (с)':<14} {'Операций':<16} {'Ускорение':<12} {'Эффективность':<15}")
    print("-" * 90)

    for size in sizes:
        seq_time = static_data[(static_data['size'] == size) &
                               (static_data['threads'] == 1)]['time_seconds']
        seq_time = seq_time.values[0] if not seq_time.empty else 0

        for threads in threads_list:
            subset = static_data[(static_data['size'] == size) &
                                 (static_data['threads'] == threads)]
            if not subset.empty:
                time_val = subset['time_seconds'].values[0]
                ops = subset['operations'].values[0]
                speedup = seq_time / time_val if time_val > 0 else 0
                efficiency = speedup / threads if threads > 0 else 0
                print(f"{size}×{size:<4} {threads:<10} {time_val:<14.4f} {ops:<16,} {speedup:<12.2f} {efficiency:<15.2%}")

    print("=" * 90)

    return df


if __name__ == "__main__":
    sizes = [200, 400, 800, 1200, 1600, 2000]

    # 1. Верификация
    is_correct = verify_results(sizes)

    if is_correct:
        print("\n" + "=" * 50)
        print(" ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ УСПЕШНО!")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print(" ОБНАРУЖЕНЫ РАСХОЖДЕНИЯ!")
        print("=" * 50)
        sys.exit(1)

    # 2. Загрузка CSV и построение графиков
    print("\n" + "=" * 60)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ OPENMP")
    print("=" * 60)

    df = load_and_plot_results("openmp_results.csv")

    if df is not None:
        print("\nГрафики сохранены:")
        print("  - openmp_time_vs_size.png")
        print("  - openmp_time_vs_threads.png")
        print("  - openmp_methods_comparison.png")
        print("  - openmp_speedup.png")
        print("  - openmp_efficiency.png")