"""
Верификация результатов MPI-умножения матриц (Лабораторная работа №3)
Сравнивает результаты MPI с эталонным умножением через NumPy
Строит графики зависимости времени от размера матриц и числа процессов
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# Порог допустимой погрешности (из-за разного порядка суммирования)
TOLERANCE = 5e-3


# ============================================================
# 1. Загрузка матриц из файлов
# ============================================================
def load_matrix(filename):
    """Загружает матрицу из текстового файла"""
    with open(filename, 'r') as f:
        rows, cols = map(int, f.readline().split())
        matrix = np.zeros((rows, cols))
        for i, line in enumerate(f):
            matrix[i] = list(map(float, line.split()))
    return matrix


# ============================================================
# 2. Верификация результатов
# ============================================================
def verify_all_results():
    """Проверяет все результаты MPI-умножения через NumPy"""

    print("=" * 70)
    print("ВЕРИФИКАЦИЯ РЕЗУЛЬТАТОВ MPI-УМНОЖЕНИЯ МАТРИЦ")
    print(f"Допустимая погрешность: {TOLERANCE:.1e}")
    print("=" * 70)

    # Ищем все сохранённые матрицы A (по ним определяем размеры)
    a_files = sorted(glob.glob("matrix_A_*.txt"))

    if not a_files:
        print("\n[ОШИБКА] Не найдены файлы matrix_A_*.txt")

        return

    results_ok = []
    results_fail = []

    for a_file in a_files:
        # Извлекаем размер матрицы из имени: matrix_A_200.txt -> 200
        size = int(a_file.replace("matrix_A_", "").replace(".txt", ""))

        # Ищем соответствующие файлы B и C
        b_file = f"matrix_B_{size}.txt"
        c_file = f"matrix_C_mpi_{size}.txt"

        print(f"\n--- Проверка матрицы {size}x{size} ---")

        if not os.path.exists(b_file):
            print(f"  [ПРОПУСК] Не найден файл {b_file}")
            continue

        if not os.path.exists(c_file):
            print(f"  [ПРОПУСК] Не найден файл {c_file}")
            continue

        try:
            # Загружаем матрицы
            A = load_matrix(a_file)
            B = load_matrix(b_file)
            C_mpi = load_matrix(c_file)
        except Exception as e:
            print(f"  [ОШИБКА] Не удалось загрузить матрицы: {e}")
            continue

        # Эталонное умножение через NumPy
        C_numpy = np.dot(A, B)

        # Сравнение
        diff = np.abs(C_mpi - C_numpy)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        elements = size * size
        wrong_elements = np.sum(diff > TOLERANCE)
        percent_wrong = 100.0 * wrong_elements / elements

        print(f"  Максимальная погрешность: {max_diff:.4e}")
        print(f"  Средняя погрешность:     {mean_diff:.4e}")
        print(f"  Элементов > {TOLERANCE:.0e}:  {wrong_elements} из {elements} ({percent_wrong:.2f}%)")

        if max_diff < TOLERANCE:
            print(f"  РЕЗУЛЬТАТ: OK  (погрешность в пределах допуска)")
            results_ok.append(size)
        else:
            print(f"  РЕЗУЛЬТАТ: ОШИБКА  (превышен допуск {TOLERANCE:.0e})")
            results_fail.append(size)

            # Показываем примеры несовпадений
            error_positions = np.where(diff > TOLERANCE)
            if len(error_positions[0]) > 0:
                print(f"  Примеры несовпадений (первые 3):")
                for idx in range(min(3, len(error_positions[0]))):
                    i, j = error_positions[0][idx], error_positions[1][idx]
                    print(f"    [{i},{j}]: MPI={C_mpi[i, j]:.10f}, "
                          f"NumPy={C_numpy[i, j]:.10f}, diff={diff[i, j]:.4e}")

    # Итог
    print("\n" + "=" * 70)
    print("ИТОГ ВЕРИФИКАЦИИ")
    print("=" * 70)
    print(f"Успешно проверено: {len(results_ok)} из {len(results_ok) + len(results_fail)}")
    if results_ok:
        print(f"  OK: {results_ok}")
    if results_fail:
        print(f"  Ошибки: {results_fail}")
    else:
        print("  Все результаты корректны! ")
    print("=" * 70)


# ============================================================
# 3. Построение графиков
# ============================================================
def plot_results(csv_file="mpi_results.csv"):
    """Строит графики по результатам из CSV-файла"""

    if not os.path.exists(csv_file):
        print(f"\n[ОШИБКА] Файл {csv_file} не найден!")
        print("Сначала запустите MPI-программу с разным количеством процессов")
        return

    # Загружаем данные
    data = pd.read_csv(csv_file)

    if data.empty:
        print(f"\n[ОШИБКА] Файл {csv_file} пуст!")
        return

    print(f"\nЗагружено {len(data)} записей из {csv_file}")
    print(f"Размеры матриц: {sorted(data['size'].unique())}")
    print(f"Количество процессов: {sorted(data['processes'].unique())}")

    # Создаём фигуру с двумя графиками
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('MPI-умножение матриц: анализ производительности', fontsize=14)

    # График 1: Время от размера матрицы (для разного числа процессов)
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']
    for idx, procs in enumerate(sorted(data['processes'].unique())):
        subset = data[data['processes'] == procs]
        color = colors[idx % len(colors)]
        ax1.plot(subset['size'], subset['time_sec'],
                 marker='o', linewidth=2, color=color,
                 label=f'{procs} процесс(ов)')

    ax1.set_xlabel('Размер матрицы (n x n)', fontsize=12)
    ax1.set_ylabel('Время (сек)', fontsize=12)
    ax1.set_title('Зависимость времени от размера матрицы', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # График 2: Ускорение относительно 1 процесса
    base_times = data[data['processes'] == 1][['size', 'time_sec']].copy()
    base_times = base_times.rename(columns={'time_sec': 'base_time'})

    merged = data.merge(base_times, on='size', how='left')
    merged['speedup'] = merged['base_time'] / merged['time_sec']

    for idx, procs in enumerate(sorted(merged['processes'].unique())):
        if procs == 1:
            continue
        subset = merged[merged['processes'] == procs]
        color = colors[idx % len(colors)]
        ax2.plot(subset['size'], subset['speedup'],
                 marker='s', linewidth=2, color=color,
                 label=f'{procs} процесс(ов)')

    # Идеальное ускорение (пунктир)
    max_procs = max(data['processes'].unique())
    sizes_range = sorted(data['size'].unique())
    for procs in sorted(data['processes'].unique()):
        if procs == 1:
            continue
        ax2.axhline(y=procs, color='gray', linestyle='--', alpha=0.2)

    ax2.set_xlabel('Размер матрицы (n x n)', fontsize=12)
    ax2.set_ylabel('Ускорение (раз)', fontsize=12)
    ax2.set_title('Ускорение относительно 1 процесса', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mpi_performance.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nГрафик сохранён в файл mpi_performance.png")

    # Выводим таблицу с ускорением
    print("\n" + "=" * 70)
    print("ТАБЛИЦА УСКОРЕНИЯ")
    print("=" * 70)
    pivot = merged.pivot_table(values='speedup', index='size', columns='processes')
    print(pivot.round(2).to_string())

    # Выводим таблицу с временами
    print("\n" + "=" * 70)
    print("ТАБЛИЦА ВРЕМЕНИ (сек)")
    print("=" * 70)
    pivot_time = data.pivot_table(values='time_sec', index='size', columns='processes')
    print(pivot_time.round(3).to_string())

    print("=" * 70)


# ============================================================
# 4. Главная функция
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ЛАБОРАТОРНАЯ РАБОТА №3: MPI-УМНОЖЕНИЕ МАТРИЦ")
    print("=" * 70)

    # Часть 1: Верификация
    print("\n[ШАГ 1] Верификация результатов...")
    verify_all_results()

    # Часть 2: Графики
    print("\n[ШАГ 2] Построение графиков...")
    plot_results("mpi_results.csv")

    print("\n" + "=" * 70)
