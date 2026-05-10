import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")


def load_matrix_from_file(filename):
    try:
        with open(filename, 'r') as f:
            first_line = f.readline().strip()
            rows, cols = map(int, first_line.split())
            matrix = np.zeros((rows, cols))
            for i, line in enumerate(f):
                values = list(map(float, line.strip().split()))
                matrix[i, :len(values)] = values
        return matrix
    except FileNotFoundError:
        print(f"  Файл {filename} не найден")
        return None
    except Exception as e:
        print(f"   Ошибка при чтении {filename}: {e}")
        return None


def verify_all_results():
    print("=" * 60)
    print("ВЕРИФИКАЦИЯ РЕЗУЛЬТАТОВ")
    print("=" * 60)

    sizes = [200, 400, 800, 1200, 1600, 2000]
    configs = ["8x8", "16x16", "32x16", "32x32"]
    EPSILON = 1e-7

    all_verified = True
    results = []

    for size in sizes:
        A = load_matrix_from_file(f"matrix_A_{size}.txt")
        B = load_matrix_from_file(f"matrix_B_{size}.txt")

        if A is None or B is None:
            print(f"Размер {size}: пропущен (нет исходных матриц)")
            continue

        C_numpy = np.dot(A, B)
        print(f"\nМатрицы {size}x{size}:")

        for config in configs:
            gpu_file = f"matrix_C_gpu_{size}_{config}.txt"
            C_gpu = load_matrix_from_file(gpu_file)

            if C_gpu is None:
                print(f"  {config}: ⚠ файл не найден")
                continue

            max_diff = np.max(np.abs(C_numpy - C_gpu))
            mean_diff = np.mean(np.abs(C_numpy - C_gpu))
            is_correct = max_diff < EPSILON

            if is_correct:
                print(f"  {config}:  OK (макс. разница: {max_diff:.2e})")
            else:
                print(f"  {config}:  ОШИБКА! (макс. разница: {max_diff:.2e})")
                all_verified = False

            results.append({
                'size': size,
                'config': config,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'verified': is_correct
            })

    print("\n" + "-" * 40)
    if all_verified:
        print(" ВСЕ РЕЗУЛЬТАТЫ ВЕРИФИЦИРОВАНЫ УСПЕШНО")
    else:
        print(" ОБНАРУЖЕНЫ РАСХОЖДЕНИЯ!")

    return pd.DataFrame(results)


def load_experiment_data():
    try:
        df = pd.read_csv('cuda_experiments_results.csv')
        print(f"\nЗагружено {len(df)} экспериментов")
        print(f"Размеры матриц: {sorted(df['size'].unique())}")
        return df
    except FileNotFoundError:
        print(" Файл cuda_experiments_results.csv не найден!")
        return None


def plot_time_vs_size(df):
    plt.figure(figsize=(12, 6))

    for (bx, by), group in df.groupby(['block_x', 'block_y']):
        label = f'{bx}x{by} ({bx * by} потоков)'
        plt.plot(group['size'], group['time_seconds'], 'o-',
                 linewidth=2, markersize=8, label=label)

    plt.xlabel('Размер матрицы (NxN)', fontsize=14)
    plt.ylabel('Время выполнения (сек)', fontsize=14)
    plt.title('Зависимость времени выполнения от размера матрицы', fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cuda_time_vs_size.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(" График сохранен: cuda_time_vs_size.png")


def plot_gflops_vs_size(df):
    plt.figure(figsize=(12, 6))

    for (bx, by), group in df.groupby(['block_x', 'block_y']):
        label = f'{bx}x{by} ({bx * by} потоков)'
        plt.plot(group['size'], group['gflops'], 'o-',
                 linewidth=2, markersize=8, label=label)

    plt.xlabel('Размер матрицы (NxN)', fontsize=14)
    plt.ylabel('Производительность (GFLOPS)', fontsize=14)
    plt.title('Зависимость производительности от размера матрицы', fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cuda_gflops_vs_size.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(" График сохранен: cuda_gflops_vs_size.png")


def plot_best_config(df):
    plt.figure(figsize=(12, 6))

    best_configs = df.loc[df.groupby('size')['gflops'].idxmax()]
    sizes = best_configs['size'].values
    gflops = best_configs['gflops'].values
    configs = [f"{int(bx)}x{int(by)}" for bx, by in
               zip(best_configs['block_x'], best_configs['block_y'])]

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sizes)))
    plt.bar(range(len(sizes)), gflops, color=colors, edgecolor='black')

    for i, (size, gflop, config) in enumerate(zip(sizes, gflops, configs)):
        plt.text(i, gflop + 1, f'{config}\n{gflop:.1f} GFLOPS',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.xticks(range(len(sizes)), [f'{s}x{s}' for s in sizes])
    plt.xlabel('Размер матрицы', fontsize=14)
    plt.ylabel('Производительность (GFLOPS)', fontsize=14)
    plt.title('Лучшая конфигурация блоков для каждого размера матрицы', fontsize=16)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('cuda_best_config.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(" График сохранен: cuda_best_config.png")


def plot_heatmap(df):
    plt.figure(figsize=(12, 8))

    pivot_data = df.pivot_table(
        values='gflops',
        index='size',
        columns=['block_x', 'block_y'],
        aggfunc='mean'
    )

    pivot_data.columns = [f'{int(bx)}x{int(by)}' for bx, by in pivot_data.columns]

    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd',
                cbar_kws={'label': 'GFLOPS'}, linewidths=0.5)

    plt.title('Тепловая карта производительности (GFLOPS)', fontsize=16)
    plt.xlabel('Конфигурация блока', fontsize=14)
    plt.ylabel('Размер матрицы', fontsize=14)
    plt.tight_layout()
    plt.savefig('cuda_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(" График сохранен: cuda_heatmap.png")


def plot_threads_comparison(df):
    plt.figure(figsize=(12, 6))

    df['threads'] = df['threads_per_block']

    for size in sorted(df['size'].unique()):
        subset = df[df['size'] == size]
        plt.plot(subset['threads'], subset['gflops'], 'o-',
                 label=f'{size}x{size}', linewidth=2, markersize=8)

    plt.xlabel('Количество потоков на блок', fontsize=14)
    plt.ylabel('Производительность (GFLOPS)', fontsize=14)
    plt.title('Влияние размера блока на производительность', fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.xticks([64, 256, 512, 1024], ['64 (8x8)', '256 (16x16)', '512 (32x16)', '1024 (32x32)'])
    plt.tight_layout()
    plt.savefig('cuda_threads_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(" График сохранен: cuda_threads_comparison.png")



if __name__ == "__main__":

    print("=" * 60)

    print("\nЧАСТЬ 1: ВЕРИФИКАЦИЯ РЕЗУЛЬТАТОВ")
    verify_df = verify_all_results()

    print("\nЧАСТЬ 2: АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ")
    df = load_experiment_data()

    if df is not None:
        print("\nЧАСТЬ 3: ПОСТРОЕНИЕ ГРАФИКОВ")
        plot_time_vs_size(df)
        plot_gflops_vs_size(df)
        plot_best_config(df)
        plot_heatmap(df)
        plot_threads_comparison(df)
    else:
        print("Не удалось загрузить экспериментальные данные.")