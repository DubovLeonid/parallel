import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# НАСТРОЙКИ
# ==========================================
filename = 'lab5.xlsx'

try:
    df = pd.read_excel(filename, sheet_name='Лист1', index_col=0)

    # Очистка: оставляем только строки с числовым индексом (количество MPI‑процессов)
    numeric_index = pd.to_numeric(df.index, errors='coerce')
    df = df[numeric_index.notna()]
    df.index = df.index.astype(int)

    # Колонки — размеры матриц
    df.columns = [int(col) for col in df.columns]

    # Сортируем
    df.sort_index(inplace=True)
    df = df[sorted(df.columns)]

except Exception as e:
    print(f"Ошибка загрузки данных: {e}")
    exit()

# Переименовываем индекс для единообразия
df.index.name = 'Processes'

# ==========================================
# ВАЖНОЕ ИСПРАВЛЕНИЕ: заменяем 0 на NaN
# ==========================================
# Это решит проблему с делением на ноль и появлением inf
df_clean = df.replace(0, np.nan)

print("Загруженные данные (Время в секундах):")
print(df_clean)  # теперь вместо 0 будет NaN
print("\nРазмеры матриц:", df.columns.tolist())
print("Количество MPI-процессов:", df.index.tolist())

# ==========================================
# РАСЧЁТ МЕТРИК (используем df_clean)
# ==========================================
# Время одного процесса — базовое последовательное время T(1)
t1 = df_clean.loc[1] if 1 in df_clean.index else None

# Ускорение S = T(1) / T(p)
speedup = pd.DataFrame(index=df_clean.index, columns=df_clean.columns, dtype=float)
if t1 is not None:
    for p in df_clean.index:
        # Деление с NaN даст NaN, а не inf
        speedup.loc[p] = t1.values / df_clean.loc[p].values
else:
    print("Внимание: Нет данных для 1 процесса. Ускорение не рассчитано.")

# Эффективность E = S / p
efficiency = pd.DataFrame(index=df_clean.index, columns=df_clean.columns, dtype=float)
if not speedup.empty:
    for p in df_clean.index:
        efficiency.loc[p] = speedup.loc[p].astype(float) / p

# ==========================================
# ВЫВОД ТАБЛИЦ В КОНСОЛЬ
# ==========================================
print("\n" + "=" * 60)
print("ТАБЛИЦА УСКОРЕНИЯ S = T(1)/T(p)")
print("=" * 60)
# Форматируем: NaN показываем как "N/A", обычные числа с двумя знаками
print(speedup.to_string(float_format=lambda x: "N/A" if pd.isna(x) else f"{x:.2f}"))

print("\n" + "=" * 60)
print("ТАБЛИЦА ЭФФЕКТИВНОСТИ E = S/p")
print("=" * 60)
print(efficiency.to_string(float_format=lambda x: "N/A" if pd.isna(x) else f"{x:.3f}"))

# ==========================================
# ПОСТРОЕНИЕ ГРАФИКОВ — КАЖДЫЙ В ОТДЕЛЬНОМ ОКНЕ
# ==========================================
plt.style.use('seaborn-v0_8-darkgrid')

# Список размеров матриц, для которых есть хоть какие-то данные (не все NaN)
valid_sizes = [size for size in df_clean.columns if df_clean[size].notna().sum() >= 2]

# --- График 1: Время от размера матрицы ---
fig1, ax1 = plt.subplots(figsize=(12, 8))
for p in df_clean.index:
    # Строим только если есть не-NaN значения
    mask = df_clean.loc[p].notna()
    if mask.sum() >= 2:  # нужно минимум 2 точки для линии
        ax1.plot(df_clean.columns[mask], df_clean.loc[p][mask],
                marker='o', linewidth=2, markersize=6, label=f'{p} процессов')
ax1.set_xlabel('Размер матрицы N', fontsize=14)
ax1.set_ylabel('Время, с', fontsize=14)
ax1.set_title('Время T(N) от размера задачи (MPI)', fontsize=16)
ax1.set_yscale('log')
ax1.legend(fontsize=12, ncol=2)
ax1.grid(True, which="both", ls="--", alpha=0.7)
plt.tight_layout()
plt.show()

# --- График 2: Время от числа процессов ---
fig2, ax2 = plt.subplots(figsize=(12, 8))
for size in valid_sizes:
    mask = df_clean[size].notna()
    if mask.sum() >= 2:
        ax2.plot(df_clean.index[mask], df_clean[size][mask],
                marker='s', linewidth=2, markersize=6, label=f'N={size}')
ax2.set_xlabel('Число MPI-процессов', fontsize=14)
ax2.set_ylabel('Время, с', fontsize=14)
ax2.set_title('Время T(p) от числа процессов (MPI)', fontsize=16)
ax2.set_yscale('log')
ax2.legend(fontsize=12, ncol=2)
ax2.grid(True, which="both", ls="--", alpha=0.7)
plt.tight_layout()
plt.show()

# --- График 3: Ускорение ---
fig3, ax3 = plt.subplots(figsize=(12, 8))
if not speedup.empty:
    for size in valid_sizes:
        valid_mask = speedup[size].notna()
        if valid_mask.sum() >= 2:
            ax3.plot(speedup.index[valid_mask], speedup[size][valid_mask],
                    marker='^', linewidth=2, markersize=6, label=f'N={size}')
    # Линейное ускорение
    max_procs = df_clean.index.max()
    ax3.plot([1, max_procs], [1, max_procs], 'k--', linewidth=2,
            label='Линейное ускорение')
    ax3.set_xlabel('Число MPI-процессов', fontsize=14)
    ax3.set_ylabel('Ускорение S', fontsize=14)
    ax3.set_title('Ускорение S = T(1)/T(p) (MPI)', fontsize=16)
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.7)
else:
    ax3.text(0.5, 0.5, 'Нет данных для расчёта', ha='center', va='center',
            transform=ax3.transAxes, fontsize=14)
plt.tight_layout()
plt.show()

# --- График 4: Эффективность ---
fig4, ax4 = plt.subplots(figsize=(12, 8))
if not efficiency.empty:
    for size in valid_sizes:
        valid_mask = efficiency[size].notna()
        if valid_mask.sum() >= 2:
            ax4.plot(efficiency.index[valid_mask], efficiency[size][valid_mask],
                    marker='D', linewidth=2, markersize=6, label=f'N={size}')
    ax4.axhline(y=1.0, color='k', linestyle='--', linewidth=2,
                label='Идеальная эффективность')
    ax4.set_xlabel('Число MPI-процессов', fontsize=14)
    ax4.set_ylabel('Эффективность E', fontsize=14)
    ax4.set_title('Эффективность E = S/p (MPI)', fontsize=16)
    ax4.legend(fontsize=12)
    ax4.grid(True, alpha=0.7)
    # Подписываем область суперлинейного ускорения
    ax4.fill_between([1, max_procs], 1.0, 1.5, alpha=0.1, color='green',
                     label='Суперлинейность (>1.0)')
    ax4.set_ylim(0, 1.3)  # для наглядности
else:
    ax4.text(0.5, 0.5, 'Нет данных для расчёта', ha='center', va='center',
            transform=ax4.transAxes, fontsize=14)
plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("ГОТОВО! Все 4 графика показаны в отдельных окнах.")
print("=" * 60)