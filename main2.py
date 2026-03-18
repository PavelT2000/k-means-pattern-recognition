import numpy as np
import matplotlib.pyplot as plt

def run_maximin_interactive():
    print("--- Лабораторная работа №2: Алгоритм Максимина ---")

    try:
        # Исходные данные: число образов (от 1000 до 100 000) [cite: 17]
        n_samples = int(input("Введите количество образов (1000 - 100000): "))
        if not (1000 <= n_samples <= 100000):
            print("Предупреждение: рекомендуется диапазон от 1000 до 100000.")
    except ValueError:
        print("Ошибка: введите целое число.")
        return

    # Признаки объектов задаются случайным образом (координаты векторов) [cite: 17]
    X = np.random.rand(n_samples, 2)

    # ШАГ 1: Произвольно выбирается один вектор и назначается первым ядром [cite: 23]
    centroids = [X[np.random.randint(0, n_samples)]]

    # ШАГ 2: Второе ядро — максимально удаленный объект от первого ядра [cite: 25]
    distances_to_first = np.linalg.norm(X - centroids[0], axis=1)
    centroids.append(X[np.argmax(distances_to_first)])

    while True:
        # ШАГ 3: Распределение объектов по критерию минимального расстояния [cite: 26]
        # Вычисляем расстояния от всех точек до всех текущих ядер
        dist_matrix = np.linalg.norm(X[:, np.newaxis] - np.array(centroids), axis=2)
        labels = np.argmin(dist_matrix, axis=1) # Индекс ближайшего ядра для каждой точки
        min_distances = np.min(dist_matrix, axis=1) # Сами минимальные расстояния

        # ШАГ 4 & 5: Поиск претендента на новое ядро [cite: 27, 28, 29]
        # Находим точку, которая максимально удалена от своего ядра (максимум среди минимумов)
        idx_max_dist = np.argmax(min_distances)
        max_dist = min_distances[idx_max_dist]

        # Вычисляем среднее расстояние между всеми существующими ядрами [cite: 30]
        if len(centroids) > 1:
            # Матрица расстояний между ядрами
            c_dist = []
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    c_dist.append(np.linalg.norm(centroids[i] - centroids[j]))
            avg_kernel_dist = np.mean(c_dist)
        else:
            avg_kernel_dist = max_dist * 2 # Для первой итерации

        # ШАГ 5 (условие): Если максимум > половины среднего расстояния между ядрами
        if max_dist > (avg_kernel_dist / 2):
            centroids.append(X[idx_max_dist])
            print(f"Найдено новое ядро. Текущее кол-во классов: {len(centroids)}")
        else:
            # ШАГ 5 (иначе): Алгоритм останавливается [cite: 30, 33]
            break

    print(f"Алгоритм завершен. Итого классов: {len(centroids)}")

    # Результат работы представить графически [cite: 19, 34]
    plot_maximin(X, labels, np.array(centroids))

def plot_maximin(X, labels, centroids):
    plt.figure(figsize=(10, 8))
    # Отображаем объекты, разделенные на классы [cite: 20]
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=1, cmap='tab20', alpha=0.6)
    # Отображаем найденные ядра классов [cite: 18, 33]
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=100, label='Ядра (Максимин)')
    plt.title(f"Результат алгоритма Максимина (Классов: {len(centroids)})")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_maximin_interactive()