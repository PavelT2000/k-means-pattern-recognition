import numpy as np
import matplotlib.pyplot as plt

def run_kmeans_interactive():
    print("--- Лабораторная работа №1: Распознавание образов ---")
    
    try:
        n_samples = int(input("Введите количество образов (от 1000 до 100000): "))
        k_clusters = int(input("Введите количество классов (от 2 до 20): "))
        
    except ValueError:
        print("Ошибка: введите целое число.")
        return

    X = np.random.rand(n_samples, 2)
    
   
    indices = np.random.choice(n_samples, k_clusters, replace=False)
    centroids = X[indices]
    
    iteration = 0
    prev_centroids = np.zeros(centroids.shape)
    
    while not np.allclose(prev_centroids, centroids):
        iteration += 1
        prev_centroids = centroids.copy()
        
        
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        if iteration == 1:
            plot_clusters(X, labels, centroids, f"Рис. 1. Первая итерация (Классов: {k_clusters})")
        
        for i in range(k_clusters):
            points_in_cluster = X[labels == i]
            if len(points_in_cluster) > 0:
                centroids[i] = points_in_cluster.mean(axis=0)

    print(f"Алгоритм завершен за {iteration} итераций.")
    plot_clusters(X, labels, centroids, f"Рис. 2. Завершающая итерация (Классов: {k_clusters})")

def plot_clusters(X, labels, centroids, title):
    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=1, cmap='tab20', alpha=0.6)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=100, label='Ядра')
    plt.title(title)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_kmeans_interactive()