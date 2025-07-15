import math
import random


### --------------
### 1. K-MEANS
### --------------

class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters

    def _euclidean(self, a, b):
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

    def fit(self, X):
        self.centroids = random.sample(X, self.k)
        for _ in range(self.max_iters):
            clusters = [[] for _ in range(self.k)]
            for x in X:
                distances = [self._euclidean(x, c) for c in self.centroids]
                cluster_idx = distances.index(min(distances))
                clusters[cluster_idx].append(x)
            new_centroids = []
            for cluster in clusters:
                if cluster:
                    new_centroids.append([
                        sum(dim) / len(dim)
                        for dim in zip(*cluster)
                    ])
                else:
                    new_centroids.append(random.choice(X))
            if all(self._euclidean(a, b) < 1e-4 for a, b in zip(self.centroids, new_centroids)):
                break
            self.centroids = new_centroids

        self.labels_ = []
        for x in X:
            distances = [self._euclidean(x, c) for c in self.centroids]
            cluster_idx = distances.index(min(distances))
            self.labels_.append(cluster_idx)

    def predict(self, x):
        distances = [self._euclidean(x, c) for c in self.centroids]
        return distances.index(min(distances))


### --------------
### 2. DBSCAN
### --------------

def _euclidean(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def region_query(points, p_idx, eps):
    neighbors = []
    for i, other in enumerate(points):
        if _euclidean(points[p_idx], other) <= eps:
            neighbors.append(i)
    return neighbors

def expand_cluster(points, labels, p_idx, cluster_id, eps, min_pts, visited):
    neighbors = region_query(points, p_idx, eps)
    if len(neighbors) < min_pts:
        labels[p_idx] = -1
        return False
    labels[p_idx] = cluster_id
    queue = list(neighbors)
    while queue:
        idx = queue.pop(0)
        if not visited[idx]:
            visited[idx] = True
            new_neighbors = region_query(points, idx, eps)
            if len(new_neighbors) >= min_pts:
                queue.extend(new_neighbors)
        if labels[idx] == 0:
            labels[idx] = cluster_id
    return True

class DBSCAN:
    def __init__(self, eps=0.5, min_pts=5):
        self.eps = eps
        self.min_pts = min_pts

    def fit(self, points):
        n = len(points)
        self.labels_ = [0] * n  # 0 = unclassified, -1 = noise, >=1 = cluster id
        visited = [False] * n
        cluster_id = 0
        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            if expand_cluster(points, self.labels_, i, cluster_id + 1, self.eps, self.min_pts, visited):
                cluster_id += 1
        self.n_clusters_ = cluster_id
        return self


### --------------
### 3. PCA
### --------------

class PCA:
    def __init__(self, n_components=2, power_iters=20):
        self.n_components = n_components
        self.power_iters = power_iters

    def fit(self, X):
        self.mean = [sum(col) / len(col) for col in zip(*X)]
        X_centered = [[xj - mj for xj, mj in zip(x, self.mean)] for x in X]
        self.components = []

        cov = self._cov_matrix(X_centered)
        for _ in range(self.n_components):
            vec = [random.random() for _ in range(len(X[0]))]
            for _ in range(self.power_iters):
                z = [sum(cov[i][j] * vec[j] for j in range(len(vec))) for i in range(len(vec))]
                norm = math.sqrt(sum(zi ** 2 for zi in z))
                vec = [zi / norm for zi in z]
            self.components.append(vec)
            cov = self._deflate(cov, vec)

    def transform(self, X):
        X_centered = [[xj - mj for xj, mj in zip(x, self.mean)] for x in X]
        return [
            [sum(xj * pcj for xj, pcj in zip(x, pc)) for pc in self.components]
            for x in X_centered
        ]

    def inverse_transform(self, X_transformed):
        return [
            [
                sum(c * pcj for c, pcj in zip(row, pc)) + mj
                for pc, mj in zip(zip(*self.components), self.mean)
            ]
            for row in X_transformed
        ]

    def _cov_matrix(self, X):
        n, d = len(X), len(X[0])
        cov = [[0.0] * d for _ in range(d)]
        for i in range(d):
            for j in range(d):
                cov[i][j] = sum(X[k][i] * X[k][j] for k in range(n)) / (n - 1)
        return cov

    def _deflate(self, cov, component):
        d = len(component)
        updated = [[0.0] * d for _ in range(d)]
        for i in range(d):
            for j in range(d):
                updated[i][j] = cov[i][j] - component[i] * component[j] * sum(
                    cov[k][k] * component[k] ** 2 for k in range(d)
                )
        return updated


### -------------------------------
### 4. HIERARCHICAL CLUSTERING
### -------------------------------

class HierarchicalClustering:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters

    def fit(self, X):
        clusters = [[i] for i in range(len(X))]

        def dist(c1, c2):
            return min(
                math.sqrt(sum((X[i][d] - X[j][d]) ** 2 for d in range(len(X[0]))))
                for i in c1 for j in c2
            )

        while len(clusters) > self.n_clusters:
            min_d = float('inf')
            to_merge = (None, None)

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    d = dist(clusters[i], clusters[j])
                    if d < min_d:
                        min_d = d
                        to_merge = (i, j)

            i, j = to_merge
            # Merge and replace instead of pop
            merged_cluster = clusters[i] + clusters[j]
            # Remove in descending order to avoid index shifting
            for index in sorted([i, j], reverse=True):
                clusters.pop(index)
            clusters.append(merged_cluster)

        self.labels_ = [0] * len(X)
        for cluster_id, cluster in enumerate(clusters):
            for idx in cluster:
                self.labels_[idx] = cluster_id