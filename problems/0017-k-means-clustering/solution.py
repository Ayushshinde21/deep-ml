import math

def k_means_clustering(points, k, initial_centroids, max_iterations):
    centroids = [tuple(c) for c in initial_centroids]

    for _ in range(max_iterations):
        clusters = [[] for _ in range(k)]

        for point in points:
            distances = [
                math.sqrt(sum(
                    (point[i] - centroid[i]) ** 2
                    for i in range(len(point))
                ))
                for centroid in centroids
            ]

            cluster_index = distances.index(min(distances))
            clusters[cluster_index].append(point)

        new_centroids = []

        for i in range(k):
            if clusters[i]:
                new_centroid = tuple(
                    sum(point[j] for point in clusters[i]) / len(clusters[i])
                    for j in range(len(points[0]))
                )
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(centroids[i])

        if new_centroids == centroids:
            break

        centroids = new_centroids

    return [
        tuple(round(value, 4) for value in centroid)
        for centroid in centroids
    ]