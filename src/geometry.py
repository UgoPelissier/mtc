import math
import numpy as np
from collections import defaultdict

# --------------------
# POINTS
# --------------------


def polygon_area(polygon: np.ndarray) -> float:
    """
    Calculate the area of a polygon given its vertices.
    Args:
        polygon (np.ndarray): An array of vertices representing the polygon.
    Returns:
        float: The area of the polygon.
    """
    num_vertices = len(polygon)
    area_sum = 0.0
    for i in range(num_vertices):
        j = (i + 1) % num_vertices
        area_sum += polygon[i][0] * polygon[j][1]
        area_sum -= polygon[j][0] * polygon[i][1]
    return abs(area_sum) / 2.0


def point_in_cavity(point: np.ndarray, cavity: np.ndarray) -> bool:
    """
    Check if a point is inside a polygon (cavity) using the ray-casting algorithm.
    Args:
        point (np.ndarray): The point to check.
        cavity (np.ndarray): An array of vertices representing the polygon.
    Returns:
        bool: True if the point is inside the polygon, False otherwise.
    """
    num_vertices = len(cavity)
    inside = False
    x, y = point
    for i in range(num_vertices):
        j = (i + 1) % num_vertices
        xi, yi = cavity[i]
        xj, yj = cavity[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
    return inside


def polygon_to_edges(polygon: np.ndarray) -> np.ndarray:
    """
    Convert a polygon to its constituent edges.
    Args:
        polygon (np.ndarray): An array of vertices representing the polygon.
    Returns:
        np.ndarray: An array of edges represented as pairs of vertex indices.
    """
    num_vertices = len(polygon)
    edges = []
    for i in range(num_vertices):
        start_idx = i
        end_idx = (i + 1) % num_vertices
        edges.append((start_idx, end_idx))
    return np.array(edges)


def point_in_triangles(index: int, triangles: np.ndarray) -> bool:
    """
    Check if a point is part of any triangle in a triangulation.
    Args:
        index (int): The index of the point.
        triangles (np.ndarray): An array of triangles, each represented by a 3x2 array of vertices.
    Returns:
        bool: True if the point is part of any triangle, False otherwise.
    """
    return index in triangles.flatten()


def point_neighbors(index: int, triangles: np.ndarray) -> np.ndarray:
    """
    Get neighboring points of a given point in a triangulation.
    Args:
        index (int): The index of the point.
        triangles (np.ndarray): An array of triangles, each represented by a 3x2 array of vertices.
    Returns:
        np.ndarray: An array of neighboring points.
    """
    neighbors = []
    for tri in triangles:
        if index in tri:
            for vertex in tri:
                if vertex != index and vertex not in neighbors:
                    neighbors.append(vertex)
    return np.array(neighbors)


def point_elements(index: int, triangles: np.ndarray) -> np.ndarray:
    """
    Get elements (triangles) connected to a given point in a triangulation.
    Args:
        index (int): The index of the point.
        triangles (np.ndarray): An array of triangles, each represented by a 3x2 array of vertices.
    Returns:
        np.ndarray: An array of elements (triangles) connected to the point.
    """
    neighbors = point_neighbors(index, triangles)
    neighbors = np.append(neighbors, index)
    elements = []
    for tri in triangles:
        if all(vertex in neighbors for vertex in tri):
            elements.append(tri)
    return np.array(elements)


def convex_hull_indices(points: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Compute the convex hull of a subset of points.

    Args:
        points (np.ndarray): (N,2) array of all points
        indices (np.ndarray): array of indices selecting the subset

    Returns:
        np.ndarray: hull_indices (original indices of hull vertices in CCW order)
    """

    # Extract subset
    pts = points[indices]

    # Remove duplicates in subset while tracking original indices
    unique_pts, unique_idx = np.unique(pts, axis=0, return_index=True)
    original_idx = np.array(indices)[unique_idx]

    if unique_pts.shape[0] <= 1:
        return original_idx.tolist()

    # Cross product
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Sort points lexicographically (and track indices)
    order = np.lexsort((unique_pts[:, 1], unique_pts[:, 0]))
    pts_sorted = unique_pts[order]
    idx_sorted = original_idx[order]

    # Build lower hull
    lower = []
    for p, i in zip(pts_sorted, idx_sorted):
        while len(lower) >= 2 and cross(points[lower[-2]], points[lower[-1]], p) < 0:
            lower.pop()
        lower.append(i)

    # Build upper hull
    upper = []
    for p, i in zip(pts_sorted[::-1], idx_sorted[::-1]):
        while len(upper) >= 2 and cross(points[upper[-2]], points[upper[-1]], p) < 0:
            upper.pop()
        upper.append(i)

    # Remove last duplicate point of each
    hull = lower[:-1] + upper[:-1]

    return np.array(hull)


# --------------------
# EDGES
# --------------------


def edge_in_triangulation(edge: np.ndarray, triangulation: np.ndarray) -> bool:
    """
    Check if an edge is present in a given triangulation.
    Args:
        edge (np.ndarray): The edge to check.
        triangulation (np.ndarray): An array of triangles, each represented by a 3x2 array of vertices.
    Returns:
        bool: True if the edge is present in the triangulation, False otherwise.
    """
    edges = triangles_to_edges(triangulation)
    for e in edges:
        if (edge[0] == e[0] and edge[1] == e[1]) or (
            edge[0] == e[1] and edge[1] == e[0]
        ):
            return True
    return False


def edges_in_triangulation(
    edges: np.ndarray, triangulation: np.ndarray, points: np.ndarray = None
) -> tuple[bool, np.ndarray, np.ndarray]:
    """
    Check if all edges are present in a given triangulation.
    Args:
        edges (np.ndarray): An array of edges to check.
        triangulation (np.ndarray): An array of triangles, each represented by a 3x2 array of vertices.
        points (np.ndarray, optional): An array of points corresponding to the vertices.
    Returns:
        tuple: A tuple containing a boolean indicating if all edges are present,
               an array of missing edges, an array of missing points, and an array of missing points' triangles.
    """
    missing_edges = []
    missing_points = []
    missing_points_triangles = []
    for edge in edges:
        if not edge_in_triangulation(edge, triangulation):
            missing_edges.append(edge)
    if len(missing_edges) == 0:
        return (
            True,
            np.array(missing_edges),
            np.array(missing_points),
            np.array(missing_points_triangles),
        )
    points_count = defaultdict(int)
    for edge in missing_edges:
        points_count[edge[0]] += 1
        points_count[edge[1]] += 1
    missing_points = [point for point, count in points_count.items() if count > 1]
    if len(missing_points) == 0:
        return (
            False,
            np.array(missing_edges),
            np.array(missing_points),
            np.array(missing_points_triangles),
        )
    missing_points_triangles = [[] for _ in range(len(missing_points))]
    if points is not None:
        for idx, missing_point in enumerate(missing_points):
            for triangle in triangulation:
                if point_in_cavity(points[missing_point], points[triangle]):
                    if missing_point - 1 in triangle and missing_point + 1 in triangle:
                        missing_points_triangles[idx] = triangle
                        break
    return (
        False,
        np.array(missing_edges),
        np.array(missing_points),
        np.array(missing_points_triangles),
    )


def edge_neighbors(edge: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """
    Get neighboring points of a given edge in a triangulation.
    Args:
        edge (np.ndarray): An array representing the edge as a pair of vertex indices.
        triangles (np.ndarray): An array of triangles, each represented by a 3x2 array of vertices.
    Returns:
        np.ndarray: An array of neighboring points.
    """
    neighbors = []
    for vertex in edge:
        neighbors.append(point_neighbors(vertex, triangles))
    common_neighbors = set(neighbors[0]).intersection(set(neighbors[1]))
    return np.array(list(common_neighbors))


def edge_elements(edge: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """
    Get elements (triangles) connected to a given edge in a triangulation.
    Args:
        edge (np.ndarray): An array representing the edge as a pair of vertex indices.
        triangles (np.ndarray): An array of triangles, each represented by a 3x2 array of vertices.
    Returns:
        np.ndarray: An array of elements (triangles) connected to the edge.
    """
    neighbors = edge_neighbors(edge, triangles)
    neighbors = np.append(neighbors, edge)
    elements = []
    for tri in triangles:
        if all(vertex in neighbors for vertex in tri):
            elements.append(tri)
    return np.array(elements)


# --------------------
# TRIANGLE
# --------------------


def area(triangle: np.ndarray, metrique: np.ndarray = np.eye(2)) -> float:
    """
    Calculate the area of a triangle given its vertices.
    Args:
        triangle (np.ndarray): A 3x2 array representing the triangle's vertices.
        metrique (np.ndarray): A 2x2 array representing the metric tensor.
    Returns:
        float: The area of the triangle.
    """
    a = triangle[0]
    b = triangle[1]
    c = triangle[2]
    return abs(
        (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])) / 2.0
    ) * math.sqrt(np.linalg.det(metrique))


def distance(p1: np.ndarray, p2: np.ndarray, metrique: np.ndarray = np.eye(2)) -> float:
    """
    Calculate the distance between two points.
    Args:
        p1 (np.ndarray): The first point.
        p2 (np.ndarray): The second point.
        metrique (np.ndarray): A 2x2 array representing the metric tensor.
    Returns:
        float: The distance between the two points.
    """
    diff = p2 - p1
    return math.sqrt(diff.T @ metrique @ diff)


def min_edge_length(triangle: np.ndarray, metrique: np.ndarray = np.eye(2)) -> float:
    """
    Calculate the minimum edge length of a triangle.
    Args:
        triangle (np.ndarray): A 3x2 array representing the vertices of the triangle.
        metrique (np.ndarray): A 2x2 array representing the metric tensor.
    Returns:
        float: The minimum edge length of the triangle.
    """
    edge_lengths = []
    for i in range(len(triangle)):
        j = (i + 1) % len(triangle)
        length = distance(triangle[i], triangle[j], metrique)
        edge_lengths.append(length)
    return min(edge_lengths)


def max_edge_length(triangle: np.ndarray, metrique: np.ndarray = np.eye(2)) -> float:
    """
    Calculate the maximum edge length of a triangle.
    Args:
        triangle (np.ndarray): A 3x2 array representing the vertices of the triangle.
        metrique (np.ndarray): A 2x2 array representing the metric tensor.
    Returns:
        float: The maximum edge length of the triangle.
    """
    edge_length = 0.0
    for i in range(len(triangle)):
        j = (i + 1) % len(triangle)
        edge_length = max(
            edge_length,
            distance(triangle[i], triangle[j], metrique),
        )
    return edge_length


def mean_edge_length(triangle: np.ndarray, metrique: np.ndarray = np.eye(2)) -> float:
    """
    Calculate the mean edge length of a triangle.
    Args:
        triangle (np.ndarray): A 3x2 array representing the vertices of the triangle.
        metrique (np.ndarray): A 2x2 array representing the metric tensor.
    Returns:
        float: The mean edge length of the triangle.
    """
    edge_length = 0.0
    for i in range(len(triangle)):
        j = (i + 1) % len(triangle)
        edge_length += distance(triangle[i], triangle[j], metrique)
    return math.sqrt(edge_length / 3.0)


def reference_area() -> float:
    """
    Get the reference area for shape factor calculation.
    Returns:
        float: The reference area.
    """
    return math.sqrt(3) / 4.0


def normalization_factor() -> float:
    """
    Get the normalization factor for shape factor calculation.
    Returns:
        float: The normalization factor.
    """
    v0 = reference_area()
    c0 = 1 / v0
    return c0


def shape_factor(triangle: np.ndarray, metrique: np.ndarray = np.eye(2)) -> float:
    """
    Calculate the shape factor of a triangle.
    Args:
        triangle (np.ndarray): A 3x2 array representing the vertices of the triangle.
        metrique (np.ndarray): A 2x2 array representing the metric tensor.
    Returns:
        float: The shape factor of the triangle.
    """
    metrique = 100 * np.eye(2)
    h = mean_edge_length(triangle, metrique)
    if h == 0:
        return 0.0
    return normalization_factor() * area(triangle, metrique) / (h * h)


def min_edge_length_shape_factor(
    triangle: np.ndarray, metrique: np.ndarray = np.eye(2)
) -> float:
    """
    Return the minimum between the edge length and the shape factor of a triangle.
    Args:
        triangle (np.ndarray): A 3x2 array representing the vertices of the triangle.
        metrique (np.ndarray): A 2x2 array representing the metric tensor.
    Returns:
        float: The minimum between the edge length and the shape factor.
    """
    h = mean_edge_length(triangle, metrique)
    if h == 0:
        return 0.0
    sf = shape_factor(triangle, metrique)
    return min(sf, h**2, 1.0 / h**2)


# --------------------
# TRIANGLES
# --------------------


def total_area(triangles: np.ndarray) -> float:
    """
    Calculate the total area of multiple triangles.
    Args:
        triangles (np.ndarray): An array of triangles, each represented by a 3x2 array of vertices.
    Returns:
        float: The total area of the triangles.
    """
    total_area = 0.0
    for triangle in triangles:
        a = triangle[0]
        b = triangle[1]
        c = triangle[2]
        total_area += abs(
            (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])) / 2.0
        )
    return total_area


def triangles_to_edges(triangles: np.ndarray) -> np.ndarray:
    """
    Convert a set of triangles to a set of edges.
    Args:
        triangles (np.ndarray): An array of triangles, each represented by a 3x2 array of vertices.
    Returns:
        np.ndarray: An array of edges.
    """
    edges = defaultdict(int)
    for tri in triangles:
        edge1 = (min(tri[0], tri[1]), max(tri[0], tri[1]))
        edge2 = (min(tri[1], tri[2]), max(tri[1], tri[2]))
        edge3 = (min(tri[2], tri[0]), max(tri[2], tri[0]))
        edges[edge1] += 1
        edges[edge2] += 1
        edges[edge3] += 1
    return np.array(list(edges.keys()))


def min_max_edge_length(triangles: np.ndarray) -> tuple[float, float]:
    """
    Calculate the minimum and maximum edge lengths among multiple triangles.
    Args:
        triangles (np.ndarray): An array of triangles, each represented by a 3x2 array of vertices.
    Returns:
        tuple[float, float]: A tuple containing the minimum and maximum edge lengths.
    """
    min_length = float("inf")
    max_length = 0.0
    for triangle in triangles:
        triangle_min = min_edge_length(triangle)
        triangle_max = max_edge_length(triangle)
        if triangle_min < min_length:
            min_length = triangle_min
        if triangle_max > max_length:
            max_length = triangle_max
    return min_length, max_length


def shape_factors(triangles: np.ndarray) -> np.ndarray:
    """
    Calculate the shape factors of multiple triangles.
    Args:
        triangles (np.ndarray): An array of triangles, each represented by a 3x2 array of vertices.
    Returns:
        np.ndarray: A sorted array of shape factors.
    """
    factors = []
    for triangle in triangles:
        factors.append(shape_factor(triangle))
    factors = np.array(factors)
    return np.sort(factors)


def min_edge_length_shape_factors(triangles: np.ndarray) -> np.ndarray:
    """
    Calculate the minimum edge length shape factors of multiple triangles.
    Args:
        triangles (np.ndarray): An array of triangles, each represented by a 3x2 array of vertices.
    Returns:
        np.ndarray: A sorted array of minimum edge length shape factors.
    """
    factors = []
    for triangle in triangles:
        factors.append(min_edge_length_shape_factor(triangle))
    factors = np.array(factors)
    return np.sort(factors)


def elements_boundary(triangles: np.ndarray) -> np.ndarray:
    """
    Get the boundary edges of a set of triangles.
    Args:
        triangles (np.ndarray): An array of triangles, each represented by a 3x2 array of vertices.
    Returns:
        np.ndarray: An array of boundary edges.
    """
    edge_count = defaultdict(int)
    for tri in triangles:
        edges = [
            (min(tri[0], tri[1]), max(tri[0], tri[1])),
            (min(tri[1], tri[2]), max(tri[1], tri[2])),
            (min(tri[2], tri[0]), max(tri[2], tri[0])),
        ]
        for edge in edges:
            edge_count[edge] += 1
    boundary_edges = np.array(
        [edge for edge, count in edge_count.items() if count == 1]
    )
    return boundary_edges, np.unique(boundary_edges.flatten())


def convex_hull(
    points: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the convex hull of a set of points given a triangulation.
    Args:
        points (np.ndarray): An array of points.
        triangles (np.ndarray): An array of triangles, each represented by a 3x2 array of vertices.
    Returns:
        tuple: A tuple containing the hull points indices, hull edges and indices of points not forming the hull.
    """
    boundary_points = np.unique(triangles.flatten())
    hull_indices = convex_hull_indices(points, boundary_points)
    hull_edges = np.array(
        [
            (hull_indices[i], hull_indices[(i + 1) % len(hull_indices)])
            for i in range(len(hull_indices))
        ]
    )
    return hull_indices, hull_edges, np.setdiff1d(boundary_points, hull_indices)


def remove_unused_points(
    points: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove unused points from the points array and update the triangles accordingly.
    Args:
        points (np.ndarray): An array of points.
        triangles (np.ndarray): An array of triangles, each represented by a 3x2 array of vertices.
    Returns:
        tuple: A tuple containing the updated points array and the updated triangles array.
    """
    used_indices = np.unique(triangles)
    new_index_map = -np.ones(points.shape[0], dtype=int)
    new_index_map[used_indices] = np.arange(len(used_indices))
    new_triangles = new_index_map[triangles]
    new_points = points[used_indices]
    return new_points, new_triangles
