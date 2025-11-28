import numpy as np

from .geometry import shape_factors, total_area


def point_of_cavity_to_others(index: int, edges: np.ndarray) -> np.ndarray:
    """
    Create triangles by connecting a given point to other edges in a cavity.
    Args:
        index (int): The index of the point.
        edges (np.ndarray): An array of edges, each represented by a pair of vertex indices.
    Returns:
        np.ndarray: An array of triangles formed by connecting the point to the edges.
    """
    triangles = []
    for edge in edges:
        if index not in edge:
            triangle = (index, edge[0], edge[1])
            triangles.append(triangle)
    return np.array(triangles)


def triangulations_minimal_area(
    points: np.ndarray, triangulations: list[np.ndarray]
) -> tuple[list[np.ndarray], list[int]]:
    """
    Select triangulations with the minimal total area.
    Args:
        points (np.ndarray): An array of points.
        triangulations (list[np.ndarray]): A list of triangulations, each represented by an array of triangles.
    Returns:
        tuple[list[np.ndarray], list[int]]: A tuple containing a list of triangulations with the minimal total area and
        their corresponding indices.
    """
    min_area = float("inf")
    best_triangulations = []
    best_triangulations_indices = []
    for index, tris in enumerate(triangulations):
        if len(tris) > 0:
            area_sum = total_area(points[tris])
            if area_sum < min_area:
                min_area = area_sum
                best_triangulations = [tris]
                best_triangulations_indices = [index]
            elif abs(area_sum - min_area) < 1e-8:
                best_triangulations.append(tris)
                best_triangulations_indices.append(index)
    return best_triangulations, best_triangulations_indices


def triangulations_maximal_min_shape_factor(
    points: np.ndarray,
    triangulations: list[np.ndarray],
) -> tuple[np.ndarray, int]:
    """
    Select the triangulation with the first minimal shape factor.
    Args:
        points (np.ndarray): An array of points.
        triangulations (list[np.ndarray]): A list of triangulations, each represented by an array of triangles.
    Returns:
        tuple[np.ndarray, int]: A tuple containing the triangulation with the maximal minimal shape factor
        and its index.
    """
    sfs = []
    for triangulation in triangulations:
        sfs.append(shape_factors(points[triangulation]))
    sfs_lengths = [len(sf) for sf in sfs]
    max_sfs_length = max(sfs_lengths)

    max_sf = sfs[0][0]
    best_triangulations = [triangulations[0]]
    best_triangulations_indices = [0]
    for i in range(max_sfs_length):
        for j in range(len(sfs)):
            if i < sfs_lengths[j]:
                if sfs[j][i] > max_sf:
                    max_sf = sfs[j][i]
                    best_triangulations = [triangulations[j]]
                    best_triangulations_indices = [j]
                elif sfs[j][i] == max_sf:
                    best_triangulations.append(triangulations[j])
                    best_triangulations_indices.append(j)
                if len(best_triangulations) == 1:
                    return best_triangulations[0], best_triangulations_indices[0]
    return best_triangulations[0], best_triangulations_indices[0]


def best_point_of_cavity_to_others(
    points, indices: list[int], edges: np.ndarray
) -> tuple[np.ndarray, int]:
    """
    Select the best triangulation formed by connecting a point to other edges in a cavity.
    Args:
        points (np.ndarray): An array of points.
        indices (list[int]): A list of point indices.
        edges (np.ndarray): An array of edges, each represented by a pair of vertex indices.
    Returns:
        tuple[np.ndarray, int]: A tuple containing the best triangulation and its corresponding index.
    """
    triangulations = []
    for index in indices:
        triangulation = point_of_cavity_to_others(index, edges)
        triangulations.append(triangulation)
    best_triangulations, best_triangulations_indices = triangulations_minimal_area(
        points, triangulations
    )
    best_triangulations_indices = [indices[i] for i in best_triangulations_indices]
    if len(best_triangulations) == 1:
        return best_triangulations[0], best_triangulations_indices[0]
    best_triangulation, best_triangulations_index = (
        triangulations_maximal_min_shape_factor(points, best_triangulations)
    )
    return best_triangulation, best_triangulations_indices[best_triangulations_index]


def remove_triangles_from_triangulation(
    triangles_to_remove: np.ndarray, triangulation: np.ndarray
) -> np.ndarray:
    """
    Remove specified triangles from a triangulation.
    Args:
        triangles_to_remove (np.ndarray): An array of triangles to be removed.
        triangulation (np.ndarray): The original triangulation.
    Returns:
        np.ndarray: The triangulation after removing the specified triangles.
    """
    return np.array(
        [
            tri
            for tri in triangulation
            if tri.tolist() not in triangles_to_remove.tolist()
        ]
    )


def add_triangles_to_triangulation(
    triangles_to_add: np.ndarray, triangulation: np.ndarray
) -> np.ndarray:
    """
    Add specified triangles to a triangulation.
    Args:
        triangles_to_add (np.ndarray): An array of triangles to be added.
        triangulation (np.ndarray): The original triangulation.
    Returns:
        np.ndarray: The triangulation after adding the specified triangles.
    """
    if len(triangulation) > 0 and len(triangles_to_add) > 0:
        return np.vstack((triangulation, triangles_to_add))
    elif len(triangulation) == 0:
        return triangles_to_add
    return triangulation
