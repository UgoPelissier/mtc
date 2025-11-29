import numpy as np

from .utils import debug, plot_polygon_and_tris

from .geometry import (
    shape_factors,
    min_edge_length_shape_factors,
    total_area,
    elements_boundary,
    convex_hull,
    point_in_triangles,
    remove_unused_points,
    point_elements,
)


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


def point_in_cavity_to_others(index: int, edges: np.ndarray) -> np.ndarray:
    """
    Create triangles by connecting a given point lying inside a cavity to other edges of this cavity.
    Args:
        index (int): The index of the point.
        edges (np.ndarray): An array of edges, each represented by a pair of vertex indices.
    Returns:
        np.ndarray: An array of triangles formed by connecting the point to the edges.
    """
    triangles = []
    for edge in edges:
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


def triangulations_first_min_shape_factor(
    points: np.ndarray,
    triangulations: list[np.ndarray],
    improve: bool = False,
) -> tuple[np.ndarray, int]:
    """
    Select the triangulation with the first minimal shape factor.
    Args:
        points (np.ndarray): An array of points.
        triangulations (list[np.ndarray]): A list of triangulations, each represented by an array of triangles.
    Returns:
        tuple[np.ndarray, int]: A tuple containing the triangulation with the first minimal shape factor
        and its index.
    """
    sfs = []
    non_degenerate_triangulations = []
    for triangulation in triangulations:
        if improve:
            degenerate = False
            sf = min_edge_length_shape_factors(points[triangulation])
            for s in sf:
                if abs(s) < 1e-5:
                    degenerate = True
            if not degenerate:
                sfs.append(sf)
                non_degenerate_triangulations.append(triangulation)
        else:
            sfs.append(shape_factors(points[triangulation]))
    sfs_lengths = [len(sf) for sf in sfs]
    max_sfs_length = max(sfs_lengths)

    max_sf = sfs[0][0]
    if improve:
        triangulations = non_degenerate_triangulations
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
    points,
    indices: list[int],
    edges: np.ndarray,
    improve: bool = False,
    inside_points: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select the best triangulation formed by connecting a point to other edges in a cavity.
    Args:
        points (np.ndarray): An array of points.
        indices (list[int]): A list of point indices.
        edges (np.ndarray): An array of edges, each represented by a pair of vertex indices.
        improve (bool, optional): Whether to add the barycenter point for improvement. Defaults to False.
        inside_points (np.ndarray, optional): An array of inside point indices. Required if improve is True.
    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the points array (possibly updated with barycenter),
        and the best triangulation.
    """
    triangulations = []
    for index in indices:
        triangulation = point_of_cavity_to_others(index, edges)
        triangulations.append(triangulation)

    tmp_points = points
    if improve:
        barycenter = np.mean(points[indices], axis=0)
        tmp_points = np.vstack((points, barycenter))
        inside_points = np.append(inside_points, len(tmp_points) - 1)
        for inside_point in inside_points:
            triangulation = point_in_cavity_to_others(inside_point, edges)
            triangulations.append(triangulation)
        best_triangulations = triangulations
    else:
        best_triangulations, _ = triangulations_minimal_area(tmp_points, triangulations)

    if len(best_triangulations) == 1:
        return points, best_triangulations[0]

    best_triangulation, _ = triangulations_first_min_shape_factor(
        tmp_points, best_triangulations, improve
    )
    if improve:
        if point_in_triangles(len(tmp_points) - 1, best_triangulation):
            points = tmp_points
    return points, best_triangulation


def cut_and_paste(
    points: np.ndarray,
    triangulation: np.ndarray,
    old_triangles: np.ndarray,
    improve: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform cut-and-paste operation on a triangulation.
    Args:
        points (np.ndarray): The array of points.
        triangulation (np.ndarray): The current triangulation.
        old_triangles (np.ndarray): The triangles to be removed.
        plot_index (int): The current plot index.
        valid_sfs_list (list, optional): A list of valid shape factors. Defaults to None.
    Returns:
        tuple[np.ndarray, np.ndarray]: The updated points and triangulation.
    """
    inside_points = None
    if improve:
        bound_points, bound_edges, inside_points = convex_hull(points, old_triangles)
        for inside_point in inside_points:
            add_old_triangles = point_elements(inside_point, triangulation)
            old_triangles = np.vstack((old_triangles, add_old_triangles))
        bound_points, bound_edges, new_inside_points = convex_hull(
            points, old_triangles
        )
        while len(new_inside_points) != len(inside_points):
            inside_points = new_inside_points
            for inside_point in inside_points:
                add_old_triangles = point_elements(inside_point, triangulation)
                old_triangles = np.vstack((old_triangles, add_old_triangles))
            bound_points, bound_edges, new_inside_points = convex_hull(
                points, old_triangles
            )
    else:
        bound_edges, bound_points = elements_boundary(old_triangles)
    points, new_triangles = best_point_of_cavity_to_others(
        points, bound_points, bound_edges, improve, inside_points
    )
    if len(new_triangles) > 0:
        triangulation = remove_triangles_from_triangulation(
            old_triangles, triangulation
        )
        triangulation = add_triangles_to_triangulation(new_triangles, triangulation)
        points, triangulation = remove_unused_points(points, triangulation)
    return points, triangulation


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
