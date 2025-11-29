import numpy as np

from .utils import debug, plot_polygon_and_tris

from .geometry import (
    area,
    polygon_area,
    polygon_to_edges,
    total_area,
    shape_factors,
    triangles_to_edges,
    edges_in_triangulation,
    point_elements,
    edge_elements,
)
from .triangulation import point_of_cavity_to_others, cut_and_paste


def remove_degenerate_triangles_from_triangulation(
    points: np.ndarray, triangulation: np.ndarray
) -> np.ndarray:
    """
    Remove degenerate triangles (with area below a threshold) from a triangulation.
    Args:
        points (np.ndarray): An array of points.
        triangulation (np.ndarray): An array of triangles, each represented by a 3x2 array of vertices.
    Returns:
        np.ndarray: The triangulation after removing degenerate triangles.
    """
    areas = np.array([area(points[tri]) for tri in triangulation])
    degenerate_triangles = areas < 1e-8
    return triangulation[~degenerate_triangles]


def check_triangulation_validity(
    points: np.ndarray,
    polygon_edges: np.ndarray,
    triangulation: np.ndarray,
    area_polygon: float,
    valid_sfs_list: list,
    verbose: bool = False,
) -> tuple[bool, list[float]]:
    """
    Check the validity of a triangulation against a polygon.
    Args:
        points (np.ndarray): An array of points.
        polygon_edges (np.ndarray): An array of polygon edges.
        triangulation (np.ndarray): An array of triangles.
        area_polygon (float): The area of the polygon.
        valid_sfs_list (list): A list to store valid shape factors.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
    Returns:
        tuple[bool, list]: A tuple containing a boolean indicating validity and the updated list of valid shape factors.
    """
    triangulation_area = total_area(points[triangulation])
    if abs(triangulation_area - area_polygon) < 1e-8:
        triangulation = remove_degenerate_triangles_from_triangulation(
            points, triangulation
        )
        all_edges_in_triangulation, _, _, _ = edges_in_triangulation(
            polygon_edges, triangulation, points
        )
        if all_edges_in_triangulation:
            if verbose:
                sfs = shape_factors(points[triangulation])
                min_sf = sfs[0]
                mean_sf = np.mean(sfs)
                max_sf = sfs[-1]
                valid_sfs_list.append([min_sf, mean_sf, max_sf])
            return True, valid_sfs_list
        else:
            return False, valid_sfs_list
    return False, valid_sfs_list


def initial_triangulation(
    points: np.ndarray, start_index: int, valid_sfs_list: list
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, int, list[float]]:
    """
    Generate an initial triangulation for a polygon.
    Args:
        polygon (np.ndarray): An array of vertices representing the polygon.
        start_index (int): The index of the starting point for triangulation.
        valid_sfs_list (list): A list to store valid shape factors.
    Returns:
        tuple[np.ndarray, float, np.ndarray, np.ndarray, int, list]: A tuple containing points, area of the polygon,
        polygon edges, triangulation, plot index, and the updated list of valid shape factors.
    """
    area_polygon = polygon_area(points)
    polygon_edges = polygon_to_edges(points)
    triangulation = point_of_cavity_to_others(start_index, polygon_edges)
    plot_index = 0
    _, valid_sfs_list = check_triangulation_validity(
        points, polygon_edges, triangulation, area_polygon, valid_sfs_list
    )
    return (
        points,
        area_polygon,
        polygon_edges,
        triangulation,
        plot_index,
        valid_sfs_list,
    )


def iterate_over_points(
    iter: int,
    points: np.ndarray,
    triangulation: np.ndarray,
    improve: bool = False,
    polygon_edges: np.ndarray = None,
    area_polygon: float = None,
    valid_sfs_list: list = None,
) -> tuple[int, np.ndarray, np.ndarray, list[float]]:
    """
    Iterate over points to improve triangulation.
    Args:
        iter (int): The current iteration count.
        points (np.ndarray): The array of points.
        triangulation (np.ndarray): The current triangulation.
        improve (bool, optional): Whether to improve triangulation. Defaults to False.
        polygon_edges (np.ndarray, optional): The edges of the polygon. Defaults to None.
        area_polygon (float, optional): The area of the polygon. Defaults to None.
        valid_sfs_list (list, optional): A list of valid shape factors. Defaults to None.
    Returns:
        tuple[int, np.ndarray, np.ndarray, list]: The updated iter, points, triangulation, and valid shape factors list.
    """
    start_points = points.copy()
    for i in range(len(start_points)):
        if np.any(np.all(np.abs(points - start_points[i]) < 1e-8, axis=1)):
            iter += 1
            old_triangles = point_elements(i, triangulation)
            if len(old_triangles) > 0:
                points, triangulation = cut_and_paste(
                    points,
                    triangulation,
                    old_triangles,
                    improve,
                )
    if not improve:
        _, valid_sfs_list = check_triangulation_validity(
            points, polygon_edges, triangulation, area_polygon, valid_sfs_list
        )
    return iter, points, triangulation, valid_sfs_list


def iterate_over_edges(
    iter: int,
    points: np.ndarray,
    triangulation: np.ndarray,
    improve: bool = False,
    polygon_edges: np.ndarray = None,
    area_polygon: float = None,
    valid_sfs_list: list = None,
) -> tuple[int, np.ndarray, np.ndarray, list[float]]:
    """
    Iterate over points to improve triangulation.
    Args:
        iter (int): The current iteration count.
        points (np.ndarray): The array of points.
        triangulation (np.ndarray): The current triangulation.
        improve (bool, optional): Whether to improve triangulation. Defaults to False.
        polygon_edges (np.ndarray, optional): The edges of the polygon. Defaults to None.
        area_polygon (float, optional): The area of the polygon. Defaults to None.
        valid_sfs_list (list, optional): A list of valid shape factors. Defaults to None.
    Returns:
        tuple[int, np.ndarray, np.ndarray, list]: The updated iter, points, triangulation, and valid shape factors list.
    """
    start_edges = triangles_to_edges(triangulation)
    edges = start_edges.copy()
    for edge in start_edges:
        edges_set = set(map(tuple, edges))
        if tuple(edge) in edges_set:
            iter += 1
            old_triangles = edge_elements(edge, triangulation)
            if len(old_triangles) > 0:
                points, triangulation = cut_and_paste(
                    points,
                    triangulation,
                    old_triangles,
                    improve,
                )
                edges = triangles_to_edges(triangulation)
    if not improve:
        _, valid_sfs_list = check_triangulation_validity(
            points, polygon_edges, triangulation, area_polygon, valid_sfs_list
        )
    return iter, points, triangulation, valid_sfs_list
