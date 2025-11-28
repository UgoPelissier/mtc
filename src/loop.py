import numpy as np
import os

from .geometry import shape_factors, total_area
from .macro_triangulation import check_triangulation_validity
from .utils import plot_polygon_and_tris


def update_mesh(
    mesh: np.ndarray,
    mesh_iteration: int,
    triangulation: np.ndarray,
    points: np.ndarray,
    iter: int,
) -> tuple[np.ndarray, int]:
    """
    Update the best mesh based on shape factors.
    Args:
        mesh (np.ndarray): The current best mesh.
        mesh_iteration (int): The iteration count of the current best mesh.
        triangulation (np.ndarray): The current triangulation.
        points (np.ndarray): The array of points.
        iter (int): The current iteration count.
    Returns:
        tuple[np.ndarray, int]: The updated best mesh and its iteration count.
    """
    if mesh is None:
        mesh = triangulation.copy()
        mesh_iteration = iter
    else:
        mesh_sfs = shape_factors(points[mesh])
        current_sfs = shape_factors(points[triangulation])
        if np.min(current_sfs) > np.min(mesh_sfs) and np.mean(current_sfs) > np.mean(
            mesh_sfs
        ):
            mesh = triangulation.copy()
            mesh_iteration = iter
    return mesh, mesh_iteration


def update_best_mesh(
    mesh: np.ndarray,
    mesh_iteration: int,
    triangulation: np.ndarray,
    points: np.ndarray,
    start_index: int,
    valid_start_indices: list,
    best_mesh: np.ndarray,
    best_mesh_index: int,
    plot_index: int,
    save_path: str,
) -> tuple[np.ndarray, int, int]:
    """
    Update the best mesh found during iterations.
    Args:
        mesh (np.ndarray): The current best mesh.
        mesh_iteration (int): The iteration count of the current best mesh.
        triangulation (np.ndarray): The current triangulation.
        points (np.ndarray): The array of points.
        start_index (int): The starting index for triangulation.
        valid_start_indices (list): A list of valid starting indices.
        best_mesh (np.ndarray): The overall best mesh found.
        best_mesh_index (int): The starting index of the overall best mesh.
        plot_index (int): The current plot index.
        save_path (str): The path to save plots.
    Returns:
        tuple[np.ndarray, int, int]: The updated best mesh, its starting index, and plot index.
    """
    print("\nMaximum number of iterations reached!")
    if mesh is not None:
        print(f"-> Valid mesh found at iteration {mesh_iteration}.")
        valid_start_indices.append(start_index)
        if best_mesh is None:
            best_mesh = mesh.copy()
            best_mesh_index = start_index
        else:
            best_mesh_sfs = shape_factors(points[best_mesh])
            current_sfs = shape_factors(points[mesh])
            if np.min(current_sfs) > np.min(best_mesh_sfs) and np.mean(
                current_sfs
            ) > np.mean(best_mesh_sfs):
                best_mesh = mesh.copy()
                best_mesh_index = start_index
        plot_index = plot_polygon_and_tris(
            points,
            mesh,
            save_path=os.path.join(save_path, f"Final_Mesh_{start_index}.png"),
            plot_index=plot_index,
        )
    else:
        print(f"\nTriangulation Area = {total_area(points[triangulation]):.10f}")
        print("-> No valid mesh found.")
    return best_mesh, best_mesh_index, plot_index


def check_update_end(
    points: np.ndarray,
    polygon_edges: np.ndarray,
    triangulation: np.ndarray,
    area_polygon: float,
    valid_sfs_list: list,
    mesh: np.ndarray,
    mesh_iteration: int,
    iter: int,
    max_iter: int,
    start_index: int,
    valid_start_indices: list,
    best_mesh: np.ndarray,
    best_mesh_index: int,
    plot_index: int,
    save_path: str,
) -> tuple[bool, list[float], np.ndarray, int, np.ndarray, int, int]:
    """
    Check triangulation validity, update mesh, and determine if the process should end.
    Args:
        points (np.ndarray): The array of points.
        polygon_edges (np.ndarray): The edges of the polygon.
        triangulation (np.ndarray): The current triangulation.
        area_polygon (float): The area of the polygon.
        valid_sfs_list (list): A list of valid shape factors.
        mesh (np.ndarray): The current best mesh.
        mesh_iteration (int): The iteration count of the current best mesh.
        iter (int): The current iteration count.
        max_iter (int): The maximum number of iterations allowed.
        start_index (int): The starting index for triangulation.
        valid_start_indices (list): A list of valid starting indices.
        best_mesh (np.ndarray): The overall best mesh found.
        best_mesh_index (int): The starting index of the overall best mesh.
        plot_index (int): The current plot index.
        save_path (str): The path to save plots.
    Returns:
        tuple: A tuple containing a boolean indicating if the process should end,
               the updated list of valid shape factors, the updated mesh,
               its iteration count, the overall best mesh, its starting index,
               and the plot index.
    """
    end = False
    valid, valid_sfs_list = check_triangulation_validity(
        points,
        polygon_edges,
        triangulation,
        area_polygon,
        valid_sfs_list,
        verbose=True,
    )
    if valid:
        mesh, mesh_iteration = update_mesh(
            mesh, mesh_iteration, triangulation, points, iter
        )
    if iter > max_iter:
        best_mesh, best_mesh_index, plot_index = update_best_mesh(
            mesh,
            mesh_iteration,
            triangulation,
            points,
            start_index,
            valid_start_indices,
            best_mesh,
            best_mesh_index,
            plot_index,
            save_path,
        )
        end = True
    return (
        end,
        valid_sfs_list,
        mesh,
        mesh_iteration,
        best_mesh,
        best_mesh_index,
        plot_index,
    )
