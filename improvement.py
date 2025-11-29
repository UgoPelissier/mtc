import os
import meshio

from src.geometry import min_max_edge_length
from src.macro_triangulation import (
    iterate_over_points,
    iterate_over_edges,
)
from src.utils import save_mesh


def improve_mesh(
    mesh: meshio.Mesh, max_mesh_size_user: float, max_iter: int, save_path: str
) -> None:
    """
    Improve the mesh by iteratively refining it until the maximum edge length
    is below a specified threshold or the maximum number of iterations is reached.
    Args:
        mesh (meshio.Mesh): The initial mesh to be improved.
        max_mesh_size_user (float): The desired maximum edge length for the mesh.
        max_iter (int): The maximum number of iterations to perform.
        save_path (str): The path to save the improved mesh.
    """
    points = mesh.points[:, :2]
    cells = mesh.cells_dict["triangle"]

    min_mesh_size, max_mesh_size = min_max_edge_length(points[cells])
    print(f"Initial mesh size: min = {min_mesh_size:.4f}, max = {max_mesh_size:.4f}")

    iter = 0
    while True:
        # Iterative improvement over points
        iter, points, cells, _ = iterate_over_points(
            iter,
            points,
            cells,
            improve=True,
        )
        min_mesh_size, max_mesh_size = min_max_edge_length(points[cells])
        print(
            f"After points iteration {iter}: min = {min_mesh_size:.4f}, max = {max_mesh_size:.4f}"
        )
        if max_mesh_size <= max_mesh_size_user or iter >= max_iter:
            break
        # Iterative improvement over edges
        iter, points, cells, _ = iterate_over_edges(
            iter,
            points,
            cells,
            improve=True,
        )
        min_mesh_size, max_mesh_size = min_max_edge_length(points[cells])
        print(
            f"After edges iteration {iter}: min = {min_mesh_size:.4f}, max = {max_mesh_size:.4f}"
        )
        if max_mesh_size <= max_mesh_size_user or iter >= max_iter:
            break
    save_mesh(
        points,
        cells,
        save_path,
    )


if __name__ == "__main__":

    # User parameters
    INITIAL_MESH_PATH = os.path.join("mesh", "convex_2", "convex_2.vtu")
    MAX_MESH_SIZE = 0.3
    MAX_ITER = 100

    # Load mesh
    mesh = meshio.read(INITIAL_MESH_PATH)

    # Improve mesh
    improve_mesh(
        mesh,
        MAX_MESH_SIZE,
        MAX_ITER,
        save_path=os.path.join("mesh", "convex_2", "improved_convex_2.vtu"),
    )
