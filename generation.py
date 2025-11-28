import os
import meshio

from src.macro_triangulation import (
    initial_triangulation,
    iterate_over_edges,
    iterate_over_points,
)
from src.loop import check_update_end
from src.utils import plot_polygon_and_tris, save_mesh


def generate_mesh(polygon: list, save_path: str, max_iter: int):
    valid_start_indices = []
    best_mesh = None
    best_mesh_index = -1
    for start_index in range(len(polygon)):
        mesh = None
        mesh_iteration = None
        valid_sfs_list = []
        print(f"\n=== Starting point index: {start_index} ===")
        if best_mesh_index != -1:
            print(f"Current best mesh start index: {best_mesh_index}")
        try:
            # Initial mesh generation
            (
                points,
                area_polygon,
                polygon_edges,
                triangulation,
                plot_index,
                valid_sfs_list,
            ) = initial_triangulation(polygon, start_index, valid_sfs_list)
            iter = 0
            while True:
                # Iterative improvement over points
                triangulation, iter, plot_index, valid_sfs_list = iterate_over_points(
                    iter,
                    points,
                    polygon_edges,
                    triangulation,
                    area_polygon,
                    plot_index,
                    valid_sfs_list,
                )
                (
                    end,
                    valid_sfs_list,
                    mesh,
                    mesh_iteration,
                    best_mesh,
                    best_mesh_index,
                    plot_index,
                ) = check_update_end(
                    points,
                    polygon_edges,
                    triangulation,
                    area_polygon,
                    valid_sfs_list,
                    mesh,
                    mesh_iteration,
                    iter,
                    max_iter,
                    start_index,
                    valid_start_indices,
                    best_mesh,
                    best_mesh_index,
                    plot_index,
                    save_path,
                )
                if end:
                    break
                # Iterative improvement over edges
                triangulation, iter, plot_index, valid_sfs_list = iterate_over_edges(
                    iter,
                    points,
                    polygon_edges,
                    triangulation,
                    area_polygon,
                    plot_index,
                    valid_sfs_list,
                )
                (
                    end,
                    valid_sfs_list,
                    mesh,
                    mesh_iteration,
                    best_mesh,
                    best_mesh_index,
                    plot_index,
                ) = check_update_end(
                    points,
                    polygon_edges,
                    triangulation,
                    area_polygon,
                    valid_sfs_list,
                    mesh,
                    mesh_iteration,
                    iter,
                    max_iter,
                    start_index,
                    valid_start_indices,
                    best_mesh,
                    best_mesh_index,
                    plot_index,
                    save_path,
                )
                if end:
                    break
        except Exception as e:
            print(f"\nFailed for starting index {start_index}: {e}")
            continue
    return valid_start_indices, best_mesh_index, best_mesh


if __name__ == "__main__":

    # Load geometry
    poly = meshio.read(os.path.join("geometry", "convex_2", "convex_2.vtu"))
    boundary_edges = poly.cells_dict["line"]
    poly = poly.points[boundary_edges[:, 0]][:, :2]

    # User selection
    POLY = poly
    FIG_SAVE_PATH = os.path.join("figure", "results")
    MESH_SAVE_PATH = os.path.join("mesh", "results")
    MAX_ITER = 100
    os.makedirs(FIG_SAVE_PATH, exist_ok=True)
    os.makedirs(MESH_SAVE_PATH, exist_ok=True)

    # Main loop over starting points
    valid_start_indices, best_mesh_index, best_mesh = generate_mesh(
        POLY, FIG_SAVE_PATH, MAX_ITER
    )

    print(f"\nValid starting indices: {valid_start_indices}")
    print(f"Best mesh found with starting index: {best_mesh_index}")
    if best_mesh is not None:
        plot_polygon_and_tris(
            POLY,
            best_mesh,
            save_path=os.path.join(FIG_SAVE_PATH, "Best_Mesh.png"),
        )
        save_mesh(
            POLY,
            best_mesh,
            save_path=os.path.join(MESH_SAVE_PATH, "mesh.vtu"),
        )
