import numpy as np
import matplotlib.pyplot as plt
import meshio


def plot_polygon_and_tris(
    polygon: np.ndarray,
    tris: np.ndarray,
    save_path: str = None,
    plot_index: int = 0,
    missing_edges: np.ndarray = None,
) -> int:
    """
    Plot a polygon and its triangulation.
    Args:
        polygon (np.ndarray): An array of vertices representing the polygon.
        tris (np.ndarray): An array of triangles, each represented by a 3x2 array of vertices.
        save_path (str, optional): The path to save the plot. Defaults to None.
        plot_index (int, optional): The current plot index. Defaults to 0.
        missing_edges (np.ndarray, optional): An array of missing edges to highlight. Defaults to None.
    Returns:
        int: The updated plot index.
    """
    for tri in tris:
        for i in range(len(tri)):
            j = (i + 1) % len(tri)
            pt1 = polygon[tri[i]]
            pt2 = polygon[tri[j]]
            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], "-", color="blue")
    # poly_pts = np.concatenate((polygon, polygon[:1]), axis=0)
    # plt.plot(poly_pts[:, 0], poly_pts[:, 1], "--", color="gray")
    if missing_edges is not None:
        for edge in missing_edges:
            pt1 = polygon[edge[0]]
            pt2 = polygon[edge[1]]
            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], "-", color="red")
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.clf()
    return plot_index + 1


def save_mesh(
    points: np.ndarray,
    triangulation: np.ndarray,
    save_path: str,
) -> None:
    """
    Save the mesh to a file.
    Args:
        points (np.ndarray): An array of points.
        triangulation (np.ndarray): An array of triangles.
        save_path (str): The path to save the mesh file.
    """
    mesh = meshio.Mesh(points, [("triangle", triangulation)])
    meshio.write(save_path, mesh)


def debug(text: str, array: np.ndarray = None) -> None:
    """
    Utility function for debugging output.
    """
    if array is None:
        print(f"{text}")
    else:
        print(f"{text}:\n{array}")


def floats_equal(list1: list[float], list2: list[float], tol=1e-8) -> bool:
    """
    Check if two lists of floats are equal within a tolerance.
     Args:
        list1 (list[float]): First list of floats.
        list2 (list[float]): Second list of floats.
        tol (float): Tolerance for comparison.
    Returns:
        bool: True if lists are equal within tolerance, False otherwise.
    """
    if len(list1) != len(list2):
        return False

    for a, b in zip(list1, list2):
        if abs(a - b) > tol:
            return False
    return True
