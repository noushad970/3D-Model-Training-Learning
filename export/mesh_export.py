import numpy as np
from skimage import measure
import trimesh

def voxel_to_mesh(voxel_grid, threshold=0.5, save_path="mesh.obj"):
    """
    Convert a voxel grid to a mesh (.obj) file.

    Parameters:
    voxel_grid : torch.Tensor or np.ndarray
        3D array of shape (1, voxel_size, voxel_size, voxel_size) or (voxel_size, voxel_size, voxel_size)
    threshold : float
        Voxels above this value are considered solid
    save_path : str
        Path to save the .obj file
    """

    # Convert to numpy array if using PyTorch tensor
    if not isinstance(voxel_grid, np.ndarray):
        voxel_grid = voxel_grid.squeeze().detach().cpu().numpy()

    # Apply threshold to determine solid voxels
    voxels = voxel_grid > threshold

    # Run marching cubes algorithm
    verts, faces, normals, _ = measure.marching_cubes(voxels, level=0.5)

    # Create mesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

    # Export to .obj
    mesh.export(save_path)
    print(f"Mesh saved to: {save_path}")

    return mesh
