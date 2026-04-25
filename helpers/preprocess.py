import numpy as np
import json

# load_model
# node_dofs_1based_6dof
# restrained_dofs_1based
# loaded_dofs_1based
# dof_map_truss_3d
# dof_map_frame_3d
# element_csl
# build_global_load_vector
# build_global_displacement_vector


def load_model(filename):
    """
    Load a JSON structural model file.

    Expected top-level keys may include:
        - model_name
        - nodes
        - elements
        - supports
        - nodal_loads
        - member_loads
        - prescribed_displacements
        - temperature_loads
        - fabrication_errors

    Returns
    -------
    model : dict
        Parsed model dictionary with integer IDs where appropriate.
    """
    with open(filename, "r", encoding="utf-8") as f:
        raw = json.load(f)

    model = {
        "model_name": raw.get("model_name", "unnamed_model"),
        "nodes": {int(k): v for k, v in raw.get("nodes", {}).items()},
        "elements": {int(k): v for k, v in raw.get("elements", {}).items()},
        "supports": {int(k): v for k, v in raw.get("supports", {}).items()},
        "nodal_loads": {int(k): v for k, v in raw.get("nodal_loads", {}).items()},
        "member_loads": {int(k): v for k, v in raw.get("member_loads", {}).items()},
        "prescribed_displacements": {
            int(k): v for k, v in raw.get("prescribed_displacements", {}).items()
        },
        "temperature_loads": {
            int(k): v for k, v in raw.get("temperature_loads", {}).items()
        },
        "fabrication_errors": {
            int(k): v for k, v in raw.get("fabrication_errors", {}).items()
        },
        "releases": {int(k): v for k, v in raw.get("releases", {}).items()},
    }

    return model


def node_dofs_1based_6dof(node_id: int) -> list[int]:
    """
    Return the 1-based global DOF numbers for a node in a 6-DOF/node system.

    DOF order per node:
        [ux, uy, uz, rx, ry, rz]

    Example
    -------
    node 1 -> [1, 2, 3, 4, 5, 6]
    node 2 -> [7, 8, 9, 10, 11, 12]
    """

    if node_id < 1:
        raise ValueError("node_id must be >= 1")

    start = 6 * (node_id - 1)
    return [start + 1, start + 2, start + 3, start + 4, start + 5, start + 6]


def restrained_dofs_1based(nodes_restrained, node_dofs_1based):
    """
    Return sorted list of restrained DOFs (1-based) from node restraints.

    Expected format:
        nodes_restrained = {
            1: ["ux", "uy", "uz", "rx", "ry", "rz"],
            2: ["uy", "uz"],
            ...
        }
    """
    dof_restrained = []

    dof_names = ["ux", "uy", "uz", "rx", "ry", "rz"]

    for node, restraints in nodes_restrained.items():
        node_dofs = node_dofs_1based(node)
        for name, dof in zip(dof_names, node_dofs):
            if name in restraints:
                dof_restrained.append(dof)

    return sorted(dof_restrained)


def loaded_dofs_1based(nodes_loaded, node_dofs_1based):
    """
    Return DOF-to-load mapping (1-based) from nodal loads.

    Expected format:
        nodes_loaded = {
            1: [Fx, Fy, Fz, Mx, My, Mz],
            ...
        }
    """
    dof_loaded = {}

    for node, load in nodes_loaded.items():
        node_dofs = node_dofs_1based(node)

        if len(load) != 6:
            raise ValueError(
                f"Node {node} load must have 6 components: [Fx, Fy, Fz, Mx, My, Mz]"
            )

        for dof, val in zip(node_dofs, load):
            if val != 0.0:
                dof_loaded[dof] = dof_loaded.get(dof, 0.0) + val

    return dof_loaded


def dof_map_truss_3d(i_node, j_node):
    """
    Return the 1-based global DOF map for a 3D truss element
    embedded in a 6-DOF-per-node global system.

    Global DOF order per node:
        [ux, uy, uz, rx, ry, rz]

    Truss element uses only translational DOFs:
        [ux_i, uy_i, uz_i, ux_j, uy_j, uz_j]

    Parameters
    ----------
    i_node, j_node : int
        1-based node IDs

    Returns
    -------
    m : (6,) ndarray of int
        Element DOF map (1-based)
    """
    di = node_dofs_1based_6dof(i_node)
    dj = node_dofs_1based_6dof(j_node)

    return np.array([di[0], di[1], di[2], dj[0], dj[1], dj[2]], dtype=int)


def dof_map_frame_3d(i_node, j_node):
    """
    Return the 1-based global DOF map for a 3D frame element
    in a 6-DOF-per-node global system.

    Global DOF order per node:
        [ux, uy, uz, rx, ry, rz]

    Frame element uses all DOFs:
        [ux_i, uy_i, uz_i, rx_i, ry_i, rz_i,
         ux_j, uy_j, uz_j, rx_j, ry_j, rz_j]

    Parameters
    ----------
    i_node, j_node : int
        1-based node IDs

    Returns
    -------
    m : (12,) ndarray of int
        Element DOF map (1-based)
    """
    di = node_dofs_1based_6dof(i_node)
    dj = node_dofs_1based_6dof(j_node)

    return np.array(di + dj, dtype=int)


def element_csL(xyz_i, xyz_j):  # for 3D
    xyz_i = np.asarray(xyz_i, dtype=float)
    xyz_j = np.asarray(xyz_j, dtype=float)

    if xyz_i.shape[0] != 3 or xyz_j.shape[0] != 3:
        raise ValueError("Each node coordinate must have 3 components: [x, y, z]")

    dx = xyz_j[0] - xyz_i[0]
    dy = xyz_j[1] - xyz_i[1]
    dz = xyz_j[2] - xyz_i[2]

    L = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    if L == 0.0:
        raise ValueError("Zero-length element.")

    l = dx / L
    m = dy / L
    n = dz / L
    return float(l), float(m), float(n), float(L)


def build_global_load_vector(nodes, nodal_loads=None):
    """
    Build the global nodal load vector for a 6-DOF-per-node system.

    DOF order per node:
        [Fx, Fy, Fz, Mx, My, Mz]
    """
    if nodal_loads is None:
        nodal_loads = {}

    ndof = 6 * len(nodes)
    F_global = np.zeros(ndof, dtype=float)

    for node_id, load in nodal_loads.items():
        if node_id not in nodes:
            raise KeyError(f"Nodal load assigned to undefined node {node_id}")

        load = np.asarray(load, dtype=float)
        if load.shape[0] != 6:
            raise ValueError(
                f"Nodal load at node {node_id} must have 6 components: "
                "[Fx, Fy, Fz, Mx, My, Mz]"
            )

        dofs = node_dofs_1based_6dof(node_id)
        for a in range(6):
            A = dofs[a] - 1
            F_global[A] += load[a]

    return F_global


def build_global_displacement_vector(nodes, prescribed_displacements=None):
    """
    Build the global displacement vector for a 6-DOF-per-node system.

    Missing prescribed values are set to zero.
    """
    if prescribed_displacements is None:
        prescribed_displacements = {}

    ndof = 6 * len(nodes)
    u_global = np.zeros(ndof, dtype=float)

    for node_id, disp in prescribed_displacements.items():
        if node_id not in nodes:
            raise KeyError(
                f"Prescribed displacement assigned to undefined node {node_id}"
            )

        disp = np.asarray(disp, dtype=float)
        if disp.shape[0] != 6:
            raise ValueError(
                f"Prescribed displacement at node {node_id} must have 6 components"
            )

        dofs = node_dofs_1based_6dof(node_id)
        for a in range(6):
            A = dofs[a] - 1
            u_global[A] = disp[a]

    return u_global
