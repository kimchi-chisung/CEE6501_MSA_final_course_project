import numpy as np

# get_plane_reference_vector
# element_local_axes_3d
# truss_transformation_matrix_3d
# frame_transformation_matrix_3d

# truss_k_local_3d
# truss_local_displacements_3d
# truss_local_end_forces_3d
# truss_axial_force_3d

# frame_k_local_3d
# frame_local_displacements_3d
# frame_local_end_forces_3d


def get_plane_reference_vector(plane=None):
    """
    Return a reference vector normal to the analysis plane.

    Parameters
    ----------
    plane : str or None
        Supported:
        - "xy" : out-of-plane = global z
        - "xz" : out-of-plane = global y
        - "yz" : out-of-plane = global x
        - None : fully automatic 3D behavior

    Returns
    -------
    v_ref : (3,) ndarray or None
    """
    if plane is None:
        return None

    plane = plane.lower()

    if plane == "xy":
        return np.array([0.0, 0.0, 1.0], dtype=float)
    elif plane == "xz":
        return np.array([0.0, 1.0, 0.0], dtype=float)
    elif plane == "yz":
        return np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        raise ValueError("plane must be one of: 'xy', 'xz', 'yz', or None")


def element_local_axes_3d(l, m, n, v_ref=None):
    """
    Build a right-handed local coordinate system for a 3D element.

    Parameters
    ----------
    l, m, n : float
        Direction cosines of the local x-axis.
    v_ref : array-like of length 3 or None
        Reference vector used to define the local y/z orientation.
        For plane problems:
        - xy plane -> v_ref = [0, 0, 1]
        - xz plane -> v_ref = [0, 1, 0]
        - yz plane -> v_ref = [1, 0, 0]

    Returns
    -------
    R : (3,3) ndarray
        Rotation matrix whose rows are the local x, y, z axes
        expressed in global coordinates.
    """
    ex = np.array([l, m, n], dtype=float)
    ex = ex / np.linalg.norm(ex)

    if v_ref is None:
        ref = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(np.dot(ex, ref)) > 0.99:
            ref = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        ref = np.asarray(v_ref, dtype=float)
        ref = ref / np.linalg.norm(ref)

        if abs(np.dot(ex, ref)) > 0.99:
            raise ValueError(
                "Reference vector v_ref is parallel (or nearly parallel) "
                "to the element axis."
            )

    # local z axis
    ez = np.cross(ex, ref)
    ez = ez / np.linalg.norm(ez)

    # local y axis
    ey = np.cross(ez, ex)
    ey = ey / np.linalg.norm(ey)

    R = np.array(
        [
            [ex[0], ex[1], ex[2]],
            [ey[0], ey[1], ey[2]],
            [ez[0], ez[1], ez[2]],
        ],
        dtype=float,
    )

    return R


def truss_transformation_matrix_3d(l, m, n, v_ref=None):
    """
    6x6 transformation matrix for a 3D truss element.

    DOF order:
        [ux_i, uy_i, uz_i, ux_j, uy_j, uz_j]
    """
    R = element_local_axes_3d(l, m, n, v_ref=v_ref)
    Z = np.zeros((3, 3), dtype=float)

    T = np.block([[R, Z], [Z, R]])
    return T


def frame_transformation_matrix_3d(l, m, n, v_ref=None):
    """
    12x12 transformation matrix for a 3D frame element.

    DOF order:
        [ux_i, uy_i, uz_i, rx_i, ry_i, rz_i,
         ux_j, uy_j, uz_j, rx_j, ry_j, rz_j]
    """
    R = element_local_axes_3d(l, m, n, v_ref=v_ref)
    Z = np.zeros((3, 3), dtype=float)

    T = np.block(
        [
            [R, Z, Z, Z],
            [Z, R, Z, Z],
            [Z, Z, R, Z],
            [Z, Z, Z, R],
        ]
    )
    return T


def truss_k_local_3d(E, A, L):
    factor = E * A / L

    k_local = factor * np.array(
        [
            [1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    return k_local


def truss_local_displacements_3d(T, u_element_global):
    u_element_global = np.asarray(u_element_global, dtype=float).reshape(6)
    return T @ u_element_global


def truss_local_end_forces_3d(k_local, u_local, Qf_local=None):
    u_local = np.asarray(u_local, dtype=float).reshape(6)

    if Qf_local is None:
        Qf_local = np.zeros(6, dtype=float)
    else:
        Qf_local = np.asarray(Qf_local, dtype=float).reshape(6)

    return k_local @ u_local + Qf_local


def truss_axial_force_3d(q_local):
    q_local = np.asarray(q_local, dtype=float).reshape(6)
    return float(q_local[3])


def frame_k_local_3d(E, A, I, J, L, G=None):
    if G is None:
        nu = 0.3
        G = E / (2.0 * (1.0 + nu))

    EA_L = E * A / L
    GJ_L = G * J / L

    EI = E * I
    c1 = 12.0 * EI / L**3
    c2 = 6.0 * EI / L**2
    c3 = 4.0 * EI / L
    c4 = 2.0 * EI / L

    k = np.array(
        [
            [EA_L, 0.0, 0.0, 0.0, 0.0, 0.0, -EA_L, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, c1, 0.0, 0.0, 0.0, c2, 0.0, -c1, 0.0, 0.0, 0.0, c2],
            [0.0, 0.0, c1, 0.0, -c2, 0.0, 0.0, 0.0, -c1, 0.0, -c2, 0.0],
            [0.0, 0.0, 0.0, GJ_L, 0.0, 0.0, 0.0, 0.0, 0.0, -GJ_L, 0.0, 0.0],
            [0.0, 0.0, -c2, 0.0, c3, 0.0, 0.0, 0.0, c2, 0.0, c4, 0.0],
            [0.0, c2, 0.0, 0.0, 0.0, c3, 0.0, -c2, 0.0, 0.0, 0.0, c4],
            [-EA_L, 0.0, 0.0, 0.0, 0.0, 0.0, EA_L, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -c1, 0.0, 0.0, 0.0, -c2, 0.0, c1, 0.0, 0.0, 0.0, -c2],
            [0.0, 0.0, -c1, 0.0, c2, 0.0, 0.0, 0.0, c1, 0.0, c2, 0.0],
            [0.0, 0.0, 0.0, -GJ_L, 0.0, 0.0, 0.0, 0.0, 0.0, GJ_L, 0.0, 0.0],
            [0.0, 0.0, -c2, 0.0, c4, 0.0, 0.0, 0.0, c2, 0.0, c3, 0.0],
            [0.0, c2, 0.0, 0.0, 0.0, c4, 0.0, -c2, 0.0, 0.0, 0.0, c3],
        ],
        dtype=float,
    )

    return k


def frame_local_displacements_3d(T, u_element_global):
    u_element_global = np.asarray(u_element_global, dtype=float).reshape(12)
    return T @ u_element_global


def frame_local_end_forces_3d(k_local, u_local, Qf_local=None):
    u_local = np.asarray(u_local, dtype=float).reshape(12)

    if Qf_local is None:
        Qf_local = np.zeros(12, dtype=float)
    else:
        Qf_local = np.asarray(Qf_local, dtype=float).reshape(12)

    return k_local @ u_local + Qf_local
