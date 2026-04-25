import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def assemble_global_stiffness_and_fef(
    ndof,
    k_list,
    T_list,
    Qf_list,
    map_list,
):
    """
    Assemble global stiffness matrix and global fixed-end force vector.

    Automatically handles 6-DOF (frame) and 4-DOF (truss/beam) elements.
    Parameters
    ----------
    ndof : int
        Total number of global degrees of freedom.

    k_list : list of ndarray
        List of local element stiffness matrices.
        Each matrix may be 6x6 (frame) or 4x4 (truss/beam).

    T_list : list of ndarray
        List of element transformation matrices corresponding
        to each k_local. Must be compatible in size.

    Qf_list : list of ndarray
        List of local fixed-end force vectors for each element.
        Size must match the element DOF count.

    map_list : list of array-like
        List of element DOF maps (1-based indexing).
        Each map defines where the element DOFs connect
        into the global DOF numbering.

    Returns
    -------
    K_global : ndarray (ndof x ndof)
        Assembled global stiffness matrix.

    F_fef_global : ndarray (ndof,)
        Assembled global fixed-end force vector.

    Notes
    -----
    - DOF maps are assumed to use 1-based indexing.
    - Internally converted to 0-based indexing for Python.
    - Assembly is dense; for large systems a sparse format
      should be used instead.
    """

    K_global = np.zeros((ndof, ndof), dtype=float)
    F_fef_global = np.zeros(ndof, dtype=float)

    nelem = len(k_list)

    for i in range(nelem):

        k_local = k_list[i]
        T = T_list[i]
        Qf_local = Qf_list[i]
        dof_map = map_list[i]  # 1-based indexing

        # Determine element DOF count automatically
        edof = k_local.shape[0]

        # Transform to global
        K = T.T @ k_local @ T
        F_fef = T.T @ Qf_local

        # Scatter-add
        for a in range(edof):
            A = dof_map[a] - 1  # convert to 0-based

            F_fef_global[A] += F_fef[a]

            for b in range(edof):
                B = dof_map[b] - 1
                K_global[A, B] += K[a, b]

    return K_global, F_fef_global


def partition_system(K, f, u, f_fef, dof_restrained_1based):
    ndof = K.shape[0]

    # Convert restrained DOFs to 0-based
    restrained_dofs = sorted(int(d) - 1 for d in dof_restrained_1based)

    # Free DOFs
    free_dofs = [i for i in range(ndof) if i not in restrained_dofs]

    # Partition stiffness matrix
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    K_fr = K[np.ix_(free_dofs, restrained_dofs)]
    K_rf = K[np.ix_(restrained_dofs, free_dofs)]
    K_rr = K[np.ix_(restrained_dofs, restrained_dofs)]

    # Partition force vector
    f_f = f[free_dofs]
    f_r = f[restrained_dofs]

    # Partition displaced vector
    u_r = u[restrained_dofs]

    # Partition fixed-end forces
    f_fef_f = f_fef[free_dofs]
    f_fef_r = f_fef[restrained_dofs]

    return (
        K_ff,
        K_fr,
        K_rf,
        K_rr,
        f_f,
        f_r,
        u_r,
        f_fef_f,
        f_fef_r,
        free_dofs,
        restrained_dofs,
    )


def assemble_global_displacements(u_f, u_r, free_dofs, restrained_dofs):
    """
    Assemble the full global displacement vector u from partitioned results.
    """
    ndof_total = len(free_dofs) + len(restrained_dofs)
    u_global = np.zeros(ndof_total)

    if u_r is None:
        u_r = np.zeros(len(restrained_dofs))

    u_global[free_dofs] = u_f
    u_global[restrained_dofs] = u_r

    return u_global


def assemble_global_forces(f_f, F_r, free_dofs, restrained_dofs):
    """
    Assemble the full global force vector f from applied loads and reactions.
    """
    ndof_total = len(free_dofs) + len(restrained_dofs)
    f_global = np.zeros(ndof_total)

    f_global[free_dofs] = f_f
    f_global[restrained_dofs] = F_r

    return f_global


def fef_local_2d_frame_point_midspan_moment_release(P, L, release=None):
    """
    Local fixed-end-force vector
    for a transverse point load P at midspan.

    DOF order:
        [u1, v1, th1, u2, v2, th2]
    """
    if release is None:
        return np.array([0, P / 2, P * L / 8, 0, P / 2, -P * L / 8], dtype=float)

    elif release == "MT1":
        return np.array([0, 3 * P / 8, 0, 0, 5 * P / 8, -P * L / 8], dtype=float)

    elif release == "MT2":
        return np.array([0, 5 * P / 8, P * L / 8, 0, 3 * P / 8, 0], dtype=float)

    elif release in ("MT1_MT2", "both"):
        return np.array([0, P / 2, 0, 0, P / 2, 0], dtype=float)

    else:
        raise ValueError("release must be None, 'MT1', 'MT2', or 'MT1_MT2'/'both'")


def fef_local_2d_frame_udl_moment_release(w, L, release=None):
    """
    Local fixed-end-force vector
    for a full-span transverse UDL on a 2D frame element.

    DOF order:
        [u1, v1, th1, u2, v2, th2]

    Sign convention:
        This returns the standard local element fixed-end-force vector
        consistent with:
            fixed: [0, wL/2, wL^2/12, 0, wL/2, -wL^2/12]
    """
    if release is None:
        return np.array(
            [0, w * L / 2, w * L**2 / 12, 0, w * L / 2, -w * L**2 / 12],
            dtype=float,
        )

    elif release == "MT1":
        return np.array(
            [0, 3 * w * L / 8, 0, 0, 5 * w * L / 8, -w * L**2 / 8], dtype=float
        )

    elif release == "MT2":
        return np.array(
            [0, 5 * w * L / 8, w * L**2 / 8, 0, 3 * w * L / 8, 0], dtype=float
        )

    elif release in ("MT1_MT2", "both"):
        return np.array([0, w * L / 2, 0, 0, w * L / 2, 0], dtype=float)

    else:
        raise ValueError("release must be None, 'MT1', 'MT2', or 'MT1_MT2'/'both'")


def k_local_2d_frame_moment_release(E, A, I_z, L, release=None):
    """
    Local stiffness matrix for a 2D planar frame element.

    DOF order:
        [u1, v1, th1, u2, v2, th2]

    Parameters
    ----------
    E : float
        Young's modulus
    A : float
        Cross-sectional area
    I_z : float
        Second moment of area
    L : float
        Element length
    release : str or None
        None  : fully fixed frame element
        "MT1" : moment release at end 1
        "MT2" : moment release at end 2
        "MT1_MT2" or "both" : moment releases at both ends

    Returns
    -------
    k : (6,6) ndarray
        Local element stiffness matrix
    """
    EA_L = E * A / L
    EI = E * I_z

    # axial part (always unchanged)
    k_axial = np.array(
        [
            [EA_L, 0.0, 0.0, -EA_L, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-EA_L, 0.0, 0.0, EA_L, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    if release is None:
        # standard 2D frame bending part
        k_bend = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [
                    0,
                    12 * EI / L**3,
                    6 * EI / L**2,
                    0,
                    -12 * EI / L**3,
                    6 * EI / L**2,
                ],
                [0, 6 * EI / L**2, 4 * EI / L, 0, -6 * EI / L**2, 2 * EI / L],
                [0, 0, 0, 0, 0, 0],
                [
                    0,
                    -12 * EI / L**3,
                    -6 * EI / L**2,
                    0,
                    12 * EI / L**3,
                    -6 * EI / L**2,
                ],
                [0, 6 * EI / L**2, 2 * EI / L, 0, -6 * EI / L**2, 4 * EI / L],
            ],
            dtype=float,
        )

    elif release == "MT1":
        # end 1 moment released
        k_bend = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 3 * EI / L**3, 0, 0, -3 * EI / L**3, 3 * EI / L**2],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, -3 * EI / L**3, 0, 0, 3 * EI / L**3, -3 * EI / L**2],
                [0, 3 * EI / L**2, 0, 0, -3 * EI / L**2, 3 * EI / L],
            ],
            dtype=float,
        )

    elif release == "MT2":
        # end 2 moment released
        k_bend = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 3 * EI / L**3, 3 * EI / L**2, 0, -3 * EI / L**3, 0],
                [0, 3 * EI / L**2, 3 * EI / L, 0, -3 * EI / L**2, 0],
                [0, 0, 0, 0, 0, 0],
                [0, -3 * EI / L**3, -3 * EI / L**2, 0, 3 * EI / L**3, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=float,
        )

    elif release in ("MT1_MT2", "both"):
        # both ends moment released -> truss-like bending behavior
        k_bend = np.zeros((6, 6), dtype=float)

    else:
        raise ValueError("release must be None, 'MT1', 'MT2', or 'MT1_MT2'/'both'")

    return k_axial + k_bend


def frame_transformation_2d(theta_deg):
    theta = np.deg2rad(theta_deg)
    c = np.cos(theta)
    s = np.sin(theta)

    T = np.array(
        [
            [c, s, 0, 0, 0, 0],
            [-s, c, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, c, s, 0],
            [0, 0, 0, -s, c, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=float,
    )

    return T


# ###################################################################################################################
# ###################################################################################################################
# ###################################################################################################################
# ###################################################################################################################
# ###################################################################################################################
# ###################################################################################################################
# ###################################################################################################################
# ###################################################################################################################
# ###################################################################################################################
# ###################################################################################################################
# ###################################################################################################################


def print_dsm_results(
    u_global,
    f_global_complete,
    dof_restrained_1based,
    dof_fictitious_1based=None,  # optional kwarg
    member_type="frame",
    disp_in_mm=False,
    dec=4,
    rad_dec=6,
):
    ndof = len(u_global)
    rows = []

    # Ensure arrays
    dof_restrained_1based = np.atleast_1d(dof_restrained_1based)

    if dof_fictitious_1based is None:
        dof_fictitious_1based = np.array([], dtype=int)
    else:
        dof_fictitious_1based = np.atleast_1d(dof_fictitious_1based)

    restrained_set = {int(d) for d in dof_restrained_1based}
    fictitious_set = {int(d) for d in dof_fictitious_1based}

    member_type = member_type.lower()

    if member_type == "frame":
        dof_labels = ["u_x", "u_y", "theta"]
        translational_idx = {0, 1}
    elif member_type == "truss":
        dof_labels = ["u_x", "u_y"]
        translational_idx = {0, 1}
    elif member_type == "beam":
        dof_labels = ["u_y", "theta"]
        translational_idx = {0}
    else:
        raise ValueError("member_type must be 'frame', 'truss', or 'beam'")

    dof_per_node = len(dof_labels)

    for i in range(ndof):
        dof_1based = i + 1
        mod = i % dof_per_node
        dof_type = dof_labels[mod]

        if mod in translational_idx:
            disp = u_global[i] * (1000 if disp_in_mm else 1)
            disp_str = f"{disp:.{dec}f}"
        else:
            disp_str = f"{u_global[i]:.{rad_dec}f}"

        load_str = f"{f_global_complete[i]:.{dec}f}"

        if dof_1based in fictitious_set:
            status = "Fictitious"
        elif dof_1based in restrained_set:
            status = "Fixed"
        else:
            status = "Free"

        rows.append([dof_1based, dof_type, status, disp_str, load_str])

    disp_unit = "mm" if disp_in_mm else "m"

    df = pd.DataFrame(
        rows,
        columns=[
            "DOF",
            "Type",
            "Status",
            f"Disp ({disp_unit} / rad)",
            "Load (kN / kN·m)",
        ],
    )

    print(df.to_string(index=False))


def print_element(e, u_global, m_1based, T, k, Qf, disp_in_mm=False, dec=4, rad_dec=6):

    idx = m_1based - 1
    u = u_global[idx]
    v = T @ u
    q = k @ v + Qf

    scale = 1000 if disp_in_mm else 1
    unit = "mm" if disp_in_mm else "m"

    # Scale translations only (0,1,3,4) — rotations (2,5) untouched
    u_out = u.copy()
    v_out = v.copy()
    for j in [0, 1, 3, 4]:
        u_out[j] *= scale
        v_out[j] *= scale

    def fmt_disp(vec):
        parts = []
        for j, val in enumerate(vec):
            if j % 3 == 2:  # rotation (rad)
                parts.append(f"{val:.{rad_dec}f}")
            else:  # translation
                parts.append(f"{val:.{dec}f}")
        return "[" + ", ".join(parts) + "]"

    def fmt_force(vec):
        # forces (kN) and moments (kN·m) both use dec
        return "[" + ", ".join(f"{val:.{dec}f}" for val in vec) + "]"

    print(f"\nE{e}")
    print(f"u [{unit},rad]: {fmt_disp(u_out)}")
    print(f"v [{unit},rad]: {fmt_disp(v_out)}")
    print(f"q [kN,kN·m]: {fmt_force(q)}")


def print_element_truss(
    e,
    u_global,
    m_1based,
    T,
    k_local,
    Qf_local=None,
    disp_in_mm=False,
    dec=3,
):
    """
    Print element-level results for a 2D truss element.

    Parameters
    ----------
    e : int
        Element number for printing.
    u_global : (ndof,) array
        Global displacement vector.
    m_1based : (4,) array-like
        Global DOF map for this element (1-based indexing).
    T : (4,4) array
        Transformation matrix (global -> local).
    k_local : (4,4) array
        Local truss stiffness matrix (typically axial-only with 1/-1 pattern).
    Qf_local : (4,) array or None
        Local fixed-end / initial-force vector. If None, assumed zero.
        (Usually zero for trusses unless you model prestrain/temperature/etc.)
    disp_in_mm : bool
        If True, print translations in mm.
    dec : int
        Decimal places for printing.

    Prints
    ------
    - Element global displacement subvector u_e (translations)
    - Element local displacement vector u'_e
    - Element local end force vector q'_e
    - Axial force N (tension positive), computed as N = Fx_j' = -Fx_i'
    """

    idx = np.asarray(m_1based, dtype=int) - 1
    u_e = u_global[idx]  # [uix, uiy, ujx, ujy]

    if Qf_local is None:
        Qf_local = np.zeros(4, dtype=float)

    u_loc = T @ u_e
    q_loc = k_local @ u_loc + Qf_local  # [Fx_i', Fy_i', Fx_j', Fy_j']

    # scale translations for printing
    scale = 1000 if disp_in_mm else 1
    unit = "mm" if disp_in_mm else "m"
    u_out = u_e * scale
    uloc_out = u_loc * scale

    def fmt(vec):
        return "[" + ", ".join(f"{v:.{dec}f}" for v in vec) + "]"

    # axial force (tension +): for a pure truss, Fy' should be ~0
    # N_i = q_loc[0]  # Fx at i in local axis
    N_j = q_loc[2]  # Fx at j in local axis
    N = N_j  # report axial as end force at j (should equal -N_i)

    print(f"\nE{e} (Truss)")
    print(f"u_global [{unit}]: {fmt(u_out)}")
    print(f"u_local  [{unit}]: {fmt(uloc_out)}")
    print(f"q_local  [kN]: {fmt(q_loc)}")
    print(f"N (tension +) = {N:.{dec}f} kN\n")


def print_matrix_scaled(K, scale=1000, decimals=1, col_width=3):
    fmt = f"{{:{col_width}.{decimals}f}}"
    print(f"K = {scale:.0e} ×")
    for i, row in enumerate(K, start=1):
        row_scaled = row / scale
        row_str = " ".join(fmt.format(val) for val in row_scaled)
        print(f"{i:02d} | {row_str}")


def print_vector_scaled(v, name="u", scale=1, decimals=3, col_width=8):
    """
    Pretty-print a vector in scaled form on one line.

    Parameters
    ----------
    v : array-like
        Vector to print
    name : str
        Name to display
    scale : float
        Scale factor shown out front
    decimals : int
        Number of decimal places
    col_width : int
        Width of value field
    """
    import numpy as np

    v = np.asarray(v, dtype=float).reshape(-1)
    fmt = f"{{:{col_width}.{decimals}f}}"

    v_scaled = v / scale
    v_str = " ".join(fmt.format(val) for val in v_scaled)

    print(f"{name} = {scale:.0e} × [{v_str}]")


def plot_truss_deformation(nodes, elements, u_global, scale=1.0):
    """
    Plot original (black) and deformed (red) truss geometry.
    """
    plt.figure()

    for e_id, (i, j, *_) in elements.items():
        xi, yi = nodes[i]
        xj, yj = nodes[j]

        ui = u_global[2 * (i - 1) : 2 * (i - 1) + 2]
        uj = u_global[2 * (j - 1) : 2 * (j - 1) + 2]

        # original
        plt.plot([xi, xj], [yi, yj], "k-", lw=2)

        # deformed
        plt.plot(
            [xi + scale * ui[0], xj + scale * uj[0]],
            [yi + scale * ui[1], yj + scale * uj[1]],
            "r-",
            lw=2,
        )

    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Original (black) and deformed (red), scale={scale}")
    plt.show()
