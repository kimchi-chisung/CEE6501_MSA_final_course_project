import numpy as np

# get_release_by_eid
# apply_frame_releases_local
# add_zero_stiffness_dofs_to_restrained


def get_release_by_eid(release_dict, eid, default=None):
    if default is None:
        default = []

    if eid in release_dict:
        return release_dict[eid]
    if str(eid) in release_dict:
        return release_dict[str(eid)]

    return default


def apply_frame_releases_local(k_local, Qf_local, releases):
    """
    Apply local end releases by static condensation.

    Parameters
    ----------
    k_local : ndarray
        Local stiffness matrix
        - (6,6) for 2D frame
        - (12,12) for 3D frame
    Qf_local : ndarray
        Local fixed-end-force vector
    releases : list[str]
        Explicit local rotational releases only.

        2D examples:
            ["i_rz"], ["j_rz"], ["i_rz", "j_rz"]

        3D examples:
            ["i_rx"], ["i_ry"], ["i_rz"], ["j_rx"], ["j_ry"], ["j_rz"]

    Returns
    -------
    k_mod, qf_mod : ndarray
    """
    k = np.asarray(k_local, dtype=float)
    qf = np.asarray(Qf_local, dtype=float).reshape(-1)

    if k.shape[0] != k.shape[1]:
        raise ValueError("k_local must be square.")
    if qf.shape[0] != k.shape[0]:
        raise ValueError("Qf_local length must match k_local size.")

    n = k.shape[0]

    if n == 6:
        dof_map = {
            "i_rz": 2,
            "j_rz": 5,
        }

    elif n == 12:
        dof_map = {
            "i_rx": 3,
            "i_ry": 4,
            "i_rz": 5,
            "j_rx": 9,
            "j_ry": 10,
            "j_rz": 11,
        }

    else:
        raise ValueError(
            "apply_frame_releases_local supports only 6x6 or 12x12 frame matrices."
        )

    release_indices = []
    for rel in releases:
        if rel not in dof_map:
            raise ValueError(f"Unknown release key: {rel}")
        release_indices.append(dof_map[rel])

    seen = set()
    release_indices = [r for r in release_indices if not (r in seen or seen.add(r))]

    if len(release_indices) == 0:
        return k.copy(), qf.copy()

    active = list(range(n))
    K_red = k.copy()
    q_red = qf.copy()

    for r_global in release_indices:
        if r_global not in active:
            continue

        p = active.index(r_global)
        u_pos = [idx for idx in range(len(active)) if idx != p]

        k_uu = K_red[np.ix_(u_pos, u_pos)]
        k_ur = K_red[np.ix_(u_pos, [p])]
        k_ru = K_red[np.ix_([p], u_pos)]
        k_rr = K_red[p, p]

        q_u = q_red[u_pos]
        q_r = q_red[p]

        if abs(k_rr) < 1e-12:
            raise ValueError(
                f"Released DOF already has near-zero diagonal stiffness at local index {r_global}."
            )

        K_red = k_uu - (k_ur @ k_ru) / k_rr
        q_red = q_u - (k_ur.flatten() * q_r) / k_rr

        active.pop(p)

    k_mod = np.zeros((n, n), dtype=float)
    qf_mod = np.zeros(n, dtype=float)

    k_mod[np.ix_(active, active)] = K_red
    qf_mod[active] = q_red

    return k_mod, qf_mod


def add_zero_stiffness_dofs_to_restrained(K_global, dof_restrained_1based, tol=1e-12):
    K = np.asarray(K_global, dtype=float)

    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K_global must be a square matrix.")

    ndof = K.shape[0]
    zero_dofs = []

    for i in range(ndof):
        row_zero = np.linalg.norm(K[i, :], ord=np.inf) < tol
        col_zero = np.linalg.norm(K[:, i], ord=np.inf) < tol
        if row_zero and col_zero:
            zero_dofs.append(i + 1)

    base = set(int(d) for d in np.atleast_1d(dof_restrained_1based))
    fict = set(zero_dofs)

    augmented = np.array(sorted(base.union(fict)), dtype=int)
    fictitious = np.array(sorted(fict), dtype=int)

    return augmented, fictitious
