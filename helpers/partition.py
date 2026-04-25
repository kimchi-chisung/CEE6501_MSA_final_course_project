import numpy as np

# partition_system


def partition_system(K, f, u, f_fef, dof_restrained_1based):
    """
    Partition the global system into free and restrained DOFs.

    Parameters
    ----------
    K : (ndof, ndof) ndarray
        Global stiffness matrix.
    f : (ndof,) ndarray
        Global applied load vector.
    u : (ndof,) ndarray
        Global displacement vector containing prescribed displacements
        at restrained DOFs and zeros elsewhere.
    f_fef : (ndof,) ndarray
        Global fixed-end-force vector.
    dof_restrained_1based : array-like
        Restrained DOFs in 1-based indexing.

    Returns
    -------
    K_ff, K_fr, K_rf, K_rr : ndarray
        Partitioned stiffness submatrices.
    f_f, f_r : ndarray
        Partitioned applied load subvectors.
    u_r : ndarray
        Prescribed displacement subvector at restrained DOFs.
    f_fef_f, f_fef_r : ndarray
        Partitioned fixed-end-force subvectors.
    free_dofs, restrained_dofs : list[int]
        Free and restrained DOFs in 0-based indexing.
    """
    K = np.asarray(K, dtype=float)
    f = np.asarray(f, dtype=float).reshape(-1)
    u = np.asarray(u, dtype=float).reshape(-1)
    f_fef = np.asarray(f_fef, dtype=float).reshape(-1)

    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square 2D array.")

    ndof = K.shape[0]

    if f.shape[0] != ndof:
        raise ValueError("Length of f must match the size of K.")
    if u.shape[0] != ndof:
        raise ValueError("Length of u must match the size of K.")
    if f_fef.shape[0] != ndof:
        raise ValueError("Length of f_fef must match the size of K.")

    restrained_dofs = sorted(set(int(d) - 1 for d in dof_restrained_1based))

    for d in restrained_dofs:
        if d < 0 or d >= ndof:
            raise IndexError(f"Restrained DOF {d+1} is out of bounds for ndof={ndof}.")

    free_dofs = [i for i in range(ndof) if i not in restrained_dofs]

    if len(free_dofs) == 0:
        raise ValueError("No free DOFs remain after applying restraints.")

    K_ff = K[np.ix_(free_dofs, free_dofs)]
    K_fr = K[np.ix_(free_dofs, restrained_dofs)]
    K_rf = K[np.ix_(restrained_dofs, free_dofs)]
    K_rr = K[np.ix_(restrained_dofs, restrained_dofs)]

    f_f = f[free_dofs]
    f_r = f[restrained_dofs]

    u_r = u[restrained_dofs]

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
