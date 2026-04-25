import numpy as np
from .elements import element_local_axes_3d

# Qf_temp_local
# Qf_fab_axial_local
# Qf_dist_axial_local
# Qf_dist_local_y
# Qf_dist_local_z
# Qf_dist_local_y_linear
# Qf_dist_local_z_linear
# Qf_point_local_x
# Qf_point_local_y
# Qf_point_local_z
# Qf_total_local

# global_vector_to_local
# global_uniform_load_to_local
# global_point_load_to_local
# global_linear_load_to_local

# _get_by_eid
# build_Qf_local_from_loads


def Qf_temp_local(
    E,
    alpha,
    A,
    Iy,
    Iz,
    T_avg=0.0,
    dTy=None,
    dy=None,
    dTz=None,
    dz=None,
):
    """
    Local thermal fixed-end-force vector for a 3D frame element.

    DOF order:
    [Fx_i, Fy_i, Fz_i, Mx_i, My_i, Mz_i,
     Fx_j, Fy_j, Fz_j, Mx_j, My_j, Mz_j]

    Convention
    ----------
    T_avg : average temperature change over the section
    dTy   : (T_bottom - T_top) in local y-direction
    dy    : distance between bottom and top points measured along local y
    dTz   : (T_bottom - T_top) in local z-direction
    dz    : distance between bottom and top points measured along local z

    This matches the sign convention in your shown formula:
        E*alpha*I*(Tb - Tt)/d
    """
    q = np.zeros(12, dtype=float)

    # axial term from average temperature
    N = E * alpha * A * T_avg
    q[0] += N
    q[6] -= N

    # bending about local y from temperature gradient through local z
    if (dTz is not None) and (dz is not None):
        My = E * alpha * Iy * (dTz / dz)
        q[4] += My
        q[10] -= My

    # bending about local z from temperature gradient through local y
    if (dTy is not None) and (dy is not None):
        Mz = E * alpha * Iz * (dTy / dy)
        q[5] += Mz
        q[11] -= Mz

    return q


def Qf_fab_axial_local(E, A, L, e_a):
    q = np.zeros(12, dtype=float)
    N = (E * A / L) * e_a
    q[0] = N
    q[6] = -N
    return q


def Qf_dist_axial_local(wx, L):
    q = np.zeros(12, dtype=float)
    q[0] = wx * L / 2.0
    q[6] = wx * L / 2.0
    return q


def Qf_dist_local_y(wy, L):
    q = np.zeros(12, dtype=float)
    q[1] = wy * L / 2.0
    q[5] = wy * L**2 / 12.0
    q[7] = wy * L / 2.0
    q[11] = -wy * L**2 / 12.0
    return q


def Qf_dist_local_z(wz, L):
    q = np.zeros(12, dtype=float)
    q[2] = wz * L / 2.0
    q[4] = -wz * L**2 / 12.0
    q[8] = wz * L / 2.0
    q[10] = wz * L**2 / 12.0
    return q


def Qf_dist_local_y_linear(w1, w2, L):
    q = np.zeros(12, dtype=float)
    q[1] = L * (7.0 * w1 + 3.0 * w2) / 20.0
    q[5] = L**2 * (3.0 * w1 + 2.0 * w2) / 60.0
    q[7] = L * (3.0 * w1 + 7.0 * w2) / 20.0
    q[11] = -(L**2) * (2.0 * w1 + 3.0 * w2) / 60.0
    return q


def Qf_dist_local_z_linear(w1, w2, L):
    q = np.zeros(12, dtype=float)
    q[2] = L * (7.0 * w1 + 3.0 * w2) / 20.0
    q[4] = -(L**2) * (3.0 * w1 + 2.0 * w2) / 60.0
    q[8] = L * (3.0 * w1 + 7.0 * w2) / 20.0
    q[10] = L**2 * (2.0 * w1 + 3.0 * w2) / 60.0
    return q


def Qf_point_local_x(P, a, L):
    q = np.zeros(12, dtype=float)
    b = L - a
    q[0] = P * b / L
    q[6] = P * a / L
    return q


def Qf_point_local_y(P, a, L):
    q = np.zeros(12, dtype=float)
    b = L - a

    q[1] = P * b**2 * (3.0 * a + b) / L**3
    q[5] = P * a * b**2 / L**2
    q[7] = P * a**2 * (a + 3.0 * b) / L**3
    q[11] = -P * a**2 * b / L**2

    return q


def Qf_point_local_z(P, a, L):
    q = np.zeros(12, dtype=float)
    b = L - a

    q[2] = P * b**2 * (3.0 * a + b) / L**3
    q[4] = -P * a * b**2 / L**2
    q[8] = P * a**2 * (a + 3.0 * b) / L**3
    q[10] = P * a**2 * b / L**2

    return q


def Qf_total_local(
    L,
    E=None,
    A=None,
    Iy=None,
    Iz=None,
    alpha=None,
    T_avg=None,
    dTy=None,
    dy=None,
    dTz=None,
    dz=None,
    e_a=None,
    wx=None,
    wy=None,
    wz=None,
    wy1=None,
    wy2=None,
    wz1=None,
    wz2=None,
    Px=None,
    Py=None,
    Pz=None,
    aP=None,
):
    q = np.zeros(12, dtype=float)

    if wx is not None:
        q += Qf_dist_axial_local(wx, L)

    if wy is not None:
        q += Qf_dist_local_y(wy, L)

    if wz is not None:
        q += Qf_dist_local_z(wz, L)

    if (wy1 is not None) and (wy2 is not None):
        q += Qf_dist_local_y_linear(wy1, wy2, L)

    if (wz1 is not None) and (wz2 is not None):
        q += Qf_dist_local_z_linear(wz1, wz2, L)

    if (Px is not None) and (aP is not None):
        q += Qf_point_local_x(Px, aP, L)

    if (Py is not None) and (aP is not None):
        q += Qf_point_local_y(Py, aP, L)

    if (Pz is not None) and (aP is not None):
        q += Qf_point_local_z(Pz, aP, L)

    if None not in (E, alpha, A, Iy, Iz):
        q += Qf_temp_local(
            E=E,
            alpha=alpha,
            A=A,
            Iy=Iy,
            Iz=Iz,
            T_avg=T_avg if T_avg is not None else 0.0,
            dTy=dTy,
            dy=dy,
            dTz=dTz,
            dz=dz,
        )

    if None not in (E, A, e_a):
        q += Qf_fab_axial_local(E, A, L, e_a)

    return q


def global_vector_to_local(vec_global, l, m, n, v_ref=None):
    """
    Convert a 3D global vector to local element components.

    Parameters
    ----------
    vec_global : array-like, shape (3,)
        Vector in global coordinates.
    l, m, n : float
        Direction cosines of local x-axis.
    v_ref : array-like or None
        Reference vector used to define local y/z axes.

    Returns
    -------
    vec_local : (3,) ndarray
        Vector components in local coordinates.
    """
    R = element_local_axes_3d(l, m, n, v_ref=v_ref)
    vec_global = np.asarray(vec_global, dtype=float).reshape(3)
    return R @ vec_global


def global_uniform_load_to_local(wx, wy, wz, l, m, n, v_ref=None):
    """
    Convert a global uniform distributed load vector to local components.

    Input wx, wy, wz are GLOBAL components.
    """
    w_global = np.array([wx, wy, wz], dtype=float)
    return global_vector_to_local(w_global, l, m, n, v_ref=v_ref)


def global_point_load_to_local(Px, Py, Pz, l, m, n, v_ref=None):
    """
    Convert a global point load vector to local components.

    Input Px, Py, Pz are GLOBAL components.
    """
    P_global = np.array([Px, Py, Pz], dtype=float)
    return global_vector_to_local(P_global, l, m, n, v_ref=v_ref)


def global_linear_load_to_local(wi_global, wj_global, l, m, n, v_ref=None):
    """
    Convert end-varying distributed load vectors at i/j ends
    from global to local coordinates.

    Parameters
    ----------
    wi_global, wj_global : array-like, shape (3,)
        Global distributed load vectors at the i-end and j-end.

    Returns
    -------
    wi_local, wj_local : (3,) ndarray
        Local components at i-end and j-end.
    """
    wi_local = global_vector_to_local(wi_global, l, m, n, v_ref=v_ref)
    wj_local = global_vector_to_local(wj_global, l, m, n, v_ref=v_ref)
    return wi_local, wj_local


def _get_by_eid(data_dict, eid, default=None):
    if eid in data_dict:
        return data_dict[eid]
    if str(eid) in data_dict:
        return data_dict[str(eid)]
    return default


def build_Qf_local_from_loads(
    elem,
    eid,
    L,
    l,
    m,
    n,
    member_loads,
    temperature_loads,
    fabrication_errors,
    v_ref=None,
):
    if elem["type"] != "3D_frame":
        return np.zeros(12, dtype=float)

    mload = _get_by_eid(member_loads, eid, {})
    tload = _get_by_eid(temperature_loads, eid, {})
    ferr = _get_by_eid(fabrication_errors, eid, {})

    Iy = elem["I"]
    Iz = elem["I"]

    q_mech = np.zeros(12, dtype=float)
    q_other = np.zeros(12, dtype=float)

    # --------------------------------------------------
    # 1) Uniform distributed load (GLOBAL -> LOCAL)
    # --------------------------------------------------
    wx_g = mload.get("wx", 0.0)
    wy_g = mload.get("wy", 0.0)
    wz_g = mload.get("wz", 0.0)

    if any(abs(v) > 0.0 for v in [wx_g, wy_g, wz_g]):
        wx_l, wy_l, wz_l = global_uniform_load_to_local(
            wx_g, wy_g, wz_g, l, m, n, v_ref=v_ref
        )
        q_mech += Qf_dist_axial_local(wx_l, L)
        q_mech += Qf_dist_local_y(wy_l, L)
        q_mech += Qf_dist_local_z(wz_l, L)

    # --------------------------------------------------
    # 2) Linearly varying distributed load (GLOBAL -> LOCAL)
    # --------------------------------------------------
    has_linear = any(key in mload for key in ["wx1", "wy1", "wz1", "wx2", "wy2", "wz2"])

    if has_linear:
        wi_global = np.array(
            [mload.get("wx1", 0.0), mload.get("wy1", 0.0), mload.get("wz1", 0.0)],
            dtype=float,
        )

        wj_global = np.array(
            [mload.get("wx2", 0.0), mload.get("wy2", 0.0), mload.get("wz2", 0.0)],
            dtype=float,
        )

        wi_local, wj_local = global_linear_load_to_local(
            wi_global, wj_global, l, m, n, v_ref=v_ref
        )

        q_mech += Qf_dist_local_y_linear(wi_local[1], wj_local[1], L)
        q_mech += Qf_dist_local_z_linear(wi_local[2], wj_local[2], L)

    # --------------------------------------------------
    # 3) Point load (GLOBAL -> LOCAL)
    # --------------------------------------------------
    Px_g = mload.get("Px", 0.0)
    Py_g = mload.get("Py", 0.0)
    Pz_g = mload.get("Pz", 0.0)
    aP = mload.get("aP", None)

    if (aP is not None) and any(abs(v) > 0.0 for v in [Px_g, Py_g, Pz_g]):
        Px_l, Py_l, Pz_l = global_point_load_to_local(
            Px_g, Py_g, Pz_g, l, m, n, v_ref=v_ref
        )
        q_mech += Qf_point_local_x(Px_l, aP, L)
        q_mech += Qf_point_local_y(Py_l, aP, L)
        q_mech += Qf_point_local_z(Pz_l, aP, L)

    # --------------------------------------------------
    # 4) Temperature load
    # --------------------------------------------------
    if "alpha" in tload:
        q_other += Qf_temp_local(
            E=elem["E"],
            alpha=tload.get("alpha", 0.0),
            A=elem["A"],
            Iy=Iy,
            Iz=Iz,
            T_avg=tload.get("T_avg", 0.0),
            dTy=tload.get("dTy", None),
            dy=tload.get("dy", None),
            dTz=tload.get("dTz", None),
            dz=tload.get("dz", None),
        )

    # --------------------------------------------------
    # 5) Fabrication error
    # --------------------------------------------------
    if "e_a" in ferr:
        q_other += Qf_fab_axial_local(
            E=elem["E"],
            A=elem["A"],
            L=L,
            e_a=ferr.get("e_a", 0.0),
        )

    # member loads only: sign flip
    return -q_mech + q_other
