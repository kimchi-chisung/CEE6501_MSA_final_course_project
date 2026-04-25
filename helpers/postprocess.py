import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from matplotlib import cm, colors
from matplotlib.lines import Line2D
from pathlib import Path

from .preprocess import element_csL
from .releases import get_release_by_eid

# print_dsm_results
# print_element
# display_compact
# print_matrix_scaled
# print_vector_scaled
# build_truss_results_dataframe_3d
# build_frame_results_dataframe_3d
# get_node_translation_3d
# get_truss_element_global_displacements_3d
# get_frame_element_global_displacements_3d
# _set_axes_equal_3d
# _get_element_xyz
# _get_node_translation
# _frame_axial_force_from_q_local
# _frame_moment_value_from_q_local
# _frame_shear_value_from_q_local
# _get_scalar_values

# plot_model_3d_plotly
# _plot_colored_structure
# plot_model_undeformed_by_type
# plot_model_undeformed_deformed
# plot_model_axial_force
# plot_model_shear_force
# plot_model_bending_moment
# plot_model_scalar_2d
# plot_model_geometry_2d
# plot_model_deformed_2d
# plot_model_axial_force_2d
# plot_model_shear_force_2d
# plot_model_bending_moment_2d
# split_elements_by_y_plane
# _get_plane_label
# plot_model_geometry_2d_two_planes
# plot_model_deformed_2d_two_planes
# plot_model_axial_force_2d_two_planes
# plot_model_shear_force_2d_two_planes
# plot_model_bending_moment_2d_two_planes
# plot_results

# build_displacement_summary
# build_reaction_summary
# build_member_force_summary
# build_equilibrium_summary
# build_release_summary
# build_result_summary
# build_result_tables


# ============================================================
# Text / table utilities
# ============================================================


def print_dsm_results(
    u_global,
    f_global_complete,
    dof_restrained_1based,
    dof_fictitious_1based=None,
    member_type="frame",
    disp_in_mm=False,
    dec=4,
    rad_dec=6,
):
    ndof = len(u_global)
    rows = []

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

    u_out = u.copy()
    v_out = v.copy()

    for j in [0, 1, 3, 4]:
        u_out[j] *= scale
        v_out[j] *= scale

    def fmt_disp(vec):
        parts = []
        for j, val in enumerate(vec):
            if j % 3 == 2:
                parts.append(f"{val:.{rad_dec}f}")
            else:
                parts.append(f"{val:.{dec}f}")
        return "[" + ", ".join(parts) + "]"

    def fmt_force(vec):
        return "[" + ", ".join(f"{val:.{dec}f}" for val in vec) + "]"

    print(f"\nE{e}")
    print(f"u [{unit},rad]: {fmt_disp(u_out)}")
    print(f"v [{unit},rad]: {fmt_disp(v_out)}")
    print(f"q [kN,kN·m]: {fmt_force(q)}")


def display_compact(df, decimals=4):
    return (
        df.style.format(precision=decimals)
        .set_properties(
            **{
                "font-size": "9pt",
                "padding": "2px",
                "white-space": "nowrap",
            }
        )
        .set_table_styles(
            [
                {"selector": "th", "props": [("font-size", "9pt")]},
            ]
        )
    )


def print_matrix_scaled(K, scale=1.0, decimals=3, col_width=10):
    K = np.asarray(K, dtype=float)
    fmt = f"{{:{col_width}.{decimals}f}}"

    print(f"K = {scale:.0e} ×")
    for i, row in enumerate(K, start=1):
        row_scaled = row / scale
        row_str = " ".join(fmt.format(val) for val in row_scaled)
        print(f"{i:2d} | {row_str}")


def print_vector_scaled(v, name="v", scale=1.0, decimals=6, col_width=12):
    v = np.asarray(v, dtype=float).reshape(-1)
    fmt = f"{{:{col_width}.{decimals}f}}"

    v_scaled = v / scale
    v_str = " ".join(fmt.format(val) for val in v_scaled)

    print(f"{name} = {scale:.0e} × [{v_str}]")


# ============================================================
# Result tables
# ============================================================


def build_truss_results_dataframe_3d(
    elements, element_lengths, results, length_unit="m"
):
    rows = []

    for e_id in sorted(
        elements.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)
    ):
        i, j = elements[e_id]["nodes"][0], elements[e_id]["nodes"][1]
        L = element_lengths[e_id]
        r = results[e_id]

        row = {
            "ele": e_id,
            "i": i,
            "j": j,
            f"L ({length_unit})": L,
            "N": r["N"],
            "sigma": r["sigma"],
        }

        row.update({f"u_{k+1}": r["u_global"][k] for k in range(6)})
        row.update({f"u_{k+1}'": r["u_local"][k] for k in range(6)})
        row.update({f"q_{k+1}'": r["q_local"][k] for k in range(6)})

        rows.append(row)

    return pd.DataFrame(rows)


def build_frame_results_dataframe_3d(
    elements, element_lengths, results, length_unit="m"
):
    rows = []

    for e_id in sorted(
        elements.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)
    ):
        i, j = elements[e_id]["nodes"][0], elements[e_id]["nodes"][1]
        L = element_lengths[e_id]
        r = results[e_id]

        row = {
            "ele": e_id,
            "i": i,
            "j": j,
            f"L ({length_unit})": L,
        }

        row.update({f"u_{k+1}": r["u_global"][k] for k in range(12)})
        row.update({f"u_{k+1}'": r["u_local"][k] for k in range(12)})
        row.update({f"q_{k+1}'": r["q_local"][k] for k in range(12)})

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Basic helpers
# ============================================================


def get_node_translation_3d(u_global, node_id):
    start = 6 * (node_id - 1)
    return np.asarray(u_global[start : start + 3], dtype=float)


def get_truss_element_global_displacements_3d(u_global, i_node, j_node):
    ui = get_node_translation_3d(u_global, i_node)
    uj = get_node_translation_3d(u_global, j_node)
    return np.hstack([ui, uj])


def get_frame_element_global_displacements_3d(u_global, i_node, j_node):
    si = 6 * (i_node - 1)
    sj = 6 * (j_node - 1)
    return np.hstack([u_global[si : si + 6], u_global[sj : sj + 6]])


def _set_axes_equal_3d(ax, xyz_points, pad_ratio=0.08):
    xyz = np.asarray(xyz_points, dtype=float)
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = 0.5 * (mins + maxs)
    spans = maxs - mins
    max_span = np.max(spans)

    if max_span == 0:
        max_span = 1.0

    half = 0.5 * max_span * (1.0 + pad_ratio)

    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))


def _get_element_xyz(nodes, elem):
    i, j = elem["nodes"]
    xi, yi, zi = nodes[str(i)] if str(i) in nodes else nodes[i]
    xj, yj, zj = nodes[str(j)] if str(j) in nodes else nodes[j]
    return i, j, xi, yi, zi, xj, yj, zj


def _get_node_translation(u_global, node_id):
    s = 6 * (node_id - 1)
    return np.asarray(u_global[s : s + 3], dtype=float)


def _frame_axial_force_from_q_local(q_l):
    return 0.5 * (q_l[0] - q_l[6])


def _frame_moment_value_from_q_local(q_l):
    candidates = [q_l[4], q_l[5], q_l[10], q_l[11]]
    return candidates[np.argmax(np.abs(candidates))]


def _frame_shear_value_from_q_local(q_l):
    candidates = [q_l[1], q_l[2], q_l[7], q_l[8]]
    return candidates[np.argmax(np.abs(candidates))]


def _get_scalar_values(elements, results_truss=None, results_frame=None, mode="axial"):
    values = {}

    for eid, elem in elements.items():
        etype = elem.get("type")

        if etype in ["3D_truss", "3D_cable"]:
            if results_truss is not None and eid in results_truss:
                if mode == "axial":
                    values[eid] = results_truss[eid].get("N", 0.0)
                elif mode in ("moment", "shear"):
                    values[eid] = 0.0
                else:
                    raise ValueError(f"Unknown mode: {mode}")
            else:
                values[eid] = 0.0

        elif etype == "3D_frame":
            if results_frame is not None and eid in results_frame:
                q_l = results_frame[eid].get("q_local", None)

                if mode == "axial":
                    values[eid] = (
                        _frame_axial_force_from_q_local(q_l)
                        if q_l is not None
                        else results_frame[eid].get("N_axial", 0.0)
                    )
                elif mode == "moment":
                    values[eid] = (
                        _frame_moment_value_from_q_local(q_l)
                        if q_l is not None
                        else results_frame[eid].get("Mmax", 0.0)
                    )
                elif mode == "shear":
                    values[eid] = (
                        _frame_shear_value_from_q_local(q_l)
                        if q_l is not None
                        else results_frame[eid].get("Vmax", 0.0)
                    )
                else:
                    raise ValueError(f"Unknown mode: {mode}")
            else:
                values[eid] = 0.0

        else:
            values[eid] = 0.0

    return values


# ============================================================
# Plotly 3D model
# ============================================================


def plot_model_3d_plotly(
    nodes,
    elements,
    supports=None,
    nodal_loads=None,
    show_node_ids=True,
    show_member_ids=False,
    load_scale=0.02,
    title="Plotly 3D Model Visualization",
):
    def get_element_nodes(e_data):
        if isinstance(e_data, dict):
            return e_data["nodes"][0], e_data["nodes"][1]
        return e_data[0], e_data[1]

    nodes3 = {}
    for nid, xyz in nodes.items():
        if len(xyz) == 2:
            x, y = xyz
            z = 0.0
        else:
            x, y, z = xyz
        nodes3[nid] = (float(x), float(y), float(z))

    fig = go.Figure()

    xs, ys, zs = [], [], []
    for eid, e_data in elements.items():
        i, j = get_element_nodes(e_data)
        xi, yi, zi = nodes3[str(i)] if str(i) in nodes3 else nodes3[i]
        xj, yj, zj = nodes3[str(j)] if str(j) in nodes3 else nodes3[j]

        xs += [xi, xj, None]
        ys += [yi, yj, None]
        zs += [zi, zj, None]

    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(width=4, color="black"),
            name="Members",
        )
    )

    if show_member_ids:
        mx, my, mz, mt = [], [], [], []
        for eid, e_data in elements.items():
            i, j = get_element_nodes(e_data)
            xi, yi, zi = nodes3[str(i)] if str(i) in nodes3 else nodes3[i]
            xj, yj, zj = nodes3[str(j)] if str(j) in nodes3 else nodes3[j]

            mx.append((xi + xj) / 2.0)
            my.append((yi + yj) / 2.0)
            mz.append((zi + zj) / 2.0)
            mt.append(str(eid))

        fig.add_trace(
            go.Scatter3d(
                x=mx,
                y=my,
                z=mz,
                mode="text",
                text=mt,
                textposition="top center",
                name="Member IDs",
                showlegend=False,
            )
        )

    node_ids = list(nodes3.keys())
    nx = [nodes3[nid][0] for nid in node_ids]
    ny = [nodes3[nid][1] for nid in node_ids]
    nz = [nodes3[nid][2] for nid in node_ids]

    fig.add_trace(
        go.Scatter3d(
            x=nx,
            y=ny,
            z=nz,
            mode="markers",
            marker=dict(size=5, color="blue"),
            name="Nodes",
        )
    )

    if show_node_ids:
        fig.add_trace(
            go.Scatter3d(
                x=nx,
                y=ny,
                z=nz,
                mode="text",
                text=[str(nid) for nid in node_ids],
                textposition="top center",
                name="Node IDs",
                showlegend=False,
            )
        )

    if supports:
        sx, sy, sz = [], [], []
        for nid in supports.keys():
            x, y, z = nodes3[str(nid)] if str(nid) in nodes3 else nodes3[nid]
            sx.append(x)
            sy.append(y)
            sz.append(z)

        fig.add_trace(
            go.Scatter3d(
                x=sx,
                y=sy,
                z=sz,
                mode="markers",
                marker=dict(size=7, color="red", symbol="diamond"),
                name="Supports",
            )
        )

    if nodal_loads:
        lx, ly, lz = [], [], []
        u, v, w = [], [], []

        for nid, load in nodal_loads.items():
            Fx, Fy, Fz = load[:3]
            x, y, z = nodes3[str(nid)] if str(nid) in nodes3 else nodes3[nid]

            lx.append(x)
            ly.append(y)
            lz.append(z)

            u.append(load_scale * Fx)
            v.append(load_scale * Fy)
            w.append(load_scale * Fz)

        fig.add_trace(
            go.Cone(
                x=lx,
                y=ly,
                z=lz,
                u=u,
                v=v,
                w=w,
                anchor="tail",
                sizemode="absolute",
                sizeref=0.2,
                showscale=False,
                name="Loads",
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.9)),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=True,
    )

    return fig


# ============================================================
# Matplotlib 3D plotting
# ============================================================


def _plot_colored_structure(
    nodes,
    elements,
    values,
    plot_title,
    cbar_label,
    cmap_name="RdBu_r",
    deformed=False,
    u_global=None,
    scale=1.0,
    elev=20,
    azim=-60,
):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    vals = np.array(
        [
            values[eid]
            for eid in sorted(
                elements.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)
            )
        ],
        dtype=float,
    )
    vmax = np.max(np.abs(vals))
    if vmax == 0:
        vmax = 1.0

    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)

    all_points = []

    for eid, elem in elements.items():
        i, j, xi, yi, zi, xj, yj, zj = _get_element_xyz(nodes, elem)

        if deformed:
            if u_global is None:
                raise ValueError("u_global must be provided when deformed=True.")
            ui = _get_node_translation(u_global, i)
            uj = _get_node_translation(u_global, j)

            x = [xi + scale * ui[0], xj + scale * uj[0]]
            y = [yi + scale * ui[1], yj + scale * uj[1]]
            z = [zi + scale * ui[2], zj + scale * uj[2]]
        else:
            x = [xi, xj]
            y = [yi, yj]
            z = [zi, zj]

        all_points.extend([[x[0], y[0], z[0]], [x[1], y[1], z[1]]])

        c = cmap(norm(values[eid]))
        lw = 2 if elem["type"] in ["3D_truss", "3D_cable"] else 3
        ax.plot(x, y, z, color=c, lw=lw)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.08, shrink=0.8, fraction=0.04)
    cbar.set_label(cbar_label)

    _set_axes_equal_3d(ax, np.array(all_points), pad_ratio=0.10)
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(plot_title)

    plt.tight_layout()
    return fig, ax


def plot_model_undeformed_by_type(
    nodes,
    elements,
    truss_color="tab:blue",
    frame_color="tab:orange",
    cable_color="tab:red",
    cable_ls="--",
    elev=20,
    azim=-60,
):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    all_points = []
    truss_labeled = False
    frame_labeled = False
    cable_labeled = False

    for eid, elem in elements.items():
        i, j, xi, yi, zi, xj, yj, zj = _get_element_xyz(nodes, elem)
        all_points.extend([[xi, yi, zi], [xj, yj, zj]])

        etype = elem["type"]

        if etype == "3D_truss":
            color = truss_color
            ls = "-"
            label = "Truss" if not truss_labeled else None
            truss_labeled = True
            lw = 2
        elif etype == "3D_frame":
            color = frame_color
            ls = "-"
            label = "Frame" if not frame_labeled else None
            frame_labeled = True
            lw = 3
        elif etype == "3D_cable":
            color = cable_color
            ls = cable_ls
            label = "Cable" if not cable_labeled else None
            cable_labeled = True
            lw = 2.5
        else:
            color = "k"
            ls = "-"
            label = None
            lw = 2

        ax.plot([xi, xj], [yi, yj], [zi, zj], color=color, ls=ls, lw=lw, label=label)

    _set_axes_equal_3d(ax, np.array(all_points), pad_ratio=0.10)
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Element type")
    ax.legend()
    plt.tight_layout()
    return fig, ax


def plot_model_undeformed_deformed(
    nodes,
    elements,
    u_global,
    scale=1.0,
    elev=20,
    azim=-60,
):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    all_points = []
    labeled_und = False
    labeled_def = False

    for eid, elem in elements.items():
        i, j, xi, yi, zi, xj, yj, zj = _get_element_xyz(nodes, elem)

        ui = _get_node_translation(u_global, i)
        uj = _get_node_translation(u_global, j)

        x_def = [xi + scale * ui[0], xj + scale * uj[0]]
        y_def = [yi + scale * ui[1], yj + scale * uj[1]]
        z_def = [zi + scale * ui[2], zj + scale * uj[2]]

        all_points.extend(
            [
                [xi, yi, zi],
                [xj, yj, zj],
                [x_def[0], y_def[0], z_def[0]],
                [x_def[1], y_def[1], z_def[1]],
            ]
        )

        lw = 2 if elem["type"] in ["3D_truss", "3D_cable"] else 3

        ax.plot(
            [xi, xj],
            [yi, yj],
            [zi, zj],
            color="k",
            lw=lw,
            label="Undeformed" if not labeled_und else None,
        )
        labeled_und = True

        ax.plot(
            x_def,
            y_def,
            z_def,
            color="r",
            lw=lw,
            label=f"Deformed (scale={scale})" if not labeled_def else None,
        )
        labeled_def = True

    _set_axes_equal_3d(ax, np.array(all_points), pad_ratio=0.10)
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Undeformed / Deformed model")
    ax.legend()
    plt.tight_layout()
    return fig, ax


def plot_model_axial_force(
    nodes,
    elements,
    results_truss=None,
    results_frame=None,
    cmap_name="RdBu_r",
    elev=20,
    azim=-60,
):
    values = _get_scalar_values(
        elements,
        results_truss=results_truss,
        results_frame=results_frame,
        mode="axial",
    )

    return _plot_colored_structure(
        nodes,
        elements,
        values,
        plot_title="Axial Force Distribution",
        cbar_label="N (kN)",
        cmap_name=cmap_name,
        deformed=False,
        u_global=None,
        elev=elev,
        azim=azim,
    )


def plot_model_shear_force(
    nodes,
    elements,
    results_truss=None,
    results_frame=None,
    cmap_name="RdBu_r",
    elev=20,
    azim=-60,
):
    values = _get_scalar_values(
        elements,
        results_truss=results_truss,
        results_frame=results_frame,
        mode="shear",
    )

    return _plot_colored_structure(
        nodes,
        elements,
        values,
        plot_title="Shear Force Distribution",
        cbar_label="V (kN)",
        cmap_name=cmap_name,
        deformed=False,
        u_global=None,
        elev=elev,
        azim=azim,
    )


def plot_model_bending_moment(
    nodes,
    elements,
    results_truss=None,
    results_frame=None,
    cmap_name="RdBu_r",
    elev=20,
    azim=-60,
):
    values = _get_scalar_values(
        elements,
        results_truss=results_truss,
        results_frame=results_frame,
        mode="moment",
    )

    return _plot_colored_structure(
        nodes,
        elements,
        values,
        plot_title="Bending Moment Distribution",
        cbar_label="M (kN·m)",
        cmap_name=cmap_name,
        deformed=False,
        u_global=None,
        elev=elev,
        azim=azim,
    )


# ============================================================
# Matplotlib 2D plotting
# ============================================================


def plot_model_scalar_2d(
    nodes,
    elements,
    values,
    title=None,
    plot_title=None,
    cbar_label=None,
    cmap_name="jet",
    linewidth=2.5,
    figsize=(16, 5),
    show_nodes=False,
):
    if title is None and plot_title is None:
        title = "Scalar Diagram"
    elif title is None:
        title = plot_title

    if cbar_label is None:
        cbar_label = ""

    fig, ax = plt.subplots(figsize=figsize)

    cmap = cm.get_cmap(cmap_name)
    vmax = max(abs(v) for v in values.values()) if values else 1.0
    if vmax == 0:
        vmax = 1.0
    norm = colors.Normalize(vmin=-vmax, vmax=vmax)

    for eid, elem in elements.items():
        i, j = elem["nodes"]

        xi, yi, zi = nodes[str(i)] if str(i) in nodes else nodes[i]
        xj, yj, zj = nodes[str(j)] if str(j) in nodes else nodes[j]

        val = values.get(int(eid), values.get(str(eid), 0.0))
        color = cmap(norm(val))

        ax.plot([xi, xj], [zi, zj], lw=linewidth, color=color)

    if show_nodes:
        for nid, coord in nodes.items():
            x, y, z = coord
            ax.plot(x, z, "ko", ms=2)

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(cbar_label)

    plt.tight_layout()
    return fig, ax


def plot_model_geometry_2d(
    nodes,
    elements,
    figsize=(16, 5),
    frame_color="black",
    truss_color="tab:blue",
    cable_color="tab:red",
    frame_ls="-",
    truss_ls="-",
    cable_ls="--",
    linewidth=2.0,
    show_nodes=False,
    title="Element type",
):
    fig, ax = plt.subplots(figsize=figsize)

    for eid, elem in elements.items():
        i, j = elem["nodes"]
        xi, yi, zi = nodes[str(i)] if str(i) in nodes else nodes[i]
        xj, yj, zj = nodes[str(j)] if str(j) in nodes else nodes[j]

        etype = elem["type"]

        if etype == "3D_frame":
            color = frame_color
            ls = frame_ls
        elif etype == "3D_truss":
            color = truss_color
            ls = truss_ls
        elif etype == "3D_cable":
            color = cable_color
            ls = cable_ls
        else:
            color = "gray"
            ls = "-"

        ax.plot([xi, xj], [zi, zj], color=color, ls=ls, lw=linewidth)

    if show_nodes:
        for nid, coord in nodes.items():
            x, y, z = coord
            ax.plot(x, z, "ko", ms=2)

    legend_handles = [
        Line2D([0], [0], color=frame_color, lw=linewidth, ls=frame_ls, label="Frame"),
        Line2D([0], [0], color=truss_color, lw=linewidth, ls=truss_ls, label="Truss"),
        Line2D([0], [0], color=cable_color, lw=linewidth, ls=cable_ls, label="Cable"),
    ]

    if show_nodes:
        legend_handles.append(
            Line2D([0], [0], marker="o", color="k", lw=0, markersize=4, label="Node")
        )

    ax.legend(handles=legend_handles, loc="best")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_model_deformed_2d(
    nodes,
    elements,
    u_global,
    scale=1.0,
    figsize=(16, 5),
    undeformed_color="black",
    deformed_color="red",
    linewidth=2.0,
    title="Undeformed / Deformed model",
):
    fig, ax = plt.subplots(figsize=figsize)

    undeformed_labeled = False
    deformed_labeled = False

    for eid, elem in elements.items():
        i, j = elem["nodes"]

        xi, yi, zi = nodes[str(i)] if str(i) in nodes else nodes[i]
        xj, yj, zj = nodes[str(j)] if str(j) in nodes else nodes[j]

        si = 6 * (i - 1)
        sj = 6 * (j - 1)

        ui = u_global[si : si + 3]
        uj = u_global[sj : sj + 3]

        ax.plot(
            [xi, xj],
            [zi, zj],
            lw=linewidth,
            color=undeformed_color,
            label="Undeformed" if not undeformed_labeled else None,
        )
        undeformed_labeled = True

        ax.plot(
            [xi + scale * ui[0], xj + scale * uj[0]],
            [zi + scale * ui[2], zj + scale * uj[2]],
            lw=linewidth,
            color=deformed_color,
            label=f"Deformed (scale={scale})" if not deformed_labeled else None,
        )
        deformed_labeled = True

    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig, ax


def plot_model_axial_force_2d(
    nodes,
    elements,
    results_truss=None,
    results_frame=None,
    cmap_name="jet",
    show_nodes=False,
):
    values = _get_scalar_values(
        elements,
        results_truss=results_truss,
        results_frame=results_frame,
        mode="axial",
    )

    return plot_model_scalar_2d(
        nodes,
        elements,
        values=values,
        plot_title="Axial Force Distribution",
        cbar_label="N (kN)",
        cmap_name=cmap_name,
        show_nodes=show_nodes,
    )


def plot_model_shear_force_2d(
    nodes,
    elements,
    results_truss=None,
    results_frame=None,
    cmap_name="jet",
    show_nodes=False,
):
    values = _get_scalar_values(
        elements,
        results_truss=results_truss,
        results_frame=results_frame,
        mode="shear",
    )

    return plot_model_scalar_2d(
        nodes,
        elements,
        values=values,
        plot_title="Shear Force Distribution",
        cbar_label="V (kN)",
        cmap_name=cmap_name,
        show_nodes=show_nodes,
    )


def plot_model_bending_moment_2d(
    nodes,
    elements,
    results_truss=None,
    results_frame=None,
    cmap_name="jet",
    show_nodes=False,
):
    values = _get_scalar_values(
        elements,
        results_truss=results_truss,
        results_frame=results_frame,
        mode="moment",
    )

    return plot_model_scalar_2d(
        nodes,
        elements,
        values=values,
        plot_title="Bending Moment Distribution",
        cbar_label="M (kN·m)",
        cmap_name=cmap_name,
        show_nodes=show_nodes,
    )


# ============================================================
# Two Planes Plotting
# ============================================================


def split_elements_by_y_plane(nodes, elements, tol=1e-8):
    """
    Split elements into:
    - plane_neg: both end nodes on negative-y plane
    - plane_pos: both end nodes on positive-y plane
    - transverse: end nodes on different y planes
    """
    plane_neg = {}
    plane_pos = {}
    transverse = {}

    for eid, elem in elements.items():
        i, j = elem["nodes"]

        yi = nodes[str(i)][1] if str(i) in nodes else nodes[i][1]
        yj = nodes[str(j)][1] if str(j) in nodes else nodes[j][1]

        if abs(yi - yj) < tol:
            y_avg = 0.5 * (yi + yj)
            if y_avg < 0:
                plane_neg[eid] = elem
            else:
                plane_pos[eid] = elem
        else:
            transverse[eid] = elem

    return plane_neg, plane_pos, transverse


def _get_plane_label(plane_key):
    if plane_key == "neg":
        return "Side Elevation (Y < 0)"
    elif plane_key == "pos":
        return "Side Elevation (Y > 0)"
    else:
        return "Side Elevation"


def plot_model_geometry_2d_two_planes(
    nodes,
    elements,
    figsize=(16, 5),
    frame_color="black",
    truss_color="tab:blue",
    cable_color="tab:red",
    frame_ls="-",
    truss_ls="-",
    cable_ls="--",
    linewidth=2.0,
    show_nodes=False,
):
    plane_neg, plane_pos, transverse = split_elements_by_y_plane(nodes, elements)

    fig1, ax1 = plot_model_geometry_2d(
        nodes,
        plane_neg,
        figsize=figsize,
        frame_color=frame_color,
        truss_color=truss_color,
        cable_color=cable_color,
        frame_ls=frame_ls,
        truss_ls=truss_ls,
        cable_ls=cable_ls,
        linewidth=linewidth,
        show_nodes=show_nodes,
        title="Element type — Side Elevation (Y < 0)",
    )

    fig2, ax2 = plot_model_geometry_2d(
        nodes,
        plane_pos,
        figsize=figsize,
        frame_color=frame_color,
        truss_color=truss_color,
        cable_color=cable_color,
        frame_ls=frame_ls,
        truss_ls=truss_ls,
        cable_ls=cable_ls,
        linewidth=linewidth,
        show_nodes=show_nodes,
        title="Element type — Side Elevation (Y > 0)",
    )

    return (fig1, ax1), (fig2, ax2), transverse


def plot_model_deformed_2d_two_planes(
    nodes,
    elements,
    u_global,
    scale=1.0,
    figsize=(16, 5),
    undeformed_color="black",
    deformed_color="red",
    linewidth=2.0,
):
    plane_neg, plane_pos, transverse = split_elements_by_y_plane(nodes, elements)

    fig1, ax1 = plot_model_deformed_2d(
        nodes,
        plane_neg,
        u_global,
        scale=scale,
        figsize=figsize,
        undeformed_color=undeformed_color,
        deformed_color=deformed_color,
        linewidth=linewidth,
        title=f"Undeformed / Deformed model — Side Elevation (Y < 0)",
    )

    fig2, ax2 = plot_model_deformed_2d(
        nodes,
        plane_pos,
        u_global,
        scale=scale,
        figsize=figsize,
        undeformed_color=undeformed_color,
        deformed_color=deformed_color,
        linewidth=linewidth,
        title=f"Undeformed / Deformed model — Side Elevation (Y > 0)",
    )

    return (fig1, ax1), (fig2, ax2), transverse


def plot_model_axial_force_2d_two_planes(
    nodes,
    elements,
    results_truss=None,
    results_frame=None,
    cmap_name="plasma",
    show_nodes=False,
    figsize=(16, 5),
):
    plane_neg, plane_pos, transverse = split_elements_by_y_plane(nodes, elements)

    values = _get_scalar_values(
        elements,
        results_truss=results_truss,
        results_frame=results_frame,
        mode="axial",
    )

    values_neg = {eid: values[eid] for eid in plane_neg.keys()}
    values_pos = {eid: values[eid] for eid in plane_pos.keys()}

    fig1, ax1 = plot_model_scalar_2d(
        nodes,
        plane_neg,
        values=values_neg,
        title="Axial Force Distribution (N) — Side Elevation (Y < 0)",
        cbar_label="N (kN)",
        cmap_name=cmap_name,
        figsize=figsize,
        show_nodes=show_nodes,
    )

    fig2, ax2 = plot_model_scalar_2d(
        nodes,
        plane_pos,
        values=values_pos,
        title="Axial Force Distribution (N) — Side Elevation (Y > 0)",
        cbar_label="N (kN)",
        cmap_name=cmap_name,
        figsize=figsize,
        show_nodes=show_nodes,
    )

    return (fig1, ax1), (fig2, ax2), transverse


def plot_model_shear_force_2d_two_planes(
    nodes,
    elements,
    results_truss=None,
    results_frame=None,
    cmap_name="plasma",
    show_nodes=False,
    figsize=(16, 5),
):
    plane_neg, plane_pos, transverse = split_elements_by_y_plane(nodes, elements)

    values = _get_scalar_values(
        elements,
        results_truss=results_truss,
        results_frame=results_frame,
        mode="shear",
    )

    values_neg = {eid: values[eid] for eid in plane_neg.keys()}
    values_pos = {eid: values[eid] for eid in plane_pos.keys()}

    fig1, ax1 = plot_model_scalar_2d(
        nodes,
        plane_neg,
        values=values_neg,
        title="Shear Force Distribution (V) — Side Elevation (Y < 0)",
        cbar_label="V (kN)",
        cmap_name=cmap_name,
        figsize=figsize,
        show_nodes=show_nodes,
    )

    fig2, ax2 = plot_model_scalar_2d(
        nodes,
        plane_pos,
        values=values_pos,
        title="Shear Force Distribution (V) — Side Elevation (Y > 0)",
        cbar_label="V (kN)",
        cmap_name=cmap_name,
        figsize=figsize,
        show_nodes=show_nodes,
    )

    return (fig1, ax1), (fig2, ax2), transverse


def plot_model_bending_moment_2d_two_planes(
    nodes,
    elements,
    results_truss=None,
    results_frame=None,
    cmap_name="plasma",
    show_nodes=False,
    figsize=(16, 5),
):
    plane_neg, plane_pos, transverse = split_elements_by_y_plane(nodes, elements)

    values = _get_scalar_values(
        elements,
        results_truss=results_truss,
        results_frame=results_frame,
        mode="moment",
    )

    values_neg = {eid: values[eid] for eid in plane_neg.keys()}
    values_pos = {eid: values[eid] for eid in plane_pos.keys()}

    fig1, ax1 = plot_model_scalar_2d(
        nodes,
        plane_neg,
        values=values_neg,
        title="Bending Moment Distribution (M) — Side Elevation (Y < 0)",
        cbar_label="M (kN·m)",
        cmap_name=cmap_name,
        figsize=figsize,
        show_nodes=show_nodes,
    )

    fig2, ax2 = plot_model_scalar_2d(
        nodes,
        plane_pos,
        values=values_pos,
        title="Bending Moment Distribution (M) — Side Elevation (Y > 0)",
        cbar_label="M (kN·m)",
        cmap_name=cmap_name,
        figsize=figsize,
        show_nodes=show_nodes,
    )

    return (fig1, ax1), (fig2, ax2), transverse


# ============================================================
# Batch plotting
# ============================================================


def plot_results(
    nodes,
    elements,
    u_global_complete=None,
    results_truss=None,
    results_cable=None,
    results_frame=None,
    supports=None,
    nodal_loads=None,
    cmap_name="plasma",
    scale_3d=50,
    scale_2d=100,
    load_scale_plotly=0.1,
    show_node_ids=True,
    show_member_ids=True,
    display_3d=True,
    save_3d=False,
    display_2d=True,
    save_2d=False,
    save_dir="outputs",
    save_prefix=None,
    model_path=None,
    split_by_y_plane_2d=False,
):
    results_truss = {} if results_truss is None else results_truss
    results_cable = {} if results_cable is None else results_cable
    results_frame = {} if results_frame is None else results_frame

    model_stem = Path(model_path).stem if model_path is not None else "results"

    if save_prefix is None:
        save_prefix = model_stem

    if save_3d or save_2d:
        save_path = Path(save_dir) / model_stem
        save_path.mkdir(parents=True, exist_ok=True)
    else:
        save_path = Path(save_dir) / model_stem

    def _finalize_matplotlib(fig, filename=None, display=True, save=False):
        if save and filename is not None:
            fig.savefig(save_path / filename, dpi=300, bbox_inches="tight")
        if display:
            plt.show()
        else:
            plt.close(fig)

    # ============================================================
    # 3D plots
    # ============================================================
    if display_3d or save_3d:
        if u_global_complete is None:
            raise ValueError(
                "u_global_complete is required when display_3d=True or save_3d=True"
            )

        fig, ax = plot_model_undeformed_by_type(nodes, elements)
        _finalize_matplotlib(
            fig,
            filename=f"{save_prefix}_undeformed_3d.png",
            display=display_3d,
            save=save_3d,
        )

        fig, ax = plot_model_undeformed_deformed(
            nodes,
            elements,
            u_global_complete,
            scale=scale_3d,
        )
        _finalize_matplotlib(
            fig,
            filename=f"{save_prefix}_deformed_3d.png",
            display=display_3d,
            save=save_3d,
        )

        fig, ax = plot_model_axial_force(
            nodes,
            elements,
            results_truss=results_truss,
            results_frame=results_frame,
            cmap_name=cmap_name,
        )
        _finalize_matplotlib(
            fig,
            filename=f"{save_prefix}_axial_3d.png",
            display=display_3d,
            save=save_3d,
        )

        fig, ax = plot_model_shear_force(
            nodes,
            elements,
            results_truss=results_truss,
            results_frame=results_frame,
            cmap_name=cmap_name,
        )
        _finalize_matplotlib(
            fig,
            filename=f"{save_prefix}_shear_3d.png",
            display=display_3d,
            save=save_3d,
        )

        fig, ax = plot_model_bending_moment(
            nodes,
            elements,
            results_truss=results_truss,
            results_frame=results_frame,
            cmap_name=cmap_name,
        )
        _finalize_matplotlib(
            fig,
            filename=f"{save_prefix}_bending_3d.png",
            display=display_3d,
            save=save_3d,
        )

        if supports is None or nodal_loads is None:
            raise ValueError("supports and nodal_loads are required for 3D plotly plot")

        fig = plot_model_3d_plotly(
            nodes,
            elements,
            supports=supports,
            nodal_loads=nodal_loads,
            show_node_ids=show_node_ids,
            show_member_ids=show_member_ids,
            load_scale=load_scale_plotly,
        )

        if save_3d:
            html_path = save_path / f"{save_prefix}_plotly_3d.html"
            fig.write_html(
                str(html_path),
                full_html=True,
                include_plotlyjs="cdn",
            )
            print(f"Saved Plotly HTML: {html_path}")

        if display_3d:
            fig.show()

    # ============================================================
    # 2D plots
    # ============================================================
    if display_2d or save_2d:
        if u_global_complete is None:
            raise ValueError(
                "u_global_complete is required when display_2d=True or save_2d=True"
            )

        if split_by_y_plane_2d:
            fig_pairs = []

            (fig1, ax1), (fig2, ax2), _ = plot_model_geometry_2d_two_planes(
                nodes,
                elements,
            )
            fig_pairs.append((fig1, f"{save_prefix}_geometry_yneg_2d.png"))
            fig_pairs.append((fig2, f"{save_prefix}_geometry_ypos_2d.png"))

            (fig1, ax1), (fig2, ax2), _ = plot_model_deformed_2d_two_planes(
                nodes,
                elements,
                u_global_complete,
                scale=scale_2d,
            )
            fig_pairs.append((fig1, f"{save_prefix}_deformed_yneg_2d.png"))
            fig_pairs.append((fig2, f"{save_prefix}_deformed_ypos_2d.png"))

            (fig1, ax1), (fig2, ax2), _ = plot_model_axial_force_2d_two_planes(
                nodes,
                elements,
                results_truss=results_truss,
                results_frame=results_frame,
                cmap_name=cmap_name,
            )
            fig_pairs.append((fig1, f"{save_prefix}_axial_yneg_2d.png"))
            fig_pairs.append((fig2, f"{save_prefix}_axial_ypos_2d.png"))

            (fig1, ax1), (fig2, ax2), _ = plot_model_shear_force_2d_two_planes(
                nodes,
                elements,
                results_truss=results_truss,
                results_frame=results_frame,
                cmap_name=cmap_name,
            )
            fig_pairs.append((fig1, f"{save_prefix}_shear_yneg_2d.png"))
            fig_pairs.append((fig2, f"{save_prefix}_shear_ypos_2d.png"))

            (fig1, ax1), (fig2, ax2), _ = plot_model_bending_moment_2d_two_planes(
                nodes,
                elements,
                results_truss=results_truss,
                results_frame=results_frame,
                cmap_name=cmap_name,
            )
            fig_pairs.append((fig1, f"{save_prefix}_bending_yneg_2d.png"))
            fig_pairs.append((fig2, f"{save_prefix}_bending_ypos_2d.png"))

            for fig, filename in fig_pairs:
                _finalize_matplotlib(
                    fig,
                    filename=filename,
                    display=display_2d,
                    save=save_2d,
                )

        else:
            fig, ax = plot_model_geometry_2d(nodes, elements, title="Structural Layout")
            _finalize_matplotlib(
                fig,
                filename=f"{save_prefix}_geometry_2d.png",
                display=display_2d,
                save=save_2d,
            )

            fig, ax = plot_model_deformed_2d(
                nodes,
                elements,
                u_global_complete,
                scale=scale_2d,
                title=f"Deformed Shape, Scale = {scale_2d}",
            )
            _finalize_matplotlib(
                fig,
                filename=f"{save_prefix}_deformed_2d.png",
                display=display_2d,
                save=save_2d,
            )

            fig, ax = plot_model_axial_force_2d(
                nodes,
                elements,
                results_truss=results_truss,
                results_frame=results_frame,
                cmap_name=cmap_name,
            )
            ax.set_title("Axial Force Distribution (N)")
            _finalize_matplotlib(
                fig,
                filename=f"{save_prefix}_axial_2d.png",
                display=display_2d,
                save=save_2d,
            )

            fig, ax = plot_model_shear_force_2d(
                nodes,
                elements,
                results_truss=results_truss,
                results_frame=results_frame,
                cmap_name=cmap_name,
            )
            ax.set_title("Shear Force Distribution (V)")
            _finalize_matplotlib(
                fig,
                filename=f"{save_prefix}_shear_2d.png",
                display=display_2d,
                save=save_2d,
            )

            fig, ax = plot_model_bending_moment_2d(
                nodes,
                elements,
                results_truss=results_truss,
                results_frame=results_frame,
                cmap_name=cmap_name,
            )
            ax.set_title("Bending Moment Distribution (M)")
            _finalize_matplotlib(
                fig,
                filename=f"{save_prefix}_bending_2d.png",
                display=display_2d,
                save=save_2d,
            )


# ============================================================
# Result table export
# ============================================================


def build_displacement_summary(nodes, u_global_complete):
    """
    Build node displacement summary.

    Returns
    -------
    df_node_disp : DataFrame
        Per-node displacement table with ux, uy, uz, rx, ry, rz and translation magnitude.
    df_disp_summary : DataFrame
        Summary table for maximum displacement quantities.
    """
    rows = []

    node_ids = sorted(
        nodes.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)
    )

    for nid in node_ids:
        s = 6 * (int(nid) - 1)
        ux, uy, uz, rx, ry, rz = u_global_complete[s : s + 6]
        u_trans = np.sqrt(ux**2 + uy**2 + uz**2)

        rows.append(
            {
                "node": nid,
                "ux": ux,
                "uy": uy,
                "uz": uz,
                "rx": rx,
                "ry": ry,
                "rz": rz,
                "u_trans": u_trans,
            }
        )

    df_node_disp = pd.DataFrame(rows)

    summary_rows = []

    for col in ["ux", "uy", "uz", "rx", "ry", "rz", "u_trans"]:
        idx = df_node_disp[col].abs().idxmax()
        row = df_node_disp.loc[idx]
        summary_rows.append(
            {
                "quantity": f"max |{col}|",
                "node": row["node"],
                "value": row[col],
                "abs_value": abs(row[col]),
            }
        )

    df_disp_summary = pd.DataFrame(summary_rows)

    return df_node_disp, df_disp_summary


def build_reaction_summary(supports, f_global_complete):
    """
    Build support reaction summary.

    Returns
    -------
    df_reaction_dof : DataFrame
        Reaction by restrained DOF.
    df_reaction_node : DataFrame
        Reaction resultant by support node.
    df_reaction_summary : DataFrame
        Maximum reaction summary.
    """
    dof_map = {
        "ux": 0,
        "uy": 1,
        "uz": 2,
        "rx": 3,
        "ry": 4,
        "rz": 5,
    }

    rows_dof = []
    rows_node = []

    for nid, restrained in supports.items():
        s = 6 * (int(nid) - 1)

        fx = fy = fz = mx = my = mz = 0.0

        for dof_name in restrained:
            local_idx = dof_map[dof_name]
            val = f_global_complete[s + local_idx]

            rows_dof.append(
                {
                    "node": nid,
                    "dof": dof_name,
                    "reaction": val,
                    "abs_reaction": abs(val),
                }
            )

            if dof_name == "ux":
                fx = val
            elif dof_name == "uy":
                fy = val
            elif dof_name == "uz":
                fz = val
            elif dof_name == "rx":
                mx = val
            elif dof_name == "ry":
                my = val
            elif dof_name == "rz":
                mz = val

        force_resultant = np.sqrt(fx**2 + fy**2 + fz**2)
        moment_resultant = np.sqrt(mx**2 + my**2 + mz**2)

        rows_node.append(
            {
                "node": nid,
                "Fx": fx,
                "Fy": fy,
                "Fz": fz,
                "Mx": mx,
                "My": my,
                "Mz": mz,
                "F_resultant": force_resultant,
                "M_resultant": moment_resultant,
            }
        )

    df_reaction_dof = pd.DataFrame(rows_dof)
    df_reaction_node = pd.DataFrame(rows_node)

    summary_rows = []

    if not df_reaction_dof.empty:
        idx = df_reaction_dof["abs_reaction"].idxmax()
        row = df_reaction_dof.loc[idx]
        summary_rows.append(
            {
                "quantity": "max |reaction| by DOF",
                "node": row["node"],
                "dof": row["dof"],
                "value": row["reaction"],
                "abs_value": row["abs_reaction"],
            }
        )

    if not df_reaction_node.empty:
        idx = df_reaction_node["F_resultant"].idxmax()
        row = df_reaction_node.loc[idx]
        summary_rows.append(
            {
                "quantity": "max support force resultant",
                "node": row["node"],
                "dof": "-",
                "value": row["F_resultant"],
                "abs_value": row["F_resultant"],
            }
        )

        idx = df_reaction_node["M_resultant"].idxmax()
        row = df_reaction_node.loc[idx]
        summary_rows.append(
            {
                "quantity": "max support moment resultant",
                "node": row["node"],
                "dof": "-",
                "value": row["M_resultant"],
                "abs_value": row["M_resultant"],
            }
        )

    df_reaction_summary = pd.DataFrame(summary_rows)

    return df_reaction_dof, df_reaction_node, df_reaction_summary


def build_member_force_summary(
    results_truss=None, results_cable=None, results_frame=None
):
    """
    Build member-force summary for truss/cable/frame.

    Returns
    -------
    df_member_summary : DataFrame
        Summary table of critical member forces.
    """
    results_truss = {} if results_truss is None else results_truss
    results_cable = {} if results_cable is None else results_cable
    results_frame = {} if results_frame is None else results_frame

    rows = []

    # -------------------------
    # Truss
    # -------------------------
    if len(results_truss) > 0:
        tension_candidates = []
        compression_candidates = []

        for eid, r in results_truss.items():
            N = r["N"]
            if N >= 0:
                tension_candidates.append((eid, N))
            else:
                compression_candidates.append((eid, N))

        if len(tension_candidates) > 0:
            eid, val = max(tension_candidates, key=lambda x: abs(x[1]))
            rows.append(
                {
                    "category": "truss",
                    "quantity": "max tension",
                    "element": eid,
                    "value": val,
                    "abs_value": abs(val),
                }
            )

        if len(compression_candidates) > 0:
            eid, val = max(compression_candidates, key=lambda x: abs(x[1]))
            rows.append(
                {
                    "category": "truss",
                    "quantity": "max compression",
                    "element": eid,
                    "value": val,
                    "abs_value": abs(val),
                }
            )

    # -------------------------
    # Cable
    # -------------------------
    if len(results_cable) > 0:
        tension_candidates = []
        compression_candidates = []

        for eid, r in results_cable.items():
            N = r["N"]
            if N >= 0:
                tension_candidates.append((eid, N))
            else:
                compression_candidates.append((eid, N))

        if len(tension_candidates) > 0:
            eid, val = max(tension_candidates, key=lambda x: abs(x[1]))
            rows.append(
                {
                    "category": "cable",
                    "quantity": "max tension",
                    "element": eid,
                    "value": val,
                    "abs_value": abs(val),
                }
            )

        if len(compression_candidates) > 0:
            eid, val = max(compression_candidates, key=lambda x: abs(x[1]))
            rows.append(
                {
                    "category": "cable",
                    "quantity": "max slack/compression",
                    "element": eid,
                    "value": val,
                    "abs_value": abs(val),
                }
            )

    # -------------------------
    # Frame
    # -------------------------
    if len(results_frame) > 0:
        axial_list = []
        shear_list = []
        moment_list = []

        for eid, r in results_frame.items():
            q = np.asarray(r["q_local"], dtype=float)

            axial_val = _frame_axial_force_from_q_local(q)
            shear_val = _frame_shear_value_from_q_local(q)
            moment_val = _frame_moment_value_from_q_local(q)

            axial_list.append((eid, axial_val))
            shear_list.append((eid, shear_val))
            moment_list.append((eid, moment_val))

        if len(axial_list) > 0:
            eid, val = max(axial_list, key=lambda x: abs(x[1]))
            rows.append(
                {
                    "category": "frame",
                    "quantity": "max axial N",
                    "element": eid,
                    "value": val,
                    "abs_value": abs(val),
                }
            )

        if len(shear_list) > 0:
            eid, val = max(shear_list, key=lambda x: abs(x[1]))
            rows.append(
                {
                    "category": "frame",
                    "quantity": "max shear V",
                    "element": eid,
                    "value": val,
                    "abs_value": abs(val),
                }
            )

        if len(moment_list) > 0:
            eid, val = max(moment_list, key=lambda x: abs(x[1]))
            rows.append(
                {
                    "category": "frame",
                    "quantity": "max moment M",
                    "element": eid,
                    "value": val,
                    "abs_value": abs(val),
                }
            )

    return pd.DataFrame(rows)


def build_equilibrium_summary(F_global, F_fef_global, f_global_complete):
    """
    Build basic global equilibrium check.

    Equation used:
        residual = f_global_complete + F_fef_global - F_global

    Returns
    -------
    df_equilibrium : DataFrame
        Residual by component and norm.
    """
    F_global = np.asarray(F_global, dtype=float).reshape(-1)
    F_fef_global = np.asarray(F_fef_global, dtype=float).reshape(-1)
    f_global_complete = np.asarray(f_global_complete, dtype=float).reshape(-1)

    residual = f_global_complete + F_fef_global - F_global

    # group by dof type in 6 dof per node system
    ux_res = residual[0::6].sum()
    uy_res = residual[1::6].sum()
    uz_res = residual[2::6].sum()
    rx_res = residual[3::6].sum()
    ry_res = residual[4::6].sum()
    rz_res = residual[5::6].sum()

    rows = [
        {"quantity": "sum residual Fx", "value": ux_res, "abs_value": abs(ux_res)},
        {"quantity": "sum residual Fy", "value": uy_res, "abs_value": abs(uy_res)},
        {"quantity": "sum residual Fz", "value": uz_res, "abs_value": abs(uz_res)},
        {"quantity": "sum residual Mx", "value": rx_res, "abs_value": abs(rx_res)},
        {"quantity": "sum residual My", "value": ry_res, "abs_value": abs(ry_res)},
        {"quantity": "sum residual Mz", "value": rz_res, "abs_value": abs(rz_res)},
        {
            "quantity": "residual norm (inf)",
            "value": np.linalg.norm(residual, ord=np.inf),
            "abs_value": np.linalg.norm(residual, ord=np.inf),
        },
        {
            "quantity": "residual norm (2)",
            "value": np.linalg.norm(residual, ord=2),
            "abs_value": np.linalg.norm(residual, ord=2),
        },
    ]

    return pd.DataFrame(rows)


def build_release_summary(elements, results_frame=None, releases=None):
    """
    Check whether released frame-end moments are approximately zero.

    Returns
    -------
    df_release : DataFrame
    """
    results_frame = {} if results_frame is None else results_frame
    releases = {} if releases is None else releases

    rows = []

    for eid, elem in elements.items():
        if elem["type"] != "3D_frame":
            continue

        elem_releases = get_release_by_eid(releases, eid, default=[])
        if len(elem_releases) == 0:
            continue

        if eid not in results_frame:
            continue

        q = np.asarray(results_frame[eid]["q_local"], dtype=float)

        dof_force_map = {
            "i_rx": ("Mx_i", q[3]),
            "i_ry": ("My_i", q[4]),
            "i_rz": ("Mz_i", q[5]),
            "j_rx": ("Mx_j", q[9]),
            "j_ry": ("My_j", q[10]),
            "j_rz": ("Mz_j", q[11]),
        }

        for rel in elem_releases:
            if rel not in dof_force_map:
                continue

            label, val = dof_force_map[rel]
            rows.append(
                {
                    "element": eid,
                    "release": rel,
                    "checked_force": label,
                    "value": val,
                    "abs_value": abs(val),
                }
            )

    return pd.DataFrame(rows)


def build_result_summary(
    nodes,
    elements,
    supports,
    u_global_complete,
    f_global_complete,
    F_global=None,
    F_fef_global=None,
    results_truss=None,
    results_cable=None,
    results_frame=None,
    releases=None,
):
    """
    Build a full result summary package.

    Returns
    -------
    summary_tables : dict[str, DataFrame]
    """
    results_truss = {} if results_truss is None else results_truss
    results_cable = {} if results_cable is None else results_cable
    results_frame = {} if results_frame is None else results_frame
    releases = {} if releases is None else releases

    df_node_disp, df_disp_summary = build_displacement_summary(
        nodes,
        u_global_complete,
    )

    df_reaction_dof, df_reaction_node, df_reaction_summary = build_reaction_summary(
        supports,
        f_global_complete,
    )

    df_member_summary = build_member_force_summary(
        results_truss=results_truss,
        results_cable=results_cable,
        results_frame=results_frame,
    )

    if F_global is not None and F_fef_global is not None:
        df_equilibrium = build_equilibrium_summary(
            F_global,
            F_fef_global,
            f_global_complete,
        )
    else:
        df_equilibrium = pd.DataFrame()

    df_release = build_release_summary(
        elements,
        results_frame=results_frame,
        releases=releases,
    )

    return {
        "node_displacement": df_node_disp,
        "summary_displacement": df_disp_summary,
        "reaction_dof": df_reaction_dof,
        "reaction_node": df_reaction_node,
        "summary_reaction": df_reaction_summary,
        "summary_member_force": df_member_summary,
        "summary_equilibrium": df_equilibrium,
        "summary_release": df_release,
    }


def build_result_tables(
    elements,
    element_lengths,
    results_truss=None,
    results_cable=None,
    results_frame=None,
    length_unit="m",
    display_tables=False,
    save_tables=False,
    save_dir="outputs",
    save_filename=None,
    model_path=None,
    nodes=None,
    supports=None,
    Qf_debug=None,
    u_f=None,
    F_r=None,
    u_global_complete=None,
    f_global_complete=None,
    F_global=None,
    F_fef_global=None,
    releases=None,
):
    results_truss = {} if results_truss is None else results_truss
    results_cable = {} if results_cable is None else results_cable
    results_frame = {} if results_frame is None else results_frame
    Qf_debug = {} if Qf_debug is None else Qf_debug
    supports = {} if supports is None else supports
    releases = {} if releases is None else releases

    def _get_result_item(results_dict, eid):
        if eid in results_dict:
            return results_dict[eid]
        if str(eid) in results_dict:
            return results_dict[str(eid)]
        try:
            eid_int = int(eid)
            if eid_int in results_dict:
                return results_dict[eid_int]
        except (ValueError, TypeError):
            pass
        return None

    def _vector_df(vec):
        if vec is None:
            return pd.DataFrame(columns=["index", "value"])
        vec = np.asarray(vec, dtype=float).reshape(-1)
        return pd.DataFrame(
            {
                "index": np.arange(1, len(vec) + 1),
                "value": vec,
            }
        )

    extra_tables = {}

    # ============================================================
    # Debug / extra tables
    # ============================================================
    if nodes is not None:
        rows = []
        for eid, elem in elements.items():
            i_node, j_node = elem["nodes"]
            l, m, n, L = element_csL(nodes[i_node], nodes[j_node])
            rows.append(
                {
                    "eid": str(eid),
                    "type": elem["type"],
                    "i_node": i_node,
                    "j_node": j_node,
                    "l": l,
                    "m": m,
                    "n": n,
                    "L": L,
                }
            )
        extra_tables["element_cosines"] = pd.DataFrame(rows)

    rows = []
    for eid, qf in Qf_debug.items():
        qf = np.asarray(qf, dtype=float).reshape(-1)
        if np.any(np.abs(qf) > 1e-12):
            row = {"eid": str(eid)}
            for i, val in enumerate(qf, start=1):
                row[f"qf_{i}"] = val
            rows.append(row)

    extra_tables["Qf_local"] = pd.DataFrame(rows)
    extra_tables["u_f"] = _vector_df(u_f)
    extra_tables["F_r"] = _vector_df(F_r)
    extra_tables["u_global_complete"] = _vector_df(u_global_complete)
    extra_tables["f_global_complete"] = _vector_df(f_global_complete)

    # ============================================================
    # Main result tables
    # ============================================================
    tables = {}

    truss_elements = {
        eid: elem for eid, elem in elements.items() if elem["type"] == "3D_truss"
    }
    cable_elements = {
        eid: elem for eid, elem in elements.items() if elem["type"] == "3D_cable"
    }
    frame_elements = {
        eid: elem for eid, elem in elements.items() if elem["type"] == "3D_frame"
    }

    if len(truss_elements) > 0 and len(results_truss) > 0:
        tables["truss"] = build_truss_results_dataframe_3d(
            truss_elements,
            element_lengths,
            results_truss,
            length_unit=length_unit,
        )

    if len(frame_elements) > 0 and len(results_frame) > 0:
        tables["frame"] = build_frame_results_dataframe_3d(
            frame_elements,
            element_lengths,
            results_frame,
            length_unit=length_unit,
        )

    if len(cable_elements) > 0 and len(results_cable) > 0:
        df_cable = build_truss_results_dataframe_3d(
            cable_elements,
            element_lengths,
            results_cable,
            length_unit=length_unit,
        )

        if "ele" in df_cable.columns:
            df_cable["state"] = df_cable["ele"].map(
                lambda eid: (
                    _get_result_item(results_cable, eid)["state"]
                    if _get_result_item(results_cable, eid) is not None
                    else None
                )
            )

        tables["cable"] = df_cable

    # ============================================================
    # Summary tables
    # ============================================================
    can_build_summary = (
        nodes is not None
        and u_global_complete is not None
        and f_global_complete is not None
    )

    if can_build_summary:
        summary_tables_raw = build_result_summary(
            nodes=nodes,
            elements=elements,
            supports=supports,
            u_global_complete=u_global_complete,
            f_global_complete=f_global_complete,
            F_global=F_global,
            F_fef_global=F_fef_global,
            results_truss=results_truss,
            results_cable=results_cable,
            results_frame=results_frame,
            releases=releases,
        )

        rename_map = {
            "summary_displacement": "max_displacement",
            "summary_reaction": "max_reaction",
            "summary_member_force": "max_member_force",
            "summary_equilibrium": "equilibrium_check",
            "summary_release": "release_check",
        }

        for old_key, df in summary_tables_raw.items():
            new_key = rename_map.get(old_key, old_key)
            if df is not None and not df.empty:
                extra_tables[new_key] = df

    # ============================================================
    # Display tables
    # ============================================================
    if display_tables:
        main_display_order = ["truss", "frame", "cable"]
        for sheet_name in main_display_order:
            if sheet_name in tables and not tables[sheet_name].empty:
                print(f"\n=== {sheet_name.upper()} RESULTS ===")
                display(display_compact(tables[sheet_name]))

        summary_display_order = [
            "max_displacement",
            "max_reaction",
            "max_member_force",
            "equilibrium_check",
            "release_check",
        ]
        for sheet_name in summary_display_order:
            if sheet_name in extra_tables and not extra_tables[sheet_name].empty:
                print(f"\n=== {sheet_name.upper()} ===")
                display(display_compact(extra_tables[sheet_name]))

        detail_display_order = [
            "node_displacement",
            "reaction_dof",
            "reaction_node",
            "element_cosines",
            "Qf_local",
            "u_f",
            "F_r",
            "u_global_complete",
            "f_global_complete",
        ]
        for sheet_name in detail_display_order:
            if sheet_name in extra_tables and not extra_tables[sheet_name].empty:
                print(f"\n=== {sheet_name.upper()} ===")
                display(display_compact(extra_tables[sheet_name]))

    # ============================================================
    # Save tables
    # ============================================================
    if save_tables:
        model_stem = Path(model_path).stem if model_path is not None else "results"
        save_path = Path(save_dir) / model_stem
        save_path.mkdir(parents=True, exist_ok=True)

        if save_filename is None:
            save_filename = f"{model_stem}.xlsx"

        filepath = save_path / save_filename

        with pd.ExcelWriter(filepath) as writer:
            # 1) main result sheets first
            for sheet_name in ["truss", "frame", "cable"]:
                if (
                    sheet_name in tables
                    and tables[sheet_name] is not None
                    and not tables[sheet_name].empty
                ):
                    tables[sheet_name].to_excel(
                        writer,
                        sheet_name=sheet_name[:31],
                        index=False,
                    )

            # 2) summary sheets next
            summary_sheet_order = [
                "max_displacement",
                "max_reaction",
                "max_member_force",
                "equilibrium_check",
                "release_check",
            ]
            for sheet_name in summary_sheet_order:
                if (
                    sheet_name in extra_tables
                    and extra_tables[sheet_name] is not None
                    and not extra_tables[sheet_name].empty
                ):
                    extra_tables[sheet_name].to_excel(
                        writer,
                        sheet_name=sheet_name[:31],
                        index=False,
                    )

            # 3) detailed/debug sheets
            already_written = set(["truss", "frame", "cable"] + summary_sheet_order)

            extra_sheet_order = [
                "node_displacement",
                "reaction_dof",
                "reaction_node",
                "element_cosines",
                "Qf_local",
                "u_f",
                "F_r",
                "u_global_complete",
                "f_global_complete",
            ]

            for sheet_name in extra_sheet_order:
                if sheet_name in already_written:
                    continue
                if (
                    sheet_name in extra_tables
                    and extra_tables[sheet_name] is not None
                    and not extra_tables[sheet_name].empty
                ):
                    extra_tables[sheet_name].to_excel(
                        writer,
                        sheet_name=sheet_name[:31],
                        index=False,
                    )
                    already_written.add(sheet_name)

            # 4) anything else
            for sheet_name, df in extra_tables.items():
                if sheet_name in already_written:
                    continue
                if df is not None and not df.empty:
                    df.to_excel(
                        writer,
                        sheet_name=sheet_name[:31],
                        index=False,
                    )

        print(f"Saved tables: {filepath}")

    return {**tables, **extra_tables}
