import numpy as np

# solve_free_displacements
# compute_reactions


def solve_free_displacements(K_ff, K_fr, f_f, u_r, f_fef_f):
    """
    Solve for free DOF displacements.

    Equation:
        K_ff u_f = f_f - f_fef_f - K_fr u_r
    """
    rhs = f_f - f_fef_f - K_fr @ u_r
    u_f = np.linalg.solve(K_ff, rhs)
    return u_f


def compute_reactions(K_rf, K_rr, u_f, u_r, f_fef_r):
    """
    Compute restrained DOF reaction forces.

    Equation:
        F_r = K_rf u_f + K_rr u_r + f_fef_r
    """
    F_r = K_rf @ u_f + K_rr @ u_r + f_fef_r
    return F_r
