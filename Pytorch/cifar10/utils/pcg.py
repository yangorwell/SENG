import torch

def pcg(apply_op, rhs, reltol=0.1, x0=None, preconditioner=None, maxiter=10, verbose=0, print_func=print):
    """Solve apply_op(x) = rhs, such that
       ||apply_op(x) - rhs|| <= reltol * ||apply_op(x0) - rhs||

       apply_op: a function to calculate Ax for given x.
         apply_op(v) should not change v in-place.
       rhs: right hand side
       reltol: relative tolerence
       x0: initial guess, default to rhs
       preconditioner: a function to evaluate M^{-1} for pcg.
           preconditioner(v) may change v in-place and return it.
       maxiter: maximum iteration
       verbose: print extra information
    """
    x = x0.clone() if x0 else torch.zeros_like(rhs)
    r = rhs - apply_op(x)
    res0_norm = r.norm()
    # z may be altered in preconditioner. no sharing memory
    z = r.clone()
    if preconditioner is not None:
        z = preconditioner(z)
    p = z
    ztr = z.dot(r)
    for itr_num in range(maxiter):
        old_ztr = ztr
        op_p = apply_op(p)
        pap = p.dot(op_p)
        if verbose and pap < 0:
            print_func("CG p'Ap = %2.1e < 0" % pap.item())
            return x
        alpha = (ztr / pap).item()
        x.add_(alpha, p)
        r.add_(-alpha, op_p)
        res_norm = r.norm()
        relres = res_norm / res0_norm
        if verbose >= 2:
            print_func("CG iter %d | res %2.1e | relres %2.1e" % (itr_num, res_norm, relres))
        if relres < reltol:
            if verbose == 1:
                print_func("CG iter %d | res %2.1e | relres %2.1e" % (itr_num, res_norm, relres))
            return x
        z = r.clone()
        if preconditioner is not None:
            z = preconditioner(z)
        ztr = z.dot(r)
        beta = (ztr / old_ztr).item()
        p = z.add(beta, p)
    if verbose == 1:
        print_func("CG iter %d | res %2.1e | relres %2.1e" % (itr_num, res_norm, relres))
    return x

if __name__ == '__main__':
    n = 1000
    r = 50
    dtype = torch.float64
    op_mat = torch.randn(n, r, dtype=dtype)
    op_mat = op_mat @ op_mat.t() + torch.eye(n) * 0.001
    exact_x = torch.randn(n, dtype=dtype)
    op = lambda x: op_mat @ x
    rhs = op(exact_x)
    precond = lambda v: v.div_(op_mat.diagonal())
    # inv = op_mat.inverse()
    # precond = lambda v: inv @ v
    x = pcg(op, rhs, reltol=1e-8, maxiter=30, preconditioner=precond, verbose=True)
