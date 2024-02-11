import twsit_utils as utils
import numpy as np
import time

def psi_function(X, tau, max_svd, nx, ny, nz, Psi):
    if tau == 0:
        return X

    if Psi == 'TV':
        return utils.tvdenoise2D(X, 2/(tau/max_svd), 3, nx, ny, nz)
    elif Psi == 'SOFT':
        return utils.soft(X, tau/max_svd)
    elif Psi == 'SOFT_DWT':
        return utils.soft_DWT(X, tau/max_svd, nx, ny, nz)
    else:
        raise ValueError("Invalid value for Psi")
    
def phi_function(X, nx, ny, nz, Phi):
    if Phi == 'TV':
        return utils.TVnorm2D(X, nx, ny, nz)
    elif Phi == 'L1':
        return np.sum(np.abs(X))
    elif Phi == 'L1_FD':
        return utils.L1Norm_FD(X, nx, ny, nz)
    elif Phi == 'L1_DWT':
        return utils.L1Norm_DWT(X, nx, ny, nz)
    else:
        raise ValueError("Invalid value for Phi")


def TwIST(y, FM, tau, nx, ny, nz,
        Psi = 'SOFT',
        Phi = 'L1',
        lam1 = 1e-4, 
        alpha = 0,
        beta = 0,
        stop_criterion = 1,
        tolA = 0.01,
        debias = 0,
        maxiter = 1000,
        maxiter_debias = 200,
        miniter = 5,
        miniter_debias = 5,
        init = 0,
        init_x = None,
        enforceMonotone = 1,
        sparse = 0,
        true_x = None,
        compute_mse = 0,
        plot_ISNR = 0,
        verbose = 1,
        tolD = 0.001,
        lamN = 1):

    if true_x is None:
        compute_mse = 0
    else:
        compute_mse = 1


    # maj_max_sv: majorizer for the maximum singular value of operator A
    max_svd = 1

    # Set the defaults for outputs that may not be computed
    debias_start = 0
    x_debias = np.array([])
    mses = np.array([])


    # twist parameters
    rho0 = (1 - lam1/lamN)/(1 + lam1/lamN)
    if alpha == 0:
        alpha = 2/(1 + np.sqrt(1 - rho0**2))
    if beta == 0:
        beta = alpha*2/(lam1 + lamN)

    # Precompute A'*y since it'll be used a lot
    Aty = utils.AT(y, FM, nx, ny, nz)

    # Initialization
    if init == 0:
        # initialize at zero, using AT to find the size of x
        x = utils.AT(np.zeros_like(y), FM, nx, ny, nz)
    elif init == 1:
        # initialize randomly, using AT to find the size of x
        x = np.random.randn(*utils.AT(np.zeros_like(y), FM, nx, ny, nz).shape)
    elif init == 2:
        # initialize x0 = A'*y
        x = Aty
    elif init == 3:
        # initial x was given as a function argument; just check size
        if np.shape(utils.A(init_x, FM, nx, ny, nz)) != np.shape(y):
            raise ValueError("Size of initial x is not compatible with A")
        else:
            x = init_x
    else:
        raise ValueError("Unknown 'Initialization' option")

    # now check if tau is an array; if it is, it has to
    # have the same size as x
    if isinstance(tau, np.ndarray):
        try:
            dummy = x*tau
        except:
            raise ValueError("Parameter tau has wrong dimensions; it should be scalar or size(x)")

    # if the true x was given, check its size
    if compute_mse and np.shape(true_x) != np.shape(x):
        raise ValueError("Initial x has incompatible size")

    # if tau is large enough, in the case of phi = l1, thus psi = soft,
    # the optimal solution is the zero vector
    if Phi == 'L1' and Psi == 'SOFT':
        max_tau = np.max(np.abs(Aty))
        x = np.zeros_like(Aty)
        objective = [0.5*np.dot(y.flatten(), y.flatten())]
        times = [0]
        if compute_mse:
            mses = np.array([np.sum(true_x.flatten()**2)])
        return x, x_debias, objective, times, debias_start, mses, max_svd

    # define the indicator vector or matrix of nonzeros in x
    nz_x = (x != 0.0)
    num_nz_x = np.sum(nz_x)

    # Compute and store the initial value of the objective function
    resid = y - utils.A(x, FM, nx, ny, nz)
    prev_f = 0.5*np.dot(resid.flatten(), resid.flatten()) + tau*phi_function(x, nx, ny, nz, Phi)

    # start the clock
    t0 = time.process_time()

    times = [time.process_time() - t0]
    objective = [prev_f]

    if compute_mse:
        mses = np.array([np.sum((x - true_x)**2)])

    cont_outer = 1
    iter = 1

    if verbose:
        print(f"\nInitial objective = {prev_f}, nonzeros = {num_nz_x}")

    # variables controlling first and second order iterations
    IST_iters = 0
    TwIST_iters = 0

    # initialize
    xm2 = x.copy()
    xm1 = x.copy()

    # TwIST iterations
    while cont_outer:
        # gradient
        grad = utils.AT(resid, FM, nx, ny, nz)
        while True:
            # IST estimate
            x = psi_function(xm1+grad/max_svd, tau, max_svd, nx, ny, nz, Psi)
            if IST_iters >= 2 or TwIST_iters != 0:
                # set to zero the past when the present is zero
                # suitable for sparse inducing priors
                if sparse:
                    mask = (x != 0)
                    xm1 = xm1*mask
                    xm2 = xm2*mask
                # two-step iteration
                xm2 = (alpha - beta)*xm1 + (1 - alpha)*xm2 + beta*x
                # compute residual
                resid = y - utils.A(xm2, FM, nx, ny, nz)
                f = 0.5*np.dot(resid.flatten(), resid.flatten()) + tau*phi_function(x, nx, ny, nz, Phi)
                if f > prev_f and enforceMonotone:
                    TwIST_iters = 0  # do an IST iteration if monotonicity fails
                else:
                    TwIST_iters += 1  # TwIST iterations
                    IST_iters = 0
                    x = xm2
                    if TwIST_iters%10000 == 0:
                        max_svd *= 0.9
                    break  # break loop while
            else:
                resid = y - utils.A(x, FM, nx, ny, nz)
                f = 0.5*np.dot(resid.flatten(), resid.flatten()) + tau*phi_function(x, nx, ny, nz, Phi)
                if f > prev_f:
                    # if monotonicity fails here is because
                    # max eig (A'A) > 1. Thus, we increase our guess
                    # of max_svs
                    max_svd = 2*max_svd
                    if verbose:
                        print(f"Incrementing S = {max_svd}")
                    IST_iters = 0
                    TwIST_iters = 0
                else:
                    TwIST_iters += 1
                    break  # break loop while

        xm2 = xm1
        xm1 = x

        # update the number of nonzero components and its variation
        nz_x_prev = nz_x
        nz_x = (x != 0.0)
        num_nz_x = np.sum(nz_x)
        num_changes_active = np.sum(nz_x != nz_x_prev)

        # Take no less than miniter and no more than maxiter iterations
        if stop_criterion == 0:
            # Compute the stopping criterion based on the change
            # of the number of non-zero components of the estimate
            criterion = num_changes_active
        elif stop_criterion == 1:
            # Compute the stopping criterion based on the relative
            # variation of the objective function.
            criterion = abs(f - prev_f) / prev_f
        elif stop_criterion == 2:
            # Compute the stopping criterion based on the relative
            # variation of the estimate.
            criterion = np.linalg.norm(x.flatten() - xm1.flatten()) / np.linalg.norm(x.flatten())
        elif stop_criterion == 3:
            # Continue if not yet reached the target value tolA
            criterion = f
        else:
            raise ValueError('Unknown stopping criterion')


        cont_outer = (iter <= maxiter) & (criterion > tolA)
        if iter <= miniter:
            cont_outer = 1

        iter += 1
        prev_f = f
        objective.append(f)
        times.append(time.process_time() - t0)

        if compute_mse:
            err = true_x - x
            mses = np.append(mses, np.dot(err.flatten(), err.flatten()))

        # print out the various stopping criteria
        if verbose:
            if plot_ISNR:
                print(f"Iteration = {iter}, ISNR = {10 * np.log10(np.sum((y - true_x)**2) / np.sum((x - true_x)**2))}, "
                      f"objective = {f}, nz = {num_nz_x}, criterion = {criterion / tolA}")
            else:
                print(f"Iteration = {iter}, objective = {f}, nz = {num_nz_x}, criterion = {criterion / tolA}")

    # end of the main loop

    # Printout results
    if verbose:
        print("\nFinished the main algorithm!\nResults:")
        print(f"||A x - y ||_2 = {np.dot(resid.flatten(), resid.flatten())}")
        print(f"||x||_1 = {np.sum(np.abs(x))}")
        print(f"Objective function = {f}")
        print(f"Number of non-zero components = {num_nz_x}")
        print(f"CPU time so far = {times[-1]}\n")

    # If the 'Debias' option is set to 1, we try to
    # remove the bias from the l1 penalty, by applying CG to the
    # least-squares problem obtained by omitting the l1 term
    # and fixing the zero coefficients at zero.
    if debias:
        if verbose:
            print('\nStarting the debiasing phase...\n')

        x_debias = x.copy()
        zeroind = (x_debias != 0)
        cont_debias_cg = True
        debias_start = iter

        # Calculate initial residual
        resid = utils.A(x_debias, FM, nx, ny, nz)
        resid = resid - y
        resid_prev = np.finfo(float).eps*np.ones_like(resid)

        rvec = utils.AT(resid, FM, nx, ny, nz)

        # Mask out the zeros
        rvec = rvec*zeroind
        rTr_cg = np.dot(rvec.flatten(), rvec.flatten())

        # Set convergence threshold for the residual || RW x_debias - y ||_2
        tol_debias = tolD*np.dot(rvec.flatten(), rvec.flatten())

        # Initialize pvec
        pvec = -rvec

        # Main loop
        while cont_debias_cg:
            # Calculate A*p = Wt * Rt * R * W * pvec
            RWpvec = utils.A(pvec, FM, nx, ny, nz)
            Apvec = utils.AT(RWpvec, FM, nx, ny, nz)

            # Mask out the zero terms
            Apvec = Apvec*zeroind

            # Calculate alpha for CG
            alpha_cg = rTr_cg/np.dot(pvec.flatten(), Apvec.flatten())

            # Take the step
            x_debias = x_debias + alpha_cg*pvec
            resid = resid + alpha_cg*RWpvec
            rvec = rvec + alpha_cg*Apvec

            rTr_cg_plus = np.dot(rvec.flatten(), rvec.flatten())
            beta_cg = rTr_cg_plus/rTr_cg
            pvec = -rvec + beta_cg*pvec

            rTr_cg = rTr_cg_plus

            iter = iter + 1

            objective[iter] = 0.5*np.dot(resid.flatten(), resid.flatten()) + tau*phi_function(x_debias.flatten(), nx, ny, nz, Phi)
            times[iter] = time.process_time() - t0

            if compute_mse:
                err = true_x - x_debias
                mses[iter] = np.dot(err.flatten(), err.flatten())

            # In the debiasing CG phase, always use convergence criterion
            # based on the residual (this is standard for CG)
            if verbose:
                print(f' Iter = {iter}, debias resid = {np.dot(resid.flatten(), resid.flatten()):.8e}, '
                    f'convergence = {rTr_cg / tol_debias:.3e}')
            cont_debias_cg = (iter - debias_start <= miniter_debias) or \
                            ((rTr_cg > tol_debias) and (iter - debias_start <= maxiter_debias))

        if verbose:
            print('\nFinished the debiasing phase!\nResults:')
            print(f'||A x - y ||_2 = {np.dot(resid.flatten(), resid.flatten()):.3e}')
            print(f'||x||_1 = {np.sum(np.abs(x_debias.flatten())):.3e}')
            print(f'Objective function = {objective[iter]:.3e}')
            nz = (x_debias != 0.0)
            print(f'Number of non-zero components = {np.sum(nz.flatten())}')
            print(f'CPU time so far = {times[iter]:.3e}\n')

    if compute_mse:
        mses = mses/len(true_x.flatten())

    return x, x_debias, objective, times, debias_start, mses, max_svd


