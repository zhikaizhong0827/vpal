import time
import torch
import numpy as np
from torch.func import vjp, jvp
from D import FiniteDifference, Laplace
import scipy.optimize


class vpal(torch.nn.Module):
    """
    Variable Projected Augmented Lagrangian (VPAL) for problems of the form:
        argmin_x  0.5 * ||A(x) - b||^2 + mu * ||D(x)||_1

    Notes
    -----
    • A : forward operator (can be linear or nonlinear, e.g., a neural network)
    • D : regularization operator (e.g., FiniteDifference for TV, Laplace)

    Design highlights
    -----------------
    1) Uses torch.func.vjp / jvp to obtain A^T r and J*v on the fly, so you
       don't need to hand-code adjoints/Jacobians. Works for complex/nonlinear A, D.
    2) After each iteration, we do
           x = x.detach().requires_grad_(True)
       to (a) cut the history graph between iterations and (b) make x a fresh
       leaf for the next jvp/vjp calls. This avoids graph blow-up and accidental
       unrolled-optimization-through-iterations.
    3) Two step-size options:
       • 'linearized' : closed-form-like step alpha ≈ ||g||^2 / (||A g||^2 + λ^2 ||D g||^2)
         via JVPs.
       • 'optimal'    : 1D coarse line-search by grid over alpha.
    """

    def __init__(self, A, D, DT=None, mu=0, lambda_=1, sigma=None, tol=1e-6, xtrue=None,
                 step_size='linearized', bnd=None, dof=0, maxIter=None, display='off'):
        super().__init__()
        # --- Parse/instantiate D ---
        if isinstance(D, str):
            if D.lower() in ['finite difference', 'finitedifference', 'diff', 'tv']:
                D = FiniteDifference()
            elif D.lower() in ['laplace', 'laplacian']:
                D = Laplace()
            else:
                raise ValueError(f"Unknown D operator: {D}")
        self.A = A                  # forward operator (nn.Module or callable)
        self.D = D                  # regularization operator (Module or callable)
        self.DT = self.D.T()        # adjoint/transpose provided by D.T()

        # --- Hyperparameters and controls ---
        self.mu = mu                # L1 weight
        self.lambda_ = lambda_      # augmented penalty (adapted during iterations)
        self.lambda2 = lambda_ ** 2 # recorded; local lambda is used in forward()
        self.sigma = sigma          # reserved (unused)
        self.tol = tol              # stop tolerance
        self.xtrue = xtrue          # optional ground-truth for logging rel. error
        self.step_size = step_size  # 'linearized' or 'optimal'
        self.bnd = bnd              # box bounds: [lo, hi] or flattened per-element 2*N
        self.dof = dof              # dof for discrepancy principle stopping
        self.display = display      # 'off' | 'iter' | 'final'
        self.maxIter = maxIter      # None => set a large fallback in forward()
        self.info = {}

    def forward(self, b, x0=None, return_info=False):
        """
        Run VPAL and return x (and optional info dict).

        Parameters
        ----------
        b : Tensor
            Observations. Shape either [B, C, H, W] or [B, N].
        x0 : Tensor or None
            Optional initialization, same batch shape as b.
        return_info : bool
            If True, returns (x, info).
        """
        # ---- 1) To tensor & choose device ----
        if not isinstance(b, torch.Tensor):
            b = torch.as_tensor(b, dtype=torch.float32)
        if x0 is not None and not isinstance(x0, torch.Tensor):
            x0 = torch.as_tensor(x0, dtype=torch.float32)

        device = (x0.device if isinstance(x0, torch.Tensor)
                  else (b.device if isinstance(b, torch.Tensor)
                        else torch.device('cpu')))
        b  = b.to(device)
        x0 = x0.to(device) if isinstance(x0, torch.Tensor) else None

        # ---- 2) Shape bookkeeping (2D latent or 4D image) ----
        if b.ndim == 4:
            B, C, H, W = b.shape
            N = H * W
        elif b.ndim == 2:
            B, N = b.shape
            C = H = W = None
        else:
            raise ValueError(f"Unsupported input shape: {b.shape}")

        # ---- 3) Initialize x (with grad) ----
        # Make x a fresh leaf so vjp/jvp can differentiate w.r.t. it this iteration.
        x = (x0.clone().detach() if x0 is not None else torch.zeros_like(b))
        x = x.requires_grad_(True)

        # ---- 4) Put operators on device, prep adjoint once ----
        self.D  = self.D.to(device)
        self.DT = self.D.T()
        try:
            self.DT = self.DT.to(device)
        except AttributeError:
            # If DT is just a callable (not Module), it doesn't have .to()
            pass

        # ---- 5) Algorithm parameters ----
        maxIter = self.maxIter or (10 * b.numel())
        tol = self.tol
        mu = self.mu
        lambda_ = torch.tensor(float(self.lambda_), dtype=torch.float32, device=device)
        lambda2  = lambda_ ** 2
        step_size = self.step_size
        display = self.display
        dof = self.dof
        bnd = self.bnd

        # ---- 6) Parse box constraints ----
        # Supports:
        #   • [lo, hi] (either side can be inf)
        #   • per-element [lo_i, hi_i] with total length 2*N (broadcasted over batch)
        bound_case = 0
        if bnd is not None:
            bnd = torch.as_tensor(bnd, device=device)
            if bnd.numel() == 2:
                if torch.isinf(bnd[1]):
                    bound_case = 1  # lower bound only
                elif torch.isinf(bnd[0]):
                    bound_case = 2  # upper bound only
                else:
                    bound_case = 3  # both sides
            elif bnd.numel() == 2 * N:
                bound_case = 4      # per-element bounds
            else:
                raise ValueError('box constraints not properly set.')

        # ---- 7) Initial residuals and multipliers ----
        Dx = self.D(x)             # D(x)
        r  = self.A(x) - b         # residual A(x) - b

        c = torch.zeros_like(Dx)   # scaled dual / Bregman variable
        y = torch.zeros_like(Dx)   # split variable approximating D(x)
        normb = torch.norm(b)
        f = float('inf')
        xOld = x.clone().detach()
        iter = 1

        # ---- 8) Info dict for logging ----
        info = {
            'tol': tol,
            'maxIter': maxIter,
            'loss': [],      # ||A x - b|| / ||b||
            'f': [],         # objective 0.5||r||^2 + mu||Dx||_1
            'alpha': [],     # step sizes
            'stop': [],      # stopping flags
            'relerr': [],    # if xtrue provided
            'tv': [],        # final ||Dx||_1
            'normr': [],     # final ||r||
            'chi2': [],      # final ||r||^2 + mu||Dx||_1
            'lambda': [],    # lambda trajectory
            'mu': [],
            'iter': 0,
            'operator': self.D
        }

        if display in ['iter', 'final']:
            print('\nVPAL algorithm (c) Matthias Chung & Rosemary Renaut, June 2021')
            if display == 'iter':
                print('\n {:<5} {:<14} {:<14}'.format('iter', 'loss', 'stop criteria'))

        # ==========================
        #        M A I N  L O O P
        # ==========================
        while True:
            start = time.time()

            # ---- 9) Gradient direction: g = A^T r + λ^2 D^T( D x - (y - c) ) ----
            # Use vjp(A, x) to get a callable for the adjoint-vector product.
            _, vjp_fn = vjp(self.A, x)
            At_r, = vjp_fn(r)
            g = At_r + lambda2 * self.DT(Dx - (y - c))

            # ---- 10) Step size alpha ----
            if step_size == 'linearized':
                # Closed-form-like with JVPs: alpha ≈ ||g||^2 / (||A g||^2 + λ^2 ||D g||^2)
                _, Ag = jvp(self.A, (x,), (g,))
                _, Dg = jvp(self.D, (x,), (g,))
                num = torch.sum(g * g)
                den = torch.sum(Ag * Ag) + lambda2 * torch.sum(Dg * Dg) + 1e-12
                alpha = num / den
            elif step_size == 'optimal':
                # Coarse 1D grid search on alpha
                _, Ag = jvp(self.A, (x,), (g,))
                _, Dg = jvp(self.D, (x,), (g,))
                gAAg = torch.sum(Ag * Ag)
                rAg = torch.sum(r * Ag)
                alphas = torch.linspace(0, 10, steps=200, device=device)

                # Vectorized proxy objective over all alphas
                Dg_expand = Dg.unsqueeze(0)  # [1, ...]
                Dx_c_expand = Dx + c - alphas.view(-1, *([1]*len(Dg.shape))) * Dg_expand

                # Prox for L1: soft-thresholding
                Z = torch.sign(Dx_c_expand) * torch.clamp(torch.abs(Dx_c_expand) - mu / lambda2, min=0)

                # Flatten to [num_alphas, -1] for norms
                norm_Dx_c_minus_Z = torch.norm((Dx_c_expand - Z).reshape(alphas.shape[0], -1), dim=1)
                norm_Z_1 = torch.norm(Z.reshape(alphas.shape[0], -1), p=1, dim=1)

                # Proxy objective (drop alpha-independent constants)
                obj = 0.5 * alphas**2 * gAAg - alphas * rAg + 0.5 * lambda2 * norm_Dx_c_minus_Z**2 + mu * norm_Z_1
                min_idx = torch.argmin(obj)
                alpha = alphas[min_idx]
            else:
                raise NotImplementedError(f"Unknown step_size: {step_size}")

            # ---- 11) Gradient step for x + box projection ----
            x = x - alpha * g

            if bound_case == 1:
                x = torch.maximum(x, bnd[0])
            elif bound_case == 2:
                x = torch.minimum(x, bnd[1])
            elif bound_case == 3:
                x = torch.clamp(x, bnd[0], bnd[1])
            elif bound_case == 4:
                x = torch.maximum(x, bnd.view(N, 2)[:, 0].view_as(x))
                x = torch.minimum(x, bnd.view(N, 2)[:, 1].view_as(x))

            # Crucial: cut previous graph and make x a fresh leaf for next jvp/vjp
            x = x.detach().requires_grad_(True)

            # ---- 12) Refresh residuals and Dx (builds the current iteration's graph) ----
            r = self.A(x) - b
            Dx = self.D(x)

            # ---- 13) Shrinkage & dual/Bregman update ----
            # y_old is a numeric snapshot; detach so it doesn't carry history.
            y_old = y.clone().detach()

            # Prox step on Dx + c
            c = Dx + c
            y = torch.sign(c) * torch.clamp(torch.abs(c) - mu / lambda2, min=0)

            # Scaled dual/Bregman update
            c = c - y

            # ---- 14) Adaptive lambda balancing primal/dual residuals ----
            r_prim = Dx - y                       # primal residual
            r_dual = self.DT(y - y_old) * lambda_ # dual residual
            norm_prim = torch.norm(r_prim)
            norm_dual = torch.norm(r_dual)
            gamma = 2.0
            delta = 10.0
            if norm_prim > delta * norm_dual:
                lambda_ = lambda_ * gamma
            elif norm_dual > delta * norm_prim:
                lambda_ = lambda_ / gamma
            lambda2 = lambda_**2
            info['lambda'].append(float(lambda_))

            # ---- 15) Objective & stopping criteria ----
            f_old = f
            f = 0.5 * torch.norm(r) ** 2 + mu * torch.norm(Dx, 1)

            stop1 = torch.abs(f_old - f) <= tol * (1 + f)
            stop2 = torch.norm(xOld - x, float('inf')) <= torch.sqrt(torch.tensor(tol, device=device)) * (1 + torch.norm(x, float('inf')))
            stop3 = iter >= maxIter
            stop4 = dof > 0 and torch.norm(r) ** 2 < dof + np.sqrt(2 * dof)

            # ---- 16) Logging and optional prints ----
            info['loss'].append((torch.norm(r) / normb).item())
            info['f'].append(f.item())
            if isinstance(alpha, torch.Tensor):
                info['alpha'].append(alpha.detach().cpu().tolist())
            else:
                info['alpha'].append(float(alpha))
            info['stop'].append([stop1.item(), stop2.item(), stop3, stop4])

            if self.xtrue is not None:
                rel = torch.norm(x - torch.tensor(self.xtrue, device=device)) / torch.norm(torch.tensor(self.xtrue, device=device))
                info['relerr'].append(rel.item())

            if display == 'iter':
                lam_list = lambda_.detach().cpu().flatten().tolist()
                print(f"{iter:5d} {f.item():14.6e} "
                      f"{int(stop1)}{int(stop2)}{int(stop3)}{int(stop4)} | "
                      f"time: {time.time()-start:.3f}s, "
                      f"λ={lam_list}, prim={norm_prim.item():.3e}, dual={norm_dual.item():.3e}")

            # ---- 17) Stopping ----
            if (stop1 and stop2) or stop3 or stop4:
                if stop3:
                    print('Warning: reached maxIter')
                if stop4:
                    print('Warning: dof criterion met')
                break

            xOld = x.clone().detach()
            iter += 1

        # ---- 18) Final metrics & return ----
        info.update({
            'tv': torch.norm(Dx, 1).item(),
            'normr': torch.norm(r).item(),
            'chi2': (torch.norm(r).item() ** 2) + mu * torch.norm(Dx, 1).item(),
            'lambda': float(lambda_.detach().cpu()),
            'mu': mu,
            'iter': iter
        })

        if return_info:
            return x, info
        return x

    # Deprecated: prefer vjp for adjoints
    def T(self, x):
        """Deprecated: use vjp(self.A, x) to obtain adjoint-vector products."""
        raise NotImplementedError("Use forward vjp instead of .T()")
