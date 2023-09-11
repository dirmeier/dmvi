import haiku as hk
from jax import numpy as jnp
from jax import random as jr


def expand_dims(v, dims):
    return v[(...,) + (None,) * (dims - 1)]


def to_sparse_list(l):
    sparse_l = []
    num = 0
    last_item = None
    for item in l:
        if last_item is None:
            last_item = item
            num = 1
        elif item != last_item:
            sparse_l.append((last_item, num))
            last_item = item
            num = 1
        else:
            num += 1
    if last_item is not None:
        sparse_l.append((last_item, num))
    return sparse_l


class DPMSolver:
    """
    Code taken and adopted from:
    https://github.com/LuChengTHU/dpm-solver/blob/main/dpm_solver_jax.py#L7
    """

    def __init__(self, n_dim, model_fn, noise_schedule, n_steps=20, order=1):
        self.model_fn = model_fn
        self.noise_schedule = noise_schedule
        self.n_dim = n_dim
        self.n_steps = n_steps
        self.predict_x0 = False
        self.order = order

    def get_time_steps(self, skip_type, t_T, t_0, N):
        if skip_type == "logSNR":
            lambda_T = self.noise_schedule.marginal_lambda(t_T)
            lambda_0 = self.noise_schedule.marginal_lambda(t_0)
            logSNR_steps = jnp.linspace(lambda_T, lambda_0, N + 1)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == "time_uniform":
            return jnp.linspace(t_T, t_0, N + 1)
        elif skip_type == "time_quadratic":
            t_order = 2
            t = jnp.power(
                jnp.linspace(t_T ** (1.0 / t_order), t_0 ** (1.0 / t_order), N + 1),
                t_order,
            )
            return t
        else:
            raise ValueError(
                "Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(
                    skip_type
                )
            )

    def get_orders_and_timesteps_for_singlestep_solver(self, steps, order, skip_type, t_T, t_0):
        """
        Get the order of each step for sampling by the singlestep DPM-Solver.

        We combine both DPM-Solver-1,2,3 to use all the function evaluations, which is named as "DPM-Solver-fast".
        Given a fixed number of function evaluations by `steps`, the sampling procedure by DPM-Solver-fast is:
            - If order == 1:
                We take `steps` of DPM-Solver-1 (i.e. DDIM).
            - If order == 2:
                - Denote K = (steps // 2). We take K or (K + 1) intermediate time steps for sampling.
                - If steps % 2 == 0, we use K steps of DPM-Solver-2.
                - If steps % 2 == 1, we use K steps of DPM-Solver-2 and 1 step of DPM-Solver-1.
            - If order == 3:
                - Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
                - If steps % 3 == 0, we use (K - 2) steps of DPM-Solver-3, and 1 step of DPM-Solver-2 and 1 step of DPM-Solver-1.
                - If steps % 3 == 1, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-1.
                - If steps % 3 == 2, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-2.

        ============================================
        Args:
            order: A `int`. The max order for the solver (2 or 3).
            steps: A `int`. The total number of function evaluations (NFE).
        Returns:
            orders: A list of the solver order of each step.
        """
        if order == 3:
            K = steps // 3 + 1
            if steps % 3 == 0:
                orders = [3,] * (
                    K - 2
                ) + [2, 1]
            elif steps % 3 == 1:
                orders = [3,] * (
                    K - 1
                ) + [1]
            else:
                orders = [3,] * (
                    K - 1
                ) + [2]
        elif order == 2:
            if steps % 2 == 0:
                K = steps // 2
                orders = [
                    2,
                ] * K
            else:
                K = steps // 2 + 1
                orders = [2,] * (
                    K - 1
                ) + [1]
        elif order == 1:
            K = 1
            orders = [
                1,
            ] * steps
        else:
            raise ValueError("'order' must be '1' or '2' or '3'.")
        if skip_type == "logSNR":
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, K)
        else:
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, steps)[
                jnp.cumsum(
                    jnp.array(
                        [
                            0,
                        ]
                        + orders
                    )
                )
            ]
        return timesteps_outer, orders

    def singlestep_dpm_solver_first_update(
        self, x, s, t, model_s=None, return_intermediate=False, is_training=True
    ):
        """
        DPM-Solver-1 (equivalent to DDIM) from time `s` to time `t`.

        Args:
            x: A jnp.DeviceArray. The initial value at time `s`.
            s: A jnp.DeviceArray. The starting time, with the shape (x.shape[0],).
            t: A jnp.DeviceArray. The ending time, with the shape (x.shape[0],).
            model_s: A jnp.DeviceArray. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s`.
        Returns:
            x_t: A jnp.DeviceArray. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        dims = x.ndim
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = ns.log_sqrt_alphas_cumprod(s), ns.log_sqrt_alphas_cumprod(t)
        sigma_s, sigma_t = ns.scale(s), ns.scale(t)
        alpha_t = jnp.exp(log_alpha_t)

        if self.predict_x0:
            phi_1 = jnp.expm1(-h)
            if model_s is None:
                model_s = self.model_fn(x, s, is_training)
            x_t = (
                expand_dims(sigma_t / sigma_s, dims) * x
                - expand_dims(alpha_t * phi_1, dims) * model_s
            )
            if return_intermediate:
                return x_t, {"model_s": model_s}
            else:
                return x_t
        else:
            phi_1 = jnp.expm1(h)
            if model_s is None:
                model_s = self.model_fn(x, s, is_training)
            x_t = (
                expand_dims(jnp.exp(log_alpha_t - log_alpha_s), dims) * x
                - expand_dims(sigma_t * phi_1, dims) * model_s
            )
            if return_intermediate:
                return x_t, {"model_s": model_s}
            else:
                return x_t

    def singlestep_dpm_solver_second_update(
        self,
        x,
        s,
        t,
        r1=0.5,
        model_s=None,
        return_intermediate=False,
        solver_type="dpm_solver",
        is_training=True,
    ):
        """
        Singlestep solver DPM-Solver-2 from time `s` to time `t`.

        Args:
            x: A jnp.DeviceArray. The initial value at time `s`.
            s: A jnp.DeviceArray. The starting time, with the shape (x.shape[0],).
            t: A jnp.DeviceArray. The ending time, with the shape (x.shape[0],).
            r1: A `float`. The hyperparameter of the second-order solver.
            model_s: A jnp.DeviceArray. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s` and `s1` (the intermediate time).
            solver_type: either 'dpm_solver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpm_solver' type.
        Returns:
            x_t: A jnp.DeviceArray. The approximated solution at time `t`.
        """
        if solver_type not in ["dpm_solver", "taylor"]:
            raise ValueError(
                f"'solver_type' must be either 'dpm_solver' or 'taylor', got {solver_type}"
            )
        if r1 is None:
            r1 = 0.5
        ns = self.noise_schedule
        dims = x.ndim
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        s1 = ns.inverse_lambda(lambda_s1)
        log_alpha_s, log_alpha_s1, log_alpha_t = (
            ns.log_sqrt_alphas_cumprod(s),
            ns.log_sqrt_alphas_cumprod(s1),
            ns.log_sqrt_alphas_cumprod(t),
        )
        sigma_s, sigma_s1, sigma_t = ns.scale(s), ns.scale(s1), ns.scale(t)
        alpha_s1, alpha_t = jnp.exp(log_alpha_s1), jnp.exp(log_alpha_t)

        if self.predict_x0:
            phi_11 = jnp.expm1(-r1 * h)
            phi_1 = jnp.expm1(-h)

            if model_s is None:
                model_s = self.model_fn(x, s, is_training=is_training)
            x_s1 = (
                expand_dims(sigma_s1 / sigma_s, dims) * x
                - expand_dims(alpha_s1 * phi_11, dims) * model_s
            )
            model_s1 = self.model_fn(x_s1, s1, is_training=is_training)
            if solver_type == "dpm_solver":
                x_t = (
                    expand_dims(sigma_t / sigma_s, dims) * x
                    - expand_dims(alpha_t * phi_1, dims) * model_s
                    - (0.5 / r1) * expand_dims(alpha_t * phi_1, dims) * (model_s1 - model_s)
                )
            elif solver_type == "taylor":
                x_t = (
                    expand_dims(sigma_t / sigma_s, dims) * x
                    - expand_dims(alpha_t * phi_1, dims) * model_s
                    + (1.0 / r1)
                    * expand_dims(alpha_t * ((jnp.exp(-h) - 1.0) / h + 1.0), dims)
                    * (model_s1 - model_s)
                )
        else:
            phi_11 = jnp.expm1(r1 * h)
            phi_1 = jnp.expm1(h)

            if model_s is None:
                model_s = self.model_fn(x, s, is_training=is_training)
            x_s1 = (
                expand_dims(jnp.exp(log_alpha_s1 - log_alpha_s), dims) * x
                - expand_dims(sigma_s1 * phi_11, dims) * model_s
            )
            model_s1 = self.model_fn(x_s1, s1, is_training=is_training)
            if solver_type == "dpm_solver":
                x_t = (
                    expand_dims(jnp.exp(log_alpha_t - log_alpha_s), dims) * x
                    - expand_dims(sigma_t * phi_1, dims) * model_s
                    - (0.5 / r1) * expand_dims(sigma_t * phi_1, dims) * (model_s1 - model_s)
                )
            elif solver_type == "taylor":
                x_t = (
                    expand_dims(jnp.exp(log_alpha_t - log_alpha_s), dims) * x
                    - expand_dims(sigma_t * phi_1, dims) * model_s
                    - (1.0 / r1)
                    * expand_dims(sigma_t * ((jnp.exp(h) - 1.0) / h - 1.0), dims)
                    * (model_s1 - model_s)
                )
        if return_intermediate:
            return x_t, {"model_s": model_s, "model_s1": model_s1}
        else:
            return x_t

    def singlestep_dpm_solver_third_update(
        self,
        x,
        s,
        t,
        r1=1.0 / 3.0,
        r2=2.0 / 3.0,
        model_s=None,
        model_s1=None,
        return_intermediate=False,
        solver_type="dpm_solver",
        is_training=True,
    ):
        """
        Singlestep solver DPM-Solver-3 from time `s` to time `t`.

        Args:
            x: A jnp.DeviceArray. The initial value at time `s`.
            s: A jnp.DeviceArray. The starting time, with the shape (x.shape[0],).
            t: A jnp.DeviceArray. The ending time, with the shape (x.shape[0],).
            r1: A `float`. The hyperparameter of the third-order solver.
            r2: A `float`. The hyperparameter of the third-order solver.
            model_s: A jnp.DeviceArray. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            model_s1: A jnp.DeviceArray. The model function evaluated at time `s1` (the intermediate time given by `r1`).
                If `model_s1` is None, we evaluate the model at `s1`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s`, `s1` and `s2` (the intermediate times).
            solver_type: either 'dpm_solver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpm_solver' type.
        Returns:
            x_t: A jnp.DeviceArray. The approximated solution at time `t`.
        """
        if solver_type not in ["dpm_solver", "taylor"]:
            raise ValueError(
                f"'solver_type' must be either 'dpm_solver' or 'taylor', got {solver_type}"
            )
        if r1 is None:
            r1 = 1.0 / 3.0
        if r2 is None:
            r2 = 2.0 / 3.0
        ns = self.noise_schedule
        dims = x.ndim
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        lambda_s2 = lambda_s + r2 * h
        s1 = ns.inverse_lambda(lambda_s1)
        s2 = ns.inverse_lambda(lambda_s2)
        log_alpha_s, log_alpha_s1, log_alpha_s2, log_alpha_t = (
            ns.log_sqrt_alphas_cumprod(s),
            ns.log_sqrt_alphas_cumprod(s1),
            ns.log_sqrt_alphas_cumprod(s2),
            ns.log_sqrt_alphas_cumprod(t),
        )
        sigma_s, sigma_s1, sigma_s2, sigma_t = (
            ns.scale(s),
            ns.scale(s1),
            ns.scale(s2),
            ns.scale(t),
        )
        alpha_s1, alpha_s2, alpha_t = (
            jnp.exp(log_alpha_s1),
            jnp.exp(log_alpha_s2),
            jnp.exp(log_alpha_t),
        )

        if self.predict_x0:
            phi_11 = jnp.expm1(-r1 * h)
            phi_12 = jnp.expm1(-r2 * h)
            phi_1 = jnp.expm1(-h)
            phi_22 = jnp.expm1(-r2 * h) / (r2 * h) + 1.0
            phi_2 = phi_1 / h + 1.0
            phi_3 = phi_2 / h - 0.5

            if model_s is None:
                model_s = self.model_fn(x, s, is_training=is_training)
            if model_s1 is None:
                x_s1 = (
                    expand_dims(sigma_s1 / sigma_s, dims) * x
                    - expand_dims(alpha_s1 * phi_11, dims) * model_s
                )
                model_s1 = self.model_fn(x_s1, s1, is_training=is_training)
            x_s2 = (
                expand_dims(sigma_s2 / sigma_s, dims) * x
                - expand_dims(alpha_s2 * phi_12, dims) * model_s
                + r2 / r1 * expand_dims(alpha_s2 * phi_22, dims) * (model_s1 - model_s)
            )
            model_s2 = self.model_fn(x_s2, s2, is_training=is_training)
            if solver_type == "dpm_solver":
                x_t = (
                    expand_dims(sigma_t / sigma_s, dims) * x
                    - expand_dims(alpha_t * phi_1, dims) * model_s
                    + (1.0 / r2) * expand_dims(alpha_t * phi_2, dims) * (model_s2 - model_s)
                )
            elif solver_type == "taylor":
                D1_0 = (1.0 / r1) * (model_s1 - model_s)
                D1_1 = (1.0 / r2) * (model_s2 - model_s)
                D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                D2 = 2.0 * (D1_1 - D1_0) / (r2 - r1)
                x_t = (
                    expand_dims(sigma_t / sigma_s, dims) * x
                    - expand_dims(alpha_t * phi_1, dims) * model_s
                    + expand_dims(alpha_t * phi_2, dims) * D1
                    - expand_dims(alpha_t * phi_3, dims) * D2
                )
        else:
            phi_11 = jnp.expm1(r1 * h)
            phi_12 = jnp.expm1(r2 * h)
            phi_1 = jnp.expm1(h)
            phi_22 = jnp.expm1(r2 * h) / (r2 * h) - 1.0
            phi_2 = phi_1 / h - 1.0
            phi_3 = phi_2 / h - 0.5

            if model_s is None:
                model_s = self.model_fn(x, s, is_training=is_training)
            if model_s1 is None:
                x_s1 = (
                    expand_dims(jnp.exp(log_alpha_s1 - log_alpha_s), dims) * x
                    - expand_dims(sigma_s1 * phi_11, dims) * model_s
                )
                model_s1 = self.model_fn(x_s1, s1, is_training=is_training)
            x_s2 = (
                expand_dims(jnp.exp(log_alpha_s2 - log_alpha_s), dims) * x
                - expand_dims(sigma_s2 * phi_12, dims) * model_s
                - r2 / r1 * expand_dims(sigma_s2 * phi_22, dims) * (model_s1 - model_s)
            )
            model_s2 = self.model_fn(x_s2, s2, is_training=is_training)
            if solver_type == "dpm_solver":
                x_t = (
                    expand_dims(jnp.exp(log_alpha_t - log_alpha_s), dims) * x
                    - expand_dims(sigma_t * phi_1, dims) * model_s
                    - (1.0 / r2) * expand_dims(sigma_t * phi_2, dims) * (model_s2 - model_s)
                )
            elif solver_type == "taylor":
                D1_0 = (1.0 / r1) * (model_s1 - model_s)
                D1_1 = (1.0 / r2) * (model_s2 - model_s)
                D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                D2 = 2.0 * (D1_1 - D1_0) / (r2 - r1)
                x_t = (
                    expand_dims(jnp.exp(log_alpha_t - log_alpha_s), dims) * x
                    - expand_dims(sigma_t * phi_1, dims) * model_s
                    - expand_dims(sigma_t * phi_2, dims) * D1
                    - expand_dims(sigma_t * phi_3, dims) * D2
                )

        if return_intermediate:
            return x_t, {
                "model_s": model_s,
                "model_s1": model_s1,
                "model_s2": model_s2,
            }
        else:
            return x_t

    def singlestep_dpm_solver_update(
        self,
        x,
        s,
        t,
        order,
        return_intermediate=False,
        solver_type="dpm_solver",
        r1=None,
        r2=None,
        is_training=True,
    ):
        if order == 1:
            return self.singlestep_dpm_solver_first_update(
                x,
                s,
                t,
                return_intermediate=return_intermediate,
                is_training=is_training,
            )
        elif order == 2:
            return self.singlestep_dpm_solver_second_update(
                x,
                s,
                t,
                return_intermediate=return_intermediate,
                solver_type=solver_type,
                r1=r1,
                is_training=is_training,
            )
        elif order == 3:
            return self.singlestep_dpm_solver_third_update(
                x,
                s,
                t,
                return_intermediate=return_intermediate,
                solver_type=solver_type,
                r1=r1,
                r2=r2,
                is_training=is_training,
            )
        else:
            raise ValueError(f"Solver order must be 1 or 2 or 3, got {order}")

    def __call__(self, sample_shape=(1,), is_training=True):
        z_T = jr.normal(hk.next_rng_key(), sample_shape + (self.n_dim,))
        t_0 = self.noise_schedule.T0
        t_T = self.noise_schedule.Tmax

        (timesteps_outer, orders,) = self.get_orders_and_timesteps_for_singlestep_solver(
            steps=self.n_steps,
            order=self.order,
            skip_type="time_uniform",
            t_T=t_T,
            t_0=t_0,
        )

        def singlestep_loop_fn(idx, val, order):
            i, x = val
            t_T_inner, t_0_inner = timesteps_outer[i], timesteps_outer[i + 1]
            timesteps_inner = self.get_time_steps(
                skip_type="time_uniform", t_T=t_T_inner, t_0=t_0_inner, N=order
            )
            lambda_inner = self.noise_schedule.marginal_lambda(timesteps_inner)
            vec_s, vec_t = jnp.tile(t_T_inner, (x.shape[0])), jnp.tile(t_0_inner, (x.shape[0]))
            h = lambda_inner[-1] - lambda_inner[0]
            r1 = None if order <= 1 else (lambda_inner[1] - lambda_inner[0]) / h
            r2 = None if order <= 2 else (lambda_inner[2] - lambda_inner[0]) / h
            x = self.singlestep_dpm_solver_update(
                x,
                vec_s,
                vec_t,
                order,
                solver_type="dpm_solver",
                r1=r1,
                r2=r2,
                is_training=is_training,
            )
            return i + 1, x

        i = 0
        x = z_T
        order_list = to_sparse_list(orders)
        for order, nums in order_list:
            i, x = hk.fori_loop(
                0,
                nums,
                lambda idx, val: singlestep_loop_fn(idx, val, order),
                (i, x),
            )

        return x
