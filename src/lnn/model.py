
# Building MLP model 
import jax
import jax.numpy as jnp
import equinox as eqx


class LagrangianNN(eqx.Module):
    """
    Neural Network model designed to learn the Lagrangian of a physical system,
    which is defined as the difference between kinetic and potential energy (T - V).

    The model consists of three core modules:
    1. kinetic_net - an MLP that maps trigonometric features of the generalized coordinates (q) 
    and the system parameters (p) to a Cholesky-factorised mass matrix. The output of this net is used to compute the kinetic energy T.
    
    2. potential_net - an MLP that takes the same trigonometric features together with 
    the parameters (p) and outputs a scalar potential energy V.
    
    4. film_net - a Feature-wise Linear Modulation (FiLM) network that generates per-layer scaling (\gamma) and shifting (β) parameters for the hidden layers of kinetic_net. 
    These parameters are conditioned on the system parameters (e.g. masses, lengths) and thus allow the kinetic energy to be modulated by the physics of the specific system.
    
    The `__call__` method of this class uses automatic differentiation (jax.grad, jax.jacobian)
    on the learned Lagrangian to derive the system's equations of motion, returning
    the generalized accelerations (q_tt).
    """
    kinetic_net:    eqx.nn.MLP
    potential_net:  eqx.nn.MLP
    film_net:       eqx.nn.MLP
    n_hidden:       int
    hidden_dim:     int

    def __init__(self,
                 pos_dim: int,
                 vel_dim: int,
                 hidden_dim: int, 
                 n_hidden: int, 
                 param_dim: int,
                 key: jnp.array, 
                 **kwds):
        super().__init__(**kwds)
        """Initializes the Lagrangian Neural Network.

        Args:
            pos_dim (int): Dimensionality of the generalized coordinates (q).
            vel_dim (int): Dimensionality of the generalized velocities (q_dot).
            param_dim (int): Dimensionality of the system parameters (p),
                             e.g., masses and lengths.
            hidden_dim (int): Width of the hidden layers in the `lagrangian_net`.
            n_hidden (int): Number of hidden layers in the `lagrangian_net`.
            key (jnp.array): JAX PRNGKey for initializing model weights.
            apply_trig_fn (bool, optional): If True, converts generalized coordinates `q`
                                            into `[sin(q_i), cos(q_i)]` pairs for input
                                            to the `lagrangian_net`. Defaults to True.
            **kwds: Additional keyword arguments for Equinox.
        """
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        trig_dim = pos_dim * 2

        kinetic_key, potential_key, film_key = jax.random.split(key, 3)
        
        # kinetic net
        output_dim_kin = vel_dim * (vel_dim+1)//2

        self.kinetic_net = eqx.nn.MLP(
            in_size=trig_dim, 
            out_size=output_dim_kin,
            width_size=hidden_dim,
            depth=n_hidden,
            activation=lambda x: x,    # identity for now; apply softplus later.
            key=kinetic_key,
        )

        # potential net
        self.potential_net = eqx.nn.MLP(
            in_size=trig_dim + param_dim,
            out_size=1,
            width_size=hidden_dim,
            depth=n_hidden,
            activation=jax.nn.tanh,
            key=potential_key
        )

        # Feature-wIse Linear Modulation (FiLM)
        # ====================================
        # outputs two parameters for each hidden layer of the lagrangian mlp to help parameterize by the mass and length of the pendulum
        self.film_net = eqx.nn.MLP(
            in_size=param_dim, 
            out_size = 2 * n_hidden,
            width_size=32,
            depth=2,
            activation=jax.nn.softplus,
            key=film_key
        )

        # Initialize FiLM net to identity such that at first, the kinetic energy is not modulated at all
        # identity_bias  = jnp.tile(jnp.array([1.0, 0.]), n_hidden)
        # model = eqx.tree_at(lambda m: m.film_net.layers[-1].bias, self, identity_bias)

    
    def apply_film(self, h, film_params, net):
        """
        Run network net by applying FILM parameters onto its hidden layers

        Args:
            h
        """
        for i in range(self.n_hidden):
            # Compute layer transformation
            h = net.layers[i](h)
            h = jax.nn.softplus(h)
            
            # FiLM scaling
            gamma = film_params[i, 0]
            beta = film_params[i, 1]

            h = gamma * h + beta
        return h

    def compute_cholesky_entries(self, q: jnp.Array, film_params: jnp.Array) -> jnp.Array:
        h = self.apply_film(q, film_params, self.kinetic_net)
        return self.kinetic_net.layers[self.n_hidden](h)
    
    def compute_potential(self, q: jnp.Array, p: jnp.Array) -> jnp.Array:
        h_pot = jnp.concatenate([q, p])
        return jnp.squeeze(self.potential_net(h_pot))

    def compute_lagrangian(self, q: jax.Array, q_t: jax.Array, p: jax.Array) -> jax.Array:    
        
        # Transform to trigonometric values 
        # to avoid numerical instability due to discontinuity at 0,2pi point (or -pi, pi)
        # =======================
        trig_q = []
        for qi in q:
            trig_q.extend([jnp.sin(qi), jnp.cos(qi)])
        trig_q = jnp.array(trig_q)
        
        # Compute FiLM parameters 
        # =======================
        # (gamma, beta) from system parameters 'p'
        # Reshape to (n_hidden, 2) where each row is [gamma_i, beta_i]
        film_params = self.film_net(p).reshape(self.n_hidden, 2)

        # Compute kinetic energy
        # =======================
        # h = self.apply_film(trig_q, film_params, self.kinetic_net)
        # chol_entries = self.kinetic_net.layers[self.n_hidden](h)
        chol_entries = self.compute_cholesky_entries(trig_q, film_params)

        # Cholesky decomposition of Mass matrix
        L = jnp.array([
            [jax.nn.softplus(chol_entries[0]),                               0.0],
            [chol_entries[1],                   jax.nn.softplus(chol_entries[2])]
        ])

        M = L.T @ L + jnp.eye(2) * 1e-6
        T = 0.5 * q_t @ M @ q_t
        
        # Compute potential energy
        # =======================
        V = self.compute_potential(trig_q, p)
        
        return T - V
    
    def __call__(self, q: jax.Array, q_t: jax.Array, p: jax.Array) -> jax.Array:
        """Derives and returns the generalized accelerations (q_tt) from the learned Lagrangian.

        This method applies Euler-Lagrange equations via automatic differentiation
        to compute the accelerations:
        M(q) * q_tt = f(q, q_t)
        where M = d^2L / (dq_t dq_t) (Mass Matrix)
        and f = dL/dq - d/dt(dL/dq_t) = dL/dq - (dL/dq_t dq)q_t - (dL/dq_t dq_t)q_tt (effectively)
        Rearranging and solving for q_tt.

        Args:
            q (jax.Array): Generalized coordinates, shape (pos_dim,).
            q_t (jax.Array): Generalized velocities, shape (vel_dim,).
            p (jax.Array): System parameters, shape (param_dim,).

        Returns:
            jax.Array: Generalized accelerations (q_tt), shape (vel_dim,).
        """
        lagrangian_fn = lambda _q, _qt: self.compute_lagrangian(_q, _qt, p)
        
        # 1. Compute dL/dq
        l_q = jax.grad(lagrangian_fn, argnums=0)(q, q_t)
        
        # 2. dL/dqt and its derivatives
        l_qt_fn = jax.grad(lagrangian_fn, argnums=1)        # get function dL/dqt
        l_qt_q = jax.jacobian(l_qt_fn, argnums=0)(q, q_t)   # l_qt_q = d^2L / (dqt dq), shape (vel_dim,vel_dim)
        l_qt_qt = jax.jacobian(l_qt_fn, argnums=1)(q, q_t)  # l_qt_qt = d^2L / (dqt dqt)  <-- The Mass Matrix, shape (vel_dim,vel_dim)
        
        # 3. Solve (L_qt_qt) * q_tt = L_q - (L_qt_q) * q_t
        l_qt_qt = l_qt_qt + jnp.eye(2) * 1e-6
        rhs     = l_q - l_qt_q @ q_t
        q_tt    = jnp.linalg.solve(l_qt_qt, rhs)    # acceleration
        return q_tt

if __name__ == '__main__':

    print('Try initialize the model')

    key = jax.random.PRNGKey(32)
    # Example initialization: 2D position, 2D velocity, 4D parameters
    model = LagrangianNN(pos_dim=2, vel_dim=2, param_dim=4, hidden_dim=128, n_hidden=2, key=key)

    print(model)

    # Basic test of __call__ method
    q_test = jnp.array([0.1, 0.2])
    qt_test = jnp.array([0.0, 0.0])
    p_test = jnp.array([1.0, 1.0, 1.0, 1.0]) # Example parameters

    q_tt_output = model(q_test, qt_test, p_test)
    print("\nExample q_tt output shape:", q_tt_output.shape)
    print("Example q_tt output value:", q_tt_output)

    # Verify input_lagrangian_net_size logic
    model_trig = LagrangianNN(pos_dim=2, vel_dim=2, param_dim=4, hidden_dim=128, n_hidden=2, key=key)

    # Check input sizes after init
    print(f"LagrangianNet kinetic input size: {model_trig.kinetic_net.in_size}") # Should be (2*2 + 2) = 6

