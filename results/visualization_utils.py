import matplotlib.pyplot as plt
import matplotlib.animation as animation
import jax.numpy as jnp
import numpy as np
import jax

def animate_single(states, pendulum, dt, title='', fname='pendulum_single.gif', 
                   fps=30, speedup=3, trail_len=80):
    """
    states: array of shape [T, 4] = [q1, q2, w1, w2]
    """
    states = states[::speedup]
    T = len(states)

    l1, l2 = pendulum.l1, pendulum.l2
    
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_xlim(-l1-l2-0.2, l1+l2+0.2)
    ax.set_ylim(-l1-l2-0.2, l1+l2+0.2)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.suptitle(title, color='white', fontsize=11)

    # pivot point
    ax.plot(0, 0, 'o', color='white', ms=6, zorder=5)

    # rods and bobs
    rod,  = ax.plot([], [], '-', color='white', lw=2)
    bob1, = ax.plot([], [], 'o', color='cyan',   ms=12, zorder=4)
    bob2, = ax.plot([], [], 'o', color='orange', ms=14, zorder=4)
    trace,= ax.plot([], [], '-', color='orange', lw=1.0, alpha=0.5)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                        color='white', fontsize=10)

    trail_x, trail_y = [], []

    def update(frame):
        q1, q2 = states[frame, 0], states[frame, 1]
        x1, y1, x2, y2 = pendulum.to_cartesian((q1, q2))

        rod.set_data([0, x1, x2], [0, y1, y2])
        bob1.set_data([x1], [y1])
        bob2.set_data([x2], [y2])

        trail_x.append(x2); trail_y.append(y2)
        start = max(0, len(trail_x) - trail_len)
        trace.set_data(trail_x[start:], trail_y[start:])

        time_text.set_text(f't = {frame * dt * speedup:.1f}s')
        return rod, bob1, bob2, trace, time_text

    anim = animation.FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=True)
    anim.save(fname, writer='pillow', fps=fps, dpi=120)
    plt.close()
    print(f'Saved: {fname}')


def animate_comparison(gt_states, model_states, pendulum, dt, 
                        title='', fname='pendulum.gif', fps=30, speedup=5, figsize=(10,6)):
    """
    gt_states, model_states: arrays of shape [T, 4] = [q1, q2, w1, w2]
    speedup: show every `speedup`-th frame to make animation faster
    """
    gt     = gt_states[::speedup]
    model  = model_states[::speedup]
    T      = len(gt)
    
    l1, l2 = pendulum.l1, pendulum.l2

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor('black')
    fig.tight_layout()
    for ax in axes:
        ax.set_facecolor('black')
        ax.set_xlim(-l1-l2-0.2, l1+l2+0.2)
        # ax.set_ylim(-l1-l2-0.2, l1+l2+0.2)
        ax.set_ylim(-l1-l2-0.2, .1)
        ax.set_aspect('equal')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
    axes[0].set_title('Ground Truth', color='white', fontsize=14)
    axes[1].set_title('LagrangianNN w/ FiLM', color='white', fontsize=14)
    fig.suptitle(title, color='white', fontsize=12)

    # initialize plot elements
    line_gt,    = axes[0].plot([], [], 'o-', color='cyan',   lw=3, ms=12)
    line_model, = axes[1].plot([], [], 'o-', color='orange', lw=3, ms=12)
    trace_gt,   = axes[0].plot([], [], '-',  color='cyan',   lw=0.75, alpha=0.6)
    trace_model,= axes[1].plot([], [], '-',  color='orange', lw=0.75, alpha=0.6)
    time_text   = fig.text(0.5, 0.02, '', ha='center', color='white', fontsize=12)

    # trail history
    trail_len = 100
    gt_trail_x, gt_trail_y = [], []
    model_trail_x, model_trail_y = [], []

    def update(frame):
        q1_gt, q2_gt = gt[frame, 0], gt[frame, 1]
        q1_m,  q2_m  = model[frame, 0], model[frame, 1]
        
        x1_gt, y1_gt, x2_gt, y2_gt = pendulum.to_cartesian((q1_gt, q2_gt))
        x1_m,  y1_m,  x2_m,  y2_m  = pendulum.to_cartesian((q1_m,  q2_m))
        
        line_gt.set_data([0, x1_gt, x2_gt], [0, y1_gt, y2_gt])
        line_model.set_data([0, x1_m, x2_m], [0, y1_m, y2_m])
        
        gt_trail_x.append(x2_gt); gt_trail_y.append(y2_gt)
        model_trail_x.append(x2_m); model_trail_y.append(y2_m)
        
        start = max(0, len(gt_trail_x) - trail_len)
        trace_gt.set_data(gt_trail_x[start:], gt_trail_y[start:])
        trace_model.set_data(model_trail_x[start:], model_trail_y[start:])
        
        t = frame * dt * speedup
        time_text.set_text(f't = {t:.2f}s')
        
        return line_gt, line_model, trace_gt, trace_model, time_text

    anim = animation.FuncAnimation(fig, update, frames=T, interval=1000/(fps), blit=True)
    anim.save(fname, writer='pillow', fps=fps, dpi=120)
    plt.close()
    print(f'Saved: {fname}')


def animate_with_phase(states, pendulum, dt, title='', fname='pendulum_phase.gif',
                       fps=30, speedup=3, trail_len=80):
    """
    states: [T, 4] = [q1, q2, w1, w2]
    """
    states = states[::speedup]
    if states.shape[1] == 8:
        states = states[:, :4]
    T = len(states)

    l1, l2 = pendulum.l1, pendulum.l2

    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor('black')
    fig.suptitle(title, color='white', fontsize=11)

    # left: pendulum
    ax_pend = fig.add_subplot(1, 3, 1)
    ax_pend.set_facecolor('black')
    ax_pend.set_xlim(-l1-l2-0.2, l1+l2+0.2)
    ax_pend.set_ylim(-l1-l2-0.2, l1+l2+0.2)
    ax_pend.set_aspect('equal')
    ax_pend.axis('off')
    ax_pend.set_title('Lagrangian NN w/ FiLM\nsimulation', color='white', fontsize=12)
    ax_pend.plot(0, 0, 'o', color='white', ms=5, zorder=5)

    # middle: q1 phase portrait
    ax_ph1 = fig.add_subplot(1, 3, 2)
    ax_ph1.set_facecolor('black')
    ax_ph1.set_xlim(states[:, 0].min()-0.1, states[:, 0].max()+0.1)
    ax_ph1.set_ylim(states[:, 2].min()-0.1, states[:, 2].max()+0.1)
    ax_ph1.set_xlabel(r'$q_1$', color='white')
    ax_ph1.set_ylabel(r'$\omega_1$', color='white')
    ax_ph1.set_title('Phase portrait — DOF 1', color='white', fontsize=10)
    ax_ph1.tick_params(colors='white')
    for spine in ax_ph1.spines.values(): spine.set_color('white')

    # right: q2 phase portrait
    ax_ph2 = fig.add_subplot(1, 3, 3)
    ax_ph2.set_facecolor('black')
    ax_ph2.set_xlim(states[:, 1].min()-0.1, states[:, 1].max()+0.1)
    ax_ph2.set_ylim(states[:, 3].min()-0.1, states[:, 3].max()+0.1)
    ax_ph2.set_xlabel(r'$q_2$', color='white')
    ax_ph2.set_ylabel(r'$\omega_2$', color='white')
    ax_ph2.set_title('Phase portrait — DOF 2', color='white', fontsize=10)
    ax_ph2.tick_params(colors='white')
    for spine in ax_ph2.spines.values(): spine.set_color('white')

    plt.tight_layout()

    # pendulum elements
    rod,  = ax_pend.plot([], [], '-',  color='white',  lw=2)
    bob1, = ax_pend.plot([], [], 'o',  color='cyan',   ms=12, zorder=4)
    bob2, = ax_pend.plot([], [], 'o',  color='orange', ms=14, zorder=4)
    trail,= ax_pend.plot([], [], '-',  color='orange', lw=1.0, alpha=0.5)
    time_text = ax_pend.text(0.02, 0.95, '', transform=ax_pend.transAxes,
                              color='white', fontsize=9)

    # phase portrait traces — build up over time
    ph1_trace, = ax_ph1.plot([], [], '-', color='cyan',   lw=1.0, alpha=0.8)
    ph2_trace, = ax_ph2.plot([], [], '-', color='orange', lw=1.0, alpha=0.8)
    
    # current position dot on phase portrait
    ph1_dot, = ax_ph1.plot([], [], 'o', color='white', ms=5, zorder=5)
    ph2_dot, = ax_ph2.plot([], [], 'o', color='white', ms=5, zorder=5)

    trail_x, trail_y = [], []

    def update(frame):
        q1, q2, w1, w2 = states[frame]
        x1, y1, x2, y2 = pendulum.to_cartesian((q1, q2))

        # pendulum
        rod.set_data([0, x1, x2], [0, y1, y2])
        bob1.set_data([x1], [y1])
        bob2.set_data([x2], [y2])

        trail_x.append(x2); trail_y.append(y2)
        start = max(0, len(trail_x) - trail_len)
        trail.set_data(trail_x[start:], trail_y[start:])
        time_text.set_text(f't = {frame * dt * speedup:.1f}s')

        # phase portraits — full history up to current frame
        ph1_trace.set_data(states[:frame+1, 0], states[:frame+1, 2])
        ph2_trace.set_data(states[:frame+1, 1], states[:frame+1, 3])

        # current position dot
        ph1_dot.set_data([q1], [w1])
        ph2_dot.set_data([q2], [w2])

        return rod, bob1, bob2, trail, time_text, ph1_trace, ph2_trace, ph1_dot, ph2_dot

    anim = animation.FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=True)
    anim.save(fname, writer='pillow', fps=fps, dpi=120)
    plt.close()
    print(f'Saved: {fname}')


def plot_phase_portrait(gt_states, model_states, title='', fname='phase.png'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title)
    
    labels = [('q1', 'w1'), ('q2', 'w2')]
    for i, ax in enumerate(axes):
        ax.plot(gt_states[:, i],    gt_states[:, i+2],    
                color='steelblue', lw=1.0, alpha=0.8, label='ground truth')
        ax.plot(model_states[:, i], model_states[:, i+2], 
                color='orange',    lw=1.0, alpha=0.8, linestyle='--', label='model')
        ax.scatter(gt_states[0, i],    gt_states[0, i+2],    
                   color='steelblue', s=50, zorder=5)
        ax.scatter(model_states[0, i], model_states[0, i+2], 
                   color='orange',    s=50, zorder=5)
        ax.set_xlabel(labels[i][0]); ax.set_ylabel(labels[i][1])
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


def animate_trajectory_on_V(states, model, norm_stats, params_phys, pendulum, fps=30, speedup=3, n_grid=60):

    X_mean, X_std = norm_stats['X_mean'], norm_stats['X_std']
    p_phys = jnp.array(params_phys)
    p_norm = (p_phys - X_mean[4:]) / X_std[4:]

    # precompute V landscape
    q1_grid = jnp.linspace(-jnp.pi, jnp.pi, n_grid)
    q2_grid = jnp.linspace(-jnp.pi, jnp.pi, n_grid)
    Q1, Q2  = jnp.meshgrid(q1_grid, q2_grid, indexing='ij')

    def model_V(q1, q2):
        trig_q = jnp.array([jnp.sin(q1), jnp.cos(q1),
                             jnp.sin(q2), jnp.cos(q2)])
        return model.compute_potential(trig_q, p_norm)

    # Build numpy arrays for plotting
    V_grid = np.array(jax.vmap(jax.vmap(model_V))(Q1, Q2))
    Q1     = np.asarray(Q1)
    Q2     = np.asarray(Q2)

    states_ds = np.asarray(states[::speedup])
    T_total = len(states_ds)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('black')

    l1, l2 = pendulum.l1, pendulum.l2

    # pendulum panel
    ax_pend = axes[0]
    ax_pend.set_facecolor('black')
    ax_pend.set_xlim(-l1-l2-0.2, l1+l2+0.2)
    ax_pend.set_ylim(-l1-l2-0.2, l1+l2+0.2)
    ax_pend.set_aspect('equal')
    ax_pend.axis('off')
    ax_pend.plot(0, 0, 'o', color='white', ms=5)

    # V landscape panel
    ax_V = axes[1]
    ax_V.set_facecolor('black')
    ax_V.contourf(Q1, Q2, V_grid.T, levels=40, cmap='viridis', alpha=0.85)
    ax_V.set_xlabel('q1', color='white')
    ax_V.set_ylabel('q2', color='white')
    ax_V.set_title('Position on V landscape', color='white', fontsize=10)
    ax_V.tick_params(colors='white')
    for spine in ax_V.spines.values(): spine.set_color('white')

    plt.tight_layout()

    rod,   = ax_pend.plot([], [], '-', color='white',  lw=2)
    bob1,  = ax_pend.plot([], [], 'o', color='cyan',   ms=12, zorder=4)
    bob2,  = ax_pend.plot([], [], 'o', color='orange', ms=14, zorder=4)
    trail, = ax_pend.plot([], [], '-', color='orange', lw=1.0, alpha=0.4)

    # trajectory trace on V landscape
    V_trace, = ax_V.plot([], [], '-', color='white', lw=1.0, alpha=0.6)
    V_dot,   = ax_V.plot([], [], 'o', color='red',   ms=8,  zorder=5)

    trail_x, trail_y = [], []

    def update(frame):
        q1, q2 = states_ds[frame, 0], states_ds[frame, 1]
        x1, y1, x2, y2 = pendulum.to_cartesian((q1, q2))

        rod.set_data([0, x1, x2], [0, y1, y2])
        bob1.set_data([x1], [y1])
        bob2.set_data([x2], [y2])
        trail_x.append(x2); trail_y.append(y2)
        start = max(0, len(trail_x) - 80)
        trail.set_data(trail_x[start:], trail_y[start:])

        V_trace.set_data(states_ds[:frame+1, 0], states_ds[:frame+1, 1])
        V_dot.set_data([q1], [q2])

        return rod, bob1, bob2, trail, V_trace, V_dot

    anim = animation.FuncAnimation(fig, update, frames=T_total,
                                    interval=1000/fps, blit=True)
    return anim