
from pathlib import Path
from typing import List, Tuple
import datetime
import jax
import optax
import jax.numpy as jnp
import equinox as eqx
from lnn import LagrangianNN
from data import load_list_of_arrays_from_h5

import numpy as np
import matplotlib.pyplot as plt
from train_utils import normalize_data, build_temporal_batch, build_input_output, train_test_split
from train_utils import save_model, load_model
from losses import energy_conservation_loss


# GLOBAL TRAINING VARIABLES
# ==========================
BATCH_SIZE = 512
PATIENCE = 50
LEARNING_RATE = 3e-3
STEPS = 50000
PRINT_EVERY = 100
EVAL_EVERY = 300
SEED = 5678

MODEL_DIR = Path(__file__).resolve().parent / "models"   # ← src/models/
MODEL_DIR.mkdir(parents=True, exist_ok=True)            # creates it if it does not exist


@eqx.filter_jit
def compute_loss(model, x, y, split_size=2):
    """
    compute loss value by running model over input data x, and evaluating predictions against ground truth y.
    Args:
        model:      Equinox/Jax model
        x:          state vector values
        y:          target variable
        split_size: int, where to split the state vector between position, velocities and parameters
    
    Returns:
        loss composed of prediction error on acceleration (huber loss) + variance of energy conservation over entire chunk of trajectory.
    """
    batch_q, batch_qt, batch_params = jnp.split(x, [split_size, split_size*2], axis=-1)    # split state into q and qdot
    preds    = jax.vmap(model)(batch_q, batch_qt, batch_params) # compute prediction with the model
    num_loss = jnp.mean(optax.huber_loss(preds, y, delta=1.0))  # compute numerical loss (on acceleration) using Huber loss
    ec_loss = energy_conservation_loss(model, x, split_size=2)  # compute energy conservation variance to help induce energy conservation
    return num_loss + 1.0 * ec_loss # combine losses


@eqx.filter_jit
def train_step(model: eqx.Module, 
               optimizer_state: optax.OptState, 
               x: jax.Array, 
               y: jax.Array, 
               optimizer: optax.GradientTransformation, 
               *args) -> Tuple[eqx.Module, optax.OptState, jax.Array]:
    """Performs a single training step for the model, including gradient computation and parameter updates.

    This function is JIT-compiled using `eqx.filter_jit` for performance.

    Args:
        model (eqx.Module): The current state of the neural network model.
        optimizer_state (optax.OptState): The current state of the Optax optimizer.
        x (jax.Array): The input data batch for the model, typically containing concatenated
                       generalized coordinates (q), generalized velocities (q_dot),
                       and system parameters (p). Shape (batch_size * temporal_chunk_len, features).
        y (jax.Array): The target data batch for the model (e.g., ground truth accelerations).
                       Shape (batch_size * temporal_chunk_len, output_dim).
        optimizer (optax.GradientTransformation): The Optax optimizer chain used for updates.
        *args: Variable-length argument list.
            args[0] (int): `split_size`, the dimensionality of `q` (generalized coordinates)
                           used for splitting the input vector `x`.

    Returns:
        tuple[eqx.Module, optax.OptState, jax.Array]: A tuple containing:
            - The updated model after applying gradients.
            - The updated optimizer state.
            - The computed loss value for the current step.
    """
    split_size = args[0]

    # Compute loss and its gradients
    loss, grads = eqx.filter_value_and_grad(compute_loss)(model, x, y, split_size=split_size)
    
    # Apply the gradients to update optimizer state and generate param updates
    updates, optimizer_state = optimizer.update(grads, optimizer_state, model, value=loss)
    
    # Apply the generated updates to the model's parameters
    model = eqx.apply_updates(model, updates)
    
    return model, optimizer_state, loss


def training_loop(
        model: eqx.Module, 
        x_train: jax.Array, 
        y_train: jax.Array, 
        x_val: jax.Array,
        y_val: jax.Array,
        steps: int, 
        lr: float, 
        batch_size: int, 
        train_key: jax.Array, 
        *args
    ) -> Tuple[eqx.Module, optax.OptState, List[float], List[float]]:
    """Executes the main training loop for the neural network model with early stopping.

    This function sets up the optimizer with a cosine decay learning rate schedule,
    iterates through a specified number of training steps, and logs both training
    and validation loss. It implements early stopping based on validation loss.

    Args:
        model (eqx.Module): The initial neural network model to be trained.
        x_train (jax.Array): The complete input training dataset.
        y_train (jax.Array): The complete target training dataset.
        x_val (jax.Array): The complete input validation dataset.
        y_val (jax.Array): The complete target validation dataset.
        steps (int): The total number of training steps (iterations).
        lr (float): The initial learning rate for the optimizer.
        batch_size (int): The intended batch size for the `build_temporal_batch`
                          function. Note: this value is reinterpreted internally
                          as `time_chunk_length` for `build_temporal_batch`,
                          and `batch_size` for that function becomes 1.
        train_key (jax.Array): A JAX PRNGKey for reproducibility of random operations.
        *args: Variable-length argument list.
            args[0] (int): `split_size`, the dimensionality of generalized coordinates (`q`).

    Returns:
        Tuple[eqx.Module, optax.OptState, List[float], List[float]]: A tuple containing:
            - The best model found during training (based on validation loss).
            - The optimizer state corresponding to the best model.
            - A list of training loss values recorded at each training step.
            - A list of validation loss values recorded at `eval_every` steps.
    """
    split_size = args[0]
    
    # OPTIMIZER
    # ==========
    
    # Scheduler for the learning rate
    scheduler = optax.cosine_decay_schedule(
        init_value=lr,
        decay_steps=steps,
        alpha=0.01
    )

    # Actual optimizer: chain AdamW with norm clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=scheduler),
    )

    # Optimizer state
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # TRAINING 
    # ======================

    # Initialization of variables
    train_loss_history, val_loss_histor = [], []

    # early stopping parameters
    best_val_loss    = jnp.inf
    best_model       = model
    best_opt_state   = opt_state
    patience_counter = 0
    
    for step in range(steps):

        step_key = jax.random.fold_in(train_key, step)  # get random key 
        
        # Get batch
        # ---------
        # NOTE: Define the batching strategy: 
        # each batch is one trajectory, with length of batch size 
        x_batch, y_batch = build_temporal_batch(x=x_train, 
                                                y=y_train, 
                                                batch_size=1, 
                                                temporal_chunk_len=batch_size, 
                                                step_key=step_key)
    
        # Training step:
        model, opt_state, loss = train_step(
            model, 
            opt_state, 
            x_batch, 
            y_batch, 
            optimizer, 
            split_size
        )

        # record train loss
        train_loss_history.append(float(loss))

        # Evaluation stage
        # ====================
        if (step+1) % EVAL_EVERY == 0:
            
            # Evaluate the loss over all samples in the validation set
            val_loss = []
            for x_val_i, y_val_i in zip(x_val, y_val):
                val_loss_i = compute_loss(model, x_val_i, y_val_i, split_size=split_size)
                val_loss.append(val_loss_i)
            val_loss = sum(val_loss)/len(val_loss)
            val_loss_history.append(float(val_loss))

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                best_model       = model        # Save the current model if it's the best
                best_opt_state   = opt_state
                patience_counter = 0    
                print(f"Validation Step {step+1:4d} | **Val Loss improved to {best_val_loss:.8f}** (Patience: {patience_counter}/{PATIENCE})")
            else:
                patience_counter += 1
                print(f"Validation Step {step+1:4d} | Val Loss: {val_loss:.8f} (No improvement, Patience: {patience_counter}/{PATIENCE})")
                if patience_counter >= PATIENCE:
                    print(f"Early stopping at step {step+1} due to no improvement in validation loss.")
                    break # Exit the training loop
        else:
            val_loss_history.append(np.nan)

        if (step + 1) % PRINT_EVERY == 0 and (step + 1) % EVAL_EVERY != 0: # Avoid double printing
            print(f"Step {step+1:4d} | Train Loss: {loss:.8f} ")
    
    return best_model, best_opt_state, train_loss_history, val_loss_history


if __name__ == '__main__':

    print("Training script for JAX models")    

    # DATASET
    # =======
    # Dataset is a list of arrays of shape [T, 5] = (time, q1, q2, qdot1, qdot2)
    
    datasets, params = load_list_of_arrays_from_h5(system='doublependulum', filename='dp_trajectories.h5')
    dt = datasets[0][1, 0] - datasets[0][0, 0]
    time_v = datasets[0][:, 0]
    pos_dim = 2
    vel_dim = 2
    param_dim = 4
    state_dim = pos_dim + vel_dim
    
    # Build dataset for training: from x = [q, dqdt], build dx/dt numerically
    # ========================================================================
    X, dXdt = build_input_output(datasets=datasets, params=params, dt=dt)
    
    # Build train/test sets
    # ==========================
    idx_train, idx_val, idx_test = train_test_split(X, n_train=0.8, n_val=0.1, seed=SEED)
    # extract
    Xtrain, Xval, Xtest = X[idx_train], X[idx_val], X[idx_test]
    dXdt_train, dXdt_val, dXdt_test = dXdt[idx_train], dXdt[idx_val], dXdt[idx_test]

    # Normalize data
    # -----------------
    # X columns: [q1, q2, w1, w2, m1, m2, l1, l2]; q1, q2 are raw angles in radians — trig transform handles them, don't normalize
    # w1, w2, m1, m2, l1, l2 — normalize by mean and std
    Xtrain_norm, Xval_norm, Xtest_norm, \
        dXdt_train_norm, dXdt_val_norm, dXdt_test_norm, \
            norm_stats = normalize_data(Xtrain, Xval, Xtest, dXdt_train, dXdt_val, dXdt_test, len(params), normalize=True)

    # MODEL INSTANTIATION
    # ========================================================================
    key = jax.random.PRNGKey(123)
    model_key, train_key = jax.random.split(key)
    model = LagrangianNN(
        pos_dim = pos_dim,
        vel_dim = vel_dim,
        param_dim= param_dim,
        hidden_dim=128,
        n_hidden=2,
        key=model_key
        )

    # TRAINING
    # =========================s
    model, opt_state, \
        loss_history, val_loss_history = training_loop(model, 
                                                       Xtrain_norm, 
                                                       dXdt_train_norm,
                                                       Xval_norm,
                                                       dXdt_val_norm,
                                                       STEPS, 
                                                       LEARNING_RATE, 
                                                       BATCH_SIZE,
                                                       train_key,
                                                       pos_dim # this is for splitting the vector.
                                                    )

    # Save model
    # -----------
    m_name = f"model_T{BATCH_SIZE}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}"
    save_model_path = MODEL_DIR / m_name
    save_model(model=model, fname=save_model_path)

    # Reload model
    # --------------
    model = LagrangianNN(pos_dim=pos_dim,vel_dim=pos_dim, param_dim=param_dim,hidden_dim=128, n_hidden=2,key=model_key)
    model = load_model(model=model, fname=save_model_path)

    # Plot and save learning curve
    # ==================
    loss_window = 20
    smoothed_loss_train = jnp.convolve(np.array(loss_history), jnp.ones(loss_window)/loss_window, mode='valid')
    smoothed_loss_val = np.array(val_loss_history)

    # Save learning curves
    np.savez_compressed(
        save_model_path,
        train_loss_history = loss_history,
        val_loss_history = val_loss_history,
        norm_stats = norm_stats
    )

    fig = plt.figure(figsize=(12,11))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.plot(smoothed_loss_train, color='tab:blue', alpha=0.6, label='train')
    ax1.plot(smoothed_loss_val, 'o', color='tab:orange', markersize=10, alpha=1.0, label='val')
    ax1.set_ylabel('loss value, -', fontsize=12)
    ax1.set_xlabel('step number, -', fontsize=12)
    ax1.legend(fontsize=12)
    ax2.plot(smoothed_loss_train, color='tab:blue', alpha=0.6, label='train')
    ax2.plot(smoothed_loss_val, 'o', color='tab:orange', markersize=10, alpha=1.0, label='val')
    ax2.set_xlabel('step number, -', fontsize=12)
    ax2.set_ylim((0., 0.1))
    fig.savefig(str(MODEL_DIR) + '/' + m_name + '_loss_history.png', dpi=300)


    plt.show()
    print('End')