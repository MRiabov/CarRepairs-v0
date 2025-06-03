import os
import pathlib
import time
import flax
import jax
import jax.numpy as jnp
import numpy as np
from repairs_components.processing.tasks import AssembleTask
import torch
import optax
from flax import nnx
from functools import partial
from matplotlib import pyplot as plt
import flashbax as fbx
from flax.struct import dataclass, PyTreeNode
from orbax.checkpoint import (
    CheckpointManager,
    PyTreeCheckpointer,
    JsonCheckpointHandler,
    CheckpointManagerOptions,
)
import orbax.checkpoint as ocp
import json
from jax import dlpack
from repairs_components.training_utils.gym_env import RepairsEnv

from genesis import gs
from examples.box_to_pos_task import MoveBoxSetup


# jax.config.update("jax_compilation_cache_dir", "/cache/jax_cache")
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update(
#     "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
# )


class SACActor(nnx.Module):
    deterministic: bool = False

    def __init__(
        self,
        action_dim,  # 6-7?
        rngs: nnx.Rngs,
        net_dtype=jnp.bfloat16,
    ):
        # simple 2‐layer MLP, no BatchNorm
        self.voxel_conv_1 = nnx.Conv(
            1, 2, kernel_size=(6, 6, 6), strides=(4, 4, 4), rngs=rngs
        )
        self.voxel_conv_2 = nnx.Conv(
            2, 4, kernel_size=(6, 6, 6), strides=(4, 4, 4), rngs=rngs
        )
        self.voxel_conv_3 = nnx.Conv(
            4, 8, kernel_size=(6, 6, 6), strides=(4, 4, 4), rngs=rngs
        )  # 3*3*3, 8 features - 216 features.
        self.video_conv_1 = nnx.Conv(
            3, 6, kernel_size=(6, 6), strides=(4, 4), rngs=rngs
        )
        self.video_conv_2 = nnx.Conv(
            6, 8, kernel_size=(6, 6), strides=(4, 4), rngs=rngs
        )
        self.video_conv_3 = nnx.Conv(
            8, 12, kernel_size=(6, 6), strides=(4, 4), rngs=rngs
        )  # 3*3*3 (12), 324
        combined_obs_dim = 216 + 324 + electronics_graph_dim[0]
        self.electronic_graph_hidden_1 = nnx.Linear(
            combined_obs_dim, 256, rngs=rngs, dtype=net_dtype
        )
        self.electronic_graph_hidden_2 = nnx.Linear(
            256, 256, rngs=rngs, dtype=net_dtype
        )
        self.out_mean = nnx.Linear(256, action_dim, rngs=rngs, dtype=net_dtype)
        self.out_log_std = nnx.Linear(256, action_dim, rngs=rngs, dtype=net_dtype)

    def __call__(self, voxel_obs, video_obs, robot_obs):
        # Voxel encoding
        x_voxel = nnx.silu(self.voxel_conv_1(voxel_obs))
        x_voxel = nnx.silu(self.voxel_conv_2(x_voxel))
        x_voxel = nnx.silu(self.voxel_conv_3(x_voxel))
        x_voxel = (
            x_voxel.reshape((x_voxel.shape[0], -1)) if x_voxel.ndim > 2 else x_voxel
        )

        # Video encoding
        x_video = nnx.silu(self.video_conv_1(video_obs))
        x_video = nnx.silu(self.video_conv_2(x_video))
        x_video = nnx.silu(self.video_conv_3(x_video))
        x_video = (
            x_video.reshape((x_video.shape[0], -1)) if x_video.ndim > 2 else x_video
        )

        # Concatenate all encodings
        x = jnp.concatenate([x_voxel, x_video, robot_obs], axis=-1)
        x = nnx.silu(self.electronic_graph_hidden_1(x))
        x = nnx.silu(self.electronic_graph_hidden_2(x))
        mean = self.out_mean(x)
        log_std = jnp.clip(self.out_log_std(x), -5.0, 2.0)
        return mean, log_std

    def sample_action(self, voxel_obs, video_obs, electronic_graph_obs, rngs: nnx.Rngs):
        mean, log_std = self.__call__(voxel_obs, video_obs, electronic_graph_obs)
        std = jnp.exp(log_std)
        if self.deterministic:
            pre_tanh = mean
        else:
            noise = jax.random.normal(rngs.action(), mean.shape)
            pre_tanh = mean + noise * std
        action = jnp.tanh(pre_tanh)
        # log‐prob of gaussian plus tanh correction:
        log_prob_gauss = -0.5 * (
            ((pre_tanh - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi)
        )
        log_prob = jnp.sum(log_prob_gauss, axis=-1)
        log_prob -= jnp.sum(jnp.log(1 - action**2 + 1e-6), axis=-1)
        return action, log_prob


class SACCritic(nnx.Module):
    def __init__(
        self,
        act_dim: int,
        electronic_graph_obs_dim,  # Note: compressed!! dim.
        rngs,
        net_dtype=jnp.bfloat16,
    ):
        # simple 2‐layer MLP for each Q‐head
        # Voxel encoder for state
        self.voxel_conv_1 = nnx.Conv(
            2, 4, kernel_size=(6, 6, 6), strides=(4, 4, 4), rngs=rngs
        )
        self.voxel_conv_2 = nnx.Conv(
            4, 8, kernel_size=(6, 6, 6), strides=(4, 4, 4), rngs=rngs
        )
        self.voxel_conv_3 = nnx.Conv(
            8, 12, kernel_size=(6, 6, 6), strides=(4, 4, 4), rngs=rngs
        )  # adjust input channels as needed

        # Video encoder for state
        self.video_conv_1 = nnx.Conv(
            3, 6, kernel_size=(6, 6), strides=(4, 4), rngs=rngs
        )
        self.video_conv_2 = nnx.Conv(
            6, 8, kernel_size=(6, 6), strides=(4, 4), rngs=rngs
        )
        self.video_conv_3 = nnx.Conv(
            8, 12, kernel_size=(6, 6), strides=(4, 4), rngs=rngs
        )

        # Linear layers for state+action encoding
        # Input dim: voxel + video + robot_obs + action
        combined_obs_dim = (
            216 + 324 + electronic_graph_obs_dim + act_dim
        )  # as described above

        self.hidden1 = nnx.Linear(combined_obs_dim, 256, rngs=rngs, dtype=net_dtype)
        self.hidden2 = nnx.Linear(256, 256, rngs=rngs, dtype=net_dtype)
        self.q1 = nnx.Linear(256, 1, rngs=rngs, dtype=net_dtype)
        self.hidden3 = nnx.Linear(combined_obs_dim, 256, rngs=rngs, dtype=net_dtype)
        self.hidden4 = nnx.Linear(256, 256, rngs=rngs, dtype=net_dtype)
        self.q2 = nnx.Linear(256, 1, rngs=rngs, dtype=net_dtype)

    def __call__(self, voxels, video, robot_obs, action):
        # Voxel encoding
        x_voxel = nnx.silu(self.voxel_conv_1(voxels))
        x_voxel = nnx.silu(self.voxel_conv_2(x_voxel))
        x_voxel = nnx.silu(self.voxel_conv_3(x_voxel))
        x_voxel = (
            x_voxel.reshape((x_voxel.shape[0], -1)) if x_voxel.ndim > 2 else x_voxel
        )

        # Video encoding
        x_video = nnx.silu(self.video_conv_1(video))
        x_video = nnx.silu(self.video_conv_2(x_video))
        x_video = nnx.silu(self.video_conv_3(x_video))
        x_video = (
            x_video.reshape((x_video.shape[0], -1)) if x_video.ndim > 2 else x_video
        )

        # Concatenate all observations with action
        x = jnp.concatenate([x_voxel, x_video, robot_obs, action], axis=-1)

        # Twin Q-networks
        # Q1 network
        x1 = nnx.silu(self.hidden1(x))
        x1 = nnx.silu(self.hidden2(x1))
        q1 = self.q1(x1)

        # Q2 network
        x2 = nnx.silu(self.hidden3(x))
        x2 = nnx.silu(self.hidden4(x2))
        q2 = self.q2(x2)

        return q1, q2


def critic_loss_fn(
    critic: SACCritic,
    actor: SACActor,
    critic_target_params,
    s,
    a,
    r,
    d,
    next_s,
    alpha,
    rng,
    gamma=0.99,
):
    video_obs, robot_obs, voxel_obs = s
    next_video_obs, next_robot_obs, next_voxel_obs = next_s
    # next action + logp
    next_a, next_logp = actor.sample_action(
        next_video_obs, next_robot_obs, next_voxel_obs, rng
    )
    # frozen target critic
    graphdef, _ = nnx.split(critic)
    target_critic = nnx.merge(graphdef, critic_target_params)
    q1_t, q2_t = target_critic(next_video_obs, next_robot_obs, next_voxel_obs, next_a)
    q_t = jnp.minimum(q1_t, q2_t) - alpha * next_logp[..., None]
    target_q = jax.lax.stop_gradient(r[..., None] + gamma * (1 - d[..., None]) * q_t)

    # Note: performance can be improved by removing target network and swapping it for batchnorm(see CrossQ)
    q1, q2 = critic(video_obs, robot_obs, voxel_obs, a)
    return jnp.mean((q1 - target_q) ** 2 + (q2 - target_q) ** 2)


def actor_loss_fn(actor, critic, s, alpha, rngs: nnx.Rngs):
    video_obs, robot_obs, voxel_obs = s
    a, logp = actor.sample_action(video_obs, robot_obs, voxel_obs, rngs)
    q1, q2 = critic(video_obs, robot_obs, voxel_obs, a)
    q = jnp.minimum(q1, q2).squeeze(-1)
    return jnp.mean(alpha * logp - q)


def temperature_loss_fn(actor, log_alpha, s, rngs: nnx.Rngs, target_entropy):
    _, logp = actor.sample_action(s, rngs)
    alpha = jnp.exp(log_alpha)
    return -jnp.mean(alpha * (logp + target_entropy))


if __name__ == "__main__":
    # Initialize Genesis
    gs.init(backend=gs.cuda)

    # Create task and environment setup
    task = AssembleTask()
    env_setup = MoveBoxSetup()

    # Environment configuration
    env_cfg = {
        "num_actions": 9,  # [x, y, z, quat_w, quat_x, quat_y, quat_z, gripper_force_left, gripper_force_right]
        "joint_names": [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
            "finger_joint1",
            "finger_joint2",
        ],
        "default_joint_angles": {
            "joint1": 0.0,
            "joint2": -0.3,
            "joint3": 0.0,
            "joint4": -2.0,
            "joint5": 0.0,
            "joint6": 2.0,
            "joint7": 0.79,  # no "hand" here? there definitely was hand.
            "finger_joint1": 0.04,
            "finger_joint2": 0.04,
        },
    }

    obs_cfg = {
        "num_obs": 3,  # RGB, depth, segmentation
        "res": (640, 480),
    }

    reward_cfg = {
        "success_reward": 10.0,
        "progress_reward_scale": 1.0,
        "progressive": True,  # TODO : if progressive, use progressive reward calc instead.
    }

    command_cfg = {}

    # Create gym environment
    print("Creating gym environment...")
    env = RepairsEnv(
        env_setup=env_setup,
        tasks=[task],
        num_envs=2,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
        num_scenes_per_task=1,
    )
    action_dim = env_cfg["num_actions"]
    num_cameras = 2
    vision_obs_dim = (num_cameras, 256, 256)  # 2 cameras
    electronics_graph_dim = (10,)  # fill in when ready
    voxel_obs_dim = (2, 256, 256, 256)  # start and finish # should be sparse?

    batch_size = 16 if jax.default_backend() == "cpu" else 256
    train_steps = (10_000_000 if jax.default_backend() == "gpu" else 3000) // batch_size
    buffer_size = 200_000
    sample_batch_size = 256

    rngs = nnx.Rngs(0, action=1, env=2, buffer=3)

    buffer_key = rngs.buffer()
    batch_rng = jax.random.split(rngs.env(), batch_size)

    buffer = fbx.make_flat_buffer(
        buffer_size, min_length=10000, sample_batch_size=256, add_batch_size=batch_size
    )
    buffer_state = buffer.init(
        {
            "video_obs": jnp.zeros(vision_obs_dim, dtype=jnp.int8),
            "electronics_graph_obs": jnp.zeros(
                electronics_graph_dim, dtype=jnp.float32
            ),
            # "voxel_obs": jnp.zeros(voxel_obs_dim, dtype=jnp.int8), #make it static for now.
            "reward": jnp.zeros((), jnp.float32),
            "action": jnp.zeros((action_dim,), dtype=jnp.float32),
            "done": jnp.zeros((), jnp.bool_),
        }
    )
    # NOTE: correct order of observations: video, robot, voxel. Everything else is a mistake.

    actor = SACActor(
        action_dim=action_dim,
        rngs=rngs,
    )
    critic = SACCritic(action_dim, electronics_graph_dim, rngs)

    actor.train()
    critic.train()
    _critic_graph_def, critic_target_params = nnx.split(critic)

    target_entropy = -action_dim

    # Using pure_callback to interface with PyTorch-based RepairsEnv
    def env_reset(key: jax.Array):
        # Function to reset environment in PyTorch land
        def _reset_impl(rng_key: jax.Array):
            obs, info = env.reset(torch.from_dlpack(jax.dlpack.to_dlpack(rng_key)))
            video_obs = jax.dlpack.from_dlpack(torch.to_dlpack(obs))
            return video_obs  # it also returns the desired state...

        # Convert back to JAX arrays
        return jax.pure_callback(
            _reset_impl,
            (
                jnp.zeros(
                    (env.num_envs, len(env.cameras), *env.obs_cfg["res"], 7),
                    dtype=jnp.float32,
                ),
                None,
            ),
            key,
        )

    def env_step(
        actions: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict]:
        # Function to step environment in PyTorch land
        def external_step(actions_jnp: jax.Array):
            # Convert jax.Array actions to torch tensors
            # dlpack is a library for converting arrays between different frameworks. Faster than torch.array or similar.
            actions_torch = torch.from_dlpack(
                jax.dlpack.to_dlpack(actions_jnp),
            )  # note: as of jax 0.6.0 jax.dlpack.to_dlpack is unnecessary, simply pass jax array directly to from_dlpack of torch.
            # Execute step in environment
            obs, rewards, dones, info = env.step(actions_torch)
            # Convert torch tensors to numpy arrays
            video_obs = jax.dlpack.from_dlpack(torch.to_dlpack(obs))
            rewards = jax.dlpack.from_dlpack(torch.to_dlpack(rewards))
            dones = jax.dlpack.from_dlpack(torch.to_dlpack(dones))
            # Return observations, rewards, dones in expected format
            electronics_graph_obs = jnp.zeros(
                (batch_size,) + electronics_graph_dim, dtype=jnp.float32
            )
            return (video_obs, electronics_graph_obs, rewards, dones, {})

        # Convert back to JAX arrays
        return jax.pure_callback(
            external_step,
            (
                jnp.zeros((batch_size,) + vision_obs_dim, dtype=jnp.float32),
                None,  # robot obs placeholder
                jnp.zeros((batch_size,), dtype=jnp.float32),
                jnp.zeros((batch_size,), dtype=jnp.bool_),
                {},
            ),
            actions,
        )

    # Initialize environment - only need to get observations
    video_obs, electronics_graph_obs = env_reset(batch_rng)
    # Initialize voxel observations with zeros # placeholder
    voxel_obs = jnp.zeros((batch_size,) + voxel_obs_dim, dtype=jnp.int8)

    log_alpha = jnp.array(jnp.log(0.2))

    # Restore with flax.nnx checkpoint manager
    checkpoint_dir = pathlib.Path(
        "/home/mriabov/Work/Projects/learning-deep-rl/checkpoints_brax_ant_sac"
    )

    # ocp.handlers.StandardCheckpointHandler()
    # ocp.handlers.DefaultCheckpointHandlerRegistry()
    # ocp.CheckpointManager( # could be also done with CheckpointManager, since it's a wrapper over a bunch of logic.
    #     directory=checkpoint_dir,
    #     handler_registry=
    # )
    checkpointer = ocp.StandardCheckpointer()
    # Setup flax.nnx checkpoint manager for actor, critic, critic_target_params, log_alpha
    # Restore if checkpoint exists
    if (
        os.path.exists(checkpoint_dir)
        and (checkpoint_dir / "actor").exists()
        and (checkpoint_dir / "log_alpha.json").exists()
    ):
        actor_graphdef, actor_params = nnx.split(actor)
        critic_graphdef, critic_params = nnx.split(critic)

        actor_ckpt = checkpointer.restore(checkpoint_dir / "actor", actor_params)
        critic_ckpt = checkpointer.restore(checkpoint_dir / "critic", critic_params)
        critic_target_ckpt = checkpointer.restore(
            checkpoint_dir / "critic_target_params", critic_params
        )

        # merge restored states into modules
        actor = nnx.merge(actor_graphdef, actor_ckpt)
        critic = nnx.merge(critic_graphdef, critic_ckpt)
        critic_target = nnx.merge(critic_graphdef, critic_target_ckpt)
        critic_graphdef, critic_target_params = nnx.split(critic_target)
        with open(checkpoint_dir / "log_alpha.json") as f:
            log_alpha_ckpt = json.load(f)
        log_alpha = jnp.array(log_alpha_ckpt["log_alpha"])
    else:
        print(f"No checkpoint found at {checkpoint_dir}, using defaults.")

    alpha_optimizer = optax.adam(3e-4)
    critic_optimizer = nnx.Optimizer(critic, optax.adam(3e-4))
    actor_optimizer = nnx.Optimizer(actor, optax.adam(3e-4))
    alpha_opt_state = alpha_optimizer.init(log_alpha)

    @partial(jax.jit, donate_argnums=(0,))
    def fill_buffer_step(carry, _):
        (
            buffer_env_states,
            (
                prev_video_obs,
                prev_electronics_graph_obs,
            ),  # not sure if it's correct that it's unused. *it was*.
            buffer_state,
            rng,
        ) = carry

        rng, action_rng, step_rng = jax.random.split(rng, 3)

        random_actions = jax.random.uniform(
            action_rng, shape=(batch_size, env.action_size), minval=-1.0, maxval=1.0
        )

        new_obs, electronics_graph_obs, rewards, dones, _infos = env_step(
            random_actions
        )

        # reset_video_obs, reset_electronics_graph_obs = env_reset(batch_rng)
        # new_video_obs = jnp.where(dones[:, None], reset_video_obs, new_obs)
        # new_electronics_graph_obs = jnp.where(
        #     dones[:, None], reset_electronics_graph_obs, electronics_graph_obs
        # ) # done by torch, implicitly.

        buffer_state = buffer.add(
            buffer_state,
            {
                "video_obs": new_obs,
                "electronics_graph_obs": electronics_graph_obs,
                # "voxel_obs": prev_obs,
                "action": random_actions,
                "reward": rewards,
                "done": dones.astype(jnp.bool_),
            },
        )

        return (
            buffer_env_states,
            (new_video_obs, new_electronics_graph_obs),
            buffer_state,
            rng,
        ), None

    # With the RepairsEnv, environment state is managed internally by the environment
    # We only need to track observations
    fill_buffer_carry = (
        (video_obs, electronics_graph_obs),
        buffer_state,
        rngs.env(),
    )

    fill_steps = 10000 // batch_size
    fill_buffer_carry, _ = jax.lax.scan(
        fill_buffer_step, fill_buffer_carry, None, length=fill_steps
    )

    obs, buffer_state, env_rng = fill_buffer_carry
    reset_video_obs, reset_electronic_graph_obs = env_reset(batch_rng)
    reset_obs = (reset_video_obs, reset_electronic_graph_obs)

    batch_rng = jax.random.split(env_rng, batch_size)

    carry = (
        # None,  # No need to track env states explicitly
        reset_obs,
        nnx.State(buffer_state),
        rngs,
        actor,
        critic,
        critic_target_params,
        0,
        log_alpha,
        actor_optimizer,
        critic_optimizer,
        alpha_opt_state,
        # buffer,
        jnp.zeros((batch_size,)),
    )

    @nnx.jit
    def train_step(carry):
        (
            # old_env_states,
            (
                prev_video_obs,
                prev_electronic_graph_obs,
                # voxel_obs,
            ),  # TODO check that it is in correct order.
            buffer_state,
            rngs,
            actor,
            critic,
            critic_target_params,
            time,
            log_alpha,
            actor_optimizer,
            critic_optimizer,
            alpha_opt_state,
            episode_cumulative_reward,
        ) = carry
        actor: SACActor  # type hints
        critic: SACCritic

        # unwrap the buffer_state nnx.State to get the raw TrajectoryBufferState back.
        experience = buffer_state.experience
        buffer_state = fbx.trajectory_buffer.TrajectoryBufferState(
            experience={
                "action": experience["action"],
                "done": experience["done"],
                "electronic_graph_obs": experience["electronic_graph_obs"],
                "video_obs": experience["video_obs"],
                "reward": experience["reward"],
            },
            current_index=buffer_state.current_index,
            is_full=buffer_state.is_full,
        )

        batch_rng = jax.random.split(rngs.env(), batch_size)
        action, _log_prob = actor.sample_action(
            prev_video_obs, prev_electronic_graph_obs, voxel_obs, rngs
        )

        video_obs, electronic_graph_obs, reward, done, _infos = env_step(action)
        assert video_obs.shape == video_obs.shape, (
            f"Obs dim mismatch: {video_obs.shape} vs {video_obs.shape}"
        )

        episode_cumulative_reward = jnp.where(
            done, 0.0, episode_cumulative_reward + reward
        )

        batch_rng = jax.random.split(rngs.env(), batch_size)
        reset_video_obs, reset_robot_obs = env_reset(batch_rng)
        new_video_obs = jnp.where(done[:, None], reset_video_obs, video_obs)
        new_electronic_graph_obs = jnp.where(
            done[:, None], reset_robot_obs, electronic_graph_obs
        )
        # voxel obs is static (yet)

        # Environment state handling is now managed by RepairsEnv
        buffer_state = buffer.add(
            buffer_state,
            {
                "video_obs": prev_video_obs,
                "electronic_graph_obs": prev_electronic_graph_obs,
                "action": action.astype(jnp.float32),
                "reward": reward,
                "done": done,
            },
        )

        batch = buffer.sample(buffer_state, rngs.buffer())
        s = (
            batch.experience.first["electronic_graph_obs"],
            batch.experience.first["video_obs"],
            voxel_obs,
        )
        a = batch.experience.first["action"]
        r = batch.experience.first["reward"]
        d = batch.experience.first["done"]
        next_s = (
            batch.experience.second["electronic_graph_obs"],
            batch.experience.second["video_obs"],
            voxel_obs,
        )

        critic_loss, critic_gradients = nnx.value_and_grad(critic_loss_fn, argnums=0)(
            critic,
            actor,
            critic_target_params,
            s,
            a,
            r,
            d,
            next_s,
            jnp.exp(log_alpha),
            rngs,
        )

        critic_optimizer.update(critic_gradients)  # stateful! (works.)

        actor_loss, actor_gradients = nnx.value_and_grad(actor_loss_fn, argnums=0)(
            actor, critic, s, jnp.exp(log_alpha), rngs
        )
        actor_optimizer.update(actor_gradients)

        temp_loss, temp_grad = nnx.value_and_grad(temperature_loss_fn, argnums=1)(
            actor, log_alpha, s, rngs, target_entropy
        )
        alpha_update, alpha_opt_state = alpha_optimizer.update(
            temp_grad, alpha_opt_state
        )
        log_alpha = optax.apply_updates(log_alpha, alpha_update)

        tau = 0.005
        critic_graphdef, critic_params = nnx.split(critic)
        critic_target_params = optax.incremental_update(
            critic_params, critic_target_params, tau
        )

        expected_episode_len = 1 / jnp.mean(d.astype(jnp.float32))

        jax.lax.cond(
            time % (2000 if jax.default_backend() == "gpu" else 200) == 0,
            lambda step=time,
            critic_loss=critic_loss,
            actor_loss=actor_loss,
            r=reward.mean(): jax.debug.print(
                "Training step {step}/100, critic_loss {critic_loss}, actor_loss {actor_loss}, reward {r}, cumulative_reward {cumulative_reward}, 1/failure_rate {failure_rate}",
                step=step,
                critic_loss=critic_loss,
                actor_loss=actor_loss,
                r=r,
                cumulative_reward=episode_cumulative_reward.mean(),
                failure_rate=expected_episode_len,
            ),
            lambda: None,
        )
        time = time + 1

        # skipping for now because it's unjittable.
        # jax.lax.cond(
        #     time % 624 == 0,  # write buffer to disk every 10000/batch_size-1 steps
        #     lambda buffer_state: vault.write(buffer_state),
        #     lambda _: None,
        #     buffer_state,
        # )

        # buffer_train_state = buffer_train_state.replace(buffer_state=buffer_state)
        # buffer_state_graphdef, buffer_state_tree = nnx.split(buffer_state)

        # cast back up
        # No need to cast metrics with RepairsEnv

        return (
            (
                (new_video_obs, new_electronic_graph_obs, voxel_obs),
                nnx.State(buffer_state),
                rngs,
                actor,
                critic,
                critic_target_params,
                time,
                log_alpha,
                actor_optimizer,
                critic_optimizer,
                alpha_opt_state,
                # buffer,
                episode_cumulative_reward,
            ),
            critic_loss,
            actor_loss,
            reward,
            done,
            expected_episode_len,
        )

    train_time_start = time.time()

    last_carry, critic_loss, actor_loss, rewards, dones, expected_episode_len = nnx.jit(
        nnx.scan(
            train_step,
            in_axes=nnx.Carry,
            out_axes=(nnx.Carry, 0, 0, 0, 0, 0),
            length=train_steps,
        )
    )(carry)

    # reward processing:
    # average cumulative reward per step:
    sum_rewards_where_done = jnp.sum(rewards, axis=0, where=dones[:, None])
    mean_of_cumulative_rewards = sum_rewards_where_done.mean(axis=-1)
    # smoothed_out_rewards = jnp.reshape(-1, 20).mean().ravel()

    # Simplified episode length calculation
    _, episode_lengths = jax.vmap(
        lambda r, d: jax.lax.scan(
            lambda c, x: jax.lax.cond(
                x[1],
                lambda c, r: (c + r, 0.0),
                lambda c, r: (0.0, c),
                c,
                x[0],
            ),  # cumsum with reset on done.
            0.0,
            (r, d),
        )
    )(rewards, dones)
    average_true_episode_length = jnp.sum(episode_lengths, axis=-1) / jnp.sum(
        dones, axis=-1
    )
    print("True episode lengths shape:", average_true_episode_length.shape)
    # print("Episode lengths shape:", episode_lengths.shape)
    print("mean_of_cumulative_rewards:", mean_of_cumulative_rewards.shape)

    # in the last 10 steps
    assert not (critic_loss[-10:] > 10e16).any(), "Losses have exploded."
    assert not (actor_loss[-10:] > 10e16).any(), "Losses have exploded."

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpointer.save(checkpoint_dir / "actor", nnx.split(last_carry[4])[1], force=True)
    checkpointer.save(
        checkpoint_dir / "critic",
        nnx.split(last_carry[5])[1],
        force=True,
    )
    checkpointer.save(
        checkpoint_dir / "critic_target_params", last_carry[6], force=True
    )

    with open(os.path.join(checkpoint_dir, "log_alpha.json"), "w") as f:
        json.dump({"log_alpha": float(log_alpha)}, f)
    print(f"Final checkpoint saved to {checkpoint_dir}")

    # backup flashbax buffer
    # unwrap the buffer_state nnx.State to get the raw TrajectoryBufferState back.
    experience = buffer_state.experience
    buffer_state = fbx.trajectory_buffer.TrajectoryBufferState(
        experience={
            "action": experience["action"],
            "done": experience["done"],
            "obs": experience["obs"],
            "reward": experience["reward"],
        },
        current_index=buffer_state.current_index,
        is_full=buffer_state.is_full,
    )

    critic_loss = np.asarray(critic_loss)
    actor_loss = np.asarray(actor_loss)
    # cum_reward = np.asarray(cum_reward[jnp.arange(98, cum_reward.shape[0], 100)])

    actor.eval()
    critic.eval()

    print("training took", time.time() - train_time_start, "seconds")
    # JIT‑compile and evaluation loop with a while‑loop
    max_steps = 1000

    @nnx.jit
    def eval_episode(rng, actor):
        # carry = (state, total_reward, steps, rng)
        def cond_fun(carry):
            state, total_reward, steps, _ = carry
            # continue while not done and steps < max_steps
            return jnp.logical_and((1 - state.done), (steps < max_steps))

        def body_fun(carry):
            state, total_reward, steps, rng = carry
            # split RNG for next step and for sampling action
            rng, action_rng = jax.random.split(rng)
            rngs_step = nnx.Rngs(rng, action=action_rng)
            # sample action
            action, _ = actor.sample_action(state.obs[None, :], rngs_step)
            action = jnp.squeeze(action, axis=0)
            # step environment
            next_state = env.step(state, action)
            # cast up because it throws otherwise.
            next_state.metrics["reward_ctrl"] = next_state.metrics[
                "reward_ctrl"
            ].astype(jnp.float32)

            return (
                next_state,
                total_reward + next_state.reward,
                steps + 1,
                rng,
            )

        # initialize
        init_state = env.reset(rng)
        init_carry = (init_state, 0.0, 0, rng)
        # run until done or max_steps
        final_state, total_reward, steps, _ = jax.lax.while_loop(
            cond_fun, body_fun, init_carry
        )
        return total_reward, steps

    # run several episodes in parallel
    num_eval = 16
    rngs_eval = jax.random.split(rngs.eval(), num_eval)
    eval_rewards, eval_steps = nnx.vmap(
        lambda r, actor: eval_episode(r, actor), in_axes=(0, None)
    )(rngs_eval, actor)

    # print results
    for i in range(num_eval):
        print(
            f"Brax Ant eval episode {i}: steps={eval_steps[i]}, "
            f"total_reward={eval_rewards[i]:.2f}"
        )
    plt.figure(figsize=(10, 4))
    # plt.plot(np.array(rewards), label="Rewards")
    plt.plot(np.array(mean_of_cumulative_rewards), label="Cumulative Rewards")
    plt.plot(np.array(average_true_episode_length), label="Episode Lengths")
    plt.legend()
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Brax Ant Evaluation Episode Reward Sequence")
    plt.show()

    actor.eval()

    @nnx.jit
    def inference_fn(actor, state, rngs):
        action, _ = actor.sample_action(state.obs[None, :], rngs)
        state = env.step(state, action[0])
        return state

    # create an env with auto-reset
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    rollout = []

    state = jit_env_reset(rngs.env())
    for _ in range(1000):
        rollout.append(state.pipeline_state)
        state = inference_fn(actor, state, rngs)
