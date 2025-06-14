import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchrl.data.replay_buffers import TensorDictReplayBuffer, TensorStorage


class SACActor(nn.Module):
    """
    PyTorch implementation of SAC Actor network.
    """

    def __init__(self, action_dim, electronics_graph_dim, device=None):
        super(SACActor, self).__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # Voxel encoder (3D conv layers)
        self.voxel_conv1 = nn.Conv3d(1, 2, kernel_size=(6, 6, 6), stride=(4, 4, 4))
        self.voxel_conv2 = nn.Conv3d(2, 4, kernel_size=(6, 6, 6), stride=(4, 4, 4))
        self.voxel_conv3 = nn.Conv3d(4, 8, kernel_size=(6, 6, 6), stride=(4, 4, 4))
        # Video encoder (2D conv layers)
        self.video_conv1 = nn.Conv2d(3, 6, kernel_size=(6, 6), stride=(4, 4))
        self.video_conv2 = nn.Conv2d(6, 8, kernel_size=(6, 6), stride=(4, 4))
        self.video_conv3 = nn.Conv2d(8, 12, kernel_size=(6, 6), stride=(4, 4))
        # Graph observations MLP
        combined_dim = 216 + 324 + electronics_graph_dim
        self.fc1 = nn.Linear(combined_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out_mean = nn.Linear(256, action_dim)
        self.out_log_std = nn.Linear(256, action_dim)

    def forward(self, voxel_obs, video_obs, graph_obs):
        # assume voxel_obs shape [B, 1, D, H, W]
        x_v = F.silu(self.voxel_conv1(voxel_obs))
        x_v = F.silu(self.voxel_conv2(x_v))
        x_v = F.silu(self.voxel_conv3(x_v))
        x_v = x_v.view(x_v.size(0), -1)
        # assume video_obs shape [B, C, H, W]
        x_vid = F.silu(self.video_conv1(video_obs))
        x_vid = F.silu(self.video_conv2(x_vid))
        x_vid = F.silu(self.video_conv3(x_vid))
        x_vid = x_vid.view(x_vid.size(0), -1)
        # concatenate all features
        x = torch.cat([x_v, x_vid, graph_obs], dim=-1)
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        mean = self.out_mean(x)
        log_std = torch.clamp(self.out_log_std(x), -5.0, 2.0)
        return mean, log_std

    def sample_action(self, voxel_obs, video_obs, graph_obs, deterministic=False):
        mean, log_std = self.forward(voxel_obs, video_obs, graph_obs)
        std = log_std.exp()
        if deterministic:
            pre_tanh = mean
        else:
            noise = torch.randn_like(mean)
            pre_tanh = mean + noise * std
        action = torch.tanh(pre_tanh)
        log_prob = -0.5 * (
            ((pre_tanh - mean) / std) ** 2
            + 2 * log_std
            + torch.log(torch.tensor(2 * torch.pi))
        )
        # correction for tanh
        log_prob = log_prob.sum(dim=-1, keepdim=True) - torch.log(
            1 - action.pow(2) + 1e-6
        ).sum(dim=-1, keepdim=True)
        return action, log_prob


class SACCritic(nn.Module):
    """
    PyTorch implementation of twin Q-function critic.
    """

    def __init__(self, action_dim, electronics_graph_dim, device=None):
        super(SACCritic, self).__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # Shared conv encoders for Q1
        self.conv3d_q1 = nn.Sequential(
            nn.Conv3d(1, 2, kernel_size=(6, 6, 6), stride=(4, 4, 4)),
            nn.SiLU(),
            nn.Conv3d(2, 4, kernel_size=(6, 6, 6), stride=(4, 4, 4)),
            nn.SiLU(),
            nn.Conv3d(4, 8, kernel_size=(6, 6, 6), stride=(4, 4, 4)),
            nn.SiLU(),
        )
        self.conv2d_q1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=(6, 6), stride=(4, 4)),
            nn.SiLU(),
            nn.Conv2d(6, 8, kernel_size=(6, 6), stride=(4, 4)),
            nn.SiLU(),
            nn.Conv2d(8, 12, kernel_size=(6, 6), stride=(4, 4)),
            nn.SiLU(),
        )
        combined_q_dim = 216 + 324 + electronics_graph_dim + action_dim
        self.q1_fc = nn.Sequential(
            nn.Linear(combined_q_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )
        # Twin Q2
        self.conv3d_q2 = nn.Sequential(
            nn.Conv3d(1, 2, kernel_size=(6, 6, 6), stride=(4, 4, 4)),
            nn.SiLU(),
            nn.Conv3d(2, 4, kernel_size=(6, 6, 6), stride=(4, 4, 4)),
            nn.SiLU(),
            nn.Conv3d(4, 8, kernel_size=(6, 6, 6), stride=(4, 4, 4)),
            nn.SiLU(),
        )
        self.conv2d_q2 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=(6, 6), stride=(4, 4)),
            nn.SiLU(),
            nn.Conv2d(6, 8, kernel_size=(6, 6), stride=(4, 4)),
            nn.SiLU(),
            nn.Conv2d(8, 12, kernel_size=(6, 6), stride=(4, 4)),
            nn.SiLU(),
        )
        self.q2_fc = nn.Sequential(
            nn.Linear(combined_q_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

    def forward(self, voxel_obs, video_obs, graph_obs, action):
        # Encoder Q1
        x_v1 = self.conv3d_q1(voxel_obs).view(voxel_obs.size(0), -1)
        x_vid1 = self.conv2d_q1(video_obs).view(video_obs.size(0), -1)
        x1 = torch.cat([x_v1, x_vid1, graph_obs, action], dim=-1)
        q1 = self.q1_fc(x1)
        # Encoder Q2
        x_v2 = self.conv3d_q2(voxel_obs).view(voxel_obs.size(0), -1)
        x_vid2 = self.conv2d_q2(video_obs).view(video_obs.size(0), -1)
        x2 = torch.cat([x_v2, x_vid2, graph_obs, action], dim=-1)
        q2 = self.q2_fc(x2)
        return q1, q2


# ===== SAC Trainer =====
class SACTrainer:
    def __init__(
        self,
        action_dim,
        electronics_graph_dim,
        device=None,
        gamma=0.99,
        tau=0.005,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        buffer_size=100000,
        batch_size=256,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.actor = SACActor(action_dim, electronics_graph_dim).to(self.device)
        self.critic = SACCritic(action_dim, electronics_graph_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.target_entropy = -action_dim
        self.gamma = gamma
        self.tau = tau
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        # TorchRL replay buffer
        tensordict = {
            "electronics_graph_obs": torch.zeros(
                electronics_graph_dim,
                dtype=torch.bool,  # bool for now
            ),  # NOTE: this in particular better be done via a list of PyG batches. But later.
            # "voxel_obs": jnp.zeros(voxel_obs_dim, dtype=jnp.int8), #make it static for now.
            "reward": torch.zeros((), dtype=torch.bfloat16),
            "action": torch.zeros((action_dim,), dtype=torch.bfloat16),
            "done": torch.zeros((), dtype=torch.bool),
        }
        self.buffer_storage = TensorStorage(
            storage=tensordict, max_size=buffer_size, device=self.device
        )
        self.replay_buffer = TensorDictReplayBuffer(
            storage=self.buffer_storage, batch_size=batch_size, sampler="random"
        )

    def select_action(self, voxel_obs, video_obs, graph_obs, deterministic=False):
        self.actor.eval()
        with torch.no_grad():
            action, _ = self.actor(
                voxel_obs.to(self.device),
                video_obs.to(self.device),
                graph_obs.to(self.device),
            )
        self.actor.train()
        return action.cpu()

    def update(self, batch):
        v, vid, g, a, r, nv, nvid, ng, d = batch
        v, vid, g, a, r, nv, nvid, ng, d = [
            x.to(self.device) for x in (v, vid, g, a, r, nv, nvid, ng, d)
        ]
        with torch.no_grad():
            na, nlp = self.actor.sample_action(nv, nvid, ng)
            q1n, q2n = self.critic_target(nv, nvid, ng, na)
            qn = torch.min(q1n, q2n) - self.log_alpha.exp() * nlp
            target = r.unsqueeze(-1) + self.gamma * (1 - d.unsqueeze(-1)) * qn
        q1, q2 = self.critic(v, vid, g, a)
        cl = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad()
        cl.backward()
        self.critic_optimizer.step()
        a2, lp = self.actor.sample_action(v, vid, g)
        q1n, q2n = self.critic(v, vid, g, a2)
        al = (self.log_alpha.exp() * lp - torch.min(q1n, q2n)).mean()
        self.actor_optimizer.zero_grad()
        al.backward()
        self.actor_optimizer.step()
        xal = -(self.log_alpha * (lp + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        xal.backward()
        self.alpha_optimizer.step()
        for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        return cl.item(), al.item(), self.log_alpha.exp().item()


# end of SACTrainer

# ===== Training Orchestrator =====
def run_training(
    env_setups,
    tasks,
    env_cfg,
    obs_cfg,
    reward_cfg,
    command_cfg,
    batch_dim,
    action_dim,
    electronics_graph_dim,
    num_steps=10000,
    prefill_steps=1000,
):
    """
    Orchestrates environment interaction, replay buffer filling, and training steps.
    """
    from repairs_components.training_utils.gym_env import RepairsEnv

    env = RepairsEnv(
        env_setups=env_setups,
        tasks=tasks,
        batch_dim=batch_dim,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
    )
    trainer = SACTrainer(action_dim, electronics_graph_dim)

    obs, _ = env.reset()
    voxel_obs, video_obs, graph_obs = obs

    # Prefill buffer with random actions
    for _ in range(prefill_steps):
        rand_action = torch.randn(batch_dim, action_dim, device=trainer.device)
        next_obs, reward, done, _ = env.step(rand_action)
        nv, nvid, ng = next_obs
        trainer.replay_buffer.add({
            "voxel_obs": voxel_obs,
            "video_obs": video_obs,
            "electronics_graph_obs": graph_obs,
            "action": rand_action,
            "reward": reward,
            "done": done,
            "next_voxel_obs": nv,
            "next_video_obs": nvid,
            "next_electronics_graph_obs": ng,
        })
        voxel_obs, video_obs, graph_obs = nv, nvid, ng

    # Main training loop
    for step in range(num_steps):
        action = trainer.select_action(voxel_obs, video_obs, graph_obs)
        next_obs, reward, done, _ = env.step(action)
        nv, nvid, ng = next_obs
        trainer.replay_buffer.add({
            "voxel_obs": voxel_obs,
            "video_obs": video_obs,
            "electronics_graph_obs": graph_obs,
            "action": action.to(trainer.device),
            "reward": reward,
            "done": done,
            "next_voxel_obs": nv,
            "next_video_obs": nvid,
            "next_electronics_graph_obs": ng,
        })
        if step >= prefill_steps:
            batch = trainer.replay_buffer.sample()
            cl, al, alpha = trainer.update(
                (
                    batch["voxel_obs"],
                    batch["video_obs"],
                    batch["electronics_graph_obs"],
                    batch["action"],
                    batch["reward"],
                    batch["next_voxel_obs"],
                    batch["next_video_obs"],
                    batch["next_electronics_graph_obs"],
                    batch["done"],
                )
            )
            if step % 1000 == 0:
                print(f"Step {step}: critic_loss={cl}, actor_loss={al}, alpha={alpha}")
        voxel_obs, video_obs, graph_obs = nv, nvid, ng

if __name__ == "__main__":
    # Configure environment and tasks
    from repairs_components.processing.tasks import AssembleTask, DisassembleTask
    # TODO: import your EnvSetup class for environment configurations

    env_setups = []  # e.g., [EnvSetup(...), ...]
    tasks = [AssembleTask(), DisassembleTask()]

    # Configuration placeholders
    env_cfg = {}
    obs_cfg = {}
    reward_cfg = {}
    command_cfg = {}

    # Training parameters
    batch_dim = 16
    action_dim = 7
    electronics_graph_dim = 128

    run_training(
        env_setups,
        tasks,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        batch_dim,
        action_dim,
        electronics_graph_dim,
    )

# ===== Training Orchestrator =====
def run_training(
    env_setups,
    tasks,
    env_cfg,
    obs_cfg,
    reward_cfg,
    command_cfg,
    batch_dim,
    action_dim,
    electronics_graph_dim,
    num_steps=10000,
    prefill_steps=1000,
):
    """
    Orchestrates environment interaction, replay buffer filling, and training steps.
    """
    from repairs_components.training_utils.gym_env import RepairsEnv

    env = RepairsEnv(
        env_setups=env_setups,
        tasks=tasks,
        batch_dim=batch_dim,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
    )
    trainer = SACTrainer(action_dim, electronics_graph_dim)

    obs, _ = env.reset()
    voxel_obs, video_obs, graph_obs = obs

    # Prefill with random
    for _ in range(prefill_steps):
        rand_action = torch.randn(batch_dim, action_dim, device=trainer.device)
        next_obs, reward, done, _ = env.step(rand_action)
        nv, nvid, ng = next_obs
        trainer.replay_buffer.add({
            "voxel_obs": voxel_obs,
            "video_obs": video_obs,
            "electronics_graph_obs": graph_obs,
            "action": rand_action,
            "reward": reward,
            "done": done,
            "next_voxel_obs": nv,
            "next_video_obs": nvid,
            "next_electronics_graph_obs": ng,
        })
        voxel_obs, video_obs, graph_obs = nv, nvid, ng

    # Main training loop
    for step in range(num_steps):
        action = trainer.select_action(voxel_obs, video_obs, graph_obs)
        next_obs, reward, done, _ = env.step(action)
        nv, nvid, ng = next_obs
        trainer.replay_buffer.add({
            "voxel_obs": voxel_obs,
            "video_obs": video_obs,
            "electronics_graph_obs": graph_obs,
            "action": action.to(trainer.device),
            "reward": reward,
            "done": done,
            "next_voxel_obs": nv,
            "next_video_obs": nvid,
            "next_electronics_graph_obs": ng,
        })
        if step >= prefill_steps:
            batch = trainer.replay_buffer.sample()
            cl, al, alpha = trainer.update(
                (
                    batch["voxel_obs"],
                    batch["video_obs"],
                    batch["electronics_graph_obs"],
                    batch["action"],
                    batch["reward"],
                    batch["next_voxel_obs"],
                    batch["next_video_obs"],
                    batch["next_electronics_graph_obs"],
                    batch["done"],
                )
            )
            if step % 1000 == 0:
                print(f"Step {step}: critic_loss={cl}, actor_loss={al}, alpha={alpha}")
        voxel_obs, video_obs, graph_obs = nv, nvid, ng

if __name__ == "__main__":
    # Configure environment and tasks
    from repairs_components.processing.tasks import AssembleTask, DisassembleTask
    # TODO: import your EnvSetup class to configure env_setups

    # TODO: Populate env_setups with your environment setup instances
    env_setups = []  # e.g., [EnvSetup(...), ...]
    tasks = [AssembleTask(), DisassembleTask()]

    # TODO: Define your configuration dictionaries
    env_cfg = {}
    obs_cfg = {}
    reward_cfg = {}
    command_cfg = {}

    # Training parameters
    batch_dim = 16
    action_dim = 7
    electronics_graph_dim = 128

    # Start training
    run_training(
        env_setups,
        tasks,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        batch_dim,
        action_dim,
        electronics_graph_dim,
    )

# ===== Training Orchestrator =====
def run_training(
    env_setups,
    tasks,
    env_cfg,
    obs_cfg,
    reward_cfg,
    command_cfg,
    batch_dim,
    action_dim,
    electronics_graph_dim,
    num_steps=10000,
    prefill_steps=1000,
):
    """
    Orchestrates environment interaction, replay buffer filling, and training steps.
    """
    from repairs_components.training_utils.gym_env import RepairsEnv

    env = RepairsEnv(
        env_setups=env_setups,
        tasks=tasks,
        batch_dim=batch_dim,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
    )
    trainer = SACTrainer(action_dim, electronics_graph_dim)

    obs, _ = env.reset()
    voxel_obs, video_obs, graph_obs = obs

    # Prefill
    for _ in range(prefill_steps):
        rand_action = torch.randn(batch_dim, action_dim, device=trainer.device)
        next_obs, reward, done, _ = env.step(rand_action)
        nv, nvid, ng = next_obs
        trainer.replay_buffer.add({
            "voxel_obs": voxel_obs,
            "video_obs": video_obs,
            "electronics_graph_obs": graph_obs,
            "action": rand_action,
            "reward": reward,
            "done": done,
            "next_voxel_obs": nv,
            "next_video_obs": nvid,
            "next_electronics_graph_obs": ng,
        })
        voxel_obs, video_obs, graph_obs = nv, nvid, ng

    # Training loop
    for step in range(num_steps):
        action = trainer.select_action(voxel_obs, video_obs, graph_obs)
        next_obs, reward, done, _ = env.step(action)
        nv, nvid, ng = next_obs
        trainer.replay_buffer.add({
            "voxel_obs": voxel_obs,
            "video_obs": video_obs,
            "electronics_graph_obs": graph_obs,
            "action": action.to(trainer.device),
            "reward": reward,
            "done": done,
            "next_voxel_obs": nv,
            "next_video_obs": nvid,
            "next_electronics_graph_obs": ng,
        })
        if step >= prefill_steps:
            batch = trainer.replay_buffer.sample()
            cl, al, alpha = trainer.update(
                (
                    batch["voxel_obs"],
                    batch["video_obs"],
                    batch["electronics_graph_obs"],
                    batch["action"],
                    batch["reward"],
                    batch["next_voxel_obs"],
                    batch["next_video_obs"],
                    batch["next_electronics_graph_obs"],
                    batch["done"],
                )
            )
            if step % 1000 == 0:
                print(f"Step {step}: critic_loss={cl}, actor_loss={al}, alpha={alpha}")
        voxel_obs, video_obs, graph_obs = nv, nvid, ng

if __name__ == "__main__":
    from repairs_components.processing.tasks import AssembleTask, DisassembleTask
    from repairs_components.training_utils.env_setup import EnvSetup

    # Setup environments and tasks
    env_setups = []  # TODO: fill with EnvSetup instances
    tasks = [AssembleTask(), DisassembleTask()]

    # Config placeholders
    env_cfg = {}
    obs_cfg = {}
    reward_cfg = {}
    command_cfg = {}

    batch_dim = 16
    action_dim = 7
    electronics_graph_dim = 128

    run_training(
        env_setups,
        tasks,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        batch_dim,
        action_dim,
        electronics_graph_dim,
    )

# ===== Training Orchestrator =====
def run_training(
    env_setups,
    tasks,
    env_cfg,
    obs_cfg,
    reward_cfg,
    command_cfg,
    batch_dim,
    action_dim,
    electronics_graph_dim,
    num_steps=10000,
    prefill_steps=1000,
):
    """
    Orchestrates environment interaction, replay buffer filling, and training steps.
    """
    # Lazy import to avoid circular dependency
    from repairs_components.training_utils.gym_env import RepairsEnv

    # Initialize environment and trainer
    env = RepairsEnv(
        env_setups=env_setups,
        tasks=tasks,
        batch_dim=batch_dim,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
    )
    trainer = SACTrainer(action_dim, electronics_graph_dim)

    # Reset env
    obs, _ = env.reset()
    voxel_obs, video_obs, graph_obs = obs

    # Prefill buffer with random actions
    for _ in range(prefill_steps):
        rand_action = torch.randn(batch_dim, action_dim, device=trainer.device)
        next_obs, reward, done, _ = env.step(rand_action)
        next_voxel, next_video, next_graph = next_obs
        trainer.replay_buffer.add(
            {
                "voxel_obs": voxel_obs,
                "video_obs": video_obs,
                "electronics_graph_obs": graph_obs,
                "action": rand_action,
                "reward": reward,
                "done": done,
                "next_voxel_obs": next_voxel,
                "next_video_obs": next_video,
                "next_electronics_graph_obs": next_graph,
            }
        )
        voxel_obs, video_obs, graph_obs = next_voxel, next_video, next_graph

    # Main training loop
    for step in range(num_steps):
        # collect and train
        action = trainer.select_action(voxel_obs, video_obs, graph_obs)
        next_obs, reward, done, _ = env.step(action)
        next_voxel, next_video, next_graph = next_obs
        trainer.replay_buffer.add(
            {
                "voxel_obs": voxel_obs,
                "video_obs": video_obs,
                "electronics_graph_obs": graph_obs,
                "action": action.to(trainer.device),
                "reward": reward,
                "done": done,
                "next_voxel_obs": next_voxel,
                "next_video_obs": next_video,
                "next_electronics_graph_obs": next_graph,
            }
        )
        if step >= prefill_steps:
            batch = trainer.replay_buffer.sample()
            cl, al, alpha = trainer.update(
                (
                    batch["voxel_obs"],
                    batch["video_obs"],
                    batch["electronics_graph_obs"],
                    batch["action"],
                    batch["reward"],
                    batch["next_voxel_obs"],
                    batch["next_video_obs"],
                    batch["next_electronics_graph_obs"],
                    batch["done"],
                )
            )
            if step % 1000 == 0:
                print(f"Step {step}: critic_loss={cl}, actor_loss={al}, alpha={alpha}")
        voxel_obs, video_obs, graph_obs = next_voxel, next_video, next_graph

if __name__ == "__main__":
    # Configure environment and tasks
    from repairs_components.processing.tasks import AssembleTask, DisassembleTask
    from repairs_components.training_utils.env_setup import EnvSetup

    # TODO: Populate env_setups with EnvSetup instances
    env_setups = []  # e.g., [EnvSetup(...), ...]
    tasks = [AssembleTask(), DisassembleTask()]

    # TODO: Define configuration dictionaries
    env_cfg = {
        # e.g., "num_actions": 7, "min_bounds": [...], "max_bounds": [...]
    }
    obs_cfg = {
        # observation configuration
    }
    reward_cfg = {
        # reward configuration
    }
    command_cfg = {
        # command input configuration
    }

    # Training parameters
    batch_dim = 16
    action_dim = 7
    electronics_graph_dim = 128

    # Start training
    run_training(
        env_setups,
        tasks,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        batch_dim,
        action_dim,
        electronics_graph_dim,
    )
    # Original tests...
    # Dummy training step...
    # Placeholder to run full training; configure env_setups and tasks below
    # run_training(env_setups, tasks, env_cfg, obs_cfg, reward_cfg, command_cfg, batch_dim, action_dim, electronics_graph_dim)
    pass


if __name__ == "__main__":
    # Configure environment and tasks
    from repairs_components.processing.tasks import AssembleTask, DisassembleTask
    from repairs_components.training_utils.env_setup import EnvSetup

    # TODO: Populate env_setups with EnvSetup instances
    env_setups = []  # e.g., [EnvSetup(...), ...]
    tasks = [AssembleTask(), DisassembleTask()]

    # TODO: Define configuration dictionaries
    env_cfg = {
        # e.g., "num_actions": 7, "min_bounds": [...], "max_bounds": [...]
    }
    obs_cfg = {
        # observation configuration
    }
    reward_cfg = {
        # reward configuration
    }
    command_cfg = {
        # command input configuration
    }

    # Training parameters
    batch_dim = 16
    action_dim = 7
    electronics_graph_dim = 128

    # Start training
    run_training(
        env_setups,
        tasks,
        env_cfg,
    # Unit tests for network shapes and sampling
    action_dim = 7
    electronics_graph_dim = 128
    batch_size = 2
    # Dummy inputs
    voxel_obs = torch.randn(batch_size, 1, 32, 32, 32)
    video_obs = torch.randn(batch_size, 3, 64, 64)
    graph_obs = torch.randn(batch_size, electronics_graph_dim)
    # Actor test
    actor = SACActor(action_dim, electronics_graph_dim)
    mean, log_std = actor(voxel_obs, video_obs, graph_obs)
    action, log_prob = actor.sample_action(voxel_obs, video_obs, graph_obs)
    # Critic test
    critic = SACCritic(action_dim, electronics_graph_dim)
    q1, q2 = critic(voxel_obs, video_obs, graph_obs, action)
    # Print shapes
    print("Actor output shapes: mean", mean.shape, ", log_std", log_std.shape)
    print("Sampled action shapes: action", action.shape, ", log_prob", log_prob.shape)
    print("Critic output shapes: q1", q1.shape, ", q2", q2.shape)
    # Dummy training step
    print("Running dummy training update...")
    trainer = SACTrainer(action_dim, electronics_graph_dim)
    # Dummy reward and done
    r = torch.randn(batch_size, device=voxel_obs.device)
    d = torch.zeros(batch_size, device=voxel_obs.device)
    cl, al, alpha = trainer.update((voxel_obs, video_obs, graph_obs, action, r, voxel_obs, video_obs, graph_obs, d))
    print("Dummy update loss: critic", cl, ", actor", al, ", alpha", alpha)
