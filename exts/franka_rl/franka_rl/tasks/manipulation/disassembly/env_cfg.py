"""Franka disassembly environment: pull bottom cap out of fixture.

The bottom cap is held in place by a spring force applied each step.
The robot must overcome this force to extract the cap.
"""

from __future__ import annotations

import os
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import (
    ActionTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets")

# --- Press-fit friction parameters ---
# Static phase: very stiff spring locks the cap until displacement > STATIC_THRESHOLD
STATIC_STIFFNESS = 5000.0  # N/m — high stiffness to prevent any motion below threshold
STATIC_THRESHOLD = 0.002   # m — 2mm deadband before sliding starts
# Sliding phase: constant friction force + velocity damping resists extraction
SLIDING_FRICTION = 5.0     # N — constant kinetic friction force opposing motion
SLIDING_DAMPING = 30.0     # N·s/m — viscous damping during sliding
# Release: cap separates completely beyond this distance
RELEASE_DISTANCE = 0.08    # m


@configclass
class DisassemblySceneCfg(InteractiveSceneCfg):
    num_envs = 1024
    env_spacing = 2.5

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
    )

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 0.0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.6, 0.4),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.3, 0.2)),
            physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    fixture = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Fixture",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 0.45]),
        spawn=UsdFileCfg(
            usd_path=os.path.join(ASSETS_DIR, "fixture.usd"),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.6,
            ),
        ),
    )

    workpiece: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Workpiece",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.052, 0.45]),
        spawn=UsdFileCfg(
            usd_path=os.path.join(ASSETS_DIR, "bottom_cap.usd"),
            scale=(0.01, 0.01, 0.01),
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            physics_material=RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.6,
            ),
        ),
    )


@configclass
class DisassemblyActionsCfg:
    arm_action: ActionTermCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=0.5,
        use_default_offset=True,
    )
    gripper_action: ActionTermCfg = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


# --- Custom observation & reward functions ---

def _workpiece_pos(env: object) -> torch.Tensor:
    return env.scene["workpiece"].data.root_pos_w[:, :3]


def _workpiece_displacement(env: object) -> torch.Tensor:
    """How far the workpiece has moved from its initial position [N, 1]."""
    current = env.scene["workpiece"].data.root_pos_w[:, :3]
    initial = torch.tensor(
        env.scene["workpiece"].cfg.init_state.pos,
        device=current.device,
    )
    return (current - initial).norm(dim=-1, keepdim=True)


def _ee_to_workpiece_distance(env: object, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    ee_pos = env.scene[robot_cfg.name].data.body_pos_w[:, robot_cfg.body_ids[0], :3]
    wp_pos = env.scene["workpiece"].data.root_pos_w[:, :3]
    return -(ee_pos - wp_pos).norm(dim=-1)


def _extraction_progress(env: object) -> torch.Tensor:
    current = env.scene["workpiece"].data.root_pos_w[:, :3]
    initial = torch.tensor(
        env.scene["workpiece"].cfg.init_state.pos,
        device=current.device,
    )
    return (current - initial).norm(dim=-1).clamp(min=0.0)


def _workpiece_extracted(env: object, threshold: float = 0.05) -> torch.Tensor:
    current = env.scene["workpiece"].data.root_pos_w[:, :3]
    initial = torch.tensor(
        env.scene["workpiece"].cfg.init_state.pos,
        device=current.device,
    )
    return (current - initial).norm(dim=-1) > threshold


@configclass
class DisassemblyObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel)
        workpiece_pos = ObservationTermCfg(func=_workpiece_pos)
        workpiece_displacement = ObservationTermCfg(func=_workpiece_displacement)
        actions = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class DisassemblyRewardsCfg:
    reaching_workpiece = RewardTermCfg(
        func=_ee_to_workpiece_distance,
        weight=0.5,
        params={"robot_cfg": SceneEntityCfg("robot", body_names=["panda_hand"])},
    )
    extraction_progress = RewardTermCfg(
        func=_extraction_progress,
        weight=5.0,
    )
    extraction_success = RewardTermCfg(
        func=_workpiece_extracted,
        weight=50.0,
        params={"threshold": 0.05},
    )
    action_penalty = RewardTermCfg(func=mdp.action_l2, weight=-0.01)
    joint_vel_penalty = RewardTermCfg(func=mdp.joint_vel_l2, weight=-0.001)


@configclass
class DisassemblyTerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    success = TerminationTermCfg(
        func=_workpiece_extracted,
        time_out=False,
        params={"threshold": 0.08},
    )


@configclass
class DisassemblyEventsCfg:
    reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.8, 1.2),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_workpiece = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("workpiece"),
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "velocity_range": {},
        },
    )


class DisassemblyEnv(ManagerBasedRLEnv):
    """Custom env that applies a spring force holding the workpiece in place."""

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        self._apply_hold_force()

    def _apply_hold_force(self):
        """Three-phase press-fit model: static lock → sliding friction → release."""
        wp = self.scene["workpiece"]
        current_pos = wp.data.root_pos_w[:, :3]
        current_vel = wp.data.root_lin_vel_w[:, :3]
        target_pos = torch.tensor(
            wp.cfg.init_state.pos, device=current_pos.device
        ).expand_as(current_pos)

        displacement = current_pos - target_pos
        dist = displacement.norm(dim=-1, keepdim=True)
        direction = displacement / (dist + 1e-8)

        force = torch.zeros_like(displacement)

        # Phase 1: Static lock — high stiffness prevents motion below threshold
        static_mask = (dist < STATIC_THRESHOLD).expand_as(force)
        force = torch.where(static_mask, -STATIC_STIFFNESS * displacement, force)

        # Phase 2: Sliding — constant friction force + damping resists extraction
        sliding_mask = ((dist >= STATIC_THRESHOLD) & (dist < RELEASE_DISTANCE)).expand_as(force)
        friction_force = -SLIDING_FRICTION * direction - SLIDING_DAMPING * current_vel
        force = torch.where(sliding_mask, friction_force, force)

        # Phase 3: Released — no force (dist >= RELEASE_DISTANCE), force stays zero

        wp.set_external_force_and_torque(
            forces=force.unsqueeze(1),
            torques=torch.zeros_like(force).unsqueeze(1),
            body_ids=[0],
        )


@configclass
class DisassemblyEnvCfg(ManagerBasedRLEnvCfg):
    """Franka disassembly: pull the bottom cap out of the fixture."""

    scene: DisassemblySceneCfg = DisassemblySceneCfg()
    actions: DisassemblyActionsCfg = DisassemblyActionsCfg()
    observations: DisassemblyObservationsCfg = DisassemblyObservationsCfg()
    rewards: DisassemblyRewardsCfg = DisassemblyRewardsCfg()
    terminations: DisassemblyTerminationsCfg = DisassemblyTerminationsCfg()
    events: DisassemblyEventsCfg = DisassemblyEventsCfg()

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0)
    episode_length_s = 10.0
    decimation = 2

    def __post_init__(self):
        super().__post_init__()
        self.sim.render_interval = self.decimation


@configclass
class DisassemblyEnvCfg_Play(DisassemblyEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.episode_length_s = 15.0
