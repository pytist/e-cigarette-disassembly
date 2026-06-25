"""Franka disassembly environment: pull bottom cap out of fixture along X axis."""

from __future__ import annotations

import os
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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

    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    # Static fixture (outer shell + mouthpiece) — does not move
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

    # Bottom cap — temporarily static for positioning
    workpiece = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workpiece",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.055, 0.45]),
        spawn=UsdFileCfg(
            usd_path=os.path.join(ASSETS_DIR, "bottom_cap.usd"),
            scale=(0.01, 0.01, 0.01),
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
    """Workpiece position in world frame [N, 3]."""
    return env.scene["workpiece"].data.root_pos_w[:, :3]


def _workpiece_displacement_x(env: object) -> torch.Tensor:
    """How far the workpiece has moved along +X from its initial position [N, 1]."""
    current_x = env.scene["workpiece"].data.root_pos_w[:, 0]
    initial_x = env.scene["workpiece"].cfg.init_state.pos[0]
    return (current_x - initial_x).unsqueeze(-1)


def _ee_to_workpiece_distance(env: object, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Negative L2 distance from end-effector to workpiece [N]."""
    ee_pos = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids[0], :3]
    wp_pos = env.scene["workpiece"].data.root_pos_w[:, :3]
    return -(ee_pos - wp_pos).norm(dim=-1)


def _disassembly_progress(env: object) -> torch.Tensor:
    """Reward for moving the workpiece along +X (the pull-out direction) [N]."""
    current_x = env.scene["workpiece"].data.root_pos_w[:, 0]
    initial_x = env.scene["workpiece"].cfg.init_state.pos[0]
    return (current_x - initial_x).clamp(min=0.0)


def _workpiece_extracted(env: object, threshold: float = 0.1) -> torch.Tensor:
    """True when workpiece has been pulled out far enough along X [N]."""
    current_x = env.scene["workpiece"].data.root_pos_w[:, 0]
    initial_x = env.scene["workpiece"].cfg.init_state.pos[0]
    return (current_x - initial_x) > threshold


@configclass
class DisassemblyObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel)
        # workpiece_pos = ObservationTermCfg(func=_workpiece_pos)  # disabled: workpiece is static for now
        # workpiece_displacement = ObservationTermCfg(func=_workpiece_displacement_x)
        actions = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class DisassemblyRewardsCfg:
    # Disabled while workpiece is static for positioning
    # reaching_workpiece = RewardTermCfg(...)
    # extraction_progress = RewardTermCfg(...)
    # extraction_success = RewardTermCfg(...)
    action_penalty = RewardTermCfg(func=mdp.action_l2, weight=-0.01)
    joint_vel_penalty = RewardTermCfg(func=mdp.joint_vel_l2, weight=-0.001)


@configclass
class DisassemblyTerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    # success = TerminationTermCfg(func=_workpiece_extracted, ...)  # disabled: workpiece is static


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
    # reset_workpiece disabled: workpiece is static for positioning


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
