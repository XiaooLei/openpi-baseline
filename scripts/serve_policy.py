import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def _metadata_with_policy_deployment_defaults(policy: _policy.Policy, metadata: dict | None) -> dict:
    """Fill metadata fields expected by the policy_deployment client tools.

    Older OpenPI configs often leave ``TrainConfig.policy_metadata`` unset, so
    the websocket server sends ``{}`` as the first frame. That is enough for the
    permissive OpenPI client, but stricter clients such as
    ``policy_deployment/scripts/ping.py`` expect protocol/action/state fields.
    Preserve any explicit metadata from the config and only add missing keys.
    """

    out = dict(metadata or {})
    model = getattr(policy, "_model", None)
    model_action_horizon = int(getattr(model, "action_horizon", 50))
    model_action_dim = int(getattr(model, "action_dim", 14))
    # The OpenPI model may use a padded internal action dimension (for this
    # Pi0/Pi05 config it is 32), but the websocket API exposed to the dual-YAM
    # client returns unpadded physical actions with 14 dims after output
    # transforms. The policy_deployment protocol should advertise that external
    # shape, not the model-internal padded shape.
    external_action_horizon = model_action_horizon
    external_action_dim = 14

    out.setdefault("protocol_version", "1.0")
    out.setdefault("policy_name", policy.__class__.__name__)
    out.setdefault("control_mode", "joints")
    out.setdefault("action_horizon", external_action_horizon)
    out.setdefault("action_dim", external_action_dim)
    # Dual-YAM policies use the same 14-dim layout for proprio state and actions:
    # [L_arm6, L_grip, R_arm6, R_grip].
    out.setdefault("state_dim", external_action_dim)
    out.setdefault("image_keys", ["cam_high", "cam_left_wrist", "cam_right_wrist"])
    # This is the observation shape sent by policy_deployment/sim/check_in_sim.py.
    out.setdefault("image_shape", [3, 480, 640])
    out.setdefault("expects_prompt", True)
    out.setdefault("accepts_compressed_images", True)
    extra = dict(out.get("extra") or {})
    extra.setdefault("model_action_dim", model_action_dim)
    out["extra"] = extra
    return out


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = _metadata_with_policy_deployment_defaults(policy, policy.metadata)

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
