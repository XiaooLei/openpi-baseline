"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import os
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.yam_policy as yam_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)
    # Additional LeRobot delta indices for observation keys. Values are frame offsets relative to the current
    # frame, e.g. {"observation.state": (-3, -2, -1, 0)} returns a short state history.
    observation_delta_indices: dict[str, Sequence[int]] = dataclasses.field(default_factory=dict)
    # Optional LeRobot episode subset. If set, the loader only reads these episode indices.
    episodes: Sequence[int] | None = None
    # Optional frame stride for training anchors. This does not change dataset FPS or action chunk timestamps.
    frame_stride: int = 1
    # Video decoder backend for LeRobot datasets. Prefer pyav in offline images because torchcodec needs system FFmpeg libs.
    video_backend: str | None = "pyav"
    # Optional child configs for mixing multiple LeRobot datasets that share the same training transforms.
    source_configs: tyro.conf.Suppress[Sequence[Any]] = ()
    # Optional target sampling ratio per source config. If omitted, sampling is proportional to source length.
    source_weights: tyro.conf.Suppress[tuple[float, ...]] = ()

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None
    local_files_path: str | None = None

class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class MixtureDataConfig(DataConfigFactory):
    """Mixes multiple data configs while reusing the transforms and assets from the first config."""

    repo_id: str = "mixture"
    data_configs: tyro.conf.Suppress[Sequence[DataConfigFactory]] = ()
    source_weights: tyro.conf.Suppress[tuple[float, ...]] = ()

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if not self.data_configs:
            raise ValueError("MixtureDataConfig requires at least one child data config.")
        source_configs = tuple(data_config.create(assets_dirs, model_config) for data_config in self.data_configs)
        if self.source_weights and len(self.source_weights) != len(source_configs):
            raise ValueError(
                f"source_weights length ({len(self.source_weights)}) must match source count ({len(source_configs)})."
            )
        if self.source_weights and any(weight <= 0 for weight in self.source_weights):
            raise ValueError(f"source_weights must all be positive, got {self.source_weights}.")

        return dataclasses.replace(
            source_configs[0],
            repo_id=self.repo_id,
            source_configs=source_configs,
            source_weights=tuple(self.source_weights),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )

@dataclasses.dataclass(frozen=True)
class DualYamDataConfig(DataConfigFactory):
    """Data class for dual-arm yam system."""

    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = ""
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: _transforms.Group = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.cam_high",
                            "cam_left_wrist": "observation.images.cam_left_wrist",
                            "cam_right_wrist": "observation.images.cam_right_wrist",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "task",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Prepare data for policy training
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[yam_policy.YamInputs(action_dim=model_config.action_dim, adapt_to_pi=self.adapt_to_pi, model_type=model_config.model_type)],
            outputs=[yam_policy.YamOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            # Left and right arm joints use delta actions, grippers use absolute actions
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)], # Convert to delta actions
                outputs=[_transforms.AbsoluteActions(delta_action_mask)], # Convert back to absolute actions during inference
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class DualYamMemoryDataConfig(DualYamDataConfig):
    """Dual-arm YAM data config that exposes sparse proprioception deltas to pi05 as discrete state context."""

    state_history_delta_indices: Sequence[int] = (-120, -60, -30, -15, 0)
    state_delta_pairs: Sequence[tuple[int, int]] = ((0, -15), (-15, -30), (-30, -60), (-60, -120))

    repack_transforms: _transforms.Group = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.cam_high",
                            "cam_left_wrist": "observation.images.cam_left_wrist",
                            "cam_right_wrist": "observation.images.cam_right_wrist",
                        },
                        "state_history": "observation.state",
                        "actions": "action",
                        "prompt": "task",
                    }
                )
            ]
        )
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        history_indices = tuple(self.state_history_delta_indices)
        if 0 not in history_indices:
            raise ValueError("state_history_delta_indices must include 0 so delta actions can use the current state.")
        current_state_index = history_indices.index(0)
        missing_delta_offsets = {
            offset for pair in self.state_delta_pairs for offset in pair if offset not in history_indices
        }
        if missing_delta_offsets:
            raise ValueError(
                f"state_delta_pairs reference offsets not present in state_history_delta_indices: "
                f"{sorted(missing_delta_offsets)}"
            )

        base_config = self.create_base_config(assets_dirs, model_config)

        data_transforms = _transforms.Group(
            inputs=[
                yam_policy.YamMemoryInputs(
                    action_dim=model_config.action_dim,
                    adapt_to_pi=self.adapt_to_pi,
                    model_type=model_config.model_type,
                    current_state_index=current_state_index,
                )
            ],
            outputs=[yam_policy.YamOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        state_history_norm = []
        if base_config.norm_stats is not None and "state" in base_config.norm_stats:
            state_history_norm = [
                _transforms.Normalize(
                    {"state_history": base_config.norm_stats["state"]},
                    use_quantiles=base_config.use_quantile_norm,
                )
            ]

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        model_transforms = _transforms.Group(
            inputs=[
                *state_history_norm,
                yam_policy.SparseYamStateDeltaContext(
                    state_history_delta_indices=history_indices,
                    delta_pairs=tuple(self.state_delta_pairs),
                ),
                *model_transforms.inputs,
            ],
            outputs=model_transforms.outputs,
        )

        observation_delta_indices = dict(base_config.observation_delta_indices)
        observation_delta_indices["observation.state"] = history_indices

        return dataclasses.replace(
            base_config,
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            observation_delta_indices=observation_delta_indices,
        )

@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.
    # Path to the filter dictionary file.
    filter_dict_path: str | None = "gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            filter_dict_path=self.filter_dict_path,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    # If true, convert absolute 7-DoF joint action targets to deltas relative to
    # the current state. Leave gripper action absolute.
    use_delta_joint_actions: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDEefDataConfig(DataConfigFactory):
    """DROID-style image dataset with EEF state and delta-EEF actions."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/eef_position": "eef_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidEefInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidEefOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)
    # Optional held-out data used only for action-MAE eval. If omitted, eval uses the training data config.
    eval_data: tyro.conf.Suppress[DataConfigFactory | None] = None

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to run evaluation (MAE between predicted and expert actions).
    eval_interval: int = 500
    # Number of batches to sample for evaluation.
    num_eval_batches: int = 10
    # Number of training-data batches to sample for held-in action-MAE eval. Set to 0 to disable.
    num_heldin_eval_batches: int = 5
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True
    # If true, will enable tensorboard logging (checkpoint_dir/tensorboard).
    tensorboard_enabled: bool = False

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_SEAL_MEMORY_TOTAL_EPISODES = 603
_SEAL_MEMORY_VAL_EPISODES = (24, 33, 38, 60, 213, 249, 271, 378, 437, 527, 550, 600)
_SEAL_MEMORY_TRAIN_EPISODES = tuple(
    ep for ep in range(_SEAL_MEMORY_TOTAL_EPISODES) if ep not in _SEAL_MEMORY_VAL_EPISODES
)
_SEAL_MEMORY_EXPERT_TRAIN_EPISODES = tuple(
    ep for ep in range(0, 379) if ep not in _SEAL_MEMORY_VAL_EPISODES
)
_SEAL_MEMORY_HIL_SUFFIX_TRAIN_EPISODES = tuple(
    ep for ep in range(379, 547) if ep not in _SEAL_MEMORY_VAL_EPISODES
)
_SEAL_MEMORY_SUCCESS_TRAIN_EPISODES = tuple(
    ep for ep in range(547, 603) if ep not in _SEAL_MEMORY_VAL_EPISODES
)
_SEAL_MEMORY70_HISTORY_INDICES = (-120, -60, -30, -15, 0)
_SEAL_MEMORY70_DELTA_PAIRS = ((0, -15), (-15, -30), (-30, -60), (-60, -120))
_SEAL_MEMORY42_HISTORY_INDICES = (-120, -60, 0)
_SEAL_MEMORY42_DELTA_PAIRS = ((0, -60), (0, -120))
_RSS_DATA_ROOT = os.environ.get(
    "RSS_DATA_ROOT",
    "/inspire/qb-ilm/project/gjjproject/public/xl/data/rss_challenge",
)
_RSS_RAW_DATA_ROOT = f"{_RSS_DATA_ROOT}/raw"
_RSS_PHASE2_RECAP_ROOT = f"{_RSS_DATA_ROOT}/recap/phase2"
_RSS_BASELINE_CHECKPOINT_ROOT = os.environ.get(
    "RSS_BASELINE_CHECKPOINT_ROOT",
    "/inspire/qb-ilm/project/gjjproject/public/xl/data/baseline_checkpoints",
)
_RSS_SECOND_SUBMIT_CHECKPOINT_ROOT = (
    os.environ.get("RSS_SECOND_SUBMIT_CHECKPOINT_ROOT") or f"{_RSS_BASELINE_CHECKPOINT_ROOT}/2nd-submit"
)
_SEAL_MIXED_DATA_PATH = f"{_RSS_RAW_DATA_ROOT}/seal-water-bottle-cap/expert-success-hil-suffix-mix-data"
_SEAL_PHASE2_TRAIN_PATH = f"{_RSS_PHASE2_RECAP_ROOT}/seal_water_bottle_cap_hil_split/train"
_SEAL_BASELINE_CHECKPOINT_DIR = f"{_RSS_BASELINE_CHECKPOINT_ROOT}/pi05_seal-water-bottle-cap/199999"
_SEAL_BASELINE_ASSETS_DIR = f"{_SEAL_BASELINE_CHECKPOINT_DIR}/assets/v21"
_SEAL_WATER120_CHECKPOINT_DIR = f"{_RSS_SECOND_SUBMIT_CHECKPOINT_ROOT}/water_120k"
_SEAL_WATER120_ASSETS_DIR = f"{_SEAL_WATER120_CHECKPOINT_DIR}/assets"
_SEAL_WATER120_ASSET_ID = "seal-water-bottle-cap/expert-success-hil-suffix-mix-data"
_INSERT_MEMORY_TOTAL_EPISODES = 1231
_INSERT_MEMORY_VAL_EPISODES = (24, 120, 240, 360, 527, 720, 830, 860, 940, 1086, 1100, 1220)
_INSERT_MEMORY_EXPERT_TRAIN_EPISODES = tuple(
    ep for ep in range(0, 831) if ep not in _INSERT_MEMORY_VAL_EPISODES
)
_INSERT_MEMORY_HIL_SUFFIX_TRAIN_EPISODES = tuple(
    ep for ep in range(831, 1087) if ep not in _INSERT_MEMORY_VAL_EPISODES
)
_INSERT_MEMORY_SUCCESS_TRAIN_EPISODES = tuple(
    ep for ep in range(1087, 1231) if ep not in _INSERT_MEMORY_VAL_EPISODES
)
_TOWER_MEMORY_TOTAL_EPISODES = 1652
_TOWER_MEMORY_VAL_EPISODES = (
    24,
    150,
    300,
    450,
    600,
    750,
    900,
    1003,
    1050,
    1200,
    1400,
    1471,
    1500,
    1580,
    1651,
)
_TOWER_MEMORY_EXPERT_TRAIN_EPISODES = tuple(
    ep for ep in range(0, 1004) if ep not in _TOWER_MEMORY_VAL_EPISODES
)
_TOWER_MEMORY_HIL_SUFFIX_TRAIN_EPISODES = tuple(
    ep for ep in range(1004, 1472) if ep not in _TOWER_MEMORY_VAL_EPISODES
)
_TOWER_MEMORY_SUCCESS_TRAIN_EPISODES = tuple(
    ep for ep in range(1472, 1652) if ep not in _TOWER_MEMORY_VAL_EPISODES
)
_SEAL_PHASE2_QUALITY60_EPISODES = (
    57,
    51,
    167,
    38,
    90,
    86,
    176,
    15,
    140,
    81,
    111,
    32,
    115,
    42,
    82,
    155,
    151,
    196,
    156,
    93,
    141,
    107,
    104,
    48,
    110,
    145,
    65,
    27,
    4,
    171,
    189,
    39,
    62,
    175,
    46,
    153,
    108,
    150,
    40,
    60,
    188,
    119,
    33,
    118,
    34,
    83,
    144,
    44,
)


def _rss_second_submit_assets(task_slug: str, checkpoint_subdir: str) -> AssetsConfig:
    return AssetsConfig(
        assets_dir=f"{_RSS_SECOND_SUBMIT_CHECKPOINT_ROOT}/{checkpoint_subdir}/assets",
        asset_id=f"{task_slug}/expert-success-hil-suffix-mix-data",
    )


def _rss_memory_source_config(
    *,
    task_slug: str,
    checkpoint_subdir: str,
    episodes: Sequence[int] | None,
    state_history_delta_indices: Sequence[int],
    state_delta_pairs: Sequence[tuple[int, int]],
    frame_stride: int = 1,
) -> DualYamMemoryDataConfig:
    return DualYamMemoryDataConfig(
        repo_id=f"{task_slug}/expert-success-hil-suffix-mix-data",
        base_config=DataConfig(
            prompt_from_task=True,
            local_files_path=f"{_RSS_RAW_DATA_ROOT}/{task_slug}/expert-success-hil-suffix-mix-data",
            episodes=episodes,
            frame_stride=frame_stride,
        ),
        assets=_rss_second_submit_assets(task_slug, checkpoint_subdir),
        state_history_delta_indices=state_history_delta_indices,
        state_delta_pairs=state_delta_pairs,
        use_delta_joint_actions=True,
        adapt_to_pi=True,
    )


def _rss_memory_phase2_source_config(
    *,
    task_slug: str,
    phase2_slug: str,
    checkpoint_subdir: str,
    state_history_delta_indices: Sequence[int],
    state_delta_pairs: Sequence[tuple[int, int]],
) -> DualYamMemoryDataConfig:
    return DualYamMemoryDataConfig(
        repo_id=f"{phase2_slug}_hil_split/train",
        base_config=DataConfig(
            prompt_from_task=True,
            local_files_path=f"{_RSS_PHASE2_RECAP_ROOT}/{phase2_slug}_hil_split/train",
        ),
        assets=_rss_second_submit_assets(task_slug, checkpoint_subdir),
        state_history_delta_indices=state_history_delta_indices,
        state_delta_pairs=state_delta_pairs,
        use_delta_joint_actions=True,
        adapt_to_pi=True,
    )


def _rss_memory_phase2_bc_config(
    *,
    task_slug: str,
    phase2_slug: str,
    checkpoint_subdir: str,
    memory_dim: int,
    state_history_delta_indices: Sequence[int],
    state_delta_pairs: Sequence[tuple[int, int]],
    max_token_len: int,
    expert_train_episodes: Sequence[int],
    hil_suffix_train_episodes: Sequence[int],
    success_train_episodes: Sequence[int],
    val_episodes: Sequence[int],
) -> TrainConfig:
    return TrainConfig(
        name=f"pi05_{task_slug}_memory{memory_dim}_phase2_bc",
        model=pi0_config.Pi0Config(pi05=True, max_token_len=max_token_len),
        data=MixtureDataConfig(
            repo_id=f"{task_slug}/memory{memory_dim}-expert-phase2",
            source_weights=(0.47, 0.21, 0.07, 0.25),
            data_configs=(
                _rss_memory_source_config(
                    task_slug=task_slug,
                    checkpoint_subdir=checkpoint_subdir,
                    episodes=expert_train_episodes,
                    frame_stride=2,
                    state_history_delta_indices=state_history_delta_indices,
                    state_delta_pairs=state_delta_pairs,
                ),
                _rss_memory_source_config(
                    task_slug=task_slug,
                    checkpoint_subdir=checkpoint_subdir,
                    episodes=hil_suffix_train_episodes,
                    state_history_delta_indices=state_history_delta_indices,
                    state_delta_pairs=state_delta_pairs,
                ),
                _rss_memory_source_config(
                    task_slug=task_slug,
                    checkpoint_subdir=checkpoint_subdir,
                    episodes=success_train_episodes,
                    state_history_delta_indices=state_history_delta_indices,
                    state_delta_pairs=state_delta_pairs,
                ),
                _rss_memory_phase2_source_config(
                    task_slug=task_slug,
                    phase2_slug=phase2_slug,
                    checkpoint_subdir=checkpoint_subdir,
                    state_history_delta_indices=state_history_delta_indices,
                    state_delta_pairs=state_delta_pairs,
                ),
            ),
        ),
        eval_data=_rss_memory_source_config(
            task_slug=task_slug,
            checkpoint_subdir=checkpoint_subdir,
            episodes=val_episodes,
            state_history_delta_indices=state_history_delta_indices,
            state_delta_pairs=state_delta_pairs,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            f"{_RSS_SECOND_SUBMIT_CHECKPOINT_ROOT}/{checkpoint_subdir}/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=2e-5,
            decay_steps=100_000,
            decay_lr=2e-5,
        ),
        num_train_steps=300_000,
        batch_size=32,
        num_workers=64,
        log_interval=50,
        eval_interval=500,
        num_eval_batches=10,
        save_interval=20_000,
        keep_period=50_000,
        policy_metadata={
            "state_history_delta_indices": tuple(state_history_delta_indices),
        },
    )


_CONFIGS = [
    # Challenge Baseline Examples
    TrainConfig(
        name="pi05_insert-mouse-battery",
        model=pi0_config.Pi0Config(pi05=True),
        data=DualYamDataConfig(
            repo_id="insert-mouse-battery/expert-data",
            base_config=DataConfig(prompt_from_task=True,  local_files_path="/Your/path/to/Posttraining-RFM-RSS2026/Challenge-phase1-dataset/insert-mouse-battery/expert-data"),
            use_delta_joint_actions=True,
            adapt_to_pi=True
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=200_000, # 200k is about 3 epochs.
        batch_size=32,
        num_workers=64,
        save_interval=40_000
    ),
    TrainConfig(
        name="pi05_seal-water-bottle-cap",
        model=pi0_config.Pi0Config(pi05=True),
        data=DualYamDataConfig(
            repo_id="seal-water-bottle-cap/expert-data",
            base_config=DataConfig(prompt_from_task=True,  local_files_path="/Your/path/to/Posttraining-RFM-RSS2026/Challenge-phase1-dataset/seal-water-bottle-cap/expert-data"),
            use_delta_joint_actions=True,
            adapt_to_pi=True
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=200_000,
        batch_size=32,
        num_workers=64,
        save_interval=40_000
    ),
    TrainConfig(
        name="pi05_seal-water-bottle-cap_memory_specialist",
        model=pi0_config.Pi0Config(pi05=True, max_token_len=320),
        data=DualYamMemoryDataConfig(
            repo_id="seal-water-bottle-cap/expert-success-hil-suffix-mix-data",
            base_config=DataConfig(
                prompt_from_task=True,
                local_files_path=_SEAL_MIXED_DATA_PATH,
                episodes=_SEAL_MEMORY_TRAIN_EPISODES,
            ),
            assets=AssetsConfig(
                assets_dir=_SEAL_BASELINE_ASSETS_DIR,
                asset_id="seal-water-bottle-cap",
            ),
            state_history_delta_indices=_SEAL_MEMORY70_HISTORY_INDICES,
            state_delta_pairs=_SEAL_MEMORY70_DELTA_PAIRS,
            use_delta_joint_actions=True,
            adapt_to_pi=True,
        ),
        eval_data=DualYamMemoryDataConfig(
            repo_id="seal-water-bottle-cap/expert-success-hil-suffix-mix-data",
            base_config=DataConfig(
                prompt_from_task=True,
                local_files_path=_SEAL_MIXED_DATA_PATH,
                episodes=_SEAL_MEMORY_VAL_EPISODES,
            ),
            assets=AssetsConfig(
                assets_dir=_SEAL_BASELINE_ASSETS_DIR,
                asset_id="seal-water-bottle-cap",
            ),
            state_history_delta_indices=_SEAL_MEMORY70_HISTORY_INDICES,
            state_delta_pairs=_SEAL_MEMORY70_DELTA_PAIRS,
            use_delta_joint_actions=True,
            adapt_to_pi=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(f"{_SEAL_BASELINE_CHECKPOINT_DIR}/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        num_train_steps=300_000,
        batch_size=32,
        num_workers=64,
        log_interval=50,
        eval_interval=500,
        num_eval_batches=10,
        save_interval=20_000,
        keep_period=50_000,
        policy_metadata={
            "state_history_delta_indices": _SEAL_MEMORY70_HISTORY_INDICES,
        },
    ),
    TrainConfig(
        name="pi05_seal-water-bottle-cap_memory42_phase2_bc",
        model=pi0_config.Pi0Config(pi05=True, max_token_len=240),
        data=MixtureDataConfig(
            repo_id="seal-water-bottle-cap/memory-expert-phase2-q60",
            source_weights=(0.47, 0.21, 0.07, 0.25),
            data_configs=(
                DualYamMemoryDataConfig(
                    repo_id="seal-water-bottle-cap/expert-success-hil-suffix-mix-data",
                    base_config=DataConfig(
                        prompt_from_task=True,
                        local_files_path=_SEAL_MIXED_DATA_PATH,
                        episodes=_SEAL_MEMORY_EXPERT_TRAIN_EPISODES,
                        frame_stride=2,
                    ),
                    assets=AssetsConfig(
                        assets_dir=_SEAL_WATER120_ASSETS_DIR,
                        asset_id=_SEAL_WATER120_ASSET_ID,
                    ),
                    state_history_delta_indices=_SEAL_MEMORY42_HISTORY_INDICES,
                    state_delta_pairs=_SEAL_MEMORY42_DELTA_PAIRS,
                    use_delta_joint_actions=True,
                    adapt_to_pi=True,
                ),
                DualYamMemoryDataConfig(
                    repo_id="seal-water-bottle-cap/expert-success-hil-suffix-mix-data",
                    base_config=DataConfig(
                        prompt_from_task=True,
                        local_files_path=_SEAL_MIXED_DATA_PATH,
                        episodes=_SEAL_MEMORY_HIL_SUFFIX_TRAIN_EPISODES,
                    ),
                    assets=AssetsConfig(
                        assets_dir=_SEAL_WATER120_ASSETS_DIR,
                        asset_id=_SEAL_WATER120_ASSET_ID,
                    ),
                    state_history_delta_indices=_SEAL_MEMORY42_HISTORY_INDICES,
                    state_delta_pairs=_SEAL_MEMORY42_DELTA_PAIRS,
                    use_delta_joint_actions=True,
                    adapt_to_pi=True,
                ),
                DualYamMemoryDataConfig(
                    repo_id="seal-water-bottle-cap/expert-success-hil-suffix-mix-data",
                    base_config=DataConfig(
                        prompt_from_task=True,
                        local_files_path=_SEAL_MIXED_DATA_PATH,
                        episodes=_SEAL_MEMORY_SUCCESS_TRAIN_EPISODES,
                    ),
                    assets=AssetsConfig(
                        assets_dir=_SEAL_WATER120_ASSETS_DIR,
                        asset_id=_SEAL_WATER120_ASSET_ID,
                    ),
                    state_history_delta_indices=_SEAL_MEMORY42_HISTORY_INDICES,
                    state_delta_pairs=_SEAL_MEMORY42_DELTA_PAIRS,
                    use_delta_joint_actions=True,
                    adapt_to_pi=True,
                ),
                DualYamMemoryDataConfig(
                    repo_id="seal_water_bottle_cap_hil_split/train",
                    base_config=DataConfig(
                        prompt_from_task=True,
                        local_files_path=_SEAL_PHASE2_TRAIN_PATH,
                        episodes=_SEAL_PHASE2_QUALITY60_EPISODES,
                    ),
                    assets=AssetsConfig(
                        assets_dir=_SEAL_WATER120_ASSETS_DIR,
                        asset_id=_SEAL_WATER120_ASSET_ID,
                    ),
                    state_history_delta_indices=_SEAL_MEMORY42_HISTORY_INDICES,
                    state_delta_pairs=_SEAL_MEMORY42_DELTA_PAIRS,
                    use_delta_joint_actions=True,
                    adapt_to_pi=True,
                ),
            ),
        ),
        eval_data=DualYamMemoryDataConfig(
            repo_id="seal-water-bottle-cap/expert-success-hil-suffix-mix-data",
            base_config=DataConfig(
                prompt_from_task=True,
                local_files_path=_SEAL_MIXED_DATA_PATH,
                episodes=_SEAL_MEMORY_VAL_EPISODES,
            ),
            assets=AssetsConfig(
                assets_dir=_SEAL_WATER120_ASSETS_DIR,
                asset_id=_SEAL_WATER120_ASSET_ID,
            ),
            state_history_delta_indices=_SEAL_MEMORY42_HISTORY_INDICES,
            state_delta_pairs=_SEAL_MEMORY42_DELTA_PAIRS,
            use_delta_joint_actions=True,
            adapt_to_pi=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(f"{_SEAL_WATER120_CHECKPOINT_DIR}/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=2e-5,
            decay_steps=100_000,
            decay_lr=2e-5,
        ),
        num_train_steps=300_000,
        batch_size=32,
        num_workers=64,
        log_interval=50,
        eval_interval=500,
        num_eval_batches=10,
        save_interval=20_000,
        keep_period=50_000,
        policy_metadata={
            "state_history_delta_indices": _SEAL_MEMORY42_HISTORY_INDICES,
        },
    ),
    TrainConfig(
        name="pi05_seal-water-bottle-cap_memory70_phase2_bc",
        model=pi0_config.Pi0Config(pi05=True, max_token_len=320),
        data=MixtureDataConfig(
            repo_id="seal-water-bottle-cap/memory70-expert-phase2-q60",
            source_weights=(0.47, 0.21, 0.07, 0.25),
            data_configs=(
                DualYamMemoryDataConfig(
                    repo_id="seal-water-bottle-cap/expert-success-hil-suffix-mix-data",
                    base_config=DataConfig(
                        prompt_from_task=True,
                        local_files_path=_SEAL_MIXED_DATA_PATH,
                        episodes=_SEAL_MEMORY_EXPERT_TRAIN_EPISODES,
                        frame_stride=2,
                    ),
                    assets=AssetsConfig(
                        assets_dir=_SEAL_WATER120_ASSETS_DIR,
                        asset_id=_SEAL_WATER120_ASSET_ID,
                    ),
                    state_history_delta_indices=_SEAL_MEMORY70_HISTORY_INDICES,
                    state_delta_pairs=_SEAL_MEMORY70_DELTA_PAIRS,
                    use_delta_joint_actions=True,
                    adapt_to_pi=True,
                ),
                DualYamMemoryDataConfig(
                    repo_id="seal-water-bottle-cap/expert-success-hil-suffix-mix-data",
                    base_config=DataConfig(
                        prompt_from_task=True,
                        local_files_path=_SEAL_MIXED_DATA_PATH,
                        episodes=_SEAL_MEMORY_HIL_SUFFIX_TRAIN_EPISODES,
                    ),
                    assets=AssetsConfig(
                        assets_dir=_SEAL_WATER120_ASSETS_DIR,
                        asset_id=_SEAL_WATER120_ASSET_ID,
                    ),
                    state_history_delta_indices=_SEAL_MEMORY70_HISTORY_INDICES,
                    state_delta_pairs=_SEAL_MEMORY70_DELTA_PAIRS,
                    use_delta_joint_actions=True,
                    adapt_to_pi=True,
                ),
                DualYamMemoryDataConfig(
                    repo_id="seal-water-bottle-cap/expert-success-hil-suffix-mix-data",
                    base_config=DataConfig(
                        prompt_from_task=True,
                        local_files_path=_SEAL_MIXED_DATA_PATH,
                        episodes=_SEAL_MEMORY_SUCCESS_TRAIN_EPISODES,
                    ),
                    assets=AssetsConfig(
                        assets_dir=_SEAL_WATER120_ASSETS_DIR,
                        asset_id=_SEAL_WATER120_ASSET_ID,
                    ),
                    state_history_delta_indices=_SEAL_MEMORY70_HISTORY_INDICES,
                    state_delta_pairs=_SEAL_MEMORY70_DELTA_PAIRS,
                    use_delta_joint_actions=True,
                    adapt_to_pi=True,
                ),
                DualYamMemoryDataConfig(
                    repo_id="seal_water_bottle_cap_hil_split/train",
                    base_config=DataConfig(
                        prompt_from_task=True,
                        local_files_path=_SEAL_PHASE2_TRAIN_PATH,
                        episodes=_SEAL_PHASE2_QUALITY60_EPISODES,
                    ),
                    assets=AssetsConfig(
                        assets_dir=_SEAL_WATER120_ASSETS_DIR,
                        asset_id=_SEAL_WATER120_ASSET_ID,
                    ),
                    state_history_delta_indices=_SEAL_MEMORY70_HISTORY_INDICES,
                    state_delta_pairs=_SEAL_MEMORY70_DELTA_PAIRS,
                    use_delta_joint_actions=True,
                    adapt_to_pi=True,
                ),
            ),
        ),
        eval_data=DualYamMemoryDataConfig(
            repo_id="seal-water-bottle-cap/expert-success-hil-suffix-mix-data",
            base_config=DataConfig(
                prompt_from_task=True,
                local_files_path=_SEAL_MIXED_DATA_PATH,
                episodes=_SEAL_MEMORY_VAL_EPISODES,
            ),
            assets=AssetsConfig(
                assets_dir=_SEAL_WATER120_ASSETS_DIR,
                asset_id=_SEAL_WATER120_ASSET_ID,
            ),
            state_history_delta_indices=_SEAL_MEMORY70_HISTORY_INDICES,
            state_delta_pairs=_SEAL_MEMORY70_DELTA_PAIRS,
            use_delta_joint_actions=True,
            adapt_to_pi=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(f"{_SEAL_WATER120_CHECKPOINT_DIR}/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=2e-5,
            decay_steps=100_000,
            decay_lr=2e-5,
        ),
        num_train_steps=300_000,
        batch_size=32,
        num_workers=64,
        log_interval=50,
        eval_interval=500,
        num_eval_batches=10,
        save_interval=20_000,
        keep_period=50_000,
        policy_metadata={
            "state_history_delta_indices": _SEAL_MEMORY70_HISTORY_INDICES,
        },
    ),
    TrainConfig(
        name="pi05_tower-of-hanoi-game",
        model=pi0_config.Pi0Config(pi05=True),
        data=DualYamDataConfig(
            repo_id="tower-of-hanoi-game/expert-data",
            base_config=DataConfig(prompt_from_task=True,  local_files_path="/Your/path/to/Posttraining-RFM-RSS2026/Challenge-phase1-dataset/tower-of-hanoi-game/expert-data"),
            use_delta_joint_actions=True,
            adapt_to_pi=True
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=200_000,
        batch_size=32,
        num_workers=64,
        save_interval=40_000
    ),
    _rss_memory_phase2_bc_config(
        task_slug="insert-mouse-battery",
        phase2_slug="insert_mouse_battery",
        checkpoint_subdir="mouse_80k",
        memory_dim=42,
        state_history_delta_indices=_SEAL_MEMORY42_HISTORY_INDICES,
        state_delta_pairs=_SEAL_MEMORY42_DELTA_PAIRS,
        max_token_len=240,
        expert_train_episodes=_INSERT_MEMORY_EXPERT_TRAIN_EPISODES,
        hil_suffix_train_episodes=_INSERT_MEMORY_HIL_SUFFIX_TRAIN_EPISODES,
        success_train_episodes=_INSERT_MEMORY_SUCCESS_TRAIN_EPISODES,
        val_episodes=_INSERT_MEMORY_VAL_EPISODES,
    ),
    _rss_memory_phase2_bc_config(
        task_slug="insert-mouse-battery",
        phase2_slug="insert_mouse_battery",
        checkpoint_subdir="mouse_80k",
        memory_dim=70,
        state_history_delta_indices=_SEAL_MEMORY70_HISTORY_INDICES,
        state_delta_pairs=_SEAL_MEMORY70_DELTA_PAIRS,
        max_token_len=320,
        expert_train_episodes=_INSERT_MEMORY_EXPERT_TRAIN_EPISODES,
        hil_suffix_train_episodes=_INSERT_MEMORY_HIL_SUFFIX_TRAIN_EPISODES,
        success_train_episodes=_INSERT_MEMORY_SUCCESS_TRAIN_EPISODES,
        val_episodes=_INSERT_MEMORY_VAL_EPISODES,
    ),
    _rss_memory_phase2_bc_config(
        task_slug="tower-of-hanoi-game",
        phase2_slug="tower_of_hanoi_game",
        checkpoint_subdir="hanoi_200k",
        memory_dim=42,
        state_history_delta_indices=_SEAL_MEMORY42_HISTORY_INDICES,
        state_delta_pairs=_SEAL_MEMORY42_DELTA_PAIRS,
        max_token_len=240,
        expert_train_episodes=_TOWER_MEMORY_EXPERT_TRAIN_EPISODES,
        hil_suffix_train_episodes=_TOWER_MEMORY_HIL_SUFFIX_TRAIN_EPISODES,
        success_train_episodes=_TOWER_MEMORY_SUCCESS_TRAIN_EPISODES,
        val_episodes=_TOWER_MEMORY_VAL_EPISODES,
    ),
    _rss_memory_phase2_bc_config(
        task_slug="tower-of-hanoi-game",
        phase2_slug="tower_of_hanoi_game",
        checkpoint_subdir="hanoi_200k",
        memory_dim=70,
        state_history_delta_indices=_SEAL_MEMORY70_HISTORY_INDICES,
        state_delta_pairs=_SEAL_MEMORY70_DELTA_PAIRS,
        max_token_len=320,
        expert_train_episodes=_TOWER_MEMORY_EXPERT_TRAIN_EPISODES,
        hil_suffix_train_episodes=_TOWER_MEMORY_HIL_SUFFIX_TRAIN_EPISODES,
        success_train_episodes=_TOWER_MEMORY_SUCCESS_TRAIN_EPISODES,
        val_episodes=_TOWER_MEMORY_VAL_EPISODES,
    ),
    TrainConfig(
        name="pi05_rss_multitask_smoke",
        model=pi0_config.Pi0Config(pi05=True),
        data=DualYamDataConfig(
            repo_id="filtered_multitask",
            base_config=DataConfig(prompt_from_task=True, local_files_path="/inspire/qb-ilm/project/gjjproject/public/xl/data/rss_challenge/filtered_multitask"),
            assets=AssetsConfig(
                assets_dir="/inspire/qb-ilm/project/gjjproject/public/xl/data/baseline_checkpoints/pi05_rss2026_multitask/199999/assets/v21",
                asset_id="rss2026_multitask",
            ),
            use_delta_joint_actions=True,
            adapt_to_pi=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/inspire/qb-ilm/project/gjjproject/public/xl/data/baseline_checkpoints/pi05_rss2026_multitask/199999/params"),
        num_train_steps=10,
        batch_size=2,
        num_workers=0,
        save_interval=100,
        overwrite=True,
        exp_name="smoke_test",
        wandb_enabled=False,
    ),
    TrainConfig(
        name="pi05_rss_generalist",
        model=pi0_config.Pi0Config(pi05=True),
        data=DualYamDataConfig(
            repo_id="mixed_multitask",
            base_config=DataConfig(prompt_from_task=True, local_files_path="/inspire/qb-ilm/project/gjjproject/public/xl/data/rss_challenge/mixed_multitask"),
            assets=AssetsConfig(
                assets_dir="/inspire/qb-ilm/project/gjjproject/public/xl/data/baseline_checkpoints/pi05_rss2026_multitask/199999/assets/v21",
                asset_id="rss2026_multitask",
            ),
            use_delta_joint_actions=True,
            adapt_to_pi=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/inspire/qb-ilm/project/gjjproject/public/xl/data/baseline_checkpoints/pi05_rss2026_multitask/199999/params"),
        num_train_steps=100_000,
        batch_size=32,
        num_workers=64,
        save_interval=20_000,
    ),
    TrainConfig(
        name="pi05_rss_multitask_ft",
        model=pi0_config.Pi0Config(pi05=True),
        data=DualYamDataConfig(
            repo_id="filtered_multitask",
            base_config=DataConfig(prompt_from_task=True, local_files_path="/inspire/qb-ilm/project/gjjproject/public/xl/data/rss_challenge/filtered_multitask"),
            assets=AssetsConfig(
                assets_dir="/inspire/qb-ilm/project/gjjproject/public/xl/data/baseline_checkpoints/pi05_rss2026_multitask/199999/assets/v21",
                asset_id="rss2026_multitask",
            ),
            use_delta_joint_actions=True,
            adapt_to_pi=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/inspire/qb-ilm/project/gjjproject/public/xl/data/baseline_checkpoints/pi05_rss2026_multitask/199999/params"),
        num_train_steps=100_000,
        batch_size=32,
        num_workers=64,
        save_interval=20_000,
    ),
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instuctions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    TrainConfig(
        # DROID whiteboard dataset collected in this workspace.
        #
        # The current LeRobot export stores 8-D actions that match absolute joint
        # positions plus gripper, so we convert the 7 joint dimensions to deltas
        # before training, matching the pi05 full-DROID joint-position path.
        name="pi05_droid_whiteboard_joint_position_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05-DROID checkpoints use 32-D padded state/action tensors.
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            repo_id="lerobot_whiteboard_v1",
            base_config=DataConfig(
                prompt_from_task=True,
                local_files_path="/inspire/qb-ilm/project/gjjproject/public/xl/data/droid_whiteboard/lerobot_whiteboard_v1",
            ),
            assets=AssetsConfig(
                # Reuse pi05-DROID normalization for compatibility with the base checkpoint.
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
            use_delta_joint_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        num_train_steps=20_000,
        batch_size=32,
        num_workers=8,
        log_interval=50,
        eval_interval=500,
        num_eval_batches=10,
        save_interval=2_000,
        keep_period=10_000,
    ),
    TrainConfig(
        # Low-memory LoRA variant of the joint-position whiteboard fine-tune.
        name="pi05_droid_whiteboard_joint_position_lora_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotDROIDDataConfig(
            repo_id="lerobot_whiteboard_v1",
            base_config=DataConfig(
                prompt_from_task=True,
                local_files_path="/inspire/qb-ilm/project/gjjproject/public/xl/data/droid_whiteboard/lerobot_whiteboard_v1",
            ),
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
            use_delta_joint_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=20_000,
        batch_size=32,
        num_workers=8,
        log_interval=50,
        eval_interval=500,
        num_eval_batches=10,
        save_interval=2_000,
        keep_period=10_000,
    ),
    TrainConfig(
        # Local smoke-test variant: same LoRA/data path as the DROID whiteboard
        # LoRA fine-tune, but loads an existing local pi05 checkpoint to avoid
        # downloading official pi05-DROID params during smoke tests.
        name="pi05_droid_whiteboard_joint_position_lora_local_smoke",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotDROIDDataConfig(
            repo_id="lerobot_whiteboard_v1",
            base_config=DataConfig(
                prompt_from_task=True,
                local_files_path="/inspire/qb-ilm/project/gjjproject/public/xl/data/droid_whiteboard/lerobot_whiteboard_v1",
            ),
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
            use_delta_joint_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/inspire/qb-ilm/project/gjjproject/public/xl/data/baseline_checkpoints/pi05_rss2026_multitask/199999/params"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=2,
        batch_size=1,
        num_workers=0,
        log_interval=1,
        eval_interval=0,
        num_eval_batches=0,
        save_interval=1,
        keep_period=10_000,
        overwrite=True,
        exp_name="smoke_joint_position_lora",
        wandb_enabled=False,
        tensorboard_enabled=False,
    ),
    TrainConfig(
        # True delta-EEF variant for DROID whiteboard. The generated dataset uses
        # 6-D Cartesian pose state plus gripper state, and 6-D next-step EEF delta
        # plus gripper action.
        name="pi05_droid_whiteboard_delta_eef_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=LeRobotDROIDEefDataConfig(
            repo_id="lerobot_whiteboard_delta_eef_v1",
            base_config=DataConfig(
                prompt_from_task=True,
                local_files_path="/inspire/qb-ilm/project/gjjproject/public/xl/data/droid_whiteboard/lerobot_whiteboard_delta_eef_v1",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        num_train_steps=20_000,
        batch_size=32,
        num_workers=8,
        log_interval=50,
        eval_interval=500,
        num_eval_batches=10,
        save_interval=2_000,
        keep_period=10_000,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    #
    # RoboArena configs.
    #
    *roboarena_config.get_roboarena_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
