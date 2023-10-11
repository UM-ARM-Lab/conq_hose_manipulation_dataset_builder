import pickle
from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from conq.cameras_utils import image_to_opencv, RGB_SOURCES
from conq.data_recorder import get_state_vec
from bosdyn.client.frame_helpers import get_a_tform_b, HAND_FRAME_NAME, VISION_FRAME_NAME
from bosdyn.client.math_helpers import quat_to_eulerZYX


class ConqHoseManipulationDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Conq hose manipulation dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps':            tfds.features.Dataset({
                    'observation':          tfds.features.FeaturesDict({
                        'hand_color_image':         tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Hand camera RGB observation.',
                        ),
                        'back_fisheye_image':       tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Back RGB observation.',
                        ),
                        'frontleft_fisheye_image':  tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Front Left RGB observation.',
                        ),
                        'frontright_fisheye_image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Front Right RGB observation.',
                        ),
                        'left_fisheye_image':       tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Left RGB observation.',
                        ),
                        'right_fisheye_image':      tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Right RGB observation.',
                        ),
                        'state':                    tfds.features.Tensor(
                            shape=(66,),
                            dtype=np.float32,
                            doc='Concatenation of [joint states (2x: 20), body vel in vision (3 lin, 3 ang),'
                                'is_holding_item (1), estimated_end_effector_force_in_hand (3),'
                                'foot states (4x: (3 pos, 1 contact)))].'
                                'See bosdyn protos for details.',
                        )
                    }),
                    'action':               tfds.features.Tensor(
                        shape=(8,),
                        dtype=np.float32,
                        doc='Robot action, consists of [3x EE position, '
                            '3x EE roll/pitch/yaw, 1x gripper, 1x is_terminal in VISION_FRAME_NAME.',
                    ),
                    'discount':             tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward':               tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first':             tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last':              tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal':          tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding':   tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        from pathlib import Path
        root = Path("/home/armlab/Documents/conq_python/data/regrasping_dataset_1697036241")
        return {
            'train': self._generate_examples(path=str(root / 'train' / 'episode_*.pkl')),
            'val':   self._generate_examples(path=str(root / 'val' / 'episode_*.pkl')),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # Load the episode data
            with open(episode_path, 'rb') as f:
                data = pickle.load(f)

            episode = []
            for i, step in enumerate(data):
                instruction = step.get('instruction', 'no instruction')
                state = step['robot_state']
                snapshot = state.kinematic_state.transforms_snapshot
                hand_in_vision = get_a_tform_b(snapshot, HAND_FRAME_NAME, VISION_FRAME_NAME)
                gripper_action = state.manipulator_state.gripper_open_percentage / 100
                ee_pos = [hand_in_vision.x, hand_in_vision.y, hand_in_vision.z]
                euler_zyx = quat_to_eulerZYX(hand_in_vision.rotation)
                ee_rpy = [euler_zyx[2], euler_zyx[1], euler_zyx[0]]
                is_terminal = i == (len(data) - 1)
                action_vec = np.concatenate([ee_pos, ee_rpy, [gripper_action], [is_terminal]], dtype=np.float32)
                state_vec = get_state_vec(state)

                missing_img = False
                for src, res in step['images'].items():
                    if res is None:
                        missing_img = True
                if missing_img:
                    print(f"Skipping step due to missing image in episode {path}")

                # compute Kona language embedding
                language_embedding = self._embed([instruction])[0].numpy()

                observation = {
                    'state': state_vec,
                }

                for rgb_src in RGB_SOURCES:
                    res = step['images'][rgb_src]
                    rgb_np = image_to_opencv(res)
                    observation[rgb_src] = rgb_np

                episode.append({
                    'observation':          observation,
                    'action':               action_vec,
                    'discount':             1.0,
                    'reward':               float(i == (len(data) - 1)),
                    'is_first':             i == 0,
                    'is_last':              is_terminal,
                    'is_terminal':          is_terminal,
                    'language_instruction': instruction,
                    'language_embedding':   language_embedding,
                })

            # create output data sample
            sample = {
                'steps':            episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)
