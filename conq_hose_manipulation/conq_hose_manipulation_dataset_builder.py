import pickle
from pathlib import Path
from typing import Iterator, Tuple, Any

import numpy as np
import tensorflow_datasets as tfds
import tensorflow_hub as hub


class ConqHoseManipulation(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Conq hose manipulation dataset."""

    VERSION = tfds.core.Version('1.10.0')
    RELEASE_NOTES = {
        '1.10.0': 'use delta actions recorded from actual actions, not backed-out from state. >50 demos now',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'hand_color_image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Hand camera RGB observation.',
                        ),
                        'frontleft_fisheye_image': tfds.features.Image(
                            shape=(726, 604, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Front Left RGB observation.',
                        ),
                        'frontright_fisheye_image': tfds.features.Image(
                            shape=(726, 604, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Front Right RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(66,),
                            dtype=np.float32,
                            doc='Concatenation of [joint states (2x: 20), body vel in vision (3 lin, 3 ang),'
                                'is_holding_item (1), estimated_end_effector_force_in_hand (3),'
                                'foot states (4x: (3 pos, 1 contact)))].'
                                'See bosdyn protos for details.',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(8,),
                        dtype=np.float32,
                        doc='Robot action, consists of [6x joint velocities, 1x gripper, 1x is_terminal].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
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
        episode_paths_dict = {}
        pkls_root = Path("pkls")
        for mode in ['train', 'val']:
            mode_paths = []
            for episode_path in pkls_root.glob(f"conq_hose_manipulation_{mode}_*.pkl"):
                mode_paths.append(episode_path)
            episode_paths_dict[mode] = self._generate_examples(mode_paths)

        return episode_paths_dict

    def _generate_examples(self, episode_paths) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            episode_str = str(episode_path)
            with open(episode_path, 'rb') as f:
                sample_i = pickle.load(f)

            return episode_str, sample_i

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # # For large datasets use Apache Beam
        # beam = tfds.core.lazy_imports.apache_beam
        # return (beam.Create(episode_paths) | beam.Map(_parse_example))
