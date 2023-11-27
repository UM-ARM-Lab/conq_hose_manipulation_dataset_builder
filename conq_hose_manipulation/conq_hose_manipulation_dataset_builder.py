import pickle
from pathlib import Path
from typing import Iterator, Tuple, Any

import cv2
import numpy as np
import rerun as rr
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from bosdyn.client.frame_helpers import get_a_tform_b, HAND_FRAME_NAME, VISION_FRAME_NAME, BODY_FRAME_NAME, \
    GRAV_ALIGNED_BODY_FRAME_NAME
from bosdyn.client.math_helpers import quat_to_eulerZYX
from conq.cameras_utils import image_to_opencv, RGB_SOURCES, ROTATION_ANGLE, rotate_image
from conq.data_recorder import get_state_vec
from facenet_pytorch.models.mtcnn import MTCNN


class ConqHoseManipulation(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Conq hose manipulation dataset."""

    VERSION = tfds.core.Version('1.1.0')
    RELEASE_NOTES = {
        '1.1.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

        self.mtcnn = MTCNN(
            image_size=160, margin=20, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device='cpu'
        )

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps':            tfds.features.Dataset({
                    'observation':                 tfds.features.FeaturesDict({
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
                    'action':                      tfds.features.Tensor(
                        shape=(8,),
                        dtype=np.float32,
                        doc='Robot action, consists of [3x hand delta position, '
                            '3x hand delta roll/pitch/yaw, 1x gripper, 1x is_terminal, in the current body frame.',
                    ),
                    'hand_in_vision':              tfds.features.Tensor(
                        shape=(8,),
                        dtype=np.float32,
                        doc='Robot action, consists of [3x hand absolute position, '
                            '3x hand absolute roll/pitch/yaw, 1x gripper, 1x is_terminal, in VISION_FRAME_NAME.',
                    ),
                    'hand_in_body_and_body_delta': tfds.features.Tensor(
                        shape=(14,),
                        dtype=np.float32,
                        doc='Robot action, consists of [3x hand absolute position, '
                            '3x hand absolute roll/pitch/yaw, '
                            '3x body delta position, '
                            '3x body delta roll/pitch/yaw, '
                            '1x gripper, 1x is_terminal, in current body frame.',
                    ),
                    'discount':                    tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward':                      tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first':                    tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last':                     tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal':                 tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction':        tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding':          tfds.features.Tensor(
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
        roots = [
            Path("/home/armlab/Documents/conq_python/data/conq_hose_manipulation_data_1700079751"),
            Path("/home/armlab/Documents/conq_python/data/conq_hose_manipulation_data_1700080718"),
            Path("/home/armlab/Documents/conq_python/data/conq_hose_manipulation_data_1700081021"),
            Path("/home/armlab/Documents/conq_python/data/conq_hose_manipulation_data_1700082505"),
            Path("/home/armlab/Documents/conq_python/data/conq_hose_manipulation_data_1700082672"),
            Path("/home/armlab/Documents/conq_python/data/conq_hose_manipulation_data_1700082856"),
            Path("/home/armlab/Documents/conq_python/data/conq_hose_manipulation_data_1700083298"),
            Path("/home/armlab/Documents/conq_python/data/conq_hose_manipulation_data_1700083463"),
            Path("/home/armlab/Documents/conq_python/data/regrasping_dataset_1697817143"),
        ]

        episode_paths_dict = {}
        for mode in ['train', 'val']:
            mode_paths = []
            for root in roots:
                mode_path = root / mode
                mode_paths.extend(list(mode_path.glob('episode_*.pkl')))
            episode_paths_dict[mode] = self._generate_examples(mode_paths)

        return episode_paths_dict

    def _generate_examples(self, episode_paths) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            episode_str = str(episode_path)
            with open(episode_path, 'rb') as f:
                data = pickle.load(f)

            episode = []
            skip = 2
            for t, step in enumerate(data[::skip]):
                # To compute we action for step, we look at the next steps' state
                if t >= len(data) - 1:
                    next_step = step
                else:
                    next_step = data[t + 1]

                instruction = step.get('instruction', 'no instruction')
                if instruction is None:
                    instruction = 'no instruction'

                state = step['robot_state']
                next_state = next_step['robot_state']
                action_vec = get_hand_delta_action_vec(data, t, state, next_state)
                hand_in_vision = get_hand_in_vision_action_vec(data, t, state, next_state)
                hand_in_body_and_body_delta = get_hand_in_body_and_body_delta_action_vec(data, t, state, next_state)
                state_vec = get_state_vec(state)

                missing_img = False
                for src, res in step['images'].items():
                    if res is None:
                        missing_img = True
                if missing_img:
                    print(f"Skipping step due to missing image in episode {episode_path}")

                # compute Kona language embedding
                language_embedding = self._embed([instruction])[0].numpy()

                observation = {
                    'state': state_vec,
                }

                for rgb_src in RGB_SOURCES:
                    res = step['images'][rgb_src]
                    rgb_np = image_to_opencv(res)
                    blurred = self.blur_faces(res, rgb_np)
                    observation[rgb_src] = blurred

                is_terminal = t == (len(data) - 1)
                episode.append({
                    'observation':                 observation,
                    'action':                      action_vec,
                    'hand_in_vision':              hand_in_vision,
                    'hand_in_body_and_body_delta': hand_in_body_and_body_delta,
                    'discount':                    1.0,
                    'reward':                      float(t == (len(data) - 1)),
                    'is_first':                    t == 0,
                    'is_last':                     is_terminal,
                    'is_terminal':                 is_terminal,
                    'language_instruction':        instruction,
                    'language_embedding':          language_embedding,
                })

            # create output data sample
            sample_i = {
                'steps':            episode,
                'episode_metadata': {
                    'file_path': episode_str
                }
            }

            return episode_str, sample_i

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # # For large datasets use Apache Beam
        # beam = tfds.core.lazy_imports.apache_beam
        # return (beam.Create(episode_paths) | beam.Map(_parse_example))

    def blur_faces(self, res, rgb_np):
        return blur_faces(self.mtcnn, res, rgb_np)


def blur_faces(mtcnn, res, rgb_np):
    angle = ROTATION_ANGLE[res.source.name]

    # NOTE: this makes the face detection work much better, but it does change the size of the images!
    rgb_np_rot = rotate_image(rgb_np, angle)
    # print(res.source.name, rgb_np_rot.shape)

    boxes, probs = mtcnn.detect(rgb_np_rot)

    if boxes is not None:
        # blur out the boxes
        blurred = rgb_np_rot.copy()
        for box in boxes:
            x0, y0, x1, y1 = box.astype(int)
            x0 = min(max(0, x0), rgb_np_rot.shape[1])
            x1 = min(max(0, x1), rgb_np_rot.shape[1])
            y0 = min(max(0, y0), rgb_np_rot.shape[0])
            y1 = min(max(0, y1), rgb_np_rot.shape[0])
            roi = rgb_np_rot[y0:y1, x0:x1]
            blurred_roi = cv2.GaussianBlur(roi, (31, 31), 0)
            blurred[y0:y1, x0:x1] = blurred_roi

            # draw the box
            # rr.log('blurred/img', rr.Image(blurred))
            # half_sizes = (box[2:] - box[:2])
            # rr.log("blurred/face_box", rr.Boxes2D(mins=box[:2], sizes=half_sizes))
            # print("detected a face!")
        return blurred

    return rgb_np_rot


def get_hand_in_vision_action_vec(data, t, state, next_state):
    next_snapshot = next_state.kinematic_state.transforms_snapshot
    next_hand_in_vision = get_a_tform_b(next_snapshot, VISION_FRAME_NAME, HAND_FRAME_NAME)

    gripper_action = state.manipulator_state.gripper_open_percentage / 100
    ee_pos = [next_hand_in_vision.x, next_hand_in_vision.y, next_hand_in_vision.z]
    try:
        euler_zyx = quat_to_eulerZYX(next_hand_in_vision.rotation)
    except ValueError:
        print(next_hand_in_vision.rotation)
        euler_zyx = [0, 0, 0]
    ee_rpy = [euler_zyx[2], euler_zyx[1], euler_zyx[0]]
    is_terminal = t == (len(data) - 1)
    action_vec = np.concatenate([ee_pos, ee_rpy, [gripper_action], [is_terminal]], dtype=np.float32)
    return action_vec


def get_hand_in_body_and_body_delta_action_vec(data, t, state, next_state):
    snapshot = state.kinematic_state.transforms_snapshot
    next_snapshot = next_state.kinematic_state.transforms_snapshot
    next_hand_in_body = get_a_tform_b(next_snapshot, BODY_FRAME_NAME, HAND_FRAME_NAME)
    body_in_vision = get_a_tform_b(snapshot, VISION_FRAME_NAME, BODY_FRAME_NAME)
    next_body_in_vision = get_a_tform_b(next_snapshot, VISION_FRAME_NAME, BODY_FRAME_NAME)
    # Transform next_body_in_vision into the body frame of the current state
    body_delta = body_in_vision.inverse() * next_body_in_vision
    # rr.log('body_delta/x', rr.TimeSeriesScalar(body_delta.x))
    # rr.log('body_delta/y', rr.TimeSeriesScalar(body_delta.y))
    # rr.log('body_delta/z', rr.TimeSeriesScalar(body_delta.z))
    # viz_common_frames(snapshot)

    gripper_action = state.manipulator_state.gripper_open_percentage / 100
    # absolute, not delta
    hand_in_body_pos = [next_hand_in_body.x, next_hand_in_body.y, next_hand_in_body.z]
    # delta
    body_delta_pos = [body_delta.x, body_delta.y, body_delta.z]
    hand_in_body_zyx = quat_to_eulerZYX(next_hand_in_body.rotation)
    body_delta_zyx = quat_to_eulerZYX(body_delta.rotation)
    hand_in_body_rpy = [hand_in_body_zyx[2], hand_in_body_zyx[1], hand_in_body_zyx[0]]
    body_delta_rpy = [body_delta_zyx[2], body_delta_zyx[1], body_delta_zyx[0]]
    is_terminal = t == (len(data) - 1)
    action_vec = np.concatenate([
        hand_in_body_pos,
        hand_in_body_rpy,
        body_delta_pos,
        body_delta_rpy,
        [gripper_action],
        [is_terminal]

    ], dtype=np.float32)
    return action_vec


def get_hand_delta_action_vec(data, t, state, next_state):
    snapshot = state.kinematic_state.transforms_snapshot
    next_snapshot = next_state.kinematic_state.transforms_snapshot
    hand_in_vision = get_a_tform_b(snapshot, VISION_FRAME_NAME, HAND_FRAME_NAME)
    next_hand_in_vision = get_a_tform_b(next_snapshot, VISION_FRAME_NAME, HAND_FRAME_NAME)
    vision2base = get_a_tform_b(snapshot, GRAV_ALIGNED_BODY_FRAME_NAME, VISION_FRAME_NAME)
    hand_in_body = vision2base * hand_in_vision
    next_hand_in_body = vision2base * next_hand_in_vision
    delta_hand_in_body = next_hand_in_body * hand_in_body.inverse()

    gripper_action = state.manipulator_state.gripper_open_percentage / 100
    ee_pos = [delta_hand_in_body.x, delta_hand_in_body.y, delta_hand_in_body.z]
    euler_zyx = quat_to_eulerZYX(delta_hand_in_body.rotation)
    ee_rpy = [euler_zyx[2], euler_zyx[1], euler_zyx[0]]
    is_terminal = t == (len(data) - 1)
    action_vec = np.concatenate([ee_pos, ee_rpy, [gripper_action], [is_terminal]], dtype=np.float32)
    return action_vec


def debug_conversion():
    rr.init('debug_conversion')
    rr.connect()

    root = Path("/home/armlab/Documents/conq_python/data/regrasping_dataset_1697817143")
    root = Path("/home/armlab/Documents/conq_python/data/conq_hose_manipulation_data_1700083463")
    mode = root / 'train'

    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device='cpu', keep_all=True
    )

    # episode_path = mode / 'episode_9.pkl'
    # rgb_src = 'left_fisheye_image'
    # with open(episode_path, 'rb') as f:
    #     data = pickle.load(f)
    #
    # for t in range(350, 400):
    #     step = data[t]
    #     res = step['images'][rgb_src]
    #     rgb_np = image_to_opencv(res)
    #     rr.log('rgb', rr.Image(rgb_np))
    #     blurred = blur_faces(mtcnn, res, rgb_np)

    for episode_path in mode.glob("episode_*.pkl"):
        print(episode_path)
        with open(episode_path, 'rb') as f:
            data = pickle.load(f)

        for t, step in enumerate(data):
            # To compute we action for step, we look at the next steps' state
            if t >= len(data) - 2:
                next_step = step
            else:
                next_step = data[t + 1]
            state = step['robot_state']
            next_state = next_step['robot_state']
            state_vec = get_state_vec(state)
            action_vec = get_hand_delta_action_vec(data, t, state, next_state)
            hand_in_vision = get_hand_in_vision_action_vec(data, t, state, next_state)
            hand_in_body_and_body_delta = get_hand_in_body_and_body_delta_action_vec(data, t, state, next_state)

            # for rgb_src in RGB_SOURCES:
            rgb_src = 'frontleft_fisheye_image'
            res = step['images'][rgb_src]
            rgb_np = image_to_opencv(res)
            blurred_np = blur_faces(mtcnn, res, rgb_np)
            # rr.log('rgb', rr.Image(blurred_np))


def check_for_bad_pkls():
    root = Path("/home/armlab/Documents/conq_python/data/")
    for mode in ['train', 'val']:
        for d in root.glob("*"):
            for episode_path in (d / mode).glob("episode_*.pkl"):
                try:
                    with open(episode_path, 'rb') as f:
                        data = pickle.load(f)
                except (pickle.UnpicklingError, EOFError):
                    print(f"{episode_path} is corrupt!")
                    # rename to add ".BAD" suffix
                    episode_path.rename(episode_path.with_suffix('.pkl.BAD'))


if __name__ == '__main__':
    # check_for_bad_pkls()
    debug_conversion()
