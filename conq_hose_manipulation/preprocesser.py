import pickle
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import tensorflow_hub as hub
from bosdyn.client.frame_helpers import get_a_tform_b, HAND_FRAME_NAME, VISION_FRAME_NAME, BODY_FRAME_NAME, \
    GRAV_ALIGNED_BODY_FRAME_NAME
from bosdyn.client.math_helpers import quat_to_eulerZYX, transform_se3velocity, SE3Velocity, SE3Pose, Quat
from matplotlib import pyplot as plt
from tqdm import tqdm

from conq.cameras_utils import image_to_opencv, ROTATION_ANGLE, rotate_image
from conq.data_recorder import get_state_vec
from conq.rerun_utils import viz_common_frames, rr_tform


def pairwise_steps(data):
    t = 0
    for i in range(len(data) - 1):
        yield t, data[i], data[i + 1]
        t += 1


def get_first_available_rgb_sources(episode_path):
    with open(episode_path, 'rb') as f:
        data = pickle.load(f)
    return list(data[0]['images'].keys())


def preprocessor_main():
    rr.init('preprocesser')
    rr.connect()

    # from facenet_pytorch.models.mtcnn import MTCNN
    # mtcnn = MTCNN(
    #     image_size=160, margin=20, min_face_size=20,
    #     thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    #     device='cuda'
    # )
    _embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    root_root = Path("~/Documents/octo_ws/conq_python/data").expanduser()
    roots = list(root_root.iterdir())

    pkls_root = Path("pkls")
    pkls_root.mkdir(exist_ok=True)

    # delete the existing files in the output directory
    for pkl_path in pkls_root.glob("*.pkl"):
        pkl_path.unlink()

    episode_paths_dict = {}
    for mode in ['train', 'val']:
        mode_paths = []
        for root in roots:
            if (root / 'unsorted').exists():
                print(f"Skipping unsorted data in {root}")
            mode_path = root / mode
            mode_paths.extend(list(mode_path.glob('episode_*.pkl')))
        episode_paths_dict[mode] = mode_paths

    available_rgb_sources = get_first_available_rgb_sources(episode_paths_dict['train'][0])

    rr_seq_t = 0
    all_dpos_mags = []
    episode_idx = 0
    for mode, episode_paths in episode_paths_dict.items():
        for episode_path in tqdm(episode_paths, desc=f'{mode=}'):
            episode_str = str(episode_path)
            with open(episode_path, 'rb') as f:
                data = pickle.load(f)

            episode = []

            # First discard
            # Throw out some data I don't want to use right now
            instruction = data[0].get('instruction', None)
            if instruction == 'drag the hose to big mess':
                continue
            elif instruction is None:
                continue

            # Downsample and trim the start & end of the data
            downsample_n = 1
            min_dpos_start_threshold = 0.005
            start_t_padding = 0
            data_subsampled = data[::downsample_n]
            trim_start_idx = None
            for t, step, next_step in pairwise_steps(data_subsampled):
                # check that the hand has started moving
                state = step['robot_state']
                next_state = next_step['robot_state']
                hand_delta_in_body = get_hand_delta_action_vec(False, state, next_state)
                if np.linalg.norm(hand_delta_in_body[:3]) > min_dpos_start_threshold:
                    trim_start_idx = t + start_t_padding  # add some padding
                    break
            if trim_start_idx is None:
                print(f"Skipping episode {episode_path} because the hand never moved!")
                continue
            trim_end_idx = len(data_subsampled) + 1

            data_trimmed = data_subsampled[trim_start_idx:trim_end_idx]
            is_terminal = False
            for t, step in enumerate(data_trimmed):
                state = step['robot_state']
                action = step['action']
                target_open_fraction = action['open_fraction']
                target_hand_in_vision = action['target_hand_in_vision']
                is_terminal = t >= (len(data_trimmed) - 1)
                action_vec = get_hand_delta_action_vec(is_terminal, state)
                state_vec = get_state_vec(state)

                snapshot = state.kinematic_state.transforms_snapshot
                dpos_mag = np.linalg.norm(action_vec[:3])
                all_dpos_mags.append(dpos_mag)

                missing_img = check_step_for_missing_images(step)
                if missing_img:
                    print(f"Skipping step due to missing image in episode {episode_path}")
                    continue

                language_embedding = _embed([instruction])[0].numpy()

                observation = {'state': state_vec, }

                for rgb_src in available_rgb_sources:
                    res = step['images'][rgb_src]
                    rgb_np = image_to_opencv(res)
                    angle = ROTATION_ANGLE[res.source.name]
                    # NOTE: this makes the face detection work much better, but it does change the size of the images!
                    rgb_np_rot = rotate_image(rgb_np, angle)
                    # FIXME: not blurring while I'm debugging because it is slow
                    # blurred = blur_faces(mtcnn, rgb_np_rot)
                    observation[rgb_src] = rgb_np_rot

                rr.set_time_sequence('step', rr_seq_t)
                viz_common_frames(snapshot)
                rr_tform('target_hand_in_vision', target_hand_in_vision)
                rr.log('dpos_mag', rr.TimeSeriesScalar(dpos_mag))
                for rgb_src in available_rgb_sources:
                    rr.log(rgb_src, rr.Image(observation[rgb_src]))

                rr_seq_t += 1

                episode.append({
                    'observation': observation,
                    'action': action_vec,
                    'discount': 1.0,
                    'reward': float(is_terminal),
                    'is_first': t == 0,
                    'is_last': is_terminal,
                    'is_terminal': is_terminal,
                    'language_instruction': instruction,
                    'language_embedding': language_embedding,
                })

            assert is_terminal, "The last step should be terminal!"

            # create output data sample
            sample_i = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_str
                }
            }

            # save as pickle
            pkl_path = pkls_root / f"conq_hose_manipulation_{mode}_{episode_idx}.pkl"
            with pkl_path.open('wb') as f:
                pickle.dump(sample_i, f)

            episode_idx += 1


def check_step_for_missing_images(step):
    missing_img = False
    for src, res in step['images'].items():
        if res is None:
            missing_img = True
    return missing_img


def blur_faces(mtcnn, rgb_np_rot):
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

            # Draw the box
            # rr.log('blurred/img', rr.Image(blurred))
            # half_sizes = (box[2:] - box[:2])
            # rr.log("blurred/face_box", rr.Boxes2D(mins=box[:2], sizes=half_sizes))
            # print("detected a face!")
        return blurred

    return rgb_np_rot


def get_joint_velocity_action_vec(is_terminal, state):
    joint_velocities = []
    for js in state.kinematic_state.joint_states:
        if 'arm' in js.name and js.name not in ['arm0.hr0', 'arm0.f1x']:
            joint_velocities.append(js.velocity.value)
            rr.log(f'joint_velocities/{js.name.replace(".", "_")}', rr.TimeSeriesScalar(js.velocity.value))
    joint_velocities = np.array(joint_velocities)

    gripper_action = state.manipulator_state.gripper_open_percentage / 100
    action_vec = np.concatenate([joint_velocities, [gripper_action], [is_terminal]], dtype=np.float32)
    return action_vec


def get_hand_velocity_in_body_action_vec(is_terminal, state):
    vel_in_vision = state.manipulator_state.velocity_of_hand_in_vision

    snapshot = state.kinematic_state.transforms_snapshot
    body_in_vision = get_a_tform_b(snapshot, VISION_FRAME_NAME, BODY_FRAME_NAME)

    vel_in_body = transform_se3velocity(body_in_vision.to_adjoint_matrix(), vel_in_vision)
    print(vel_in_vision)
    print(vel_in_body)
    l = SE3Velocity.from_proto(vel_in_vision).linear
    np.sqrt(l.x ** 2 + l.y ** 2 + l.z ** 2)
    l = SE3Velocity.from_proto(vel_in_body).linear
    np.sqrt(l.x ** 2 + l.y ** 2 + l.z ** 2)
    print()

    vel_in_body_np = np.array([
        vel_in_body.linear.x,
        vel_in_body.linear.y,
        vel_in_body.linear.z,
        vel_in_body.angular.x,
        vel_in_body.angular.y,
        vel_in_body.angular.z
    ])

    gripper_action = state.manipulator_state.gripper_open_percentage / 100
    action_vec = np.concatenate([vel_in_body_np, [gripper_action], [is_terminal]], dtype=np.float32)
    return action_vec


def get_hand_in_vision_action_vec(is_terminal, state, next_state):
    next_snapshot = next_state.kinematic_state.transforms_snapshot
    next_hand_in_vision = get_a_tform_b(next_snapshot, VISION_FRAME_NAME, HAND_FRAME_NAME)

    gripper_action = state.manipulator_state.gripper_open_percentage / 100
    ee_pos = [next_hand_in_vision.x, next_hand_in_vision.y, next_hand_in_vision.z]
    try:
        euler_zyx = quat_to_eulerZYX(next_hand_in_vision.rotation)
    except ValueError:
        print('Bad rotation!!!', next_hand_in_vision.rotation)
        euler_zyx = [0, 0, 0]
    ee_rpy = [euler_zyx[2], euler_zyx[1], euler_zyx[0]]
    action_vec = np.concatenate([ee_pos, ee_rpy, [gripper_action], [is_terminal]], dtype=np.float32)
    return action_vec


def get_hand_in_body_and_body_delta_action_vec(is_terminal, state, next_state):
    snapshot = state.kinematic_state.transforms_snapshot
    next_snapshot = next_state.kinematic_state.transforms_snapshot
    next_hand_in_body = get_a_tform_b(next_snapshot, BODY_FRAME_NAME, HAND_FRAME_NAME)
    body_in_vision = get_a_tform_b(snapshot, VISION_FRAME_NAME, BODY_FRAME_NAME)
    next_body_in_vision = get_a_tform_b(next_snapshot, VISION_FRAME_NAME, BODY_FRAME_NAME)
    # Transform next_body_in_vision into the body frame of the current state
    body_delta = body_in_vision.inverse() * next_body_in_vision

    gripper_action = state.manipulator_state.gripper_open_percentage / 100
    # absolute, not delta
    hand_in_body_pos = [next_hand_in_body.x, next_hand_in_body.y, next_hand_in_body.z]
    # delta
    body_delta_pos = [body_delta.x, body_delta.y, body_delta.z]
    hand_in_body_zyx = quat_to_eulerZYX(next_hand_in_body.rotation)
    body_delta_zyx = quat_to_eulerZYX(body_delta.rotation)
    hand_in_body_rpy = [hand_in_body_zyx[2], hand_in_body_zyx[1], hand_in_body_zyx[0]]
    body_delta_rpy = [body_delta_zyx[2], body_delta_zyx[1], body_delta_zyx[0]]
    action_vec = np.concatenate([
        hand_in_body_pos,
        hand_in_body_rpy,
        body_delta_pos,
        body_delta_rpy,
        [gripper_action],
        [is_terminal]

    ], dtype=np.float32)
    return action_vec


def get_hand_delta_action_vec(is_terminal, state, next_state):
    snapshot = state.kinematic_state.transforms_snapshot
    next_snapshot = next_state.kinematic_state.transforms_snapshot
    hand_in_vision = get_a_tform_b(snapshot, VISION_FRAME_NAME, HAND_FRAME_NAME)
    next_hand_in_vision = get_a_tform_b(next_snapshot, VISION_FRAME_NAME, HAND_FRAME_NAME)
    vision2base = get_a_tform_b(snapshot, GRAV_ALIGNED_BODY_FRAME_NAME, VISION_FRAME_NAME)
    hand_in_body = vision2base * hand_in_vision
    next_hand_in_body = vision2base * next_hand_in_vision
    delta_hand_in_body = hand_in_body.inverse() * next_hand_in_body

    # NOTE: this will only grab very gently, because we're reading where the gripper _IS_ which is not the same
    # as where it was being commanded (e.g. fully closed), which might be what we actually want.
    # TODO: possibly make the data_recorder record the commanded gripper position?
    gripper_action = state.manipulator_state.gripper_open_percentage / 100
    ee_pos = [delta_hand_in_body.x, delta_hand_in_body.y, delta_hand_in_body.z]
    euler_zyx = quat_to_eulerZYX(delta_hand_in_body.rotation)
    ee_rpy = [euler_zyx[2], euler_zyx[1], euler_zyx[0]]
    action_vec = np.concatenate([ee_pos, ee_rpy, [gripper_action], [is_terminal]], dtype=np.float32)

    return action_vec


def pkl_itr():
    root = Path("/home/pmitrano/Documents/octo_ws/conq_python/data/")
    for mode in ['train', 'val']:
        for d in root.glob("*"):
            for episode_path in (d / mode).glob("episode_*.pkl"):
                with open(episode_path, 'rb') as f:
                    data = pickle.load(f)
                yield episode_path, data


def check_control_rate():
    rows = []
    for episode_path, data in pkl_itr():
        # compute the dt between steps
        ts = []
        for step in data:
            ts.append(step['time'])
        ts = np.array(ts)
        dts = np.diff(ts)
        hz = 1 / dts

        for hz_i in hz:
            rows.append([str(episode_path), hz_i])

    import pandas as pd
    df = pd.DataFrame(rows, columns=['episode_path', 'hz'])

    print(f"mean Hz: {df['hz'].agg('mean'):.2f}")
    print(f"median Hz: {df['hz'].agg('median'):.2f}")

    # do a simple MA filter to smooth the data and plot it
    w = 10
    df['hz_ma'] = df['hz'].rolling(w).mean()
    df['hz_ma'].plot()
    plt.show()


def change_instruction(new_instruction):
    for episode_path, data in pkl_itr():
        with open(episode_path, 'rb') as f:
            data = pickle.load(f)

        for step in data:
            step['instruction'] = new_instruction

        with open(episode_path, 'wb') as f:
            pickle.dump(data, f)


def check_for_bad_pkls():
    for episode_path, data in pkl_itr():
        try:
            with open(episode_path, 'rb') as f:
                data = pickle.load(f)
        except (pickle.UnpicklingError, EOFError):
            print(f"{episode_path} is corrupt!")
            # rename to add ".BAD" suffix
            episode_path.rename(episode_path.with_suffix('.pkl.BAD'))


if __name__ == '__main__':
    # change_instruction("grasp hose")
    check_control_rate()
    # check_for_bad_pkls()
    preprocessor_main()
