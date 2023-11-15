import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import tensorflow_datasets as tfds
import tqdm

from example_transform.transform import transform_step


def main():
    rr.init("viz_dataset")
    rr.connect()

    # create TF dataset
    dataset_name = "conq_hose_manipulation"
    print(f"Visualizing data from dataset: {dataset_name}")
    ds = tfds.load(dataset_name, split='val')

    # visualize episodes
    for i, episode in enumerate(ds.take(1)):
        for step in episode['steps']:
            rgb_np = step['observation']['hand_color_image'].numpy()
            rr.log('instruction', rr.TextLog(step['language_instruction'].numpy().decode()))
            rr.log('image', rr.Image(rgb_np))

    # visualize action and state statistics
    actions, states = [], []
    for episode in tqdm.tqdm(ds.take(1)):
        for step in episode['steps']:
            actions.append(step['action'].numpy())
            states.append(step['observation']['state'].numpy())
    actions = np.array(actions)
    states = np.array(states)
    action_mean = actions.mean(0)
    state_mean = states.mean(0)
    print(action_mean)
    print(state_mean)

    def vis_stats(vector, vector_mean, tag):
        assert len(vector.shape) == 2
        assert len(vector_mean.shape) == 1
        assert vector.shape[1] == vector_mean.shape[0]

        n_elems = vector.shape[1]
        fig = plt.figure(tag, figsize=(5 * n_elems, 5))
        for elem in range(n_elems):
            plt.subplot(1, n_elems, elem + 1)
            plt.hist(vector[:, elem], bins=20)
            plt.title(vector_mean[elem])

    vis_stats(actions, action_mean, 'action_stats')
    vis_stats(states, state_mean, 'state_stats')


if __name__ == '__main__':
    main()
