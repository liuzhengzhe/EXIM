import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    gt_uncliped_path = r'E:\03001627_3.npy'
    gt_clipped_path = r'E:\03001627_0.1_bior6.8_3.npy'

    np_uncliped = np.load(gt_uncliped_path)
    np_clipped = np.load(gt_clipped_path)

    sample_index = 0
    sample_clipped = np_clipped[sample_index].reshape(-1)
    sample_unclipped = np_uncliped[sample_index].reshape(-1)

    kwargs = dict(alpha=0.5, bins=100,  density=True, stacked=True)

    plt.hist(sample_clipped, **kwargs, color='g', label='clipped')
    plt.hist(sample_unclipped, **kwargs, color='b', label='unclipped')

    plt.gca().set(title='Frequency Histogram of Coefficient', ylabel='Frequency')
    plt.legend()

    plt.show()