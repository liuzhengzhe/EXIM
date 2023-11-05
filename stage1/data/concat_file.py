import os
import numpy as np

if __name__ == '__main__':

    folder_path = r''
    save_path = r''

    loading_paths = [ os.path.join(folder_path, filename) for filename in os.listdir(folder_path)]

    results = []
    for idx, path in enumerate(loading_paths):
        values = np.load(path)
        results.append(values[None, :])
        print(f"{idx} done!")

    results = np.concatenate(results, axis = 0)

    np.save(save_path, results)