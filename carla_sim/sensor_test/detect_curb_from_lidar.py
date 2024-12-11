import numpy as np
import matplotlib.pyplot as plt
import glob
import os


def find_threshold(data, threshold_factor=3):
    """
    Finds the threshold in x where the y-values change abruptly.

    Args:
        data: A list of tuples (x, y) representing the data points.
        threshold_factor: A factor to multiply the standard deviation to determine the threshold.

    Returns:
        The threshold value of x.
    """

    # Sort the data by x-coordinate
    data.sort(key=lambda x: x[0])
    if len(data) > 0:
        if np.max(np.array(data)[:, 1]) < -0.8:
            return 5
    # Calculate differences between consecutive y-values
    differences = np.diff([y for _, y in data])

    # Calculate standard deviation of differences
    std_dev = np.std(differences)
    # Find the index where the difference exceeds the threshold
    try:
        threshold_index = np.argmax(np.abs(differences) > threshold_factor * std_dev)
    except ValueError:
        return 5

    # Return the x-value at the threshold index
    return data[threshold_index + 1][0]  # +1 to get the x-value after the change


folder_path = './output'
lidar_extension = '*LiDAR.npy'


lidar_data_list = sorted(glob.glob(os.path.join(folder_path, lidar_extension)))
threshold_list = []
count_yes = 0
for l_idx, lidar_file in enumerate(lidar_data_list):
    try:
        test_data = np.load(lidar_file, allow_pickle=True)
    except:
        break
    points = test_data[:, :-1]
    points[:, :1] = -points[:, :1]
    points = points[points[:, 0] < 0.5, :]
    points = points[points[:, 1] < 5, :]
    points = points[points[:, 0] > -0.5, :]
    points = points[points[:, 1] > 0, :]

    points = points[:, 1:]
    # plt.figure(figsize=(4, 4), layout='constrained')
    # plt.plot(points[:, 0], points[:, 1], 'o')
    # plt.xlabel('Y_coordinate (m)', fontsize=14)
    # plt.ylabel('Z_coordinate (m)', fontsize=14)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.ylim([-0.9, -0.7])
    # plt.savefig(rf'./temp_curb/{l_idx}.png')

    threshold = find_threshold(points.tolist())

    threshold_list.append(threshold)
# plt.figure(figsize=(6, 4), layout='constrained')
# plt.plot(threshold_list)
# plt.xlabel('Tick', fontsize=15)
# plt.yticks(np.arange(3, 5.5, 0.5), fontsize=15)
# plt.ylabel('Curb distance', fontsize=15)
# plt.xticks(fontsize=15)
# plt.show()
