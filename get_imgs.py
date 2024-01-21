import src.am4ip.dataset as ds
import numpy as np
import matplotlib.pyplot as plt
import os

EDA_dataset = ds.CropSegmentationDataset(
                                        transform = np.array,
                                        target_transform = np.array,
                                        merge_small_items=False)

# EDA_dataset.
folder_name = './imgs'
if not os.path.exists(folder_name):
        os.makedirs(folder_name)
for i in range(0, EDA_dataset.__len__()):
    input_img, target = EDA_dataset.__getitem__(i)
    # plt.figure(figsize=(12,8))
    # plt.subplot(221)
    plt.imshow(input_img)
    # plt.title('Input Image')
    plt.axis('off')
    # plt.subplot(222)
    # plt.imshow(np.array(target)*60, cmap="gray")
    # plt.title('Label')
    # plt.axis('off')
    # plt.subplot(223)
    # # plt.imshow()
    # plt.title('Blended Image later')
    # plt.axis('off')
    # plt.subplot(224)
    # # plt.imshow()
    # plt.title('')
    # plt.axis('off')
    plt.savefig(f'{folder_name}/{i}.png')
