import os

import pydicom
import numpy as np
from matplotlib import pyplot as plt


def main(sort_slices=True):
    # Carregar fitxer a fitxer
    slices = [pydicom.dcmread(f'CT_Lung/{file}')
              for file in os.listdir('CT_Lung/')]

    if sort_slices:
        # Re-ordenar segons SliceLocation
        assert all(hasattr(slc, 'SliceLocation') for slc in slices)
        slices = sorted(slices, key=lambda s: s.SliceLocation)

    img = np.stack([slc.pixel_array for slc in slices])
    img = np.flip(img, axis=0)  # Per a visualitzar-ho amb la orientació correcta
    pixel_len = [slices[0].SliceThickness, slices[0].PixelSpacing[0], slices[0].PixelSpacing[1]]

    cm = plt.cm.get_cmap('bone')
    cm_min = np.amin(img)
    cm_max = np.amax(img)

    voxel = [d//2 for d in img.shape]

    # Visualització plans medials
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.imshow(img[:, voxel[1], :], cmap=cm, vmin=cm_min, vmax=cm_max,
               aspect=pixel_len[0]/pixel_len[2])
    ax2.imshow(img[:, :, voxel[2]], cmap=cm, vmin=cm_min, vmax=cm_max,
               aspect=pixel_len[0]/pixel_len[1])
    ax3.imshow(img[voxel[0], :, :], cmap=cm, vmin=cm_min, vmax=cm_max,
               aspect=pixel_len[1]/pixel_len[2])
    ax4.axis('off')
    fig.show()

    # Visualització talls d'interés
    fig, axes = plt.subplots(4, 4)
    for idx_slice, ax in zip(range(40, 56), axes.flat):
        ax.imshow(img[idx_slice, :, :], cmap=cm, vmin=cm_min, vmax=cm_max,
                   aspect=pixel_len[1]/pixel_len[2])
    ax4.axis('off')
    fig.show()


if __name__ == '__main__':
    main(sort_slices=True)
    main(sort_slices=False)
