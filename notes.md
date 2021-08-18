# Status of DSS Network: Aug 16 2021

### Implemented changes:
- `models.py`: Added pooling and upconv layers to the U-Net in order to accept 512x512 sized images (originally 64x64). The network is able to complete training with a placeholder dataset of size (3, 5, 512, 512, 4).
- Created `run.py`: select and run a network
- Created `load_sets.py`: load `.uvh5` files across 5 nights and, using an arbitrarily-defined Night 1 as reference, calculate the minimum shift for each antenna pair such that the total distance from each shifted image to the Night 1 reference is minimized. This calculation can be done via FFT or regular convolution. The shifts are then returned.
- Created `shift_sets.py`: apply shifts to sets of images and crop as needed.

### To-do:
- Create a dataset with ground truths to train on. The easiest way would be to add simulated masks to the existing real data from the 5 nights. This step requires:
    - Decide whether to use overlapping masks or not (i.e. whether a particular area should be masked in multiple images in a set, or not). This will require analyzing the existing real data.
    - Generate simulated masks that respect the proportions of the masks from the 5 nights (which are different than the masks previously used)
    - Decide whether the DSS network should output 1 or 5 images. The U-Net is currently set to output 1, but Adrian prefers 5 for greater flexibility.
- Train the U-Net!

Finally, feel free to reach out via my McGill email with any questions about the code.