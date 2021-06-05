import numpy as np
import random
import argparse
import os.path
import h5py
import itertools
import time
import sys

from hera_sim import foregrounds, noise, sigchain, rfi, simulate

########################################
##### Generate simulated vis plots #####
########################################

def generate_one_vis_plot(lsts, fqs, bl_len_ns):
    '''
    Generate one visibility waterfall plot
    Returns a 2D (x,y) plot, as well as a channel-separated plot (x,y,2)
    '''
    # point-source and diffuse foregrounds
    Tsky_mdl = noise.HERA_Tsky_mdl['xx']
    vis = foregrounds.diffuse_foreground(lsts, fqs, bl_len_ns, Tsky_mdl)
    vis += foregrounds.pntsrc_foreground(lsts, fqs, bl_len_ns, nsrcs=200)

    # noise
    tsky = noise.resample_Tsky(fqs,lsts,Tsky_mdl=noise.HERA_Tsky_mdl['xx'])
    t_rx = 150.
#     OMEGA_P = (0.72)*np.ones(1024) # from before; fraction of sky telescope is looking at; normalizes noise
    OMEGA_P = noise.bm_poly_to_omega_p(fqs) # default; from the hera_sim docs
    nos_jy = noise.sky_noise_jy(tsky + t_rx, fqs, lsts, OMEGA_P)
    vis += nos_jy
    
    # crosstalk, gains
    xtalk = sigchain.gen_whitenoise_xtalk(fqs)
#     g = sigchain.gen_gains(fqs, [1,2,3,4], 0) # default 0.1 # leave out for now
    vis = sigchain.apply_xtalk(vis, xtalk)
#     vis = sigchain.apply_gains(vis, g, (1,2))

    # separate new_vis into real and imaginary channels
    real = vis.real
    imag = vis.imag
    new_vis = np.zeros((vis.shape[0], vis.shape[1], 2))
    new_vis[:, :, 0] = real
    new_vis[:, :, 1] = imag

    return vis, new_vis


def generate_vis_plots(n, lsts, fqs, bl_len_ns):
    '''
    Generate n visibility waterfall plots
    '''
    temp = np.array([generate_one_vis_plot(lsts, fqs, bl_len_ns)[1] for i in range(n)]) # [1] to get separated vis
    print("TEMP", temp.shape)
    vis = np.ones((temp.shape[0], temp.shape[1], temp.shape[2], 3))
    vis[:, :, :, :2] = temp
    print("vis", vis.shape)
    return vis


def generate_simulated_vis_wrapper(n_examples, num_jd, start_freq, end_freq, num_reduced_channels, save):
    '''
    Wrapper for generating simulated visibility plots.
    Output: list of visibility waterfall plot of size: (n_examples, freq, time, 3)
    where the last channel is (real, imag, EMPTY)
    '''
    # set up visibility parameters
    lsts = np.linspace(0, 0.5*np.pi, num_jd, endpoint=False) # local sidereal times; start range, stop range, number of snapshots
    # num_jd = 1500 to match the mask; π/2 ~ 6h
    fqs = np.linspace(start_freq, end_freq, num_reduced_channels, endpoint=False) # frequencies in GHz; start freq, end freq, number of channels
    # fqs = np.linspace(.1, .2, 1024, endpoint=False) # original
    bl_len_ns = np.array([48.73,0,0]) # ENU coordinates # ORIGINALLY 30

    # generate vis list and masks
    vis_list = generate_vis_plots(n_examples, lsts, fqs, bl_len_ns)

    if save:
        prefix = f"{int(time.time())}_{len(vis_list)}_examples_visibilities"
        np.save(os.path.join("visibilities", f"{prefix}_1sample.npy"), vis_list[0])
        np.save(os.path.join("visibilities", f"{prefix}.npy"), vis_list)
        
        print(f"Dataset saved as {prefix}.npy")
    print(f"Data shape: {vis_list.shape}.")

    return vis_list


##############################
##### main function ##########
##############################

def main():
    # cmd line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_examples", type=int, help="Number of examples to generate")
    parser.add_argument("--num_jd", type=int, default=1500, help="Number of time samples")
    parser.add_argument("--num_channels", type=int, default=0.10927734375, help="Number of freq channels")
    parser.add_argument("--start_freq", type=int, default=0.18916015625, help="Start frequency")
    parser.add_argument("--end_freq", type=int, default=818, help="End frequency")
    # default numbers come from a run on a real mask from Saurabh

    parser.add_argument("--no-save", default=True, action="store_false", help="Use this flag to run tests without saving generated files to disk")
    args = parser.parse_args()

    n_examples = args.n_examples
    num_jd = args.num_jd
    num_channels = args.num_channels
    start_freq = args.start_freq
    end_freq = args.end_freq
    save = args.no_save

    # get visibilities
    _ = generate_simulated_vis_wrapper(n_examples, num_jd, start_freq, end_freq, num_channels, save)

    print("generate_vis.py complete.")


if __name__ == "__main__":
    main()