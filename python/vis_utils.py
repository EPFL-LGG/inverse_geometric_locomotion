import cv2
import ffmpeg
import glob
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import os
from scipy.interpolate import CubicSpline, interp1d
import warnings

retval = cv2.VideoWriter_fourcc(*'mp4v')
    
def produce_video_from_path(
        im_pattern, save_folder, anim_fn, 
        framerate=25, transparent=False, overwrite_anim=False
    ):
    '''Concatenates images into one video
    
    Args:
        im_pattern: string representing the pattern of the images to be concatenated
        save_folder: string representing the folder where the video will be saved
        anim_fn: string representing the name of the video
        framerate: int representing the framerate of the video
        transparent: bool representing whether the images have transparency
        overwrite_anim: bool representing whether to overwrite the video if it already exists
    '''
    video_name = os.path.join(save_folder, anim_fn)
    if transparent:
        (
            ffmpeg
            .input(im_pattern, pattern_type='sequence', framerate=framerate)
            .filter('crop', w='trunc(iw/2)*2', h='trunc(ih/2)*2')
            .output(video_name, vcodec='png')
            .run(overwrite_output=overwrite_anim)
        )
    else:
        (
            ffmpeg
            .input(im_pattern, pattern_type='sequence', framerate=framerate)
            .filter('crop', w='trunc(iw/2)*2', h='trunc(ih/2)*2')
            .output(video_name, vcodec='libopenh264', pix_fmt='yuv420p')
            .run(overwrite_output=overwrite_anim)
        )


def plot_sdf_2d(xs, ys, sdfs, n_levels=20, show_text=True, filename=None):
    
    '''
    Args:
        xs (torch.tensor of shape (n_plots_x,)): x coordinates of the points on the grid
        ys (torch.tensor of shape (n_plots_y,)): y coordinates of the points on the grid
        sdfs (torch.tensor of shape (n_plots_x * n_plots_y,)): the corresponding sdfs to plot
        n_levels: the number of levels to show
        show_text: whether to show the text or not
        filename: the name of the file to save
    '''
    
    assert xs.shape[0] == sdfs.shape[0] and ys.shape[0] == sdfs.shape[1], "check the dimensions of xs, ys, and sdfs"
    
    min_sdf, max_sdf = torch.min(sdfs), torch.max(sdfs)
    max_abs_sdf = max(torch.abs(min_sdf), torch.abs(max_sdf))
    levels = np.linspace(-max_abs_sdf, max_abs_sdf, n_levels)
    
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    ax.contourf(xs, ys, sdfs, levels=levels, cmap='coolwarm', zorder=-2)
    ax.contour(xs, ys, sdfs, levels=[0.0], colors='k', linewidths=3.0, zorder=-1)
    ax.set_aspect('equal')
    if not show_text:
        ax.axis('off')
    if not filename is None: plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.show()

# def render_video(file_name_prefix, output_file, path_to_output, fps=25, transparent=False):
#     '''Probably a duplicate of produce_video_from_path'''
#     # Extract png size in pixels
#     im = Image.open(os.path.join(path_to_output, file_name_prefix + "{:04d}.png".format(0)))  # assume all images are the same size
#     width, height = im.size

#     if transparent:
#         vcodec = 'png'  # Use 'png' for transparency, but outputs .mov or .mkv, not .mp4
#         pix_fmt = 'rgba'  # Use 'rgba' for transparency
#         output_file.replace('.mp4', '.mov')
#     else:
#         vcodec = 'libopenh264'
#         pix_fmt = 'yuv420p'  # Standard for MP4, no transparency
#         output_file.replace('.mov', '.mp4')

#     # Render video
#     (
#         ffmpeg
#         .input(os.path.join(path_to_output, file_name_prefix + "*.png"), pattern_type='glob', framerate=fps)
#         .output(os.path.join(path_to_output, output_file), vcodec=vcodec, pix_fmt=pix_fmt, vf="color=white:{}x{} [bg]; [bg][0:v] overlay=shortest=1".format(width, height))
#         .run(overwrite_output=True)
#     )

def checkerboard_rectangle_aligned(
    ax, center=None, rect_width=1.5, rect_height=1.0, n_rows=8, n_cols=8, 
    colors=('white', 'black'), lwidth=2, zorder=0,
):
    '''
    # Example usage to test the function
    fig, ax = plt.subplots(figsize=(6, 6))
    checkerboard_rectangle_aligned(ax, center=(0, 0), rect_width=1.5, rect_height=1.0, n_rows=8, n_cols=8)
    plt.show()
    '''
    if center is None:
        center = (0, 0)
    x0, y0 = center
    size_x = rect_width / n_cols
    size_y = rect_height / n_rows
    rect_x_min = x0 - rect_width / 2
    rect_y_min = y0 - rect_height / 2

    for i in range(n_rows):
        for j in range(n_cols):
            x_min = rect_x_min + j * size_x
            y_min = rect_y_min + i * size_y
            color = colors[(i + j) % 2]
            rect = plt.Rectangle((x_min, y_min), size_x, size_y, facecolor=color, edgecolor=None, zorder=zorder)
            ax.add_patch(rect)
    # Draw the rectangle boundary
    rect_patch = plt.Rectangle(
        (rect_x_min, rect_y_min), rect_width, rect_height, fill=False, edgecolor='k', linewidth=lwidth, zorder=zorder+0.5
    )
    ax.add_patch(rect_patch)
    ax.set_aspect('equal')
    ax.set_xlim(rect_x_min, rect_x_min + rect_width)
    ax.set_ylim(rect_y_min, rect_y_min + rect_height)
    ax.axis('off')

def print_json_data(js):
    '''Reads data generated by the experiments and prints it in a readable manner'''
    n_ts, n_vs = np.array(js['pos_']).shape[:2]
    print("Number of timesteps: ", n_ts)
    print("Number of vertices: ", n_vs)
    
    if 'optimization_settings' in js.keys():
        optim_settings = js['optimization_settings']
        if 'n_cp' in optim_settings.keys():
            print("Number of control points: ", optim_settings['n_cp'])
        else:
            print("Number of control points not available.")
        if 'close_gait' in optim_settings.keys():
            print("Is it a closed gate: ", optim_settings['close_gait'])
        else:
            print("Gait closedness not available.")
        if 'n_modes' in optim_settings.keys():
            print("Number of modes: ", optim_settings['n_modes'])
        else:
            print("Number of modes is not available.")
        
    else:
        print("Optimization settings not available.")
    
    if 'optimization_duration' in js.keys():
        print("Optimization duration (min:sec): {:02d}:{:02d}".format(
            int(js['optimization_duration'] // 60), 
            int(js['optimization_duration'] % 60))
        )
    else:
        print("Optimization duration not available.")
