from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
from matplotlib.patches import Arrow
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import numpy as np
import torch
import os
from utils import quaternion_to_matrix_torch
from vis_utils import checkerboard_rectangle_aligned

def plot_animated_snake(
    pos, path_to_save,
    g=None, gt=None, gcp=None,
    broken_joint_ids=None, obstacle=None,
    n_pos_per_curve=None,
    exponent=1.0, xy_lim=None, 
    show_orientation=False, show_snake_trail=False,
    show_g_trail=False, show_g_start=False,
    show_joints=False,
    arrow_params=None, min_aspect_ratio=0.2,
    padding_mult_xy=None,
):
    '''Plot an animated snake
    
    Args:
        pos: (n_steps, n_points, 3) array representing the positions of the snake
        path_to_save: string representing the folder where the images will be saved
        g: (n_steps, 7) array representing the rigid transformations
        gt: (7,) array representing the target rigid transformations
        gcp: (n_cp, 7) array representing the checkpoints
        broken_joint_ids: list of int representing the indices of the broken joints
        obstacle: an object of type ImplicitFunction that can be found in obstacle_implicits.py
        exponent: float representing the exponent of the alpha in the plot
        xy_lim: (2, 2) array representing the limits of the x and y axes
        show_orientation: boolean telling if the orientation should be shown as arrows
        show_snake_trail: boolean telling if the snake trail should be shown
        show_g_trail: boolean telling if the previous center of mass positions should be shown
        show_g_start: boolean telling if the initial positioning of the snake should be shown
        show_joints: boolean specifying whether the joints should be shown at the vertices
        arrow_params: dictionary containing parameters for the arrows, e.g. length and width
        min_aspect_ratio: float representing the minimum aspect ratio of the plot
        padding_mult_xy: (2,) array representing the padding in the x and y directions as a fraction of the range of the x and y axes
    '''

    if padding_mult_xy is None:
        padding_mult_xy = np.array([0.05, 0.05])

    if broken_joint_ids is None:
        broken_joint_ids = []

    if n_pos_per_curve is None:
        n_pos_per_curve = [pos.shape[1]]
    slice_pos = np.cumsum([0] + n_pos_per_curve)
    
    if xy_lim is None:
        xy_lim = np.zeros(shape=(2, 2))
        xy_lim[:, 0] = np.min(pos.reshape(-1, 3)[:, :2], axis=0)
        xy_lim[:, 1] = np.max(pos.reshape(-1, 3)[:, :2], axis=0)
        range_xy = xy_lim[:, 1] - xy_lim[:, 0]
        xy_lim[:, 0] -= padding_mult_xy * range_xy
        xy_lim[:, 1] += padding_mult_xy * range_xy
    
    n_steps = pos.shape[0]
    alphas = np.linspace(0.1, 1.0, n_steps) ** exponent
    blue = tuple((np.array([59.0, 130.0, 219.0]) / 255.0).tolist())
    trail_color = mcolors.to_hex(tuple(3 * [150.0 / 255.0]))

    if show_orientation:
        assert arrow_params is not None
        if g is not None:
            g_orientation = arrow_params["length"] * quaternion_to_matrix_torch(torch.tensor(g)[..., :4])[..., 0, :2].numpy()
        if gt is not None:
            gt_orientation = arrow_params["length"] * quaternion_to_matrix_torch(torch.tensor(gt)[..., :4])[..., 0, :2].numpy()
        if gcp is not None:
            gcp_orientation = arrow_params["length"] * quaternion_to_matrix_torch(torch.tensor(gcp)[..., :4])[..., 0, :2].numpy()

    fig_area = 50
    aspect_ratio = max(min_aspect_ratio, (xy_lim[1, 1] - xy_lim[1, 0]) / (xy_lim[0, 1] - xy_lim[0, 0]))
    figure_width = 2 * round(np.sqrt(fig_area / aspect_ratio) / 2.0, 1)
    figure_height = 2 * round(np.sqrt(fig_area * aspect_ratio) / 2.0, 1)
    linewidth_snake = 7.0
    joints_size = 10.0 * linewidth_snake
    g_size = 6.0 * linewidth_snake

    operational_pos_ids = []
    n_tails = len(n_pos_per_curve)
    for tail_id in range(n_tails):
        operational_pos_ids += [i for i in np.arange(slice_pos[tail_id]+1, slice_pos[tail_id+1]-1) if (i-1-2*tail_id) not in broken_joint_ids]
    broken_pos_ids = [i+1 for i in broken_joint_ids]

    fig = plt.figure(figsize=(figure_width, figure_height))
    gs = fig.add_gridspec(1, 1)
    ax_tmp = fig.add_subplot(gs[0, 0])
    for id_step in range(n_steps):
        
        if show_snake_trail:
            for pos_id_start, pos_id_end in zip(slice_pos[:-1], slice_pos[1:]):
                line_collection = mcoll.LineCollection(pos[:id_step+1, pos_id_start:pos_id_end, :2], alpha=alphas[-id_step-1:], color=blue, linewidths=linewidth_snake, zorder=1)
                ax_tmp.add_collection(line_collection)
        else:
            for pos_id_start, pos_id_end in zip(slice_pos[:-1], slice_pos[1:]):
                ax_tmp.plot(pos[id_step, pos_id_start:pos_id_end, 0], pos[id_step, pos_id_start:pos_id_end, 1], lw=linewidth_snake, c=blue, alpha=alphas[-1], zorder=1)


        if show_joints:
            ax_tmp.scatter(pos[id_step, operational_pos_ids, 0], pos[id_step, operational_pos_ids, 1], marker='o', s=joints_size, c="k", zorder=1.5)
            ax_tmp.scatter(pos[id_step, broken_pos_ids, 0], pos[id_step, broken_pos_ids, 1], marker='o', s=joints_size, c="w", edgecolors="k", linewidths=2.0, zorder=1.5)

            if n_tails > 1:
                ax_tmp.scatter(pos[id_step, 0, 0], pos[id_step, 0, 1], marker='o', s=1.2*joints_size, c="k", zorder=1.5)
        
        if g is not None:
            if show_g_trail:
                ax_tmp.scatter(g[:id_step+1, 4], g[:id_step+1, 5], marker='x', s=g_size, c=trail_color, alpha=alphas[-id_step-1:], zorder=0.5)
                if show_orientation:
                    arrows_collection = [
                        Arrow(
                            g[id_step_in, 4], g[id_step_in, 5], g_orientation[id_step_in, 0], 
                            g_orientation[id_step_in, 1], width=arrow_params["width"]
                        ) for id_step_in in range(id_step+1)
                    ]
                    ax_tmp.add_collection(PatchCollection(arrows_collection, color=trail_color, alpha=alphas[-id_step-1:], zorder=0.5))
            else:
                ax_tmp.scatter(g[id_step, 4], g[id_step, 5], marker='x', s=g_size, c=trail_color, alpha=alphas[-1], zorder=0.5)
                if show_orientation:
                    ax_tmp.add_patch(Arrow(g[id_step, 4], g[id_step, 5], g_orientation[id_step, 0], g_orientation[id_step, 1], color=trail_color, width=arrow_params["width"], zorder=0.5))

            if show_g_start:
                g_start_size = 2.0 * g_size
                ax_tmp.scatter(g[0, 4], g[0, 5], marker='o', s=g_start_size, c='tab:red', alpha=1.0, zorder=0)
                if show_orientation:
                    ax_tmp.add_patch(Arrow(g[0, 4], g[0, 5], g_orientation[0, 0], g_orientation[0, 1], color='tab:red', width=arrow_params["width"], zorder=0))
            
        if gt is not None:
            rect_width = 0.03 * (xy_lim[0, 1] - xy_lim[0, 0])
            rect_height = 3.0 / 5.0 * rect_width
            lwidth = 17.0 * rect_width
            checkerboard_rectangle_aligned(
                ax_tmp, center=(gt[4].item(), gt[5].item()), 
                rect_width=rect_width, rect_height=rect_height, n_rows=5, n_cols=5, colors=('black', 'white'), 
                lwidth=lwidth, zorder=0
            )
            if show_orientation:
                ax_tmp.add_patch(Arrow(gt[4], gt[5], gt_orientation[0], gt_orientation[1], color='k', width=arrow_params["width"], zorder=0))
                
        if gcp is not None:
            ax_tmp.scatter(gcp[:, 4], gcp[:, 5], marker='o', s=30.0, c='tab:green', alpha=1.0, zorder=0)
            if show_orientation:
                arrows_collection = [
                    Arrow(
                        gcp[id_step_in, 4], gcp[id_step_in, 5], gcp_orientation[id_step_in, 0], 
                        gcp_orientation[id_step_in, 1], color='tab:green',
                        width=arrow_params["width"], zorder=0
                    ) for id_step_in in range(id_step+1)
                ]
                ax_tmp.add_collection(PatchCollection(arrows_collection))

        ax_tmp.set_xlim(xy_lim[0])
        ax_tmp.set_ylim(xy_lim[1])

        if obstacle is not None:

            n_plot = 1000
            n_levels = 15
            x_plot = torch.linspace(ax_tmp.get_xlim()[0], ax_tmp.get_xlim()[1], n_plot)
            y_plot = torch.linspace(ax_tmp.get_ylim()[0], ax_tmp.get_ylim()[1], n_plot)
            xyz_plot = torch.stack([
                torch.tile(x_plot, dims=(n_plot,)),
                torch.repeat_interleave(y_plot, repeats=n_plot, dim=0),
                torch.zeros(size=(n_plot*n_plot,))
            ], dim=1)

            sdfs = obstacle.evaluate_implicit_function(xyz_plot).reshape(n_plot, n_plot)

            min_sdf, max_sdf = torch.min(sdfs), torch.max(sdfs)
            max_abs_sdf = max(torch.abs(min_sdf), torch.abs(max_sdf))
            levels = np.linspace(-max_abs_sdf, max_abs_sdf, n_levels)
            ax_tmp.contourf(x_plot, y_plot, sdfs, levels=levels, cmap='coolwarm', zorder=-2)
            ax_tmp.contour(x_plot, y_plot, sdfs, levels=[0.0], colors='k', linewidths=3.0, zorder=-1)

        ax_tmp.set_aspect('equal')
        ax_tmp.axis('off')

        # Save the figure to a file
        plt.savefig(os.path.join(path_to_save, 'step_{}.png'.format(str(id_step).zfill(5))))
        ax_tmp.clear()
        
    plt.close(fig)
    print("Images saved to {}".format(path_to_save))