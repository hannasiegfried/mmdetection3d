# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
import os
import sys
import time
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import mmcv
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from mmdet.visualization import DetLocalVisualizer, get_palette
from mmengine.dist import master_only
from mmengine.logging import print_log
from mmengine.structures import InstanceData
from mmengine.visualization import Visualizer as MMENGINE_Visualizer
from mmengine.visualization.utils import (check_type, color_val_matplotlib,
                                          tensor2ndarray)
from torch import Tensor

from mmdet3d.registry import VISUALIZERS
from mmdet3d.structures import (BaseInstance3DBoxes, Box3DMode,
                                CameraInstance3DBoxes, Coord3DMode,
                                DepthInstance3DBoxes, DepthPoints,
                                Det3DDataSample, LiDARInstance3DBoxes,
                                PointData, points_cam2img)
from .vis_utils import (proj_camera_bbox3d_to_img, proj_depth_bbox3d_to_img,
                        proj_lidar_bbox3d_to_img, to_depth_mode)

try:
    import open3d as o3d
    from open3d import geometry
    from open3d.visualization import Visualizer
except ImportError:
    o3d = geometry = Visualizer = None

import pyvista as pv
from fw_lidar.viewer_pyvista import ViewerPyVista
from fw_lidar.visu_img import color_tof


@VISUALIZERS.register_module()
class PyVistaVisualizer(DetLocalVisualizer):
    """PyVista Local Visualizer.

    - 3D detection and segmentation drawing methods

      - draw_bboxes_3d: draw 3D bounding boxes on point clouds
      - draw_proj_bboxes_3d: draw projected 3D bounding boxes on image
      - draw_seg_mask: draw segmentation mask via per-point colorization

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        points (np.ndarray, optional): Points to visualize with shape (N, 3+C).
            Defaults to None.
        image (np.ndarray, optional): The origin image to draw. The format
            should be RGB. Defaults to None.
        pcd_mode (int): The point cloud mode (coordinates): 0 represents LiDAR,
            1 represents CAMERA, 2 represents Depth. Defaults to 0.
        vis_backends (List[dict], optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
            Defaults to None.
        bbox_color (str or Tuple[int], optional): Color of bbox lines.
            The tuple of color should be in BGR order. Defaults to None.
        text_color (str or Tuple[int]): Color of texts. The tuple of color
            should be in BGR order. Defaults to (200, 200, 200).
        mask_color (str or Tuple[int], optional): Color of masks. The tuple of
            color should be in BGR order. Defaults to None.
        line_width (int or float): The linewidth of lines. Defaults to 3.
        frame_cfg (dict): The coordinate frame config while Open3D
            visualization initialization.
            Defaults to dict(size=1, origin=[0, 0, 0]).
        alpha (int or float): The transparency of bboxes or mask.
            Defaults to 0.8.
        multi_imgs_col (int): The number of columns in arrangement when showing
            multi-view images.

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> from mmengine.structures import InstanceData
        >>> from mmdet3d.structures import (DepthInstance3DBoxes
        ...                                 Det3DDataSample)
        >>> from mmdet3d.visualization import Det3DLocalVisualizer

        >>> det3d_local_visualizer = Det3DLocalVisualizer()
        >>> image = np.random.randint(0, 256, size=(10, 12, 3)).astype('uint8')
        >>> points = np.random.rand(1000, 3)
        >>> gt_instances_3d = InstanceData()
        >>> gt_instances_3d.bboxes_3d = DepthInstance3DBoxes(
        ...     torch.rand((5, 7)))
        >>> gt_instances_3d.labels_3d = torch.randint(0, 2, (5,))
        >>> gt_det3d_data_sample = Det3DDataSample()
        >>> gt_det3d_data_sample.gt_instances_3d = gt_instances_3d
        >>> data_input = dict(img=image, points=points)
        >>> det3d_local_visualizer.add_datasample('3D Scene', data_input,
        ...                                       gt_det3d_data_sample)

        >>> from mmdet3d.structures import PointData
        >>> det3d_local_visualizer = Det3DLocalVisualizer()
        >>> points = np.random.rand(1000, 3)
        >>> gt_pts_seg = PointData()
        >>> gt_pts_seg.pts_semantic_mask = torch.randint(0, 10, (1000, ))
        >>> gt_det3d_data_sample = Det3DDataSample()
        >>> gt_det3d_data_sample.gt_pts_seg = gt_pts_seg
        >>> data_input = dict(points=points)
        >>> det3d_local_visualizer.add_datasample('3D Scene', data_input,
        ...                                       gt_det3d_data_sample,
        ...                                       vis_task='lidar_seg')
    """

    def __init__(
        self,
        name: str = 'visualizer',
        points: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
        pcd_mode: int = 0,
        vis_backends: Optional[List[dict]] = None,
        save_dir: Optional[str] = None,
        bbox_color: Optional[Union[str, Tuple[int]]] = None,
        text_color: Union[str, Tuple[int]] = (200, 200, 200),
        mask_color: Optional[Union[str, Tuple[int]]] = None,
        line_width: Union[int, float] = 3,
        frame_cfg: dict = dict(size=1, origin=[0, 0, 0]),
        alpha: Union[int, float] = 0.8,
        multi_imgs_col: int = 3,
        fig_show_cfg: dict = dict(figsize=(18, 12))
    ) -> None:
        super().__init__(
            name=name,
            image=image,
            vis_backends=vis_backends,
            save_dir=save_dir,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            line_width=line_width,
            alpha=alpha)
        if points is not None:
            self.set_points(points, pcd_mode=pcd_mode, frame_cfg=frame_cfg)
        self.multi_imgs_col = multi_imgs_col
        self.fig_show_cfg.update(fig_show_cfg)

        self.flag_pause = False
        self.flag_next = False
        self.flag_exit = False
        self.viewer = None
        self.point_size_intermetric = 3

    def _clear_o3d_vis(self) -> None:
        """Clear open3d vis."""

        if hasattr(self, 'o3d_vis'):
            del self.o3d_vis
            del self.points_colors
            del self.view_control
            if hasattr(self, 'pcd'):
                del self.pcd

    def _initialize_vis(self, filename: str, show=True) -> None:
        """Initialize open3d vis according to frame_cfg.

        Args:
            frame_cfg (dict): The config to create coordinate frame in open3d
                vis.

        Returns:
            :obj:`o3d.visualization.Visualizer`: Created open3d vis.
        """
        cam={'position': (4.603877123316819, -54.6337574202994, 41.179474213289794), 'focal_point': (-6.131375256054346, 123.03976976361925, -35.33510426221356), 'view_up': (0.04581749076706278, 0.3974478377888943, 0.9164802091571551)}
        path, file = os.path.split(filename)
        self.viewer = ViewerPyVista(**cam, save_path=path, filename=file, off_screen=False)

    @master_only
    def set_points(self,
                   points: np.ndarray,
                   pcd_mode: int = 0,
                   vis_mode: str = 'replace',
                   frame_cfg: dict = dict(size=1, origin=[0, 0, 0]),
                   points_color: Tuple[float] = (0.8, 0.8, 0.8),
                   points_size: int = 2,
                   mode: str = 'xyz') -> None:
        """Set the point cloud to draw.

        Args:
            points (np.ndarray): Points to visualize with shape (N, 3+C).
            pcd_mode (int): The point cloud mode (coordinates): 0 represents
                LiDAR, 1 represents CAMERA, 2 represents Depth. Defaults to 0.
            vis_mode (str): The visualization mode in Open3D:

                - 'replace': Replace the existing point cloud with input point
                  cloud.
                - 'add': Add input point cloud into existing point cloud.

                Defaults to 'replace'.
            frame_cfg (dict): The coordinate frame config for Open3D
                visualization initialization.
                Defaults to dict(size=1, origin=[0, 0, 0]).
            points_color (Tuple[float]): The color of points.
                Defaults to (1, 1, 1).
            points_size (int): The size of points to show on visualizer.
                Defaults to 2.
            mode (str): Indicate type of the input points, available mode
                ['xyz', 'xyzrgb']. Defaults to 'xyz'.
        """
        assert points is not None
        assert vis_mode in ('replace', 'add')
        check_type('points', points, np.ndarray)
        #colors = color_tof(points_color[points_color != 0])
        colors = color_tof(points_color, vmax=1)
        self.viewer.add_point_cloud(points[:, :3], colors, point_size=self.point_size_intermetric)
        self.pcd = points[:, :3]

    # TODO: assign 3D Box color according to pred / GT labels
    # We draw GT / pred bboxes on the same point cloud scenes
    # for better detection performance comparison
    def draw_bboxes_3d(self,
                       bboxes_3d: BaseInstance3DBoxes,
                       bbox_color: Tuple[float] = (0, 1, 0),
                       points_in_box_color: Tuple[float] = (1, 0, 0),
                       rot_axis: int = 2,
                       center_mode: str = 'lidar_bottom',
                       mode: str = 'xyz') -> None:
        """Draw bbox on visualizer and change the color of points inside
        bbox3d.

        Args:
            bboxes_3d (:obj:`BaseInstance3DBoxes`): 3D bbox
                (x, y, z, x_size, y_size, z_size, yaw) to visualize.
            bbox_color (Tuple[float]): The color of 3D bboxes.
                Defaults to (0, 1, 0).
            points_in_box_color (Tuple[float]): The color of points inside 3D
                bboxes. Defaults to (1, 0, 0).
            rot_axis (int): Rotation axis of 3D bboxes. Defaults to 2.
            center_mode (str): Indicates the center of bbox is bottom center or
                gravity center. Available mode
                ['lidar_bottom', 'camera_bottom']. Defaults to 'lidar_bottom'.
            mode (str): Indicates the type of input points, available mode
                ['xyz', 'xyzrgb']. Defaults to 'xyz'.
        """
        # Before visualizing the 3D Boxes in point cloud scene
        # we need to convert the boxes to Depth mode
        check_type('bboxes', bboxes_3d, BaseInstance3DBoxes)

        if not isinstance(bboxes_3d, DepthInstance3DBoxes):
            bboxes_3d = bboxes_3d.convert_to(Box3DMode.DEPTH)

        # convert bboxes to numpy dtype
        bboxes_3d = tensor2ndarray(bboxes_3d.tensor)

        # in_box_color = np.array(points_in_box_color)

        for i in range(len(bboxes_3d)):
            center = bboxes_3d[i, 0:3]
            dim = bboxes_3d[i, 3:6]
            yaw = np.zeros(3)
            yaw[rot_axis] = bboxes_3d[i, 6]
            rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)

            if center_mode == 'lidar_bottom':
                # bottom center to gravity center
                center[rot_axis] += dim[rot_axis] / 2
            elif center_mode == 'camera_bottom':
                # bottom center to gravity center
                center[rot_axis] -= dim[rot_axis] / 2

            #x_len, y_len, z_len = center
            x_len, y_len, z_len = dim
            box = pv.Cube(center=center, x_length=x_len, y_length=y_len, z_length=z_len)
            points = box.points - center  # Shift to origin for rotation
            rotated_points = points @ rot_mat.T
            box.points = rotated_points + center  # Shift back to final center

            self.viewer.plotter.add_mesh(box, color=list(bbox_color[0]), opacity=1, style='wireframe')

            # change the color of points which are in box
            # if self.pcd is not None and mode == 'xyz':
            #     indices = box3d.get_point_indices_within_bounding_box(
            #         self.pcd.points)
            #     self.points_colors[indices] = np.array(bbox_color[i]) / 255.

        # update points colors
        #if self.pcd is not None:
        #    self.pcd.colors = o3d.utility.Vector3dVector(self.points_colors)
        #    self.o3d_vis.update_geometry(self.pcd)


    def _draw_instances_3d(self,
                           data_input: dict,
                           instances: InstanceData,
                           input_meta: dict,
                           vis_task: str,
                           show_pcd_rgb: bool = False,
                           palette: Optional[List[tuple]] = None) -> dict:
        """Draw 3D instances of GT or prediction.

        Args:
            data_input (dict): The input dict to draw.
            instances (:obj:`InstanceData`): Data structure for instance-level
                annotations or predictions.
            input_meta (dict): Meta information.
            vis_task (str): Visualization task, it includes: 'lidar_det',
                'multi-modality_det', 'mono_det'.
            show_pcd_rgb (bool): Whether to show RGB point cloud.
            palette (List[tuple], optional): Palette information corresponding
                to the category. Defaults to None.

        Returns:
            dict: The drawn point cloud and image whose channel is RGB.
        """

        # Only visualize when there is at least one instance
        if not instances or len(instances) == 0:
            return None
        
        if "gt_bboxes_3d" in instances:
            bboxes_3d = instances["gt_bboxes_3d"]
            labels_3d = instances["gt_labels_3d"]
        else:
            bboxes_3d = instances.bboxes_3d  # BaseInstance3DBoxes
            labels_3d = instances.labels_3d

        data_3d = dict()

        if vis_task in ['lidar_det', 'multi-modality_det']:
            assert 'points' in data_input
            points = data_input['points']
            check_type('points', points, (np.ndarray, Tensor))
            points = tensor2ndarray(points)

            if not isinstance(bboxes_3d, DepthInstance3DBoxes):
                points, bboxes_3d_depth = to_depth_mode(points, bboxes_3d)
            else:
                bboxes_3d_depth = bboxes_3d.clone()

            if 'axis_align_matrix' in input_meta:
                points = DepthPoints(points, points_dim=points.shape[1])
                rot_mat = input_meta['axis_align_matrix'][:3, :3]
                trans_vec = input_meta['axis_align_matrix'][:3, -1]
                points.rotate(rot_mat.T)
                points.translate(trans_vec)
                points = tensor2ndarray(points.tensor)

            max_label = int(max(labels_3d) if len(labels_3d) > 0 else 0)
            bbox_color = palette if self.bbox_color is None \
                else self.bbox_color
            bbox_palette = get_palette(bbox_color, max_label + 1)
            colors = [bbox_palette[label] for label in labels_3d]

            self.set_points(
                points, pcd_mode=2, points_color=data_input['pred_score'])
            self.draw_bboxes_3d(bboxes_3d_depth, bbox_color=colors)

            data_3d['bboxes_3d'] = tensor2ndarray(bboxes_3d_depth.tensor)
            data_3d['points'] = points

        return data_3d

    @master_only
    def show(self,
             save_path: Optional[str] = None,
             drawn_img_3d: Optional[np.ndarray] = None,
             drawn_img: Optional[np.ndarray] = None,
             win_name: str = 'image',
             wait_time: int = -1,
             continue_key: str = 'right',
             vis_task: str = 'lidar_det') -> None:
        """Show the drawn point cloud/image.

        Args:
            save_path (str, optional): Path to save open3d visualized results.
                Defaults to None.
            drawn_img_3d (np.ndarray, optional): The image to show. If
                drawn_img_3d is not None, it will show the image got by
                Visualizer. Defaults to None.
            drawn_img (np.ndarray, optional): The image to show. If drawn_img
                is not None, it will show the image got by Visualizer.
                Defaults to None.
            win_name (str): The image title. Defaults to 'image'.
            wait_time (int): Delay in milliseconds. 0 is the special value that
                means "forever". Defaults to 0.
            continue_key (str): The key for users to continue. Defaults to ' '.
        """

        # In order to show multi-modal results at the same time, we show image
        # firstly and then show point cloud since the running of
        # Open3D will block the process
        if hasattr(self, '_image'):
            if drawn_img is None and drawn_img_3d is None:
                # use the image got by Visualizer.get_image()
                if vis_task == 'multi-modality_det':
                    import matplotlib.pyplot as plt
                    is_inline = 'inline' in plt.get_backend()
                    img = self.get_image() if drawn_img is None else drawn_img
                    self._init_manager(win_name)
                    fig = self.manager.canvas.figure
                    # remove white edges by set subplot margin
                    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
                    fig.clear()
                    ax = fig.add_subplot()
                    ax.axis(False)
                    ax.imshow(img)
                    self.manager.canvas.draw()
                    if is_inline:
                        return fig
                    else:
                        fig.show()
                    self.manager.canvas.flush_events()
                else:
                    super().show(drawn_img_3d, win_name, wait_time,
                                 continue_key)
            else:
                if vis_task == 'multi-modality_det':
                    import matplotlib.pyplot as plt
                    is_inline = 'inline' in plt.get_backend()
                    img = drawn_img if drawn_img_3d is None else drawn_img_3d
                    self._init_manager(win_name)
                    fig = self.manager.canvas.figure
                    # remove white edges by set subplot margin
                    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
                    fig.clear()
                    ax = fig.add_subplot()
                    ax.axis(False)
                    ax.imshow(img)
                    self.manager.canvas.draw()
                    if is_inline:
                        return fig
                    else:
                        fig.show()
                    self.manager.canvas.flush_events()
                else:
                    if drawn_img_3d is not None:
                        super().show(drawn_img_3d, win_name, wait_time,
                                     continue_key)
                    if drawn_img is not None:
                        super().show(drawn_img, win_name, wait_time,
                                     continue_key)

        if hasattr(self, 'o3d_vis'):
            if hasattr(self, 'view_port'):
                self.view_control.convert_from_pinhole_camera_parameters(
                    self.view_port)
            self.flag_exit = not self.o3d_vis.poll_events()
            self.o3d_vis.update_renderer()
            # if not hasattr(self, 'view_control'):
            #     self.o3d_vis.create_window()
            #     self.view_control = self.o3d_vis.get_view_control()
            self.view_port = \
                self.view_control.convert_to_pinhole_camera_parameters()  # noqa: E501
            if wait_time != -1:
                self.last_time = time.time()
                while time.time(
                ) - self.last_time < wait_time and self.o3d_vis.poll_events():
                    self.o3d_vis.update_renderer()
                    self.view_port = \
                        self.view_control.convert_to_pinhole_camera_parameters()  # noqa: E501
                while self.flag_pause and self.o3d_vis.poll_events():
                    self.o3d_vis.update_renderer()
                    self.view_port = \
                        self.view_control.convert_to_pinhole_camera_parameters()  # noqa: E501

            else:
                while not self.flag_next and self.o3d_vis.poll_events():
                    self.o3d_vis.update_renderer()
                    self.view_port = \
                        self.view_control.convert_to_pinhole_camera_parameters()  # noqa: E501
                self.flag_next = False
            self.o3d_vis.clear_geometries()
            try:
                del self.pcd
            except (KeyError, AttributeError):
                pass
            if save_path is not None:
                if not (save_path.endswith('.png')
                        or save_path.endswith('.jpg')):
                    save_path += '.png'
                self.o3d_vis.capture_screen_image(save_path)
            if self.flag_exit:
                self.o3d_vis.destroy_window()
                self.o3d_vis.close()
                self._clear_o3d_vis()
                sys.exit(0)

    def escape_callback(self, vis):
        self.o3d_vis.clear_geometries()
        self.o3d_vis.destroy_window()
        self.o3d_vis.close()
        self._clear_o3d_vis()
        sys.exit(0)

    def space_action_callback(self, vis, action, mods):
        if action == 1:
            if self.flag_pause:
                print_log(
                    'Playback continued, press [SPACE] to pause.',
                    logger='current')
            else:
                print_log(
                    'Playback paused, press [SPACE] to continue.',
                    logger='current')
            self.flag_pause = not self.flag_pause
        return True

    def right_callback(self, vis):
        self.flag_next = True
        return False

    # TODO: Support Visualize the 3D results from image and point cloud
    # respectively
    @master_only
    def add_datasample(self,
                       name: str,
                       data_input: dict,
                       data_sample: Optional[Det3DDataSample] = None,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       show: bool = False,
                       wait_time: float = 0,
                       out_file: Optional[str] = None,
                       o3d_save_path: Optional[str] = None,
                       vis_task: str = 'mono_det',
                       pred_score_thr: float = 0.3,
                       step: int = 0,
                       show_pcd_rgb: bool = False) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are displayed
          in a stitched image where the left image is the ground truth and the
          right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and the images
          will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be saved to
          ``out_file``. It is usually used when the display is not available.

        Args:
            name (str): The image identifier.
            data_input (dict): It should include the point clouds or image
                to draw.
            data_sample (:obj:`Det3DDataSample`, optional): Prediction
                Det3DDataSample. Defaults to None.
            draw_gt (bool): Whether to draw GT Det3DDataSample.
                Defaults to True.
            draw_pred (bool): Whether to draw Prediction Det3DDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn point clouds and image.
                Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str, optional): Path to output file. Defaults to None.
            o3d_save_path (str, optional): Path to save open3d visualized
                results. Defaults to None.
            vis_task (str): Visualization task. Defaults to 'mono_det'.
            pred_score_thr (float): The threshold to visualize the bboxes
                and masks. Defaults to 0.3.
            step (int): Global step value to record. Defaults to 0.
            show_pcd_rgb (bool): Whether to show RGB point cloud. Defaults to
                False.
        """
        print(data_sample.lidar_path)
        assert vis_task in (
            'mono_det', 'multi-view_det', 'lidar_det', 'lidar_seg',
            'multi-modality_det'), f'got unexpected vis_task {vis_task}.'
        classes = self.dataset_meta.get('classes', None)
        # For object detection datasets, no palette is saved
        palette = self.dataset_meta.get('palette', None)
        ignore_index = self.dataset_meta.get('ignore_index', None)
        if vis_task == 'lidar_seg' and ignore_index is not None and 'pts_semantic_mask' in data_sample.gt_pts_seg:  # noqa: E501
            keep_index = data_sample.gt_pts_seg.pts_semantic_mask != ignore_index  # noqa: E501
        else:
            keep_index = None

        gt_data_3d = None
        pred_data_3d = None
        gt_img_data = None
        pred_img_data = None

        self.o3d_vis = self._initialize_vis(o3d_save_path)

        if draw_gt and data_sample is not None:
            if 'gt_instances_3d' in data_sample:
                gt_data_3d = self._draw_instances_3d(
                    data_input, data_sample.eval_ann_info,
                    data_sample.metainfo, vis_task, show_pcd_rgb, [(0, 255, 0)])

        if draw_pred and data_sample is not None:
            if 'pred_instances_3d' in data_sample:
                pred_instances_3d = data_sample.pred_instances_3d
                # .cpu can not be used for BaseInstance3DBoxes
                # so we need to use .to('cpu')
                print("print prediction")
                pred_instances_3d = pred_instances_3d[
                    pred_instances_3d.scores_3d > pred_score_thr].to('cpu')
                pred_data_3d = self._draw_instances_3d(data_input,
                                                       pred_instances_3d,
                                                       data_sample.metainfo,
                                                       vis_task, show_pcd_rgb,
                                                       palette)

        #gt_points = np.load("/media/hasiegf/data/carla_pcdet/points/" + data_sample.lidar_path + ".npy")
        #gt_points = gt_points.reshape(-1, 4)
        #self.set_points(self.rotate_points(gt_points, data_sample), pcd_mode=2, points_color=(230,230,230), vis_mode="add")
        self.viewer.show()
        self.viewer.save_screenshot()
        #self.viewer.show()

    def rotate_points(self, points, data_sample):
        points = tensor2ndarray(points)
        points, _ = to_depth_mode(points, None)

        if 'axis_align_matrix' in data_sample.metainfo:
            points = DepthPoints(points, points_dim=points.shape[1])
            rot_mat = data_sample.metainfo['axis_align_matrix'][:3, :3]
            trans_vec = data_sample.metainfo['axis_align_matrix'][:3, -1]
            points.rotate(rot_mat.T)
            points.translate(trans_vec)
            points = tensor2ndarray(points.tensor)

        return points