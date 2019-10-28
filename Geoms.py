# Geometrical plotting helper functions
# - Moon Ki Jung

from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial.transform import Slerp, Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#%%
class Geom(ABC):
    def __init__(self, name=None, axes=None):
        super().__init__()
        self.name = name
        self.arr_pos = None
        self.ax = axes
        self.vis_objs = []
    
    def set_name(self, name):
        self.name = name
    
    def attach_axes(self, axes):
        self.ax = axes
    
    def get_number_of_frames(self):
        return self.arr_pos.shape[0]
    
    @abstractmethod
    def set_from_points(self, point_coords):
        pass
    
    @abstractmethod
    def draw_vis_objs(self, fr):
        pass
    
    @abstractmethod
    def update_vis_objs(self, fr):
        pass

class Line(Geom):
    def __init__(self, name=None, axes=None):
        super().__init__(name, axes)
        self.pts_names = None
        self.n_pts = None
        
    def set_point_names(self, point_names):
        self.pts_names = point_names        
        
    def set_from_points(self, point_coords):
        self.n_pts = point_coords.shape[1]
        self.arr_pos = point_coords
        
    def set_from_2points(self, arr_p0, arr_p1):
        if arr_p0.shape != arr_p1.shape:
            print("Dimenstions of p0 and p1 are not compatible!")
            return False
        n_frames = arr_p0.shape[0]
        self.n_pts = 2
        self.arr_pos = np.empty((n_frames, 2, 3), dtype=np.float32)
        self.arr_pos[:,0,:] = arr_p0
        self.arr_pos[:,1,:] = arr_p1
        
    def draw_vis_objs(self, fr):
        if self.ax is None:
            return None
        pts_pair = np.empty((2, 3), dtype=np.float32)
        pts_pair[:,:] = self.arr_pos[fr,:,:]
        vis_obj = self.ax.plot3D(pts_pair[:,0], pts_pair[:,1], pts_pair[:,2], 'k-')[0]
        #vis_obj.set_label(self.pts_names[0]+"-"+self.pts_names[1])
        self.vis_objs.append(vis_obj)
        return self.vis_objs
    
    def update_vis_objs(self, fr):
        if self.ax is None:
            return None        
        pts_pair = np.empty((2, 3), dtype=np.float32)
        pts_pair[:,:] = self.arr_pos[fr,:,:]
        self.vis_objs[0].set_data_3d(pts_pair[:,0], pts_pair[:,1], pts_pair[:,2])
        return self.vis_objs
        
class PointCloud(Geom):
    def __init__(self, name=None, axes=None):
        super().__init__(name, axes)
        self.pts_names = None        
        self.n_pts = None
        self.marker_size = 8
        self.marker_color = 'b'
        self.marker_edge = 'b'

    def set_point_names(self, point_names):
        self.pts_names = point_names
        
    def set_from_points(self, point_coords):
        self.n_pts = point_coords.shape[1]
        self.arr_pos = point_coords        

    def set_marker_style(self, size_in, edge_in, face_in):
        self.marker_size = size_in
        self.marker_edge = edge_in
        self.marker_color = face_in

    def draw_vis_objs(self, fr):
        if self.ax is None:
            return None
        vis_obj = self.ax.plot(xs=self.arr_pos[fr,:,0],
                               ys=self.arr_pos[fr,:,1],
                               zs=self.arr_pos[fr,:,2],
                               linestyle='', marker='o', markersize=self.marker_size,
                               mfc=self.marker_color, mec=self.marker_edge)[0]
        vis_obj.set_label(self.name)
        self.vis_objs.append(vis_obj)
        return self.vis_objs
   
    def update_vis_objs(self, fr):
        if self.ax is None:
            return None        
        self.vis_objs[0].set_data_3d(
                self.arr_pos[fr,:,0],
                self.arr_pos[fr,:,1],
                self.arr_pos[fr,:,2])
        return self.vis_objs
        
class CoordSys(Geom):
    def __init__(self, name='', axes=None, size=0.1, linestyle='-', linewidth=1.5):
        super().__init__(name, axes)
        self.arr_rot = None
        self.arr_axes_pos = None
        self.colors = [(1,0,0),(0,1,0),(0,0,1)]
        self.size = size
        self.linestyle = linestyle
        self.linewidth = linewidth
        
    def set_size(self, size):
        self.size = size
        
    def set_colors(self, colors):
        for i in range(len(self.colors)):
            self.colors[i] = colors[i]
            
    def set_pos(self, arr_pos=None):
        if arr_pos is not None:
            self.arr_pos = arr_pos
        else:
            self.arr_pos = None
            
    def set_rot(self, arr_rot=None):
        if arr_rot is not None:
            self.arr_rot = arr_rot
        else:
            self.arr_rot = None
            
    def set_rot_from_quat(self, arr_quat_xyzw, interpol=False):
        if arr_quat_xyzw.shape[1] != 4:
            print("Dimensions of input quaternions are wrong!")
            return False
        n_frames = arr_quat_xyzw.shape[0]
        self.arr_rot = np.empty((n_frames, 3, 3), dtype=np.float32)
        r_obj = Rotation.from_quat(arr_quat_xyzw)
        if not interpol:
            self.arr_rot = r_obj.as_dcm()
        else:
            arr_frs = np.linspace(1, 1+n_frames, n_frames, dtype=int)
            r_interpol_obj = Slerp(arr_frs, r_obj)
            r_interpol = r_interpol_obj(arr_frs)
            self.arr_rot = r_interpol.as_dcm()
            
    def set_rot_from_3points(self, arr_p0, arr_p1, arr_p2):
        if arr_p0.shape != arr_p1.shape:
            print("Dimenstions of p0 and p1 are not compatible!")
            return False
        if arr_p1.shape != arr_p2.shape:
            print("Dimenstions of p1 and p2 are not compatible!")
            return False
        if arr_p2.shape != arr_p0.shape:
            print("Dimenstions of p2 and p0 are not compatible!")
            return False
        arr_vec0 = arr_p1-arr_p0
        arr_vec1 = arr_p2-arr_p0
        arr_vec0_norm = np.linalg.norm(arr_vec0, axis=1, keepdims=True)
        arr_vec1_norm = np.linalg.norm(arr_vec1, axis=1, keepdims=True)
        arr_vec0_unit = np.divide(arr_vec0, arr_vec0_norm, where=(arr_vec0_norm!=0))
        arr_vec1_unit = np.divide(arr_vec1, arr_vec1_norm, where=(arr_vec1_norm!=0))
        arr_vec2 = np.cross(arr_vec0_unit, arr_vec1_unit)
        arr_vec2_norm = np.linalg.norm(arr_vec2, axis=1, keepdims=True)
        arr_vec2_unit = np.divide(arr_vec2, arr_vec2_norm, where=(arr_vec2_norm!=0))
        arr_vec_x = arr_vec0_unit
        arr_vec_z = arr_vec2_unit
        arr_vec_y = np.cross(arr_vec_z, arr_vec_x)
        arr_mat_rot = np.array([arr_vec_x.T, arr_vec_y.T, arr_vec_z.T]).T
        self.arr_rot = arr_mat_rot
        
    def set_rot_from_points(self, point_coords):
        arr_p0 = point_coords[:,0,:]
        arr_p1 = point_coords[:,1,:]
        arr_p2 = point_coords[:,2,:]
        self.set_rot_from_3points(arr_p0, arr_p1, arr_p2)        
            
    def set_from_3points(self, arr_p0, arr_p1, arr_p2):
        self.set_pos(arr_p0)
        self.set_rot_from_3points(arr_p0, arr_p1, arr_p2)
        
    def set_from_points(self, point_coords):
        arr_p0 = point_coords[:,0,:]
        arr_p1 = point_coords[:,1,:]
        arr_p2 = point_coords[:,2,:]
        self.set_from_3points(arr_p0, arr_p1, arr_p2)
        
#    def apply_extrinsic_rotation(self, rot_mat):
#        for i in range(self.arr_rot.shape[0]):
#            self.arr_rot[i] = np.dot(rot_mat, self.arr_rot[i])
        
    def apply_intrinsic_rotation(self, rot_mat):
        self.arr_rot = np.dot(self.arr_rot, rot_mat)        
        
    def update_axes_pos(self):
        n_frames = self.get_number_of_frames()
        self.arr_axes_pos = np.empty((n_frames, 3, 3), dtype=np.float32)
        for fr in range(n_frames):
            for idx in range(3):
                self.arr_axes_pos[fr,idx,:] = self.arr_pos[fr]+self.arr_rot[fr,:,idx]*self.size
                
    def get_axis_dir(self, axis, scale_factor=1.0):
        n_frames = self.get_number_of_frames()
        arr_ret = np.empty((n_frames, 3), dtype=np.float32)
        for i in range(n_frames):
            arr_ret[i] = self.arr_rot[i,:,axis]*scale_factor
        return arr_ret
    
    def get_axis_pos(self, axis, scale_factor=1.0):
        n_frames = self.get_number_of_frames()
        arr_ret = np.empty((n_frames, 3), dtype=np.float32)
        for i in range(n_frames):
            arr_ret[i] = self.arr_pos[i]+self.arr_rot[i,:,axis]*scale_factor
        return arr_ret
                
    def draw_vis_objs(self, fr):
        if self.ax is None: return None
        if self.arr_axes_pos is None: return None
        pts_pairs = np.empty((3, 2, 3), dtype=np.float32)
        coords = ['X', 'Y', 'Z']
        for i in range(3):
            pts_pairs[i,0,:] = self.arr_pos[fr,:]
            pts_pairs[i,1,:] = self.arr_axes_pos[fr,i,:]
            vis_obj_axis = self.ax.plot3D(pts_pairs[i,:,0], pts_pairs[i,:,1], pts_pairs[i,:,2],
                                          color=self.colors[i], ls=self.linestyle, lw=self.linewidth)[0]
            vis_obj_axis.set_label(self.name+'_'+coords[i])
            self.vis_objs.append(vis_obj_axis)
        return self.vis_objs
    
    def update_vis_objs(self, fr):
        if self.ax is None: return None
        if self.arr_axes_pos is None: return None
        pts_pairs = np.empty((3, 2, 3), dtype=np.float32)
        for i in range(3):
            pts_pairs[i,0,:] = self.arr_pos[fr,:]
            pts_pairs[i,1,:] = self.arr_axes_pos[fr,i,:]
            self.vis_objs[i].set_data_3d(pts_pairs[i,:,0], pts_pairs[i,:,1], pts_pairs[i,:,2])
        return self.vis_objs