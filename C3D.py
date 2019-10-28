# C3DServer API accessing functions
# - Moon Ki Jung

import pythoncom
import win32com.client as win32
import numpy as np
import pandas as pd



class C3DServer:
    itf = None
    reg_mode = 0
    
    def __init__(self):
        self.itf = win32.Dispatch('C3DServer.C3D')
        self.reg_mode = self.itf.GetRegistrationMode()
        if self.reg_mode == 0:
            print('Unregistered C3DServer')
        elif self.reg_mode == 1:
            print('Evaluation C3DServer')
        elif self.reg_mode == 2:
            print('Registered C3DServer')
        print('Version = ', self.itf.GetVersion())
        print(self.itf.GetRegUserName())
        print(self.itf.GetRegUserOrganization())

    def open_c3d(self, file_name):
        #print("Opening a C3D file:{}".format(file_name))
        if self.itf is None:
            self.itf = win32.Dispatch('C3DServer.C3D')        
        return self.itf.Open(file_name,3)        

    def save_c3d(self, file_name=''):
        return self.itf.SaveFile(file_name, -1)

    def close_c3d(self):
        self.itf.close()
        #del self.itf
        return
        
    def get_start_frame_number(self):
        return self.itf.GetVideoFrame(0)
    
    def get_end_frame_number(self):
        return self.itf.GetVideoFrame(1)
    
    def get_total_frame_counts(self):
        n_frames = self.itf.GetVideoFrame(1)-self.itf.GetVideoFrame(0)+1
        return n_frames
    
    def get_video_frame_rate(self):
        return self.itf.GetVideoFrameRate()
    
    def get_analog_frame_rate(self):
        return self.itf.GetVideoFrameRate()*self.itf.GetAnalogVideoRatio()
    
    def get_analog_video_ratio(self):
        return self.itf.GetAnalogVideoRatio()
    
    def get_video_frame_indices(self):
        start_frame = self.get_start_frame_number()
        end_frame = self.get_end_frame_number()
        n_frames = self.get_total_frame_counts()
        arr_indices = np.linspace(start = start_frame, stop = end_frame, num = n_frames, dtype=np.int32)
        return arr_indices
    
    def get_video_time_array(self):
        start_frame = self.get_start_frame_number()
        end_frame = self.get_end_frame_number()
        video_frame_rate = self.get_video_frame_rate()
        start_time = (start_frame-1)/video_frame_rate
        end_time = (end_frame-1)/video_frame_rate
        video_steps = self.get_total_frame_counts()
        arr_time = np.linspace(start = start_time, stop = end_time, num = video_steps)
        return arr_time
    
    def get_analog_time_array(self):
        start_frame = self.get_start_frame_number()
        end_frame = self.get_end_frame_number()
        video_frame_rate = self.get_video_frame_rate()
        analog_frame_rate = self.get_analog_frame_rate()
        start_time = (start_frame-1)/video_frame_rate
        end_time = (end_frame-1)/video_frame_rate+(self.itf.GetAnalogVideoRatio()-1)/analog_frame_rate
        #analog_steps = round((end_time-start_time)*analog_frame_rate)+1
        analog_steps = self.get_total_frame_counts()*self.itf.GetAnalogVideoRatio()
        arr_time = np.linspace(start = start_time, stop = end_time, num = analog_steps)
        return arr_time
    
    def get_video_time_array_subset(self, list_selection_mark):
        return self.get_video_time_array()[list_selection_mark]
    
    def get_analog_time_array_subset(self, list_selection_mark):
        return self.get_analog_time_array()[list_selection_mark]
    
    def get_marker_names(self):
        mkr_names = []
        par_idx = self.itf.GetParameterIndex('POINT', 'LABELS')
        n_items = self.itf.GetParameterLength(par_idx)
        for i in range(n_items):
            mkr_names.append(self.itf.GetParameterValue(par_idx, i))
        return mkr_names
    
    def get_marker_index(self, mkr_name):
        par_idx = self.itf.GetParameterIndex('POINT', 'LABELS')
        n_items = self.itf.GetParameterLength(par_idx)
        mkr_idx = -1
        for i in range(n_items):
            tgt_name = self.itf.GetParameterValue(par_idx, i)
            if tgt_name == mkr_name:
                mkr_idx = i
                break
        if mkr_idx == -1:
            print("There is no %s marker in this C3D file!" % mkr_name)
        return mkr_idx
    
    def get_marker_data(self, mkr_name, residual=True, *args):
        start_frame = self.itf.GetVideoFrame(0)
        end_frame = self.itf.GetVideoFrame(1)
        n_frames = self.get_total_frame_counts()
        if len(args) == 2:
            start_frame = args[0]
            end_frame = args[1]
        mkr_idx = self.get_marker_index(mkr_name)
        arr_ret_pts = np.zeros((n_frames, 4), dtype=np.float32)
        arr_ret_pts[:] = np.nan
        if mkr_idx != -1:
            for i in range(3):
                arr_ret_pts[:, i] = np.array(self.itf.GetPointDataEx(mkr_idx, i, start_frame, end_frame, 0))
            if residual:
                arr_ret_pts[:, 3] = np.array(self.itf.GetPointResidualEx(mkr_idx, start_frame, end_frame))
        return arr_ret_pts
    
    def get_marker_coords(self, mkr_name, *args):
        start_frame = self.itf.GetVideoFrame(0)
        end_frame = self.itf.GetVideoFrame(1)
        n_frames = self.get_total_frame_counts()
        if len(args) == 2:
            start_frame = args[0]
            end_frame = args[1]
        mkr_idx = self.get_marker_index(mkr_name)
        arr_ret_pts = np.zeros((n_frames, 3), dtype=np.float32)
        arr_ret_pts[:] = np.nan
        if mkr_idx != -1:
            for i in range(3):
                arr_ret_pts[:, i] = np.array(self.itf.GetPointDataEx(mkr_idx, i, start_frame, end_frame, 0))
        return arr_ret_pts
    
    def get_marker_residual(self, mkr_name, *args):
        start_frame = self.itf.GetVideoFrame(0)
        end_frame = self.itf.GetVideoFrame(1)
        n_frames = self.get_total_frame_counts()
        if len(args) == 2:
            start_frame = args[0]
            end_frame = args[1]
        mkr_idx = self.get_marker_index(mkr_name)
        arr_ret_resid = np.zeros((n_frames), dtype=np.float32)
        arr_ret_resid[:] = np.nan
        if mkr_idx != -1:
            arr_ret_resid[:] = np.array(self.itf.GetPointResidualEx(mkr_idx, start_frame, end_frame))
        return arr_ret_resid
    
    def get_all_marker_coords_df(self, scale_factor=1.0, blocked_nan=False):
        mkr_names = self.get_marker_names()
        n_frames = self.get_total_frame_counts()
        mkr_coord_names = [a+'_'+b for a in mkr_names for b in ['X', 'Y', 'Z']]
        df = pd.DataFrame(np.zeros((n_frames, len(mkr_coord_names))), columns=mkr_coord_names)
        for mkr_name in mkr_names:
            mkr_data = self.get_marker_data(mkr_name, blocked_nan)
            mkr_pts = mkr_data[:,0:3]
            if blocked_nan:
                mkr_blocked = np.where(mkr_data[:,3]==-1, True, False)
                mkr_pts[mkr_blocked,:] = np.nan            
            mkr_coords = [mkr_name+'_'+c for c in ['X', 'Y', 'Z']]
            df[mkr_coords] = mkr_pts[:,0:3]
        df[mkr_coord_names] = df[mkr_coord_names]*scale_factor
        return df
    
    def get_all_marker_coords_arr2d(self, scale_factor=1.0, blocked_nan=False):
        mkr_names = self.get_marker_names()
        n_frames = self.get_total_frame_counts()
        arr_mkr_pts = np.zeros((n_frames, len(mkr_names)*3))
        for idx, mkr_name in enumerate(mkr_names):
            mkr_data = self.get_marker_data(mkr_name, blocked_nan)
            mkr_pts = mkr_data[:,0:3]
            if blocked_nan:
                mkr_blocked = np.where(mkr_data[:,3]==-1, True, False)
                mkr_pts[mkr_blocked,:] = np.nan
            arr_mkr_pts[:,3*idx+0:3*idx+3] = mkr_pts[:,0:3]
        arr_mkr_pts = arr_mkr_pts*scale_factor
        return arr_mkr_pts
    
    def get_all_marker_coords_arr3d(self, scale_factor=1.0, blocked_nan=False):
        mkr_names = self.get_marker_names()
        n_frames = self.get_total_frame_counts()
        arr_mkr_pts = np.zeros((n_frames, len(mkr_names), 3))
        for idx, mkr_name in enumerate(mkr_names):
            mkr_data = self.get_marker_data(mkr_name, blocked_nan)
            mkr_pts = mkr_data[:,0:3]
            if blocked_nan:
                mkr_blocked = np.where(mkr_data[:,3]==-1, True, False)
                mkr_pts[mkr_blocked,:] = np.nan
            arr_mkr_pts[:,idx,0:3] = mkr_pts[:,0:3]
        arr_mkr_pts = arr_mkr_pts*scale_factor
        return arr_mkr_pts    
    
    def get_marker_unit(self):
        par_idx = self.itf.GetParameterIndex('POINT', 'UNITS')
        n_items = self.itf.GetParameterLength(par_idx)
        if n_items < 1:
            return None
        unit = self.itf.GetParameterValue(par_idx, n_items-1)
        return unit
    
    def get_analog_names(self):
        analog_names = []
        par_idx = self.itf.GetParameterIndex('ANALOG', 'LABELS')
        n_items = self.itf.GetParameterLength(par_idx)
        for i in range(n_items):
            analog_names.append(self.itf.GetParameterValue(par_idx, i))
        return analog_names
    
    def get_analog_index(self, sig_name):
        par_idx = self.itf.GetParameterIndex('ANALOG', 'LABELS')
        n_items = self.itf.GetParameterLength(par_idx)
        sig_idx = -1    
        for i in range(n_items):
            tgt_name = self.itf.GetParameterValue(par_idx, i)
            if tgt_name == sig_name:
                sig_idx = i
                break
        if sig_idx == -1:
            print("There is no %s analog channel in this C3D file!" % sig_name)
        return sig_idx
    
    def get_analog_unit(self, sig_name):
        sig_idx = self.get_analog_index(sig_name)
        if sig_idx == -1:
            return None
        else:
            par_idx = self.itf.GetParameterIndex('ANALOG', 'UNITS')
            par_val = self.itf.GetParameterValue(par_idx, sig_idx)
            return par_val
              
    def get_analog_data(self, sig_name, *args):
        start_frame = self.itf.GetVideoFrame(0)
        end_frame = self.itf.GetVideoFrame(1)
        if len(args) == 2:
            start_frame = args[0]
            end_frame = args[1]
        sig_idx = self.get_analog_index(sig_name)
        arr_ret = np.zeros(0, dtype=np.float32)
        if sig_idx != -1:
            arr_ret =\
            np.array(self.itf.GetAnalogDataEx(sig_idx, start_frame, end_frame, "1", 0, 0, "0"), dtype=np.float32)
        return arr_ret
    
    def change_marker_name(self, mkr_name_old, mkr_name_new):
        par_idx = self.itf.GetParameterIndex('POINT', 'LABELS')
        mkr_idx = -1
        mkr_idx = self.get_marker_index(mkr_name_old)
        if mkr_idx == -1:
            return False
        else:
            self.itf.SetParameterValue(par_idx, mkr_idx, mkr_name_new)
        return True

    def change_analog_name(self, sig_name_old, sig_name_new):
        par_idx = self.itf.GetParameterIndex('ANALOG', 'LABELS')
        sig_idx = -1
        sig_idx = self.get_analog_index(sig_name_old)
        if sig_idx == -1:
            return False
        else:
            self.itf.SetParameterValue(par_idx, sig_idx, sig_name_new)
        return True    
            
    def add_marker(self, mkr_name, mkr_pts):  
        start_frame = self.itf.GetVideoFrame(0)
        n_total_frames = self.get_total_frame_counts()    
        input_pts_dim = mkr_pts.ndim
        input_pts_shape = mkr_pts.shape
        if input_pts_dim != 2 or input_pts_shape[0] != n_total_frames:
            print("The dimension of the input is not compatible!")
            return False
        ret = 0
        # Add an parameter to the 'POINT:LABELS' section
        par_idx_point_labels = self.itf.GetParameterIndex('POINT', 'LABELS')
        ret = self.itf.AddParameterData(par_idx_point_labels, 1)
        cnt_point_labels = self.itf.GetParameterLength(par_idx_point_labels)
        variant = win32.VARIANT(pythoncom.VT_BSTR, np.string_(mkr_name))
        ret = self.itf.SetParameterValue(par_idx_point_labels, cnt_point_labels-1, variant)
        # Add a null parameter in the 'POINT:DESCRIPTIONS' section
        par_idx_point_desc = self.itf.GetParameterIndex('POINT', 'DESCRIPTIONS')
        ret = self.itf.AddParameterData(par_idx_point_desc, 1)
        cnt_point_desc = self.itf.GetParameterLength(par_idx_point_desc)
        variant = win32.VARIANT(pythoncom.VT_BSTR, np.string_(mkr_name))
        ret = self.itf.SetParameterValue(par_idx_point_desc, cnt_point_desc-1, variant)
        # Add a marker
        new_mkr_idx = self.itf.AddMarker()
        cnt_mkrs = self.itf.GetNumber3DPoints()
        arr_zeros = np.zeros((n_total_frames, ), dtype=np.float32)
        arr_masks = np.array(["0000000"]*n_total_frames, dtype = np.string_)
        variant = win32.VARIANT(pythoncom.VT_ARRAY|pythoncom.VT_R4, np.nan_to_num(mkr_pts[:, 0]))
        ret = self.itf.SetPointDataEx(cnt_mkrs-1, 0, start_frame, variant)
        variant = win32.VARIANT(pythoncom.VT_ARRAY|pythoncom.VT_R4, np.nan_to_num(mkr_pts[:, 1]))
        ret = self.itf.SetPointDataEx(cnt_mkrs-1, 1, start_frame, variant)
        variant = win32.VARIANT(pythoncom.VT_ARRAY|pythoncom.VT_R4, np.nan_to_num(mkr_pts[:, 2]))
        ret = self.itf.SetPointDataEx(cnt_mkrs-1, 2, start_frame, variant)
        variant = win32.VARIANT(pythoncom.VT_ARRAY|pythoncom.VT_R4, arr_zeros)
        ret = self.itf.SetPointDataEx(cnt_mkrs-1, 3, start_frame, variant)
        variant = win32.VARIANT(pythoncom.VT_ARRAY|pythoncom.VT_BSTR, arr_masks)
        ret = self.itf.SetPointDataEx(cnt_mkrs-1, 4, start_frame, variant)
        for idx, val in enumerate(mkr_pts[:, 0]):
            if val == 1:
                variant =  win32.VARIANT(pythoncom.VT_R4, val)
                ret = self.itf.SetPointData(cnt_mkrs-1, 0, start_frame+idx, variant)
        for idx, val in enumerate(mkr_pts[:, 1]):
            if val == 1:
                variant = win32.VARIANT(pythoncom.VT_R4, val)
                ret = self.itf.SetPointData(cnt_mkrs-1, 1, start_frame+idx, variant)
        for idx, val in enumerate(mkr_pts[:, 2]):
            if val == 1:
                variant = win32.VARIANT(pythoncom.VT_R4, val)
                ret = self.itf.SetPointData(cnt_mkrs-1, 2, start_frame+idx, variant)
        # Increase the value 'POINT:USED' by the 1
        par_idx_point_used = self.itf.GetParameterIndex('POINT', 'USED')
        cnt_point_used = self.itf.GetParameterValue(par_idx_point_used, 0)
        par_idx_point_labels = self.itf.GetParameterIndex('POINT', 'LABELS')
        cnt_point_labels = self.itf.GetParameterLength(par_idx_point_labels)
        if cnt_point_used != cnt_point_labels:
            ret = self.itf.SetParameterValue(par_idx_point_used, 0, cnt_point_labels)
        if ret == 1:
            return True
        else:
            return False
        
    def update_marker_coords(self, mkr_name, mkr_pts):  
        ret = 0
        start_frame = self.get_start_frame_number()
        n_total_frames = self.get_total_frame_counts()  
        input_pts_dim = mkr_pts.ndim
        input_pts_shape = mkr_pts.shape
        if input_pts_dim != 2 or input_pts_shape[0] != n_total_frames:
            print("The dimension of the input is not compatible!")
            return False
        mkr_idx = self.get_marker_index(mkr_name)
        if mkr_idx == -1:
            return False
        variant = win32.VARIANT(pythoncom.VT_ARRAY|pythoncom.VT_R4, np.nan_to_num(mkr_pts[:, 0]))
        ret = self.itf.SetPointDataEx(mkr_idx, 0, start_frame, variant)
        variant = win32.VARIANT(pythoncom.VT_ARRAY|pythoncom.VT_R4, np.nan_to_num(mkr_pts[:, 1]))
        ret = self.itf.SetPointDataEx(mkr_idx, 1, start_frame, variant)
        variant = win32.VARIANT(pythoncom.VT_ARRAY|pythoncom.VT_R4, np.nan_to_num(mkr_pts[:, 2]))
        ret = self.itf.SetPointDataEx(mkr_idx, 2, start_frame, variant)
        var_const = win32.VARIANT(pythoncom.VT_R4, 1)
        for idx, val in enumerate(mkr_pts[:, 0]):
            if val == 1:
                ret = self.itf.SetPointData(mkr_idx, 0, start_frame+idx, var_const)
        for idx, val in enumerate(mkr_pts[:, 1]):
            if val == 1:
                ret = self.itf.SetPointData(mkr_idx, 1, start_frame+idx, var_const)
        for idx, val in enumerate(mkr_pts[:, 2]):
            if val == 1:
                ret = self.itf.SetPointData(mkr_idx, 2, start_frame+idx, var_const)
        if ret == 1:
            return True
        else:
            return False
        
    def update_marker_residual(self, mkr_name, mkr_resid):  
        ret = 0
        start_frame = self.get_start_frame_number()
        n_total_frames = self.get_total_frame_counts()   
        input_resid_dim = mkr_resid.ndim
        input_resid_shape = mkr_resid.shape
        if input_resid_dim != 1 or input_resid_shape[0] != n_total_frames:
            print("The dimension of the input is not compatible!")
            return False
        mkr_idx = self.get_marker_index(mkr_name)
        if mkr_idx == -1:
            return False
        variant = win32.VARIANT(pythoncom.VT_ARRAY|pythoncom.VT_R4, mkr_resid)
        ret = self.itf.SetPointDataEx(mkr_idx, 3, start_frame, variant)
        var_const = win32.VARIANT(pythoncom.VT_R4, 1)
        for idx, val in enumerate(mkr_resid):
            if val == 1:
                ret = self.itf.SetPointData(mkr_idx, 3, start_frame+idx, var_const)
        if ret == 1:
            return True
        else:
            return False        
        
    def recover_marker_rigidbody(self, tgt_mkr_name, cl_mkr_names):
        print("Trying to recover %s marker ... " % tgt_mkr_name, end="")
        cnt_frames = self.get_total_frame_counts()
        mkr_unit = self.get_marker_unit()
        mkr_scale_factor = 0.001 if mkr_unit == 'mm' else 1.0
        dict_cl_mkr_pts = {}
        dict_cl_mkr_valid = {}
        arr_cl_mkr_valid = np.ones((cnt_frames), dtype=bool)
        for mkr in cl_mkr_names:
            arr_mkr_data = self.get_marker_data(mkr, True)
            dict_cl_mkr_pts[mkr] = arr_mkr_data[:, 0:3]*mkr_scale_factor
            dict_cl_mkr_valid[mkr] = np.where(arr_mkr_data[:,3]==-1, False, True)
            arr_cl_mkr_valid = np.logical_and(arr_cl_mkr_valid, dict_cl_mkr_valid[mkr])
        arr_tgt_mkr_data = self.get_marker_data(tgt_mkr_name, True)
        arr_tgt_mkr_pts = arr_tgt_mkr_data[:,0:3]*mkr_scale_factor
        arr_tgt_mkr_resid = arr_tgt_mkr_data[:,3]
        arr_tgt_mkr_valid = np.where(arr_tgt_mkr_resid == -1, False, True)
        arr_all_mkr_valid = np.logical_and(arr_cl_mkr_valid, arr_tgt_mkr_valid)
        arr_cl_mkr_only_valid = np.logical_and(arr_cl_mkr_valid, np.logical_not(arr_tgt_mkr_valid))
        if not np.any(arr_tgt_mkr_valid):
            print("Skipped.")
            #print('No target marker trajectory is available!')
            return False
        if not np.any(arr_all_mkr_valid):
            print("Skipped.")
            #print('No time frame where all necessary markers are available!')
            return False
        if not np.any(arr_cl_mkr_only_valid):
            print("Skipped.")
            #print('No need to fill the gap of the target marker!')
            return False
        arr_p0 = dict_cl_mkr_pts[cl_mkr_names[0]]
        arr_p1 = dict_cl_mkr_pts[cl_mkr_names[1]]
        arr_p2 = dict_cl_mkr_pts[cl_mkr_names[2]]
        arr_tgt = arr_tgt_mkr_pts
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
        arr_s_rel_tgt = np.einsum('ij,ijk->ik', (arr_tgt-arr_p0)[arr_all_mkr_valid], arr_mat_rot[arr_all_mkr_valid])
        arr_all_mkr_idx = np.where(arr_all_mkr_valid)[0]
        arr_cl_mkr_only_idx = np.where(arr_cl_mkr_only_valid)[0]
        arr_tgt_recovered = np.zeros((arr_cl_mkr_only_idx.size, 3))
        for cnt, idx in np.ndenumerate(arr_cl_mkr_only_idx):
            search_idx = np.searchsorted(arr_all_mkr_idx, idx)
            if search_idx >= arr_all_mkr_idx.shape[0] or search_idx == 0:
                s_rel_idx = (np.abs(arr_all_mkr_idx-idx)).argmin()
                s_rel_tgt = arr_s_rel_tgt[s_rel_idx]
            else:
                idx_right = search_idx
                idx_left = idx_right-1
                mkr_idx_right = arr_all_mkr_idx[idx_right]
                mkr_idx_left = arr_all_mkr_idx[idx_left]
                a = np.float32(idx-mkr_idx_left)
                b = np.float32(mkr_idx_right-idx)
                s_rel_tgt = (b*arr_s_rel_tgt[idx_left]+a*arr_s_rel_tgt[idx_right])/(a+b)
            arr_tgt_recovered[cnt] = arr_p0[idx]+np.dot(s_rel_tgt, arr_mat_rot[idx].T)
        arr_tgt[arr_cl_mkr_only_valid] = arr_tgt_recovered
        arr_tgt_unscaled = arr_tgt/mkr_scale_factor
        arr_tgt_mkr_resid[arr_cl_mkr_only_valid] = 0.0
        self.update_marker_coords(tgt_mkr_name, arr_tgt_unscaled)
        self.update_marker_residual(tgt_mkr_name, arr_tgt_mkr_resid)
        print("Updated.")
        return True
    
    def export_all_mkr_coords_vicon_csv(self, file_path, sep=','):
        arr_mkr_pts = self.get_all_marker_coords_arr2d(blocked_nan=True)
        arr_fr_idx = self.get_video_frame_indices()
        arr_sub_fr_idx = np.array(np.zeros(arr_fr_idx.shape), dtype=np.int32)
        mkr_names = self.get_marker_names()
        mkr_unit = self.get_marker_unit()
        video_fps = self.get_video_frame_rate()
        header_row_0 = "Trajectories"
        header_row_1 = str(int(video_fps))
        header_mkr_names = []
        for item in mkr_names:
            header_mkr_names.append("")
            header_mkr_names.append("")
            header_mkr_names.append(item)
        header_row_2 = sep.join(header_mkr_names)
        header_coord_names = []
        header_coord_names.append("Frame")
        header_coord_names.append("Sub Frame")
        for i in range(len(mkr_names)):
            header_coord_names.append("X")
            header_coord_names.append("Y")
            header_coord_names.append("Z")
        header_row_3 = sep.join(header_coord_names)
        header_units = []
        header_units.append("")
        header_units.append("")
        for i in range(len(mkr_names)):
            for j in range(3):
                header_units.append(mkr_unit)
        header_row_4 = sep.join(header_units)
        header_str = ""
        header_str = header_str+header_row_0+"\n"
        header_str = header_str+header_row_1+"\n"           
        header_str = header_str+header_row_2+"\n"
        header_str = header_str+header_row_3+"\n"
        header_str = header_str+header_row_4
        arr_mkr_data = np.hstack((np.column_stack((arr_fr_idx, arr_sub_fr_idx)), arr_mkr_pts))
        np.savetxt(file_path, arr_mkr_data, fmt='%.6g', delimiter=sep, comments='', header=header_str)

        