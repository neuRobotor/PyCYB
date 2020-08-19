##############################################################
# Joint Coordinate System (JCS) as per Grood and Suntay (1983)
##############################################################
import numpy as np
from typing import List
import abs


def norm(a):
    return a / np.linalg.norm(a, axis=1)[:, None]


def vec_dot(a, b):
    return np.einsum('ij,ij->i', a, b)


class Segment:
    parent_joints: List['JCS']

    def __init__(self, lateral, frontal, longitudinal, name=None, normalize=True):
        self._lateral = Segment.check_input(lateral)
        self._frontal = Segment.check_input(frontal)
        self._longitudinal = Segment.check_input(longitudinal)
        self.axes = [self.lateral, self.frontal, self.longitudinal]
        self.name = name
        self.parent_joints = []
        if normalize:
            self.normalize_axes()

    def normalize_axes(self):
        for idx in range(3):
            self.axes[idx] = norm(self.axes[idx])

    def update_parents(self):
        for joint in self.parent_joints:
            joint.update()

    #region Axis getters-setters, parent JCS update
    @property
    def lateral(self):
        return self._lateral

    @lateral.setter
    def lateral(self, value):
        self._lateral = Segment.check_input(value)
        self.update_parents()

    @property
    def frontal(self):
        return self._frontal

    @frontal.setter
    def frontal(self, value):
        self._frontal = Segment.check_input(value)
        self.update_parents()

    @property
    def longitudinal(self):
        return self._longitudinal

    @longitudinal.setter
    def longitudinal(self, value):
        self._longitudinal = Segment.check_input(value)
        self.update_parents()
    #endregion

    @staticmethod
    def check_input(a: np.ndarray):
        if type(a) is not np.ndarray:
            raise Exception('Use numpy ndarrays for Segment objects')
        if 3 not in a.shape:
            raise Exception('JCS only works in 3D')
        if a.ndim > 2:
            raise Exception('Too many dimensions of input array')
        return np.atleast_2d(a) if a.shape[-1] is 3 else a.T


class JCS:
    _segment_a: Segment
    _segment_b: Segment

    def __init__(self, segment_a: Segment, segment_b: Segment,
                 body_fixed_axes=(0, 2), reference_axes=(1,1),
                 side='R', name=None):
        self._segment_a = segment_a
        self._segment_b = segment_b
        self._segment_a.parent_joints.append(self)
        self._segment_b.parent_joints.append(self)
        self.body_fixed_axes = body_fixed_axes
        self.reference_axes = reference_axes
        self.remainder_axes = self.get_remainder_axes()

        self.name = name
        self.side = side

        self.floating_axis, self.flexion, self.abduction, self.rotation = self.update()

    def get_remainder_axes(self):
        if np.any([self.body_fixed_axes[i] == self.reference_axes[i] for i in range(2)]):
            raise Exception('Duplicate axis used in joint')
        self.remainder_axes = \
            tuple([tuple({0,1,2}-set(used_axes))[0] for used_axes in zip(self.body_fixed_axes, self.reference_axes)])
        return self.remainder_axes

    def update(self):
        self.floating_axis = norm(np.cross(self.e3, self.e1, axis=1))
        sin_alpha = -vec_dot(self.floating_axis, self.e1s)
        cos_alpha = vec_dot(self.e1r, self.floating_axis)

        cos_beta = vec_dot(self.e1, self.e3)

        sin_gamma = (-1 if self.side is 'R' else 1) * vec_dot(self.floating_axis, self.e3s)
        cos_gamma = vec_dot(self.e1r, self.floating_axis)

        self.flexion = np.arctan2(sin_alpha, cos_alpha)
        self.abduction = (1 if self.side is 'R' else -1) * (np.arccos(cos_beta) - np.pi/2)
        self.rotation = np.arctan2(sin_gamma, cos_gamma)
        return self.floating_axis, self.flexion, self.abduction, self.rotation

    #region Property setters and getters (and update)
    @property
    def e1(self):
        return self._segment_a.axes[self.body_fixed_axes[0]]

    @property
    def e1r(self):
        return self._segment_a.axes[self.reference_axes[0]]

    @property  # Sign determining axis
    def e1s(self):
        return self._segment_a.axes[self.remainder_axes[0]]

    @property
    def e3(self):
        return self._segment_b.axes[self.body_fixed_axes[1]]

    @property
    def e3r(self):
        return self._segment_b.axes[self.reference_axes[1]]

    @property  # Sign determining axis
    def e3s(self):
        return self._segment_b.axes[self.remainder_axes[1]]

    @property
    def segment_a(self):
        # Getter of segment a
        return self._segment_a

    @segment_a.setter
    def segment_a(self, value):
        self._segment_a = value
        self._segment_a.parent_joints.append(self)
        self.update()

    @property
    def segment_b(self):
        # Getter of segment a
        return self._segment_b

    @segment_b.setter
    def segment_b(self, value):
        self._segment_b = value
        self._segment_b.parent_joints.append(self)
        self.update()

    #endregion