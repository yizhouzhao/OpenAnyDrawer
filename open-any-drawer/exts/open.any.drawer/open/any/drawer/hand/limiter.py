# hand limiter
import carb
import sys
from pxr import Usd, Sdf, PhysxSchema, UsdPhysics, Vt, Gf, UsdGeom, UsdShade
from omni.physx.scripts import physicsUtils, particleUtils
from omni.physx.scripts import deformableUtils, utils
import math
from copy import copy

# helpers
def computeMeshWorldBoundsFromPoints(mesh: UsdGeom.Mesh) -> Vt.Vec3fArray:
    mesh_pts = mesh.GetPointsAttr().Get()
    extent = UsdGeom.PointBased.ComputeExtent(mesh_pts)
    transform = mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    for i in range(len(extent)):
        extent[i] = transform.Transform(extent[i])
    return extent

def get_quat_from_extrinsic_xyz_rotation(angleXrad: float = 0.0, angleYrad: float = 0.0, angleZrad: float = 0.0):
    # angles are in radians
    rotX = rotate_around_axis(1, 0, 0, angleXrad)
    rotY = rotate_around_axis(0, 1, 0, angleYrad)
    rotZ = rotate_around_axis(0, 0, 1, angleZrad)
    return rotZ * rotY * rotX

def rotate_around_axis(x: float, y: float, z: float, angle: float) -> Gf.Quatf:
    s = math.sin(0.5 * angle)
    return Gf.Quatf(math.cos(0.5 * angle), s * x, s * y, s * z)

class QuaternionRateOfChangeLimiter:
    def __init__(self, initQuat: Gf.Quatf = Gf.Quatf(1.0), maxRateRad: float = 0.01, iirAlpha: float = 0.0):
        self.maxDeltaPerFrame = maxRateRad
        self.cosThreshdold = math.cos(maxRateRad)
        self.resetValue = initQuat
        self.reset()
        self.alpha = 1.0 - iirAlpha  # 1 - alpha due to slerp (alpha = 0 -> immediate step to new goal)

    def reset(self):
        self.targetValue = self.resetValue
        self.currentValue = self.resetValue
        self.filteredTarget = None

    def set_target(self, targetValue: Gf.Quatf):
        self.targetValue = targetValue

    def set_target_and_update(self, targetValue: Gf.Quatf):
        self.targetValue = targetValue
        self.update()

    @property
    def current_value(self):
        return self.currentValue

    def update(self):
        if self.filteredTarget is None:
            self.filteredTarget = self.targetValue
        else:
            self.filteredTarget = Gf.Quatf(Gf.Slerp(self.alpha, self.filteredTarget, self.targetValue))
        toTarget = self.currentValue.GetInverse() * self.filteredTarget
        # shortest rotation
        if toTarget.GetReal() < 0.0:
            toTarget = -toTarget
        angle = math.acos(max(-1, min(1, toTarget.GetReal()))) * 2.0
        if angle > self.maxDeltaPerFrame:
            angle = self.maxDeltaPerFrame
            axis = toTarget.GetImaginary()
            axis.Normalize()
            sin = math.sin(0.5 * angle)
            toTarget = Gf.Quatf(math.cos(angle * 0.5), sin * axis[0], sin * axis[1], sin * axis[2])
        self.currentValue = self.currentValue * toTarget


class JointGeometry:
    def __init__(self, bbCenterWeight=None, quat=None, posOffsetW=None, axis=None, limits=None, joint_type="revolute"):
        self.bbCenterWeight = bbCenterWeight
        self.quat = quat
        self.posOffsetW = posOffsetW
        self.axis = axis
        self.limits = limits
        self.type = joint_type
        self.defaultDriveAngles = {"rotX": 0.0, "rotY": 0.0, "rotZ": 0.0}


class VectorRateOfChangeLimiter:
    def __init__(self, initVector: Gf.Vec3f = Gf.Vec3f(0.0), maxRatePerAxis: float = 0.01, iirAlpha: float = 0.0):
        self.maxDeltaPerFrame = maxRatePerAxis
        self.resetValue = initVector
        self.reset()
        self.alpha = iirAlpha

    def reset(self):
        # need to copy to avoid creating just a ref
        self.targetValue = copy(self.resetValue)
        self.currentValue = copy(self.resetValue)
        self.filteredTarget = None

    def set_target(self, targetValue: Gf.Vec3f):
        self.targetValue = targetValue

    def set_target_and_update(self, targetValue: Gf.Vec3f):
        self.targetValue = targetValue
        self.update()

    @property
    def current_value(self):
        return self.currentValue

    def update(self):
        if self.filteredTarget is None:
            self.filteredTarget = self.targetValue
        else:
            self.filteredTarget = self.alpha * self.filteredTarget + (1.0 - self.alpha) * self.targetValue
        for i in range(3):
            toTarget = self.filteredTarget[i] - self.currentValue[i]
            if abs(toTarget) > self.maxDeltaPerFrame:
                if toTarget < 0.0:
                    toTarget = -self.maxDeltaPerFrame
                else:
                    toTarget = self.maxDeltaPerFrame
            self.currentValue[i] += toTarget

class JointAngleRateOfChangeLimiter:
    def __init__(self, jointDriveAPI, initValue: float = 0.0, maxRateRad: float = 0.01):
        self.maxDeltaPerFrame = maxRateRad
        self.jointDriveAPI = jointDriveAPI
        self.resetValue = initValue
        self.reset()

    def set_current_angle_in_drive_api(self):
        targetDeg = self.currentValue * 180.0 / math.pi
        self.jointDriveAPI.GetTargetPositionAttr().Set(targetDeg)

    def reset(self):
        self.targetValue = self.resetValue
        self.currentValue = self.resetValue

    def set_target(self, targetValue):
        self.targetValue = targetValue

    def set_target_and_update(self, targetValue):
        self.targetValue = targetValue
        self.update()

    def update(self):
        toTarget = self.targetValue - self.currentValue
        if abs(toTarget) > self.maxDeltaPerFrame:
            if toTarget < 0:
                toTarget = -self.maxDeltaPerFrame
            else:
                toTarget = self.maxDeltaPerFrame
        self.currentValue += toTarget