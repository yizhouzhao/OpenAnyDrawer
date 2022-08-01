import omni.ext
import omni.ui as ui

from .open_env import OpenEnv

# go to directory: open-any-drawer/exts/open.any.drawer/open/any/drawer/
#  # start notebook from: /home/yizhou/.local/share/ov/pkg/isaac_sim-2022.1.0/jupyter_notebook.sh


# hand helper
import carb
import sys
from pxr import Usd, Sdf, PhysxSchema, UsdPhysics, Vt, Gf, UsdGeom, UsdShade
from omni.physx.scripts import physicsUtils, particleUtils
from omni.physx.scripts import deformableUtils, utils
import math
from copy import copy

from .hand.limiter import *

# Any class derived from `omni.ext.IExt` in top level module (defined in `python.modules` of `extension.toml`) will be
# instantiated when extension gets enabled and `on_startup(ext_id)` will be called. Later when extension gets disabled
# on_shutdown() is called.
class MyExtension(omni.ext.IExt):
    # ext_id is current extension id. It can be used with extension manager to query additional information, like where
    # this extension is located on filesystem.
    def on_startup(self, ext_id):
        print("[open.any.drawer] MyExtension startup")

        self.env = OpenEnv()

        self._window = ui.Window("Open any drawer", width=300, height=300)
        with self._window.frame:
            with ui.VStack():
                ui.Button("Add Franka Robot", clicked_fn= self.env.add_robot)

                with ui.HStack(height = 20):
                    ui.Label("object index: ", width = 80)
                    self.object_id_ui = omni.ui.IntField(height=20, width = 40, style={ "margin": 2 })
                    self.object_id_ui.model.set_value(0)
                    ui.Label("object scale: ", width = 80)
                    self.object_scale_ui = omni.ui.FloatField(height=20, width = 40, style={ "margin": 2 })
                    self.object_scale_ui.model.set_value(0.1)
                    ui.Button("Add Object", clicked_fn=self.add_object)

                ui.Button("Add Ground", clicked_fn=self.add_ground)

                ui.Button("Debug", clicked_fn= self.debug)
                ui.Button("Debug2", clicked_fn= self.debug2)
                ui.Button("Rig D6", clicked_fn= self.debug_rig_d6)
                ui.Button("Test instructor", clicked_fn= self.debug_instructor)
                ui.Button("Batch generation", clicked_fn= self.debug_batch_gen)
                
                

    def add_ground(self):
        from utils import add_ground_plane

        add_ground_plane("/World/Game")

    def add_object(self):
        object_id = self.object_id_ui.model.get_value_as_int()
        object_scale = self.object_scale_ui.model.get_value_as_float()
        self.env.add_object(object_id, scale = object_scale)

        selection = omni.usd.get_context().get_selection()
        selection.clear_selected_prim_paths()
        selection.set_prim_path_selected("/World/game", True, True, True, True)

        viewport = omni.kit.viewport_legacy.get_viewport_interface()
        if viewport:
            viewport.get_viewport_window().focus_on_selected()

    def on_shutdown(self):
        print("[open.any.drawer] MyExtension shutdown")

    def debug(self):
        print("debug")

        #########################################################
        ################### constants ###########################
        #########################################################
        self._physicsMaterialPath = None

        self._material_static_friction = 1.0
        self._material_dynamic_friction = 1.0
        self._material_restitution = 0.0

        # Joint drives / params:
        radToDeg = 180.0 / math.pi
        self._drive_max_force = 1e20
        self._revolute_drive_stiffness =  10000000 / radToDeg  # 50000.0
        self._spherical_drive_stiffness =  22000000 / radToDeg  # 50000.0
        self._revolute_drive_damping = 0.2 * self._revolute_drive_stiffness
        self._spherical_drive_damping = 0.2 * self._spherical_drive_stiffness
        self._maxJointVelocity = 3.0 * radToDeg
        self._jointFriction = 0  # 0.01

        mHand = 0.1 * 20.0 + 0.1 + 0.1
        dh = 0.05
        self._d6LinearSpring = mHand * 100 / dh
        self._d6LinearDamping = 20 * math.sqrt(self._d6LinearSpring * mHand)
        self._d6RotationalSpring = self._d6LinearSpring * 100.0 * 100.0 / radToDeg
        self._d6RotationalDamping = self._d6LinearDamping * 100.0 * 50.0 / radToDeg

        self._jointAngleRateLimitRad = 150 / 60 * math.pi / 180.0

        # driving and dofs
        self._drives = []
        self._driveGuards = []
        self._numDofs = 0
        self._thumbIndices = []

        #########################################################
        ################### hand ###########################
        #########################################################

        # import skeleton hand
        stage = omni.usd.get_context().get_stage()
        self._stage = stage

        default_prim_path = stage.GetDefaultPrim().GetPath()
        self._bones_root_path = default_prim_path.AppendPath("Hand/Bones")
        self._tips_root_path = default_prim_path.AppendPath("Hand/Tips")

        abspath = "https://omniverse-content-staging.s3.us-west-2.amazonaws.com/DoNotDelete/PhysicsDemoAssets/103.1/DeformableHand/skeleton_hand_with_tips.usd"
        assert stage.DefinePrim(default_prim_path.AppendPath("Hand")).GetReferences().AddReference(abspath)

        ## set up geo
        self._setup_geometry()
        self._setup_mesh_tree()
        self._rig_hand()
        self._rig_D6_anchor()
        self._setup_skeleton_hand_db_tips(stage)


    def _setup_geometry(self):
        boneNames = ["proximal", "middle", "distal"]
        self._jointGeometry = {}
        # self._tableRestOffset = 0.005
        # self._tableHeightOffset = Gf.Vec3f(0.0, -2.3 + self._tableRestOffset, 0.0)
        # self._handPosOffset = Gf.Vec3f(0.0, 64.0, 0.0)
        self._handInitPos = Gf.Vec3f(0.0, 1.0, 0.5)
        # self._flamePosition = Gf.Vec3d(27.992, 100.97, -13.96)
        # self._mugInitPos = Gf.Vec3f(10, 74.17879 + self._mugRestOffset, 0)
        # self._candlePosition = Gf.Vec3f(28, 74, -14)
        # self._mugInitRot = self.get_quat_from_extrinsic_xyz_rotation(angleYrad=-0.7 * math.pi)

        # self._fluidPositionOffset = Gf.Vec3f(0, 1.05, 0)
        # scale = 1.1
        # self._mugScale = Gf.Vec3f(scale)
        # self._mugOffset = Gf.Vec3f(1.1, 0, 15) * scale

        revoluteLimits = (-20, 120)

        # Thumb:
        metacarpal = JointGeometry()
        metacarpal.bbCenterWeight = 0.67
        metacarpal.posOffsetW = Gf.Vec3d(-1.276, 0.28, 0.233)
        # this quaternion is the joint pose in the inertial coordinate system
        # and will be converted to the bone frame in the joint rigging
        angleY = -0.45
        angleZ = -0.5
        quat = self.get_quat_from_extrinsic_xyz_rotation(angleYrad=angleY, angleZrad=angleZ)
        metacarpal.quat = quat  # first y then z, extrinsic
        metacarpal.type = "spherical"
        metacarpal.axis = "X"
        metacarpal.limits = (90, 90)
        metacarpal.defaultDriveAngles["rotY"] = angleY
        metacarpal.defaultDriveAngles["rotZ"] = angleZ

        proximal = JointGeometry()
        proximal.bbCenterWeight = 0.67
        proximal.quat = Gf.Quatf(0, 0, 0, 1) * quat
        proximal.axis = "Y"
        proximal.limits = revoluteLimits

        distal = copy(proximal)
        distal.bbCenterWeight = 0.55
        self._jointGeometry["Thumb"] = {
            "metacarpal": copy(metacarpal),
            "proximal": copy(proximal),
            "distal": copy(distal),
        }

        sphericalLimits = (60, 90)

        # Index:
        proximal = JointGeometry()
        proximal.bbCenterWeight = 0.67
        proximal.quat = Gf.Quatf(1.0)
        proximal.type = "spherical"
        proximal.axis = "X"
        proximal.limits = sphericalLimits

        middle = JointGeometry()
        middle.bbCenterWeight = 0.67
        xAngleRad = 5.0 * math.pi / 180.0
        middle.quat = self.get_quat_from_extrinsic_xyz_rotation(angleXrad=xAngleRad)
        middle.axis = "Z"
        middle.limits = revoluteLimits

        distal = copy(middle)
        distal.bbCenterWeight = 0.55

        geoms = [copy(g) for g in [proximal, middle, distal]]
        self._jointGeometry["Index"] = dict(zip(boneNames, geoms))

        # middle:
        proximal = JointGeometry()
        proximal.bbCenterWeight = 0.67
        proximal.quat = Gf.Quatf(1.0)
        proximal.type = "spherical"
        proximal.limits = sphericalLimits
        proximal.axis = "X"

        middle = JointGeometry()
        middle.bbCenterWeight = 0.67
        middle.quat = Gf.Quatf(1.0)
        middle.axis = "Z"
        middle.limits = revoluteLimits

        distal = copy(middle)
        distal.bbCenterWeight = 0.55

        geoms = [copy(g) for g in [proximal, middle, distal]]
        self._jointGeometry["Middle"] = dict(zip(boneNames, geoms))

        # ring:
        proximal = JointGeometry()
        proximal.bbCenterWeight = 0.67
        proximal.quat = Gf.Quatf(1.0)
        proximal.type = "spherical"
        proximal.limits = sphericalLimits
        proximal.axis = "X"

        middle = JointGeometry()
        middle.bbCenterWeight = 0.6
        middle.quat = Gf.Quatf(1.0)
        middle.limits = revoluteLimits
        xAngleRad = -5.0 * math.pi / 180.0
        middle.quat = self.get_quat_from_extrinsic_xyz_rotation(angleXrad=xAngleRad)
        middle.axis = "Z"

        distal = copy(middle)
        distal.bbCenterWeight = 0.55

        geoms = [copy(g) for g in [proximal, middle, distal]]
        self._jointGeometry["Ring"] = dict(zip(boneNames, geoms))

        # pinky:
        proximal = JointGeometry()
        proximal.bbCenterWeight = 0.67
        yAngleRad = 8.0 * math.pi / 180.0
        proximal.quat = self.get_quat_from_extrinsic_xyz_rotation(angleXrad=xAngleRad, angleYrad=yAngleRad)
        proximal.type = "spherical"
        proximal.limits = sphericalLimits
        proximal.axis = "X"
        proximal.defaultDriveAngles["rotY"] = yAngleRad

        middle = JointGeometry()
        middle.bbCenterWeight = 0.67
        middle.quat = Gf.Quatf(1.0)
        middle.limits = revoluteLimits
        middle.axis = "Z"
        yAngleRad = 8.0 * math.pi / 180.0
        xAngleRad = -5.0 * math.pi / 180.0
        middle.quat = self.get_quat_from_extrinsic_xyz_rotation(angleXrad=xAngleRad, angleYrad=yAngleRad)

        distal = copy(middle)
        distal.bbCenterWeight = 0.55

        geoms = [copy(g) for g in [proximal, middle, distal]]
        self._jointGeometry["Pinky"] = dict(zip(boneNames, geoms))

    def _setup_mesh_tree(self):
        self._articulation_root = UsdGeom.Mesh.Get(self._stage, self._bones_root_path.AppendChild("l_carpal_mid"))
        assert self._articulation_root
        boneNames = ["metacarpal", "proximal", "middle", "distal"]
        fingerNames = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        self._fingerMeshes = {}
        for fingerName in fingerNames:
            self._fingerMeshes[fingerName] = {}
            groupPath = self._bones_root_path.AppendChild(f"l_{fingerName.lower()}Skeleton_grp")
            for boneName in boneNames:
                if fingerName == "Thumb" and boneName == "middle":
                    continue
                bonePath = groupPath.AppendChild(f"l_{boneName}{fingerName}_mid")
                boneMesh = UsdGeom.Mesh.Get(self._stage, bonePath)
                assert boneMesh, f"Mesh {bonePath.pathString} invalid"
                self._fingerMeshes[fingerName][boneName] = boneMesh

    ################################## rigging #########################################

    def _rig_hand(self):
        self._set_bones_to_rb()
        UsdPhysics.ArticulationRootAPI.Apply(self._articulation_root.GetPrim())
        physxArticulationAPI = PhysxSchema.PhysxArticulationAPI.Apply(self._articulation_root.GetPrim())
        physxArticulationAPI.GetSolverPositionIterationCountAttr().Set(15)
        physxArticulationAPI.GetSolverVelocityIterationCountAttr().Set(0)
        self._setup_physics_material(self._articulation_root.GetPath())
        self._rig_hand_base()
        self._rig_fingers()

    def _rig_hand_base(self):
        basePath = self._articulation_root.GetPath()
        parentWorldBB = self._computeMeshWorldBoundsFromPoints(self._articulation_root)
        self._base_mesh_world_pos = Gf.Vec3f(0.5 * (parentWorldBB[0] + parentWorldBB[1]))
        baseLocalToWorld = self._articulation_root.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        for fingerName, finger in self._fingerMeshes.items():
            if fingerName == "Thumb":
                # skip thumb
                continue
            for boneName, bone in finger.items():
                if boneName == "metacarpal":
                    fixedJointPath = bone.GetPath().AppendChild("baseFixedJoint")
                    fixedJoint = UsdPhysics.FixedJoint.Define(self._stage, fixedJointPath)
                    fixedJoint.CreateBody0Rel().SetTargets([basePath])
                    fixedJoint.CreateBody1Rel().SetTargets([bone.GetPath()])

                    childWorldBB = self._computeMeshWorldBoundsFromPoints(bone)
                    childWorldPos = Gf.Vec3f(0.5 * (childWorldBB[0] + childWorldBB[1]))
                    childLocalToWorld = bone.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

                    jointWorldPos = 0.5 * (self._base_mesh_world_pos + childWorldPos)
                    jointParentPosition = baseLocalToWorld.GetInverse().Transform(jointWorldPos)
                    jointChildPosition = childLocalToWorld.GetInverse().Transform(jointWorldPos)

                    fixedJoint.CreateLocalPos0Attr().Set(jointParentPosition)
                    fixedJoint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))

                    fixedJoint.CreateLocalPos1Attr().Set(jointChildPosition)
                    fixedJoint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

                    fixedJoint.CreateBreakForceAttr().Set(sys.float_info.max)
                    fixedJoint.CreateBreakTorqueAttr().Set(sys.float_info.max)

    def _rig_joint(self, boneName, fingerName, parentBone):
        if boneName not in self._jointGeometry[fingerName]:
            return

        childBone = self._fingerMeshes[fingerName][boneName]
        jointGeom = self._jointGeometry[fingerName][boneName]
        jointType = jointGeom.type.lower()

        parentWorldBB = self._computeMeshWorldBoundsFromPoints(parentBone)
        parentWorldPos = Gf.Vec3d(0.5 * (parentWorldBB[0] + parentWorldBB[1]))
        parentLocalToWorld = parentBone.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        childWorldBB = self._computeMeshWorldBoundsFromPoints(childBone)
        childWorldPos = Gf.Vec3d(0.5 * (childWorldBB[0] + childWorldBB[1]))
        childLocalToWorld = childBone.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        jointWorldPos = parentWorldPos + jointGeom.bbCenterWeight * (childWorldPos - parentWorldPos)
        if jointGeom.posOffsetW is not None:
            jointWorldPos += jointGeom.posOffsetW
        jointParentPosition = parentLocalToWorld.GetInverse().Transform(jointWorldPos)
        jointChildPosition = childLocalToWorld.GetInverse().Transform(jointWorldPos)

        if jointType == "revolute":
            jointPath = childBone.GetPath().AppendChild("RevoluteJoint")
            joint = UsdPhysics.RevoluteJoint.Define(self._stage, jointPath)
        elif jointType == "spherical":
            jointPath = childBone.GetPath().AppendChild("SphericalJoint")
            joint = UsdPhysics.SphericalJoint.Define(self._stage, jointPath)

        joint.CreateBody0Rel().SetTargets([parentBone.GetPath()])
        joint.CreateBody1Rel().SetTargets([childBone.GetPath()])
        joint.CreateAxisAttr(jointGeom.axis)

        # for the sphericals, the relative orientation does not matter as they are externally driven.
        # for the revolutes, it is key that they are oriented correctly and that parent and child are identical
        # in order to avoid offsets - offsets will be added in the joint commands
        jointPose = Gf.Quatf(parentLocalToWorld.GetInverse().RemoveScaleShear().ExtractRotationQuat())
        jointPose *= jointGeom.quat
        # this is assuming that parent and child's frames coincide
        joint.CreateLocalPos0Attr().Set(jointParentPosition)
        joint.CreateLocalRot0Attr().Set(jointPose)

        joint.CreateLocalPos1Attr().Set(jointChildPosition)
        joint.CreateLocalRot1Attr().Set(jointPose)

        physxJointAPI = PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim())
        physxJointAPI.GetMaxJointVelocityAttr().Set(self._maxJointVelocity)
        physxJointAPI.GetJointFrictionAttr().Set(self._jointFriction)

        if jointType == "revolute":
            # for revolute create drive
            driveAPI = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
            driveAPI.CreateTypeAttr("force")
            driveAPI.CreateMaxForceAttr(self._drive_max_force)
            driveAPI.CreateDampingAttr(self._revolute_drive_damping)
            driveAPI.CreateStiffnessAttr(self._revolute_drive_stiffness)
            dofIndex = len(self._drives)
            self._numDofs += 1
            if fingerName == "Thumb":
                self._thumbIndices.append(dofIndex)
            self._drives.append(driveAPI)
            targetAngle = jointGeom.defaultDriveAngles["rot" + jointGeom.axis]
            self._driveGuards.append(
                JointAngleRateOfChangeLimiter(driveAPI, targetAngle, self._jointAngleRateLimitRad)
            )
        elif jointType == "spherical":
            # add 6d external joint and drive:
            d6path = childBone.GetPath().AppendChild("D6DriverJoint")
            d6j = UsdPhysics.Joint.Define(self._stage, d6path)
            d6j.CreateExcludeFromArticulationAttr().Set(True)
            d6j.CreateBody0Rel().SetTargets([parentBone.GetPath()])
            d6j.CreateBody1Rel().SetTargets([childBone.GetPath()])
            d6j.CreateExcludeFromArticulationAttr().Set(True)
            d6j.CreateLocalPos0Attr().Set(jointParentPosition)
            parentWorldToLocal = Gf.Quatf(parentLocalToWorld.GetInverse().RemoveScaleShear().ExtractRotationQuat())
            d6j.CreateLocalRot0Attr().Set(parentWorldToLocal)
            d6j.CreateLocalPos1Attr().Set(jointChildPosition)
            childPose = parentWorldToLocal * jointGeom.quat
            d6j.CreateLocalRot1Attr().Set(childPose)
            d6j.CreateBreakForceAttr().Set(1e20)
            d6j.CreateBreakTorqueAttr().Set(1e20)

            axes = [x for x in "XYZ" if jointGeom.axis != x]
            assert len(axes) == 2, "Error in spherical drives setup"
            drives = ["rot" + x for x in axes]

            # lock the joint axis:
            limitAPI = UsdPhysics.LimitAPI.Apply(d6j.GetPrim(), "rot" + jointGeom.axis)
            limitAPI.CreateLowAttr(1.0)
            limitAPI.CreateHighAttr(-1.0)

            for d in drives:
                driveAPI = UsdPhysics.DriveAPI.Apply(d6j.GetPrim(), d)
                driveAPI.CreateTypeAttr("force")
                driveAPI.CreateMaxForceAttr(self._drive_max_force)
                driveAPI.CreateDampingAttr(self._spherical_drive_damping)
                driveAPI.CreateStiffnessAttr(self._spherical_drive_stiffness)
                dofIndex = len(self._drives)
                self._numDofs += 1
                if fingerName == "Thumb":
                    self._thumbIndices.append(dofIndex)
                self._drives.append(driveAPI)
                targetAngle = jointGeom.defaultDriveAngles[d]
                self._driveGuards.append(
                    JointAngleRateOfChangeLimiter(driveAPI, targetAngle, self._jointAngleRateLimitRad)
                )

    def _rig_fingers(self):
        for fingerName, finger in self._fingerMeshes.items():
            parentBone = self._articulation_root
            for boneName, bone in finger.items():
                self._rig_joint(boneName, fingerName, parentBone)
                parentBone = bone

    def _rig_D6_anchor(self):
        # create anchor:
        self._anchorXform = UsdGeom.Xform.Define(
            self._stage, self._stage.GetDefaultPrim().GetPath().AppendChild("AnchorXform")
        )
        # these are global coords because world is the xform's parent
        xformLocalToWorldTrans = self._handInitPos
        xformLocalToWorldRot = Gf.Quatf(1.0)
        self._anchorXform.AddTranslateOp().Set(xformLocalToWorldTrans)
        self._anchorXform.AddOrientOp().Set(xformLocalToWorldRot)
        self._anchorPositionRateLimiter = VectorRateOfChangeLimiter(
            xformLocalToWorldTrans, 1.666, 0.5 ** (1 / 6)
        )
        self._anchorQuatRateLimiter = QuaternionRateOfChangeLimiter(
            xformLocalToWorldRot, 1.666, 0.5 ** (1 / 6)
        )
        xformPrim = self._anchorXform.GetPrim()
        physicsAPI = UsdPhysics.RigidBodyAPI.Apply(xformPrim)
        physicsAPI.CreateRigidBodyEnabledAttr(True)
        physicsAPI.CreateKinematicEnabledAttr(True)

        # setup joint to floating hand base
        component = UsdPhysics.Joint.Define(
            self._stage, self._stage.GetDefaultPrim().GetPath().AppendChild("AnchorToHandBaseD6")
        )

        baseLocalToWorld = self._articulation_root.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        jointPosition = baseLocalToWorld.GetInverse().Transform(xformLocalToWorldTrans)
        jointPose = Gf.Quatf(baseLocalToWorld.GetInverse().RemoveScaleShear().ExtractRotationQuat())

        component.CreateExcludeFromArticulationAttr().Set(True)
        component.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0))
        component.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
        component.CreateBody0Rel().SetTargets([self._anchorXform.GetPath()])

        component.CreateBody1Rel().SetTargets([self._articulation_root.GetPath()])
        component.CreateLocalPos1Attr().Set(jointPosition)
        component.CreateLocalRot1Attr().Set(jointPose)

        component.CreateBreakForceAttr().Set(sys.float_info.max)
        component.CreateBreakTorqueAttr().Set(sys.float_info.max)

        rootJointPrim = component.GetPrim()
        for dof in ["transX", "transY", "transZ"]:
            driveAPI = UsdPhysics.DriveAPI.Apply(rootJointPrim, dof)
            driveAPI.CreateTypeAttr("force")
            driveAPI.CreateMaxForceAttr(self._drive_max_force)
            driveAPI.CreateTargetPositionAttr(0.0)
            driveAPI.CreateDampingAttr(self._d6LinearDamping)
            driveAPI.CreateStiffnessAttr(self._d6LinearSpring)

        for rotDof in ["rotX", "rotY", "rotZ"]:
            driveAPI = UsdPhysics.DriveAPI.Apply(rootJointPrim, rotDof)
            driveAPI.CreateTypeAttr("force")
            driveAPI.CreateMaxForceAttr(self._drive_max_force)
            driveAPI.CreateTargetPositionAttr(0.0)
            driveAPI.CreateDampingAttr(self._d6RotationalDamping)
            driveAPI.CreateStiffnessAttr(self._d6RotationalSpring)

    ########################### soft body #################################################

    def _setup_skeleton_hand_db_tips(self, stage):

        # SB and fluid:
        self._sb_hand_schema_parameters = {
            "youngsModulus": 1.0e5,
            "poissonsRatio": 0.3,
            "dampingScale": 1.0,
            "dynamicFriction": 1.0,
            "solver_position_iteration_count": 15,
            "collisionRestOffset": 0.1,
            "collisionContactOffset": 0.5,
            "self_collision": False,
            "vertex_velocity_damping": 0.005,
            "sleep_damping": 0.001,  # disable
            "sleep_threshold": 0.001,  # disable
            "settling_threshold": 0.001,  # disable
        }
        self._sb_tips_schema_parameters = self._sb_hand_schema_parameters
        self._sb_tips_schema_parameters["collisionRestOffset"] = 0.00001

        self._sb_tips_resolution = 8
        self._sb_hand_resolution = 20

        # create and attach softbodies
        sbTipsStringPaths = [
            "LeftHandThumbTipScaled/geom",
            "LeftHandIndexTipScaled/geom",
            "LeftHandMiddleTipScaled/geom",
            "LeftHandRingTipScaled/geom",
            "LeftHandPinkyTipScaled/geom",
        ]
        sbTipsPaths = [self._tips_root_path.AppendPath(x) for x in sbTipsStringPaths]

        sbTips_material_path = omni.usd.get_stage_next_free_path(stage, "/sbTipsMaterial", True)
        deformableUtils.add_deformable_body_material(
            stage,
            sbTips_material_path,
            youngs_modulus=self._sb_tips_schema_parameters["youngsModulus"],
            poissons_ratio=self._sb_tips_schema_parameters["poissonsRatio"],
            damping_scale=self._sb_tips_schema_parameters["dampingScale"],
            dynamic_friction=self._sb_tips_schema_parameters["dynamicFriction"],
        )

        self._deformableTipMass = 0.01
        for sbTipPath in sbTipsPaths:
            self.set_softbody(
                sbTipPath,
                self._sb_tips_schema_parameters,
                sbTips_material_path,
                self._deformableTipMass,
                self._sb_tips_resolution,
            )

        # rigid attach
        attachmentBoneStringPaths = [
            "l_thumbSkeleton_grp/l_distalThumb_mid",
            "l_indexSkeleton_grp/l_distalIndex_mid",
            "l_middleSkeleton_grp/l_distalMiddle_mid",
            "l_ringSkeleton_grp/l_distalRing_mid",
            "l_pinkySkeleton_grp/l_distalPinky_mid",
            "l_thumbSkeleton_grp/l_metacarpalThumb_mid",
            "l_indexSkeleton_grp/l_metacarpalIndex_mid",
            "l_middleSkeleton_grp/l_metacarpalMiddle_mid",
            "l_ringSkeleton_grp/l_metacarpalRing_mid",
            "l_pinkySkeleton_grp/l_metacarpalPinky_mid",
            "l_thumbSkeleton_grp/l_proximalThumb_mid",
            "l_indexSkeleton_grp/l_proximalIndex_mid",
            "l_middleSkeleton_grp/l_proximalMiddle_mid",
            "l_ringSkeleton_grp/l_proximalRing_mid",
            "l_pinkySkeleton_grp/l_proximalPinky_mid",
            "l_indexSkeleton_grp/l_middleIndex_mid",
            "l_middleSkeleton_grp/l_middleMiddle_mid",
            "l_ringSkeleton_grp/l_middleRing_mid",
            "l_pinkySkeleton_grp/l_middlePinky_mid",
            "l_carpal_mid",
        ]

        # color of tips:
        color_rgb = [161, 102, 94]
        sbColor = Vt.Vec3fArray([Gf.Vec3f(color_rgb[0], color_rgb[1], color_rgb[2]) / 256.0])
        attachmentBonePaths = [self._bones_root_path.AppendPath(x) for x in attachmentBoneStringPaths]
        for sbTipPath, bonePath in zip(sbTipsPaths, attachmentBonePaths):
            sbMesh = UsdGeom.Mesh.Get(stage, sbTipPath)
            sbMesh.CreateDisplayColorAttr(sbColor)
            boneMesh = UsdGeom.Mesh.Get(stage, bonePath)
            self.create_softbody_rigid_attachment(sbMesh, boneMesh, 0)

        softbodyGroupPath = "/World/physicsScene/collisionGroupSoftBodyTips"
        boneGroupPath = "/World/physicsScene/collisionGroupHandBones"
        softbodyGroup = UsdPhysics.CollisionGroup.Define(stage, softbodyGroupPath)
        boneGroup = UsdPhysics.CollisionGroup.Define(stage, boneGroupPath)

        filteredRel = softbodyGroup.CreateFilteredGroupsRel()
        filteredRel.AddTarget(boneGroupPath)

        filteredRel = boneGroup.CreateFilteredGroupsRel()
        filteredRel.AddTarget(softbodyGroupPath)

        for sbTipPath in sbTipsPaths:
            self.assign_collision_group(sbTipPath, softbodyGroupPath)
        # filter all SB tips vs bone rigid bodies collisions
        self.assign_collision_group(self._articulation_root.GetPath(), boneGroupPath)
        for finger in self._fingerMeshes.values():
            for bone in finger.values():
                self.assign_collision_group(bone.GetPath(), boneGroupPath)

    def assign_collision_group(self, primPath: Sdf.Path, groupPath: Sdf.Path):
        stage = self._stage
        physicsUtils.add_collision_to_collision_group(stage, primPath, groupPath)

    def set_softbody(
        self, mesh_path: Sdf.Path, schema_parameters: dict, material_path: Sdf.Path, mass: float, resolution: int
    ):

        success = omni.kit.commands.execute(
            "AddDeformableBodyComponentCommand",
            skin_mesh_path=mesh_path,
            voxel_resolution=resolution,
            solver_position_iteration_count=schema_parameters["solver_position_iteration_count"],
            self_collision=schema_parameters["self_collision"],
            vertex_velocity_damping=schema_parameters["vertex_velocity_damping"],
            sleep_damping=schema_parameters["sleep_damping"],
            sleep_threshold=schema_parameters["sleep_threshold"],
            settling_threshold=schema_parameters["settling_threshold"],
        )

        prim = self._stage.GetPrimAtPath(mesh_path)
        physxCollisionAPI = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        assert physxCollisionAPI.CreateRestOffsetAttr().Set(schema_parameters["collisionRestOffset"])
        assert physxCollisionAPI.CreateContactOffsetAttr().Set(schema_parameters["collisionContactOffset"])

        massAPI = UsdPhysics.MassAPI.Apply(prim)
        massAPI.CreateMassAttr().Set(mass)

        physicsUtils.add_physics_material_to_prim(self._stage, self._stage.GetPrimAtPath(mesh_path), material_path)

        assert success

    def create_softbody_rigid_attachment(self, soft_body, gprim, id):
        assert PhysxSchema.PhysxDeformableBodyAPI(soft_body)
        assert UsdPhysics.CollisionAPI(gprim)

        # get attachment to set parameters:
        attachmentPath = soft_body.GetPath().AppendChild(f"rigid_attachment_{id}")

        attachment = PhysxSchema.PhysxPhysicsAttachment.Define(self._stage, attachmentPath)
        attachment.GetActor0Rel().SetTargets([soft_body.GetPath()])
        attachment.GetActor1Rel().SetTargets([gprim.GetPath()])
        PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment.GetPrim())

        attachment = PhysxSchema.PhysxAutoAttachmentAPI.Get(self._stage, attachmentPath)
        attachment.GetEnableDeformableVertexAttachmentsAttr().Set(True)
        attachment.GetEnableRigidSurfaceAttachmentsAttr().Set(True)

    ########################################## physics ###################################

    def _setup_physics_material(self, path: Sdf.Path):
        if self._physicsMaterialPath is None:
            self._physicsMaterialPath = self._stage.GetDefaultPrim().GetPath().AppendChild("physicsMaterial")
            UsdShade.Material.Define(self._stage, self._physicsMaterialPath)
            material = UsdPhysics.MaterialAPI.Apply(self._stage.GetPrimAtPath(self._physicsMaterialPath))
            material.CreateStaticFrictionAttr().Set(self._material_static_friction)
            material.CreateDynamicFrictionAttr().Set(self._material_dynamic_friction)
            material.CreateRestitutionAttr().Set(self._material_restitution)

        collisionAPI = UsdPhysics.CollisionAPI.Get(self._stage, path)
        prim = self._stage.GetPrimAtPath(path)
        if not collisionAPI:
            collisionAPI = UsdPhysics.CollisionAPI.Apply(prim)
        # apply material
        physicsUtils.add_physics_material_to_prim(self._stage, prim, self._physicsMaterialPath)

    def _apply_mass(self, mesh: UsdGeom.Mesh, mass: float):
        massAPI = UsdPhysics.MassAPI.Apply(mesh.GetPrim())
        massAPI.GetMassAttr().Set(mass)

    def _setup_rb_parameters(self, prim, restOffset, contactOffset):
        physxCollisionAPI = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        self._setup_physics_material(prim.GetPath())
        assert physxCollisionAPI.GetRestOffsetAttr().Set(restOffset)
        assert physxCollisionAPI.GetContactOffsetAttr().Set(contactOffset)
        assert prim.CreateAttribute("physxMeshCollision:minThickness", Sdf.ValueTypeNames.Float).Set(0.001)
        physxRBAPI = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        # physxRBAPI.CreateSolverPositionIterationCountAttr().Set(self._solverPositionIterations)
        # physxRBAPI.CreateSolverVelocityIterationCountAttr().Set(self._solverVelocityIterations)

    def _set_bone_mesh_to_rigid_body_and_config(self, mesh: UsdGeom.Mesh, approximationShape="convexHull"):
        prim = mesh.GetPrim()
        utils.setRigidBody(prim, approximationShape=approximationShape, kinematic=False)
        self._setup_rb_parameters(prim, restOffset=0.0, contactOffset=0.1)

    def _set_bones_to_rb(self):
        self._set_bone_mesh_to_rigid_body_and_config(self._articulation_root)
        self._apply_mass(self._articulation_root, 0.1)
        for _, finger in self._fingerMeshes.items():
            for _, bone in finger.items():
                self._set_bone_mesh_to_rigid_body_and_config(bone)
                self._apply_mass(bone, 0.1)

    ########################################## utils #####################################
    @staticmethod
    def get_quat_from_extrinsic_xyz_rotation(angleXrad: float = 0.0, angleYrad: float = 0.0, angleZrad: float = 0.0):
        # angles are in radians
        rotX = rotate_around_axis(1, 0, 0, angleXrad)
        rotY = rotate_around_axis(0, 1, 0, angleYrad)
        rotZ = rotate_around_axis(0, 0, 1, angleZrad)
        return rotZ * rotY * rotX

    
    @staticmethod
    def _computeMeshWorldBoundsFromPoints(mesh: UsdGeom.Mesh) -> Vt.Vec3fArray:
        mesh_pts = mesh.GetPointsAttr().Get()
        extent = UsdGeom.PointBased.ComputeExtent(mesh_pts)
        transform = mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        for i in range(len(extent)):
            extent[i] = transform.Transform(extent[i])
        return extent

    
    def debug2(self):
        print("debug2")
        from .hand.helper import HandHelper
        self.hand_helper = HandHelper()

    def debug_rig_d6(self):
        self._stage = omni.usd.get_context().get_stage()
        self._damping_stiffness = 1e4
        # create anchor:
        self._anchorXform = UsdGeom.Xform.Define(
            self._stage, Sdf.Path("/World/allegro/AnchorXform")
        )
        # these are global coords because world is the xform's parent
        xformLocalToWorldTrans = Gf.Vec3f(0)
        xformLocalToWorldRot = Gf.Quatf(1.0)
        self._anchorXform.AddTranslateOp().Set(xformLocalToWorldTrans)
        self._anchorXform.AddOrientOp().Set(xformLocalToWorldRot)
      
        xformPrim = self._anchorXform.GetPrim()
        physicsAPI = UsdPhysics.RigidBodyAPI.Apply(xformPrim)
        physicsAPI.CreateRigidBodyEnabledAttr(True)
        physicsAPI.CreateKinematicEnabledAttr(True)

        # setup joint to floating hand base
        component = UsdPhysics.Joint.Define(
            self._stage, Sdf.Path("/World/allegro/AnchorToHandBaseD6")
        )

        
        self._articulation_root = self._stage.GetPrimAtPath("/World/Hand/Bones/l_carpal_mid") # /World/allegro/allegro_mount
        baseLocalToWorld = UsdGeom.Xformable(self._articulation_root).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        jointPosition = baseLocalToWorld.GetInverse().Transform(xformLocalToWorldTrans)
        jointPose = Gf.Quatf(baseLocalToWorld.GetInverse().RemoveScaleShear().ExtractRotationQuat())

        component.CreateExcludeFromArticulationAttr().Set(True)
        component.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0))
        component.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
        component.CreateBody0Rel().SetTargets([self._anchorXform.GetPath()])

        component.CreateBody1Rel().SetTargets([self._articulation_root.GetPath()])
        component.CreateLocalPos1Attr().Set(jointPosition)
        component.CreateLocalRot1Attr().Set(jointPose)

        component.CreateBreakForceAttr().Set(sys.float_info.max)
        component.CreateBreakTorqueAttr().Set(sys.float_info.max)

        rootJointPrim = component.GetPrim()
        for dof in ["transX", "transY", "transZ"]:
            driveAPI = UsdPhysics.DriveAPI.Apply(rootJointPrim, dof)
            driveAPI.CreateTypeAttr("force")
            # driveAPI.CreateMaxForceAttr(self._drive_max_force)
            driveAPI.CreateTargetPositionAttr(0.0)
            driveAPI.CreateDampingAttr(self._damping_stiffness)
            driveAPI.CreateStiffnessAttr(self._damping_stiffness)

        for rotDof in ["rotX", "rotY", "rotZ"]:
            driveAPI = UsdPhysics.DriveAPI.Apply(rootJointPrim, rotDof)
            driveAPI.CreateTypeAttr("force")
            # driveAPI.CreateMaxForceAttr(self._drive_max_force)
            driveAPI.CreateTargetPositionAttr(0.0)
            driveAPI.CreateDampingAttr(self._damping_stiffness)
            driveAPI.CreateStiffnessAttr(self._damping_stiffness)

    def debug_instructor(self):
        print("debug instru")

        from .task.instructor import SceneInstructor

        self.scene_instr = SceneInstructor()
        self.scene_instr.analysis()
        self.scene_instr.build_handle_desc_ui()
        self.scene_instr.add_semantic_to_handle()

        self.scene_instr.export_data()

    def debug_batch_gen(self):
        print("debug_batch_gen")

        from .task.instructor import SceneInstructor
        import omni.replicator.core as rep

        object_id = self.object_id_ui.model.set_value(6)
        object_id = self.object_id_ui.model.get_value_as_int()
        object_scale = self.object_scale_ui.model.get_value_as_float()
        self.env.add_object(object_id, scale = object_scale)

        self.scene_instr = SceneInstructor()
        self.scene_instr.analysis()
        # self.scene_instr.build_handle_desc_ui()
        
        print("scene_instr.is_obj_valid: ", self.scene_instr.is_obj_valid)
        if self.scene_instr.is_obj_valid:
            self.scene_instr.add_semantic_to_handle()
            self.scene_instr.output_path = f"/home/yizhou/Research/temp/{object_id}"
            self.scene_instr.export_data()
        
        
        # print("print(rep.orchestrator.get_is_started())", rep.orchestrator.get_is_started())
        
