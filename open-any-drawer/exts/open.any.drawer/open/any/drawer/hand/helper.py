# hand helper
import carb
import omni
import sys
from pxr import Usd, Sdf, PhysxSchema, UsdPhysics, Vt, Gf, UsdGeom, UsdShade
from omni.physx.scripts import physicsUtils, particleUtils
from omni.physx.scripts import deformableUtils, utils
import math
from copy import copy

from .limiter import *


class HandHelper():
    def __init__(self) -> None:
        self.stage  = omni.usd.get_context().get_stage()
        
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
        self._revolute_drive_stiffness = 0 # 10000000 / radToDeg  # 50000.0
        self._spherical_drive_stiffness = 0 # 22000000 / radToDeg  # 50000.0
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

        self.import_hand()
        self._setup_geometry()
        self._setup_mesh_tree()
        # print("_fingerMeshes", self._fingerMeshes)
        self._rig_hand()

    def import_hand(self):
        # import skeleton hand
        

        default_prim_path = Sdf.Path("/World") # stage.GetDefaultPrim().GetPath()
        self._hand_prim_path = default_prim_path.AppendPath("Hand")
        self._bones_root_path = default_prim_path.AppendPath("Hand/Bones")
        self._tips_root_path = default_prim_path.AppendPath("Hand/Tips")

        abspath = "https://omniverse-content-staging.s3.us-west-2.amazonaws.com/DoNotDelete/PhysicsDemoAssets/103.1/DeformableHand/skeleton_hand_with_tips.usd"
        assert self.stage.DefinePrim(self._hand_prim_path).GetReferences().AddReference(abspath)

        self._hand_prim = self.stage.GetPrimAtPath(self._hand_prim_path.pathString)
        hand_xform = UsdGeom.Xformable(self._hand_prim)
        hand_xform.ClearXformOpOrder()
        precision = UsdGeom.XformOp.PrecisionFloat
        hand_xform.AddTranslateOp(precision=precision).Set(Gf.Vec3f(0,1.0,0.5))
        hand_xform.AddOrientOp(precision=precision).Set(Gf.Quatf(1,0,0,0))
        hand_xform.AddScaleOp(precision=precision).Set(Gf.Vec3f(0.01))

         # Physics scene
        physicsScenePath = default_prim_path.AppendChild("physicsScene")
        scene = UsdPhysics.Scene.Define(self.stage, physicsScenePath)
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)
        # utils.set_physics_scene_asyncsimrender(scene.GetPrim())
        # physxAPI = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        # physxAPI.CreateSolverTypeAttr("TGS")
        # physxAPI.CreateGpuMaxNumPartitionsAttr(4)

    def _setup_geometry(self):
        boneNames = ["proximal", "middle", "distal"]
        self._jointGeometry = {}
        # self._tableRestOffset = 0.005
        # self._tableHeightOffset = Gf.Vec3f(0.0, -2.3 + self._tableRestOffset, 0.0)
        self._handPosOffset = Gf.Vec3f(0.0, 64.0, 0.0)
        self._handInitPos = Gf.Vec3f(0.0, 94.0, 0.0)
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
        quat = get_quat_from_extrinsic_xyz_rotation(angleYrad=angleY, angleZrad=angleZ)
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
        middle.quat = get_quat_from_extrinsic_xyz_rotation(angleXrad=xAngleRad)
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
        middle.quat = get_quat_from_extrinsic_xyz_rotation(angleXrad=xAngleRad)
        middle.axis = "Z"

        distal = copy(middle)
        distal.bbCenterWeight = 0.55

        geoms = [copy(g) for g in [proximal, middle, distal]]
        self._jointGeometry["Ring"] = dict(zip(boneNames, geoms))

        # pinky:
        proximal = JointGeometry()
        proximal.bbCenterWeight = 0.67
        yAngleRad = 8.0 * math.pi / 180.0
        proximal.quat = get_quat_from_extrinsic_xyz_rotation(angleXrad=xAngleRad, angleYrad=yAngleRad)
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
        middle.quat = get_quat_from_extrinsic_xyz_rotation(angleXrad=xAngleRad, angleYrad=yAngleRad)

        distal = copy(middle)
        distal.bbCenterWeight = 0.55

        geoms = [copy(g) for g in [proximal, middle, distal]]
        self._jointGeometry["Pinky"] = dict(zip(boneNames, geoms))

    def _setup_mesh_tree(self):
        self._baseMesh = UsdGeom.Mesh.Get(self.stage, self._bones_root_path.AppendChild("l_carpal_mid"))
        assert self._baseMesh
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
                boneMesh = UsdGeom.Mesh.Get(self.stage, bonePath)
                assert boneMesh, f"Mesh {bonePath.pathString} invalid"
                self._fingerMeshes[fingerName][boneName] = boneMesh

    ################################## rigging #########################################

    def _rig_hand(self):
        self._set_bones_to_rb()
        UsdPhysics.ArticulationRootAPI.Apply(self._baseMesh.GetPrim())
        physxArticulationAPI = PhysxSchema.PhysxArticulationAPI.Apply(self._baseMesh.GetPrim())
        physxArticulationAPI.GetSolverPositionIterationCountAttr().Set(15)
        physxArticulationAPI.GetSolverVelocityIterationCountAttr().Set(0)
        self._setup_physics_material(self._baseMesh.GetPath())
        self._rig_hand_base()
        self._rig_fingers()

    def _rig_hand_base(self):
        basePath = self._baseMesh.GetPath()
        parentWorldBB = computeMeshWorldBoundsFromPoints(self._baseMesh)
        self._base_mesh_world_pos = Gf.Vec3f(0.5 * (parentWorldBB[0] + parentWorldBB[1]))
        baseLocalToWorld = self._baseMesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        for fingerName, finger in self._fingerMeshes.items():
            if fingerName == "Thumb":
                # skip thumb
                continue
            for boneName, bone in finger.items():
                if boneName == "metacarpal":
                    fixedJointPath = bone.GetPath().AppendChild("baseFixedJoint")
                    fixedJoint = UsdPhysics.FixedJoint.Define(self.stage, fixedJointPath)
                    fixedJoint.CreateBody0Rel().SetTargets([basePath])
                    fixedJoint.CreateBody1Rel().SetTargets([bone.GetPath()])

                    childWorldBB = computeMeshWorldBoundsFromPoints(bone)
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

        print("jointType", parentBone, jointType, childBone, jointType)

        parentWorldBB = computeMeshWorldBoundsFromPoints(parentBone)
        parentWorldPos = Gf.Vec3d(0.5 * (parentWorldBB[0] + parentWorldBB[1]))
        parentLocalToWorld = parentBone.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        childWorldBB = computeMeshWorldBoundsFromPoints(childBone)
        childWorldPos = Gf.Vec3d(0.5 * (childWorldBB[0] + childWorldBB[1]))
        childLocalToWorld = childBone.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        jointWorldPos = parentWorldPos + jointGeom.bbCenterWeight * (childWorldPos - parentWorldPos)
        
        print("jointWorldPos", jointWorldPos, parentWorldPos)
        
        if jointGeom.posOffsetW is not None:
            jointWorldPos += (jointGeom.posOffsetW / 100)
            # print("jointGeom.posOffsetW", jointGeom.posOffsetW)
        jointParentPosition = parentLocalToWorld.GetInverse().Transform(jointWorldPos)
        jointChildPosition = childLocalToWorld.GetInverse().Transform(jointWorldPos)

        if jointType == "revolute":
            jointPath = childBone.GetPath().AppendChild("RevoluteJoint")
            joint = UsdPhysics.RevoluteJoint.Define(self.stage, jointPath)
        elif jointType == "spherical":
            jointPath = childBone.GetPath().AppendChild("SphericalJoint")
            joint = UsdPhysics.SphericalJoint.Define(self.stage, jointPath)

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
            d6j = UsdPhysics.Joint.Define(self.stage, d6path)
            d6j.CreateExcludeFromArticulationAttr().Set(True)
            d6j.CreateBody0Rel().SetTargets([parentBone.GetPath()])
            d6j.CreateBody1Rel().SetTargets([childBone.GetPath()])
            d6j.CreateExcludeFromArticulationAttr().Set(True)
            d6j.CreateLocalPos0Attr().Set(jointParentPosition)
            parentWorldToLocal = Gf.Quatf(parentLocalToWorld.GetInverse().RemoveScaleShear().ExtractRotationQuat())
            
            print("D6DriverJoint parentWorldToLocal", jointParentPosition, jointChildPosition)
            
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
            print("fingerName", fingerName)
            parentBone = self._baseMesh
            for boneName, bone in finger.items():
                self._rig_joint(boneName, fingerName, parentBone)
                parentBone = bone

            return 

    def _rig_D6_anchor(self):
        # create anchor:
        self._anchorXform = UsdGeom.Xform.Define(
            self.stage, self.stage.GetDefaultPrim().GetPath().AppendChild("AnchorXform")
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
            self.stage, self.stage.GetDefaultPrim().GetPath().AppendChild("AnchorToHandBaseD6")
        )

        baseLocalToWorld = self._baseMesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        jointPosition = baseLocalToWorld.GetInverse().Transform(xformLocalToWorldTrans)
        jointPose = Gf.Quatf(baseLocalToWorld.GetInverse().RemoveScaleShear().ExtractRotationQuat())

        component.CreateExcludeFromArticulationAttr().Set(True)
        component.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0))
        component.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
        component.CreateBody0Rel().SetTargets([self._anchorXform.GetPath()])

        component.CreateBody1Rel().SetTargets([self._baseMesh.GetPath()])
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
    
     ########################################## physics ###################################

    def _setup_physics_material(self, path: Sdf.Path):
        if self._physicsMaterialPath is None:
            self._physicsMaterialPath = self.stage.GetDefaultPrim().GetPath().AppendChild("physicsMaterial")
            UsdShade.Material.Define(self.stage, self._physicsMaterialPath)
            material = UsdPhysics.MaterialAPI.Apply(self.stage.GetPrimAtPath(self._physicsMaterialPath))
            material.CreateStaticFrictionAttr().Set(self._material_static_friction)
            material.CreateDynamicFrictionAttr().Set(self._material_dynamic_friction)
            material.CreateRestitutionAttr().Set(self._material_restitution)

        collisionAPI = UsdPhysics.CollisionAPI.Get(self.stage, path)
        prim = self.stage.GetPrimAtPath(path)
        if not collisionAPI:
            collisionAPI = UsdPhysics.CollisionAPI.Apply(prim)
        # apply material
        physicsUtils.add_physics_material_to_prim(self.stage, prim, self._physicsMaterialPath)

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
        self._setup_rb_parameters(prim, restOffset=0.0, contactOffset=0.01) #! change contact offset

    def _set_bones_to_rb(self):
        self._set_bone_mesh_to_rigid_body_and_config(self._baseMesh)
        self._apply_mass(self._baseMesh, 0.01) #! change mass
        for _, finger in self._fingerMeshes.items():
            for _, bone in finger.items():
                self._set_bone_mesh_to_rigid_body_and_config(bone)
                self._apply_mass(bone, 0.01) #! change mass