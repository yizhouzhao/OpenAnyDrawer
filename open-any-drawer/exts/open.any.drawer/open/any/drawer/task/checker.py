# task check

import omni
from task.newJointCheck import JointCheck

class TaskChecker():
    def __init__(self, target_obj, target_joint, joint_type, IS_RUNTIME = False) -> None:
        
        self.target_obj =target_obj
        self.target_joint = target_joint
        self.joint_type = joint_type

        self.target_prim_path = "/World/Game/" + self.target_obj
        self.joint_checker = JointCheck(self.target_prim_path, self.target_joint)

        self.init_value = 0.0 # from start 
        self.target_value = 0.25 # to target

        # reverse joint direction check if necessary
        if self.joint_type == "PhysicsRevoluteJoint":
            self.check_joint_direction()

        # other constant
        self.total_step = 0
        self.print_every = 30
        self.checking_interval = 30

        # register events
        if not IS_RUNTIME:
            self.create_task_callback()
     
    def check_joint_direction(self):
        """
        Check joint positive rotation to upper or negative rotation to lower
        """
        is_upper = abs(self.joint_checker.upper) > abs(self.joint_checker.lower)
        if not is_upper:
            # if is lower, reverse init_value and target value
            self.init_value = 1 - self.init_value if self.init_value != -1 else -1
            self.target_value = 1 - self.target_value

    ################################### UPDATE ###########################################

    def create_task_callback(self):
        self.timeline = omni.timeline.get_timeline_interface()
        stream = self.timeline.get_timeline_event_stream()
        self._timeline_subscription = stream.create_subscription_to_pop(self._on_timeline_event)
        # subscribe to Physics updates:
        self._physics_update_subscription = omni.physx.get_physx_interface().subscribe_physics_step_events(
            self._on_physics_step
        )

    def _on_timeline_event(self, e):
        """
        set up timeline event
        """
        if e.type == int(omni.timeline.TimelineEventType.STOP):
            self.it = 0
            self.time = 0
            self.reset()
    
    def reset(self):
        """
        Reset event
        """
        self._physics_update_subscription = None
        self._timeline_subscription = None
        # self._setup_callbacks()

    def _on_physics_step(self, dt):
        self.start_checking()

    def start_checking(self):
        
        self.total_step += 1
        if self.total_step % self.checking_interval == 0:
            percentage =  self.joint_checker.compute_percentage()

            # log
            if self.total_step % self.print_every == 0:
                print("current: {:.1f}; target: {:.1f}; delta percentage: {:.1f}:".format(percentage, self.target_value * 100, self.target_value * 100 - percentage) )
                
            
            if percentage / 100.0 > self.target_value:
                print("success")
                # self.timeline.pause()
