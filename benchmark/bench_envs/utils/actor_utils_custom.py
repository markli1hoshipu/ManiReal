import sapien
from sapien import Entity

class Simple_Actor:
    def __init__(self, actor: Entity, mass=0.01, scale=None):
        self.actor = actor
        self.scale = scale
        self.set_mass(mass)
    
    def get_pose(self) -> sapien.Pose:
        """Get the sapien.Pose of the actor."""
        return self.actor.get_pose()

    def get_name(self):
        return self.actor.get_name()

    def set_name(self, name):
        self.actor.set_name(name)

    def set_mass(self, mass):
        for component in self.actor.get_components():
            if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                component.mass = mass