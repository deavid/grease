# do de imports ...
import grease
from grease import component, controller, geometry, mode

# of course there will be more, but let's do some code :-)

# I prefer things "defined" rather than created at runtime. I'm new
# to this in Python and i started to see it in SqlAlchemy, the main idea
# is to have some part of the code working more like a config.ini rather
# than code. (Define what you want rather than writting code that creates
# those things)

# I'll try to put the things in the __class__ rather than in the instance.



# MyGameWorld would be a child of grease.World, but i avoided it to 
# allow introspection using bpython -i of this file.
class MyGameWorld(object): 
    # The world: To define the components in __class__ we need to define first
    # which components we want:
    components = grease.WorldComponents()
    # Here you restrict for the names, which types should be
    components.position = component.Position
    components.movement = component.Movement
    components.shape = component.Shape
    components.renderable = component.Renderable
    components.collision = component.Collision
    # they can exist or not on an entity, but if they do, they must pass
    # the check: assert(isinstance(entity.movement, self.movement))
    # you could subclass, or use metaclass abc magic.
    
    # this avoids conflicts in the systems and renderers. World systems and 
    # renderers should work only on the components defined here. They
    # could check it and require to you to define here their required
    # components in order to work.
    
    def configure(self):
        self.systems.movement = controller.EulerMovement()
        self.systems.collision = collision.Circular(
            handlers=[collision.dispatch_events])
        self.systems.sweeper = Sweeper()
        self.systems.gun = Gun()
        self.systems.wrapper = PositionWrapper()

        self.renderers.camera = renderer.Camera(
            position=(window.width / 2, window.height / 2))
        self.renderers.vector = renderer.Vector(line_width=1.5)
    

# --- Entities ..

class SpaceEntity(grease.Entity):
    position = component.Position(x=0, y=0)
    movement = component.Movement()
    # In grease.Entity.__new__ these generic components 
    # could be simply "copied" to the new Entity and to the new world.
    # copying, of course, their values as a default.
    def __init__(self, world, position = None):
        # You can also set the properties at run-time 
        if position:
            self.position.position = position
        # but you cannot add more components here.


        
class Ship(SpaceEntity):
    SHAPE_VERTS = [
        (-8, -12), (-4, -10), (0, -8), (4, -10), (8, -12), # flame
        (0, 12), (-8, -12), (0, -8), (8, -12)]   
    shape = component.Shape(SHAPE_VERTS, closed = False) # New shape.
    renderable = component.Renderable(color="#7f7")
    collision = component.Collision
    gun = component.Component(
            firing=bool, 
            last_fire_time=float, 
            cool_down=float
            ) # ad-hoc component example.
    # And you can also run things here. It's a small function.
    gun.cool_down = 2.0 # for example, setting defaults...
    
    
        
    
world = MyGameWorld()
