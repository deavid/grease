cimport cython
ctypedef int bool

cdef extern from "chipmunk/chipmunk.h":
    ctypedef float cpFloat 
    cpFloat INFINITY
    
    ctypedef struct cpVect:
        cpFloat x,y
        
    ctypedef void* cpDataPointer
    ctypedef bool cpBool
    ctypedef int cpGroup
    ctypedef int cpLayers
    ctypedef int cpCollisionType
    ctypedef struct cpBB:
        cpFloat l, b, r ,t
        
    cpVect cpv(cpFloat x, cpFloat y)
    
    ctypedef struct cpBody:
        # *** Integration Functions.

        # Function that is called to integrate the body's velocity. (Defaults to cpBodyUpdateVelocity)
        #cpBodyVelocityFunc velocity_func;
    
        # Function that is called to integrate the body's position. (Defaults to cpBodyUpdatePosition)
        #cpBodyPositionFunc position_func;
    
        # *** Mass Properties
    
        # Mass and it's inverse.
        # Always use cpBodySetMass() whenever changing the mass as these values must agree.
        cpFloat m, m_inv
    
        # Moment of inertia and it's inverse.
        # Always use cpBodySetMoment() whenever changing the moment as these values must agree.
        cpFloat i, i_inv
    
        # *** Positional Properties
    
        # Linear components of motion (position, velocity, and force)
        cpVect p, v, f
    
        # Angular components of motion (angle, angular velocity, and torque)
        # Always use cpBodySetAngle() to set the angle of the body as a and rot must agree.
        cpFloat a, w, t
    
        # Cached unit length vector representing the angle of the body.
        # Used for fast vector rotation using cpvrotate().
        cpVect rot
    
        # *** User Definable Fields
    
        # User defined data pointer.
        cpDataPointer data
    
        # *** Other Fields
    
        # Maximum velocities this body can move at after integrating velocity
        cpFloat v_limit, w_limit
        
    cpBody *cpBodyAlloc()
    cpBody *cpBodyInit(cpBody *body, cpFloat m, cpFloat i)
    cpBody *cpBodyNew(cpFloat m, cpFloat i)

    void cpBodyDestroy(cpBody *body)
    void cpBodyFree(cpBody *body)
    
    void cpBodySetMass(cpBody *body, cpFloat m)
    void cpBodySetMoment(cpBody *body, cpFloat i)
    
    
    ctypedef struct cpSpace:
        # Number of iterations to use in the impulse solver to solve contacts.
        int iterations
    
        # Number of iterations to use in the impulse solver to solve elastic collisions.
        int elasticIterations
    
        # Default gravity to supply when integrating rigid body motions.
        cpVect gravity
    
        # Default damping to supply when integrating rigid body motions.
        cpFloat damping
    
        # Speed threshold for a body to be considered idle.
        # The default value of 0 means to let the space guess a good threshold based on gravity.
        cpFloat idleSpeedThreshold
    
        # Time a group of bodies must remain idle in order to fall asleep
        # The default value of INFINITY disables the sleeping algorithm.
        cpFloat sleepTimeThreshold
    
        cpBody staticBody
        

    cdef void cpInitChipmunk()

    # Basic allocation/destruction functions.
    cdef cpSpace* cpSpaceAlloc()
    cdef cpSpace* cpSpaceInit(cpSpace *space)
    cdef cpSpace* cpSpaceNew()

    cdef void cpSpaceDestroy(cpSpace *space)
    cdef void cpSpaceFree(cpSpace *space)

    # Convenience function. Frees all referenced entities. (bodies, shapes and constraints)
    cdef void cpSpaceFreeChildren(cpSpace *space)
    
    # dim is the minimum cell-size, the average size of the object in the space
    # count is the minimum cells in the space. object * 10x is the recommended value.
    void cpSpaceResizeStaticHash(cpSpace *space, cpFloat dim, int count)
    void cpSpaceResizeActiveHash(cpSpace *space, cpFloat dim, int count)


    ctypedef struct cpShape:
        # cpBody that the shape is attached to.
        cpBody *body

        # Cached BBox for the shape.
        cpBB bb
    
        # Sensors invoke callbacks, but do not generate collisions
        cpBool sensor
    
        # *** Surface properties.
    
        # Coefficient of restitution. (elasticity)
        cpFloat e
        # Coefficient of friction.
        cpFloat u
        # Surface velocity used when solving for friction.
        cpVect surface_v

        # *** User Definable Fields

        # User defined data pointer for the shape.
        cpDataPointer data
    
        # User defined collision type for the shape.
        cpCollisionType collision_type
        # User defined collision group for the shape.
        cpGroup group
        # User defined layer bitmask for the shape.
        cpLayers layers

    void cpShapeDestroy(cpShape *shape)
    void cpShapeFree(cpShape *shape)
    # Destroy and Free functions are shared by all shape types. Allocation 
    #  and initialization functions are specific to each shape type. See below.
    ctypedef cpShape cpCircleShape
    ctypedef cpShape cpSegmentShape
    ctypedef cpShape cpPolyShape


    # ** Working With Circle Shapes: 

    cpCircleShape *cpCircleShapeAlloc()
    cpCircleShape *cpCircleShapeInit(cpCircleShape *circle, cpBody *body, cpFloat radius, cpVect offset)
    cpShape *cpCircleShapeNew(cpBody *body, cpFloat radius, cpVect offset)

    # body is the body to attach the circle to, offset is the offset from the body’s center of gravity in body local coordinates.

    cpVect cpCircleShapeGetOffset(cpShape *circleShape)
    cpFloat cpCircleShapeGetRadius(cpShape *circleShape)

    #Getters for circle shape properties. Passing as non-circle shape will throw an assertion.
    
    # ** Working With Segment Shapes:

    cpSegmentShape* cpSegmentShapeAlloc()
    cpSegmentShape* cpSegmentShapeInit(cpSegmentShape *seg, cpBody *body, cpVect a, cpVect b, cpFloat radius)
    cpShape* cpSegmentShapeNew(cpBody *body, cpVect a, cpVect b, cpFloat radius)

    # body is the body to attach the segment to, a and b are the endpoints, and radius is the thickness of the segment.

    cpVect cpSegmentShapeGetA(cpShape *shape)
    cpVect cpSegmentShapeGetA(cpShape *shape)
    cpVect cpSegmentShapeGetNormal(cpShape *shape)
    cpFloat cpSegmentShapeGetRadius(cpShape *shape)

    # Getters for segment shape properties. Passing a non-segment shape will throw an assertion.
    
    # ** Working With Polygon Shapes:
    cpPolyShape *cpPolyShapeAlloc()
    cpPolyShape *cpPolyShapeInit(cpPolyShape *poly, cpBody *body, int numVerts, cpVect *verts, cpVect offset)
    cpShape *cpPolyShapeNew(cpBody *body, int numVerts, cpVect *verts, cpVect offset)

    # body is the body to attach the poly to, verts is an array of cpVect structs defining a convex hull with a clockwise winding, offset is the offset from the body’s center of gravity in body local coordinates. An assertion will be thrown the vertexes are not convex or do not have a clockwise winding.

    int cpPolyShapeGetNumVerts(cpShape *shape)
    cpVect cpPolyShapeGetVert(cpShape *shape, int index)

    # Getters for poly shape properties. Passing a non-poly shape or an index that does not exist will throw an assertion.

    
    # MATH::    
    # Clamp f to be between min and max.
    cpFloat cpfclamp(cpFloat f, cpFloat minimum, cpFloat maximum)

    #Linearly interpolate between f1 and f2.
    cpFloat cpflerp(cpFloat f1, cpFloat f2, cpFloat t)

    #Linearly interpolate from f1 towards f2 by no more than d.
    cpFloat cpflerpconst(cpFloat f1, cpFloat f2, cpFloat d)
    

import numpy as np
cimport numpy as np

np.import_array()


cpdef int init():
    cpInitChipmunk()
    return 0
    



cdef class Body:
    cpdef cpBody* cpbody
    cpdef int automanaged
    
    def __cinit__(self, float m = 1, float i = 1, int autocreate = 1):
        self.automanaged = autocreate
        if self.automanaged:
            self.cpbody = cpBodyNew(m,i)
    
    def __dealloc__(self):
        if self.automanaged:
            cpBodyFree(self.cpbody)
    
    @property
    def mass(self):
        return self.cpbody.m
    
    @mass.setter
    def setmass(self, m):
        cpBodySetMass(self.cpbody,m)
    
    m = mass
    
    @property
    def inertia(self):
        return self.cpbody.i
    
    @inertia.setter
    def setinertia(self, i):
        cpBodySetMoment(self.cpbody,i)
    
    i = inertia
    
    

cdef class Shape:
    cpdef cpShape* cpshape
    cpdef int automanaged
    def __cinit__(self):
        self.cpshape = NULL
        self.automanaged = 0
    
    def __dealloc__(self):
        if self.automanaged:
            cpShapeFree(self.cpshape)
            
cdef class CircleShape(Shape):
    def __cinit__(self, Body body, cpFloat radius, offset_xy):
        cdef cpVect offset = cpv(offset_xy[0],offset_xy[1])
        self.body = body
        self.cpshape = cpCircleShapeNew(body.cpbody, radius, offset)
        self.automanaged = 1
    
    
    
cdef class Space:
    cpdef cpSpace* cpspace
    cpdef Body staticBody
    
    def __cinit__(self):
        self.cpspace = cpSpaceNew()
        self.staticBody = Body(autocreate = 0)
        self.staticBody.cpbody = &self.cpspace.staticBody
    
    def __dealloc__(self):
        self.staticBody = None
        cpSpaceFree(self.cpspace)
    
    @property
    def iterations(self):
        return self.cpspace.iterations
    
    @iterations.setter
    def setiterations(self, value):
        self.cpspace.iterations = int(value)
    
    @property
    def elasticIterations(self):
        return self.cpspace.elasticIterations
    
    @elasticIterations.setter
    def setelasticIterations(self, value):
        self.cpspace.elasticIterations = int(value)
        
    @property
    def gravity(self):
        return (self.cpspace.gravity.x,self.cpspace.gravity.y)
    
    @gravity.setter
    def setgravity(self, x, y):
        self.cpspace.gravity.x = float(x)
        self.cpspace.gravity.y = float(y)
    
    @property
    def damping(self):
        return self.cpspace.damping
    
    @damping.setter
    def setdamping(self, value):
        self.cpspace.damping = float(value)

    @property
    def idleSpeedThreshold(self):
        return self.cpspace.idleSpeedThreshold

    @idleSpeedThreshold.setter
    def setidleSpeedThreshold(self, value):
        self.cpspace.idleSpeedThreshold = float(value)

    @property
    def sleepTimeThreshold(self):
        if self.cpspace.sleepTimeThreshold == INFINITY:
            return None
        else:
            return self.cpspace.sleepTimeThreshold

    @sleepTimeThreshold.setter
    def setsleepTimeThreshold(self, value):
        if value is None:
            self.cpspace.sleepTimeThreshold = INFINITY
        else:
            self.cpspace.sleepTimeThreshold = float(value)

    def resizeHash(self, dim, count, static_dim = None, static_count = None):
        dim = float(dim)
        count = int(count)
        if static_dim is None: static_dim = dim
        if static_count is None: static_count = count
        
        cpSpaceResizeStaticHash(self.cpspace, static_dim, static_count)
        cpSpaceResizeActiveHash(self.cpspace, dim, count)
        
        
