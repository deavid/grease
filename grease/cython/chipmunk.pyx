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
        
    ctypedef struct cpConstraint:
        cpBody *a, *b
        cpFloat maxForce
        cpFloat biasCoef
        cpFloat maxBias
    
        cpDataPointer data

    cpBody *cpBodyAlloc()
    cpBody *cpBodyInit(cpBody *body, cpFloat m, cpFloat i)
    cpBody *cpBodyNew(cpFloat m, cpFloat i)

    void cpBodyDestroy(cpBody *body)
    void cpBodyFree(cpBody *body)
    
    void cpBodySetMass(cpBody *body, cpFloat m)
    void cpBodySetMoment(cpBody *body, cpFloat i)

    # ** Moment of inertia helper functions **
    # Use the following functions to approximate the moment of inertia for your body, adding the results together if you want to use more than one.

    # Calculate the moment of inertia for a hollow circle, r1 and r2 are the inner and outer diameters in no particular order. (A solid circle has an inner diameter of 0)
    cpFloat cpMomentForCircle(cpFloat m, cpFloat r1, cpFloat r2, cpVect offset)

    # Calculate the moment of inertia for a line segment. The endpoints a and b are relative to the body.
    cpFloat cpMomentForSegment(cpFloat m, cpVect a, cpVect b)

    # Calculate the moment of inertia for a solid polygon shape assuming it’s center of gravity is at it’s centroid. The offset is added to each vertex.
    cpFloat cpMomentForPoly(cpFloat m, int numVerts, cpVect *verts, cpVect offset)

    # Calculate the moment of inertia for a solid box centered on the body.    
    cpFloat cpMomentForBox(cpFloat m, cpFloat width, cpFloat height)

    
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

    # ** Simulating the Space: **

    # Update the space for the given time step. Using a fixed time step 
    # is highly recommended. Doing so will increase the efficiency of the 
    # contact persistence, requiring an order of magnitude fewer 
    # iterations and CPU usage.
    void cpSpaceStep(cpSpace *space, cpFloat dt)

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
    cpShape *cpBoxShapeNew(cpBody *body, cpFloat width, cpFloat height)
    # body is the body to attach the poly to, verts is an array of cpVect structs defining a convex hull with a clockwise winding, offset is the offset from the body’s center of gravity in body local coordinates. An assertion will be thrown the vertexes are not convex or do not have a clockwise winding.

    int cpPolyShapeGetNumVerts(cpShape *shape)
    cpVect cpPolyShapeGetVert(cpShape *shape, int index)

    # Getters for poly shape properties. Passing a non-poly shape or an index that does not exist will throw an assertion.

    # **** Operations: ***
    #  These functions add and remove shapes, bodies and constraints from 
    #  space. See the section on Static Shapes above for an explanation of 
    #  what a static shape is and how it differs from a normal shape. Also, 
    #  you cannot call the any of these functions from within a callback 
    #  other than a post-step callback (which is different than a post-solve 
    #  callback!). Attempting to add or remove objects from the space while 
    #  cpSpaceStep() is still executing will throw an assertion. See the 
    #  callbacks section for more information.

    void cpSpaceAddShape(cpSpace *space, cpShape *shape)
    void cpSpaceAddStaticShape(cpSpace *space, cpShape *shape)
    void cpSpaceAddBody(cpSpace *space, cpBody *body)
    void cpSpaceAddConstraint(cpSpace *space, cpConstraint *constraint)

    void cpSpaceRemoveShape(cpSpace *space, cpShape *shape)
    void cpSpaceRemoveStaticShape(cpSpace *space, cpShape *shape)
    void cpSpaceRemoveBody(cpSpace *space, cpBody *body)
    void cpSpaceRemoveConstraint(cpSpace *space, cpConstraint *constraint)



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
    
    def set_position(self, x, y):
        self.cpbody.p = cpv(x,y)
    
    def position(self):
        return (self.cpbody.p.x,self.cpbody.p.y)
    
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
    cpdef Body body
    cpdef int automanaged
    def __cinit__(self):
        self.cpshape = NULL
        self.automanaged = 0
    
    def __dealloc__(self):
        if self.automanaged:
            cpShapeFree(self.cpshape)

    cpdef elasticity(self): return self.cpshape.e
    cpdef set_elasticity(self, value): self.cpshape.e = float(value)

    cpdef friction(self): return self.cpshape.u
    cpdef set_friction(self, value): self.cpshape.u = float(value)
    
            
cdef class CircleShape(Shape):
    def __cinit__(self, Body body, cpFloat radius, offset_xy):
        cdef cpVect offset = cpv(offset_xy[0],offset_xy[1])
        self.body = body
        self.cpshape = cpCircleShapeNew(body.cpbody, radius, offset)
        self.automanaged = 1
    
cdef class BoxShape(Shape):
    def __cinit__(self, Body body, cpFloat width, cpFloat height):
        self.body = body
        self.cpshape = cpBoxShapeNew(body.cpbody, width, height)
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
    
    def addBody(self, Body body):
        cpSpaceAddBody(self.cpspace, body.cpbody)
    
    def addShape(self, Shape shape):
        cpSpaceAddShape(self.cpspace, shape.cpshape)
        
    def newBox(self, width, height, mass, elasticity = 0, friction = 0.5):
        body = Body(mass, cpMomentForBox(mass, width, height))
        self.addBody(body)
        shape = BoxShape(body, width, height)
        shape.set_elasticity(elasticity)
        shape.set_friction(friction)
        self.addShape(shape)
        return body, shape
        
    def step(self, float dt, int iterations):
        cdef int i
        for i in range(iterations):
            cpSpaceStep(self.cpspace, dt)
            
    
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
        
    def get_gravity(self): return (self.cpspace.gravity.x,self.cpspace.gravity.y)
    def set_gravity(self, object xy): self.cpspace.gravity.x, self.cpspace.gravity.y = xy
    gravity = property(get_gravity,set_gravity)
    
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
        
        
