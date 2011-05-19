import random as pyrandom
import pyglet
import pyglet.gl as pgl
from numpy import *
import itertools as it
import time

import squirtle 

from grease.cython import arraygl

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
    
window = pyglet.window.Window(resizable=True,vsync=True)
context = window.context
config = context.config

def vertexlist(points,data='xyrgb'):
    assert(data == 'xyrgb')
    p_x = data.index('x')
    p_y = data.index('y')
    p_r = data.index('r')
    p_g = data.index('g')
    p_b = data.index('b')
    x = hstack((points[:,p_x], points[:,p_x] * 1.3, points[:,p_x] * 1.5))
    y = hstack((points[:,p_y], points[:,p_y] * 1.3, points[:,p_y] * 1.5))
    r = hstack((points[:,p_r], points[:,p_r] * 0.6, points[:,p_r] * 0.4))
    g = hstack((points[:,p_g], points[:,p_g] * 0.6, points[:,p_g] * 0.4))
    b = hstack((points[:,p_b], points[:,p_b] * 0.6, points[:,p_b] * 0.4))
    size = x.shape[0]
    pxy = vstack((x,y)).transpose()
    p_xy = pxy.ravel()
    prgb = vstack((r,g,b)).transpose()
    p_rgb = prgb.ravel()

    vertices = tuple(p_xy)
    colors = tuple(p_rgb)
    return (size, (
            ('v2f', vertices ),
            ('c3f', colors )
            )
        )
    
def circle(steps = 32, radius = 1, c = 'rgb'):
    i = linspace(0,2*pi,steps+1)
    x = cos(i) * ((cos(i*steps/4)/4.0+0.75)*radius) 
    y = sin(i) * ((cos(i*steps/4)/4.0+0.75)*radius)
    g = cos(i*3) *0.3 + 0.6
    b = sin(i*3) *0.3 + 0.4
    r = 1 - b
    c0 = zeros(x.shape[0])
    c1 = ones(x.shape[0])
    lc = {'r' : r,'g' : g,'b' : b, '0' : c0, '1': c1}
    return vstack((x,y,lc[c[0]],lc[c[1]],lc[c[2]])).transpose()





@window.event
def on_resize(width, height):
    glViewport(0, 0, width, height)
    glMatrixMode(pgl.gl.GL_PROJECTION)
    glLoadIdentity()
    
    aspect = float(width) / float(height)
    if aspect < 1:
        glOrtho(-1,1, -1/aspect, 1/aspect, -1, 1)
    else:
        glOrtho(-1*aspect,1*aspect, -1, 1, -1, 1)
    glMatrixMode(pgl.gl.GL_MODELVIEW)
    return pyglet.event.EVENT_HANDLED



entity_type_counter = it.count()
class BaseEntity(object):
    _id_counter = it.count()
    _last_id = -1
    _var_counter = it.count()
    _min_resize = 8
    components = "n t x y z w r dx dy".split()
    _data_width = len(components)
    _data_array = zeros( (_min_resize, _data_width) )*nan
    entitytype  = entity_type_counter.next()
    index = None
    
    @classmethod
    def get_id(cls):
        newid = cls._id_counter.next()
        cls._last_id = newid
        return newid
    """
    @classmethod
    def get_idarray(cls):
        return hstack( (arange(cls._last_id+1),
            zeros(cls._data_array.shape[0]-cls._last_id-1)*nan) )
    """
    @classmethod
    def get_count(cls):
        return cls._data_array.shape[0]

    @classmethod
    def get_data(cls):
        return cls._data_array
    """ 
    @classmethod
    def get_iddata(cls):
        return hstack((array([cls.get_idarray()]).transpose(),cls.get_data()))
    """    
    @classmethod
    def set_entity(cls, pos, point):
        if pos >= cls._data_array.shape[0]:
            cls._data_array.resize(pos+cls._min_resize, cls._data_width)
        cls._data_array[pos] = point
        return pos
        
    def __init__(self,**kwargs):
        self.id = self.get_id()
        Entity.set_entity(self.id,zeros(self._data_width))
        self.n = self.id
        self.t = self.entitytype
        for k,v in list(kwargs.iteritems()):
            if k in self.components: 
                setattr(self,k,v)
                del kwargs[k]
        
    def __setattr__(self, k,v):
        if k in self.components:
            i = self.components.index(k)
            v = float(v)
            self._data_array[self.id][i] = v
        else:
            return object.__setattr__(self,k,v)
        
    def __getattr__(self, k):
        if k in self.components:
            i = self.components.index(k)
            return self._data_array[self.id][i]
        else:
            return object.__getattr__(self,k)
                    
    @classmethod
    def array(cls, klist):
        return [ cls._data_array[:cls._last_id+1,cls.components.index(k)] for k in klist.split(",") ]

    @classmethod
    def idx(cls):
        _where = cls._data_array[:cls._last_id+1, cls.components.index('t')] == cls.entitytype 
        return _where

    @classmethod
    def idx_update(cls):
        cls.index = cls.idx().copy()

    @classmethod
    def farray(cls, klist):
        _where = cls.idx()
    
        return [ cls._data_array[_where,cls.components.index(k)] for k in klist.split(",") ]
        

        
class Entity(BaseEntity):
    entitytype  = entity_type_counter.next()
    centerpoint = array( (0,0,0,0,0) )
    mycircle = vstack((centerpoint,circle(48)))
    shape = pyglet.graphics.Batch()
    shape_count, shape_data = vertexlist(mycircle)
    shape.add(shape_count,pyglet.gl.GL_TRIANGLE_FAN,None,*shape_data)
    displaylist = None
    entitytype  = entity_type_counter.next()
    def __init__(self, *args, **kwargs):
        BaseEntity.__init__(self, *args, **kwargs)
        
    @classmethod
    def draw(cls):
        if not cls.displaylist:
            glLoadIdentity()
            cls.displaylist = glGenLists(1)
            glNewList(cls.displaylist,GL_COMPILE)
            cls.shape.draw()
            glEndList()
            
            
        #X,Y,Z,W,R = cls.farray("x,y,z,w,r")
        #arraygl.draw_array2(cls.displaylist,X,Y,Z,W,R)
        T,X,Y,Z,W,R = cls.array("t,x,y,z,w,r")
        #print R.min(), R.max()
        #arraygl.draw_array3(cls.displaylist,cls.entitytype,T.copy(),X.copy(),Y.copy(),Z.copy(),W.copy(),R.copy())
        arraygl.draw_array3(cls.displaylist,cls.entitytype,T,X,Y,Z,W,R)
    

    @classmethod
    def update(cls,dt, depth=0):
        cls.idx_update()
        i = cls.index
        X,Y,W,DX,DY = cls.array("x,y,w,dx,dy")
        X[i] += DX[i] * dt
        Y[i] += DY[i] * dt
        DX[i] -= X[i] * dt
        DY[i] -= Y[i]* dt
        W[i] += (DX[i] + DY[i]) * dt * 360
        return
        cd = sqrt(x**2 + y**2)
        xn = x / cd
        yn = y / cd
        
        dx -= xn * 12.0 * dt * r
        dy -= yn * 12.0 * dt * r
        dx /= 1.01**dt
        dy /= 1.01**dt
        return 
        x1, x2 = ix_(x,x)
        y1, y2 = ix_(y,y)
        r1, r2 = ix_(r,r)
        dx2 = x1-x2
        dy2 = y1-y2
        d2 = (dx2) ** 2 + (dy2) ** 2
        r2 = (r1+r2) ** 2 +.001
        
        f = d2 / r2
        
        idx1 , idx2 = where(f < 1)
            
        pairs = array([idx1,idx2]).transpose()[idx1<idx2]
        pairs_t = pairs.transpose()

        h = sqrt(d2[pairs_t[0],pairs_t[1]])
        rr = sqrt(r2[pairs_t[0],pairs_t[1]])
        f = h/rr/2.0
        corr = 30
        limit = 0.25
        f[f<limit]=limit
        x1 = x[pairs_t[0]]
        y1 = y[pairs_t[0]]
        dx1 = dx[pairs_t[0]]
        dy1 = dy[pairs_t[0]]
        x2 = x[pairs_t[1]]
        y2 = y[pairs_t[1]]
        dx2 = dx[pairs_t[1]]
        dy2 = dy[pairs_t[1]]
        
        dnx = (x1-x2) / h
        dny = (y1-y2) / h
        rf = r[pairs_t[0]]**2 / r[pairs_t[1]]**2
        jx = dnx * dt / f * rf / f
        jy = dny * dt / f * rf / f
        dx[pairs_t[0]] += jx
        dy[pairs_t[0]] += jy
        dx[pairs_t[1]] -= jx
        dy[pairs_t[1]] -= jy
        x[pairs_t[0]] += jx / corr
        y[pairs_t[0]] += jy / corr
        x[pairs_t[1]] -= jx / corr
        y[pairs_t[1]] -= jy / corr
        global draw_times
        if draw_times < 2:
            draw_times+=1
            
        """
        t = 1.6
        for p1, p2 in pairs:
            f1 = f[p1][p2]
            if f1 < 0.02: f1 = 0.02
            f1 *= 100
            h2 = d2[p1][p2]
            dnx = dx2[p1][p2] / h2 
            dny = dy2[p1][p2] / h2
            dx[p1] /= t
            dy[p1] /= t
            dx[p2] /= t
            dy[p2] /= t
            
            dx[p1] += dnx * dt / f1 * (r[p2]**2 / r[p1]**2)
            dy[p1] += dny * dt / f1 * (r[p2]**2 / r[p1]**2)
            dx[p2] += -dnx * dt / f1* (r[p1]**2 / r[p2]**2)
            dy[p2] += -dny * dt / f1* (r[p1]**2 / r[p2]**2)
            x[p1] += dnx * dt / f1 * (r[p2]**2 / r[p1]**2) / 10.0
            y[p1] += dny * dt / f1 * (r[p2]**2 / r[p1]**2) / 10.0
            x[p2] += -dnx * dt / f1* (r[p1]**2 / r[p2]**2) / 10.0
            y[p2] += -dny * dt / f1* (r[p1]**2 / r[p2]**2) / 10.0
            
            dx[p1] *= t
            dy[p1] *= t
            dx[p2] *= t
            dy[p2] *= t
        """

class EntityRed(Entity):
    entitytype  = entity_type_counter.next()
    centerpoint = array( (0,0,1,0,0) )
    mycircle = vstack((centerpoint,circle(64,c='r00')))
    shape = pyglet.graphics.Batch()
    shape_count, shape_data = vertexlist(mycircle)
    shape.add(shape_count,pyglet.gl.GL_TRIANGLE_FAN,None,*shape_data)
    displaylist = None

class EntityGreen(Entity):
    entitytype  = entity_type_counter.next()
    centerpoint = array( (0,0,0,1,0) )
    mycircle = vstack((centerpoint,circle(96,c='0g0')))
    shape = pyglet.graphics.Batch()
    shape_count, shape_data = vertexlist(mycircle)
    shape.add(shape_count,pyglet.gl.GL_TRIANGLE_FAN,None,*shape_data)
    displaylist = None
    

class EntityBlue(Entity):
    entitytype  = entity_type_counter.next()
    centerpoint = array( (0,0,0,0,1) )
    mycircle = vstack((centerpoint,circle(128,c='00b')))
    shape = pyglet.graphics.Batch()
    shape_count, shape_data = vertexlist(mycircle)
    shape.add(shape_count,pyglet.gl.GL_TRIANGLE_FAN,None,*shape_data)
    displaylist = None

class MyShape(object):
    def __init__(self, filename, quality = 4):
        self.shape = squirtle.SVG(filename, bezier_points=quality, 
            circle_points=quality * 3, anchor_x='center', anchor_y='center', reversed_order=True)
        max_size = max([self.shape.width,self.shape.height])
        
        self.main_scale = 1.0 /  max_size
        self.displaylist = self.shape.disp_list
        #print self.shape.width
        #self.shape.width = 1
        #self.shape.height = 1
        #print self.shape.width
        
    def draw(self):
        self.shape.draw(x=0,y=0, scale=self.main_scale)
    
"""
class EntityNucleus(Entity):
    entitytype  = entity_type_counter.next()
    shape = MyShape("svg/nucleus.svg", 5)
    displaylist = None
"""
class EntityShip(Entity):
    entitytype  = entity_type_counter.next()
    shape = MyShape("svg/ship.svg", 6)
    displaylist = None
        
def entityblue_update(cls,dt, depth=0):
    cls.parent_update(dt,depth)
    W, = cls.array("w")
    i = cls.idx()
    W[i] += 90 * dt
    
EntityBlue.parent_update = EntityBlue.update
EntityBlue.update = classmethod(entityblue_update)
    
#entity_types=[Entity,EntityGreen, EntityRed, EntityBlue,EntityNucleus]
entity_types=[EntityShip]

for i in range(1000):
    EType = pyrandom.choice(entity_types)    
    e = EType()
    a = random.uniform(0,360)
    d = random.normal(0.0,1.0)
    e.x = cos(a) * d
    e.y = sin(a) * d
    e.z = random.uniform(-0.5,0.5)
    e.w = a
    
    t = 0.06
    e.r = random.normal(t,t) 
    if e.r < 0.02: e.r = 0.02

    a = random.uniform(0,360)
    d = random.normal(0.5,1.5)
    e.dx = cos(a) * d + e.x
    e.dy = sin(a) * d + e.y

#print Entity.components
#print Entity.get_data()
fps_display = pyglet.clock.ClockDisplay()
frame_time = time.time()
@window.event
def on_draw():
    global drawing_time, draw_times, frame_time
    t1 = time.time()
    draw_times += 1
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    for EType in entity_types:
        EType.draw()
    
    t2 = time.time()
    drawing_time += t2-t1
    FPS = 70
    MIN_NAP = 0.5 # ms
    NAP_OFFSET = 0.5 # ms
    frame_delta = time.time() - frame_time + NAP_OFFSET/1000.0
    desired_delta = 1.0/FPS
    naptime = max(desired_delta - frame_delta, MIN_NAP/1000.0)
    time.sleep(naptime)
    frame_time = time.time()

# vertex: v2f: vertexes XY in float
# color: c4f : RGBA in float

def update(dt):
    global updating_time
    t1 = time.time()
    for EType in entity_types:
        EType.update(dt)
    
    t2 = time.time()
    updating_time += t2-t1

def fps(dt):
    global start_time, drawing_time, updating_time, draw_times
    if dt:
        elapsed = time.time() - start_time
        print "%.2f FPS. (%d dps)" % (pyglet.clock.get_fps(),draw_times), "drawing: %.2f%% (%.2fms)  updating: %.2f%%" % (
            100*drawing_time/elapsed, 1000*drawing_time/draw_times, 100*updating_time/elapsed)
    start_time = time.time()
    drawing_time =  updating_time = draw_times = 0
"""
xlist = linspace(3,12,33)
n = arange(33)
print xlist.sum(), n.shape, xlist.shape
print arraygl.numpylist(n,xlist)
"""
#squirtle.setup_gl()
glEnable(GL_DEPTH_TEST)
glEnable(GL_LINE_SMOOTH)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

pyglet.clock.schedule_interval(fps, 1)    
pyglet.clock.schedule(update)    
#pyglet.clock.schedule_interval(update, 1/65.0)    
fps(0)

pyglet.app.run()