import random as pyrandom
import itertools as it
import time
"""
print "importing squirtle . . . "
import squirtle 

print "importing grease.cython.arraygl . . . "
from grease.cython import arraygl
"""
print "importing grease.cython.chipmunk . . . "
from grease.cython import chipmunk

print "initializing . . ."
chipmunk.init()
myspace = chipmunk.Space()
myspace.gravity = (0, -900)
boxbody, boxshape = myspace.newBox(10,10,5)
boxbody.set_position(5,10)
boxbody2, boxshape2 = myspace.newBox(10,10,15)
boxbody2.set_position(2,12)
for i in range(10):
    print boxbody.position()
    myspace.step(0.1, i)

