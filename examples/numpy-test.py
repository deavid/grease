# encoding: UTF-8
import numpy as n
import random
import time

def create_entity():
    x = random.uniform(0,150)
    y = random.uniform(0,150)
    r = random.uniform(3,30)
    return n.array([x,y,r])

def ecollide(entity1,entity2):
    e1 = entities[entity1]
    e2 = entities[entity2]
    d2xy = (e1 - e2)[:2] ** 2
    d2 = d2xy.sum()
    r2 = (e1 + e2)[2] ** 2
    return d2 / r2
    

entities = n.vstack([ create_entity() for x in range(200) ])

def getpairs(entities):
    Ex = entities[:,0] 
    Ey = entities[:,1] 
    Er = entities[:,2] 

    Ex1, Ex2 = n.ix_(Ex, Ex)
    Ey1, Ey2 = n.ix_(Ey, Ey)
    Er1, Er2 = n.ix_(Er, Er)

    PairH2 = (Ex1-Ex2)**2+(Ey1-Ey2)**2 / ((Er1+Er2) ** 2) 
    return PairH2

def getcollisions(entities):
    pairs = getpairs(entities)
    Ca,Cb = n.array(n.where(pairs<1))
    Cbool = Ca<Cb
    collisions = n.array( [ Ca[Cbool], Cb[Cbool] ] ).transpose()
    return collisions
    
def getbbox(entities):
    x,y,r = entities.transpose()
    left = x - r
    right = x + r
    top = y - r
    bottom = y + r
    return left,right,top,bottom
    
def getboxrange(entities):
    x,y,r = entities.transpose()
    left = x.min()
    right = x.max()
    top = y.min()
    bottom = y.max()
    return left,right,top,bottom

def getboxcenter(box):
    l,r,t,b = box
    x = (l+r)/2
    y = (t+b)/2
    return x,y
    
def getentitiesatbox(entities,box):    
    el,er,et,eb = getbbox(entities)
    bl,br,bt,bb = box
    return n.where( (er > bl) & (el < br) & (et < bb) & (eb > bt))[0]

def getboxes(box):
    bxy = getboxcenter(box)
    box1 = (box[0],bxy[0],box[2],bxy[1])    
    box2 = (bxy[0],box[1],bxy[1],box[3])    

    box3 = (box[0],bxy[0],bxy[1],box[3])    
    box4 = (bxy[0],box[1],box[2],bxy[1])    
    return box1,box2,box3,box4
    
def getdivideandconquer(entities):
    box = getboxrange(entities)
    boxes = getboxes(box)
    return [ getentitiesatbox(entities, b) for b in boxes ]
    
def autodivide(entities):
    return _autodivide(entities, n.arange(entities.shape[0]))
    
def _autodivide(data, entities, depth = 0):
    print entities
    if entities.size < 10 + depth ** 2: return [entities]
    sections = []
    for section in getdivideandconquer(data[entities]):
        sections+=_autodivide(data, section, depth+1)
    return sections
    
    
x = autodivide(entities)    
#collisions = getcollisions(entities)
#print " %d collisions found in a set of %d circles."  % (len(collisions), len(entities))



