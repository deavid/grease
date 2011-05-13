#############################################################################
#
# Copyright (c) 2010 by Casey Duncan and contributors
# All Rights Reserved.
#
# This software is subject to the provisions of the MIT License
# A copy of the license should accompany this distribution.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
#
#############################################################################
"""Grease tutorial game revision 3"""

import os, sys
import math
import random
import itertools
import pyglet
from pyglet.window import key
import grease
from grease import component, controller, geometry, mode
from grease.geometry import Vec2d

#from grease import collision
#from grease import controller
#from grease import renderer

try:
    from grease.cython import collision
    from grease.cython import controller
    from grease.cython import renderer
except Exception, e:
    print "Unexpected error when trying to load cython-compiled parts:"
    print repr(e)
    print "Python-versions of grease have been loaded instead."
    print "This can lead to less performance."

from grease.controls import KeyControls

## Utility functions ##
path = lambda *x: os.path.join(os.path.dirname(__file__),*x)

pyglet.font.add_file(path('font', 'Vectorb.ttf'))

def load_sound(name, streaming=False):
    """Load a sound from the `sfx` directory"""
    print "Loading SFX %s ..." % name
    return pyglet.media.load(
        path('sfx', name), streaming=streaming)

def looping_sound(name):
    """Load a sound from the `sfx` directory and configure it too loop
    continuously
    """
    player = pyglet.media.Player()
    player.queue(load_sound(name, streaming=False))
    player.eos_action = player.EOS_LOOP
    return player

def mediumpoint(obj1, obj2):
    px1, py1 = obj1.position.position
    px2, py2 = obj2.position.position
    px = (px1 + px2) / 2.0
    py = (py1 + py2) / 2.0

    dx1, dy1 = obj1.movement.velocity
    dx2, dy2 = obj2.movement.velocity
    dx = (dx1 + dx2) / 2.0
    dy = (dy1 + dy2) / 2.0
    return ((px,py),(dx,dy))


## Define entity classes ##

class BlasteroidsEntity(grease.Entity):
    """Entity base class"""

    def explode(self):
        if self.states.exploded == True: return
        self.states.exploded = True
        """Segment the entity shape into itty bits"""
        shape = self.shape.verts.transform(angle=self.position.angle)
        for segment in shape.segments():
            debris = Debris(self.world)
            debris.shape.verts = segment
            debris.position.position = self.position.position
            debris.movement.velocity = self.movement.velocity
            debris.movement.velocity += segment[0].normalized() * random.gauss(50, 20)
            debris.movement.rotation = random.gauss(0, 45)
            debris.renderable.color = self.renderable.color


class Debris(grease.Entity):
    """Floating space junk"""


class PlayerShip(BlasteroidsEntity):
    """Thrust ship piloted by the player"""

    THRUST_ACCEL = 50
    TURN_RATE = 150
    SHAPE_VERTS = [
        (-8, -12), (-4, -10), (0, -8), (4, -10), (8, -12), # flame
        (0, 12), (-8, -12), (0, -8), (8, -12)]
    COLOR = "#7f7"
    COLLISION_RADIUS = 7.5
    COLLIDE_INTO_MASK = 0x1
    GUN_COOL_DOWN = 2.0
    GUN_SOUND = load_sound('pewpew.wav')
    THRUST_SOUND = looping_sound('thrust.wav')
    DEATH_SOUND = load_sound('dead.wav')
    POWERUP_SOUND = load_sound('powerup-extra.wav')

    def __init__(self, world, invincible=False):
        self.position.position = (0, 0)
        self.position.angle = 0
        self.movement.velocity = (0, 0)
        self.movement.rotation = 0
        self.states.exploded = False
        self.states.invincible = False
        self.shape.verts = self.SHAPE_VERTS
        self.shape.closed = False
        self.renderable.color = self.COLOR
        self.collision.radius = self.COLLISION_RADIUS
        self.gun.cool_down = self.GUN_COOL_DOWN
        self.gun.sound = self.GUN_SOUND
        self.gun.shots = 3
        self.gun.spread = 15.35
        self.gun.precision = 1
        self.set_invincible(invincible)
        self.engine.thrust = self.THRUST_ACCEL
        self.engine.turnrate = self.TURN_RATE
        if "--cheat1" in sys.argv:
            self.engine.thrust *= 1.5
            self.engine.turnrate *= 1.2
            self.gun.precision = 15
            self.gun.spread = 10
            self.gun.shots = 32
            self.gun.cool_down /= 15.0
        
        if "--cheat2" in sys.argv:
            self.engine.thrust *= 1.5
            self.engine.turnrate *= 1.2
            self.gun.precision = 15
            self.gun.spread = 0.1
            self.gun.shots = 64
            self.gun.cool_down /= 10.0
        
    
    def turn(self, direction):
        self.movement.rotation = self.engine.turnrate * direction
    
    def thrust_on(self):
        thrust_vec = geometry.Vec2d(0, self.engine.thrust)
        thrust_vec.rotate(self.position.angle)
        self.movement.accel = thrust_vec
        self.shape.verts[2] = (0, -16 - random.random() * 16)
        self.THRUST_SOUND.play()
    
    def thrust_off(self):
        self.movement.accel = (0, 0)
        self.shape.verts[2] = (0, -8)
        self.THRUST_SOUND.pause()

    def brake(self):
        thrust_vec = geometry.Vec2d(0, -self.engine.thrust)
        thrust_vec.rotate(self.position.angle)
        self.movement.accel = thrust_vec
        self.shape.verts[2] = (0, 0 - random.random() * 16)
        self.THRUST_SOUND.play()
    
    def set_invincible(self, invincible):
        """Set the invincibility status of the ship. If invincible is
        True then the ship will not collide with any obstacles and will
        blink to indicate this. If False, then the normal collision 
        behavior is restored
        """
        self.states.invincible = invincible
        if invincible:
            self.collision.into_mask = 0
            self.collision.from_mask = 0
            self.world.clock.schedule_interval(self.blink, 0.15)
            self.world.clock.schedule_once(lambda dt: self.set_invincible(False), 3)
            self.world.clock.schedule_once(lambda dt: self.set_invincible(False), 3.5) # Bug? sometimes the player disappears
        else:
            self.world.clock.unschedule(self.blink)
            self.renderable.color = self.COLOR
            self.collision.from_mask = 0xffffffff
            self.collision.into_mask = self.COLLIDE_INTO_MASK
    
    def blink(self, dt):
        """Blink the ship to show invincibility"""
        if not self.states.invincible: return
        if self.renderable:
            del self.renderable
        else:
            self.renderable.color = self.COLOR
    
    def collect(self, obj):
        assert(obj.taxonomy.type1 == "collectable")
        if obj.taxonomy.subtype1 == "gun-powerup":
            self.gun.cool_down /= 1.3
            self.gun.spread /= 1.5
            if self.gun.cool_down < self.GUN_COOL_DOWN / (1.0 + self.gun.shots / 2.0):
                self.gun.cool_down *= 2
                self.gun.shots += 1
                self.POWERUP_SOUND.play()
        elif obj.taxonomy.subtype1 == "electro-gun":
            self.gun.spread += 2
            self.gun.precision *= 1.3
            self.gun.shots += 1
            self.gun.cool_down /= 1.1
            self.engine.thrust *= 1.1
            self.POWERUP_SOUND.play()
        elif obj.taxonomy.subtype1 == "engine-powerup":
            if self.engine.thrust < self.THRUST_ACCEL * 2:
                self.engine.thrust *= 1.2
            
            if self.engine.turnrate < self.TURN_RATE * 2:
                self.engine.turnrate += 5
                
            
        
        
    
    def on_collide(self, other, point, normal):
        if isinstance(other, Collectable):
            self.collect(other)
        else:
            self.explode()
            self.THRUST_SOUND.pause()
            self.DEATH_SOUND.play()
            self.world.systems.game.player_died()


class Collectable(BlasteroidsEntity):
    """Something that is collectable (powerups, 1life, etc)"""

    COLLIDE_INTO_MASK = 0x2
    UNIT_CIRCLE = [(math.sin(math.radians(a)), math.cos(math.radians(a))) 
        for a in range(0, 360, 360/6)]

    HIT_SOUNDS = [
        load_sound('hit1c.wav'),
        load_sound('hit2c.wav'),
        load_sound('hit3c.wav'),
    ]
    POWERUP_SOUNDS = [
        load_sound('powerup1.wav'),
    ]
    TYPES = [
        "gun-powerup",
        "engine-powerup",
        #"gun-spread",
    ]

    def __init__(self, world, collect_type=None, position=None, parent_velocity=None):
        radius = 9
        if position is None:
            self.position.position = (
                random.choice([-1, 1]) * random.randint(50, window.width / 2), 
                random.choice([-1, 1]) * random.randint(50, window.height / 2))
        else:
            self.position.position = position
        if collect_type:
            self.taxonomy.subtype1 = collect_type
        else:
            self.taxonomy.subtype1 = random.choice(self.TYPES)
        if self.taxonomy.subtype1 == "gun-powerup":
            self.renderable.color = "#fa3"
        elif self.taxonomy.subtype1 == "electro-gun":        
            self.renderable.color = "#f0f"
            radius *= 1.5
        elif self.taxonomy.subtype1 == "engine-powerup":        
            self.renderable.color = "#3af"
            
        self.movement.velocity = (random.gauss(0, 20), random.gauss(0, 20))
        if parent_velocity is not None:
            self.movement.velocity += parent_velocity
        self.movement.rotation = random.gauss(0, 65)
        verts = [(x*radius, y*radius)
            for x, y in self.UNIT_CIRCLE]
        self.shape.verts = verts
        self.states.exploded = False
        self.states.created = self.world.time
        self.collision.radius = radius
        self.collision.from_mask = PlayerShip.COLLIDE_INTO_MASK | self.COLLIDE_INTO_MASK
        #self.collision.from_mask = PlayerShip.COLLIDE_INTO_MASK
        self.collision.into_mask = self.COLLIDE_INTO_MASK
        self.taxonomy.type1 = "collectable"


    def on_collide(self, other, point, normal):
        if isinstance(other, PlayerShip):
            random.choice(self.POWERUP_SOUNDS).play()
            self.delete()
        elif isinstance(other, Shot):
            random.choice(self.HIT_SOUNDS).play()
            ox, oy = self.movement.velocity
            dx, dy = other.movement.velocity
            ox += dx / 5
            oy += dy / 5
            self.movement.velocity = (ox,oy)
            other.movement.velocity = -other.movement.velocity/2
            dx, dy = other.movement.velocity
            other.position.position[0] += dx / 10.0
            other.position.position[1] += dy / 10.0
            dx += random.gauss(0,50)
            dy += random.gauss(0,50)
            other.movement.velocity = (dx,dy)
            self.explode()
            self.delete()
        elif isinstance(other, (Asteroid,Collectable)):
            if self.collision.radius == 0: return
            if other.collision.radius == 0: return
            nx, ny = normal
            ppx, ppy = point
            px1, py1 = self.position.position
            px2, py2 = other.position.position
            d = math.sqrt((px1-px2)**2 + (py1-py2)**2)
            d1 = d - self.collision.radius - other.collision.radius
            if d1 > 0: d1 += 1
            else: d1 = d / ( self.collision.radius + other.collision.radius )
            if d1 < 0.001: d1 = 0.001
            
            #self.position.position[0] += nx / d1 * 1.0
            #self.position.position[1] += ny / d1 * 1.0
            self.movement.velocity *= 0.5
            other.movement.velocity *= 0.5
            self.movement.velocity[0] += nx / d1 * 8.0
            self.movement.velocity[1] += ny / d1 * 8.0
            other.movement.velocity[0] += nx / d1 * -8.0
            other.movement.velocity[1] += ny / d1 * -8.0
            #other.position.position[0] += nx / d1 * -2.0
            #other.position.position[1] += ny / d1 * -2.0
            if self.world.time - self.states.created < 1: return
            if self.world.time - other.states.created < 1: return
            if isinstance(other, Collectable):
                if d < (self.collision.radius + other.collision.radius) / 1.5 + 8:
                    subtypes = set([])
                    subtypes.add(self.taxonomy.subtype1)
                    subtypes.add(other.taxonomy.subtype1)
                    if subtypes == set(["gun-powerup","engine-powerup"]):
                        self.delete()
                        other.delete()
                        other.collision.radius = 0
                        self.collision.radius = 0
                        ppos, pvel = mediumpoint(self,other)
                        Collectable(self.world,"electro-gun",ppos,pvel)


class Asteroid(BlasteroidsEntity):
    """Big floating space rock"""

    COLLIDE_INTO_MASK = 0x2
    HIT_SOUNDS = [
        load_sound('hit1.wav'),
        load_sound('hit2.wav'),
        load_sound('hit3.wav'),
    ]
    HIT_SMALL_SOUNDS = [
        load_sound('hit1s.wav'),
        load_sound('hit2s.wav'),
    ]

    UNIT_CIRCLE = [(math.sin(math.radians(a)), math.cos(math.radians(a))) 
        for a in range(0, 360, 18)]
    UNIT_CIRCLE_2 = [(math.sin(math.radians(a)), math.cos(math.radians(a))) 
        for a in range(0, 360, 18*2)]
    UNIT_CIRCLE_4 = [(math.sin(math.radians(a)), math.cos(math.radians(a))) 
        for a in range(0, 360, 18*4)]
    
    def __init__(self, world, radius=65, position=None, parent_velocity=None, points=25):
        if position is None:
            self.position.position = (
                random.choice([-1, 1]) * random.randint(50, window.width / 2), 
                random.choice([-1, 1]) * random.randint(50, window.height / 2))
        else:
            self.position.position = position
        self.states.exploded = False
        self.movement.velocity = (random.gauss(0, 700 / radius), random.gauss(0, 700 / radius))
        if parent_velocity is not None:
            self.movement.velocity += parent_velocity
        self.movement.rotation = random.gauss(0, 15)
        dx, dy = self.movement.velocity
        vel = math.sqrt(dx ** 2 + dy ** 2) 
        mod_vel = vel / math.sqrt(vel)
        dx /= mod_vel
        dy /= mod_vel
        self.states.created = self.world.time
        self.movement.velocity = (dx, dy)
        if radius < 15: circle = self.UNIT_CIRCLE_4
        elif radius < 30: circle = self.UNIT_CIRCLE_2
        else: circle = self.UNIT_CIRCLE
        
        verts = [(random.gauss(x*radius, radius / 7), random.gauss(y*radius, radius / 7))
            for x, y in circle]
        self.shape.verts = verts
        self.renderable.color = "#aaa"
        world.clock.schedule_once(self.begin_collide, 0.1)
        world.clock.schedule_interval(self.set_radius, 0.05, world, radius)
        self.collision.radius = 0.01
        self.collision.from_mask = 0
        self.collision.into_mask = 0
        self.award.points = points
        
    def set_radius(self,dt,world, radius):
        try:
            self.collision.radius += radius / 10.0
            if self.collision.radius > radius:
                self.collision.radius = radius
                world.clock.unschedule(self.set_radius)
        except AttributeError:
            world.clock.unschedule(self.set_radius)
            
            
        
        
    def begin_collide(self,dt):
        self.collision.from_mask = PlayerShip.COLLIDE_INTO_MASK | self.COLLIDE_INTO_MASK | Shot.COLLIDE_INTO_MASK
        self.collision.into_mask = self.COLLIDE_INTO_MASK
        

    def on_collide(self, other, point, normal):
        if isinstance(other, Asteroid):
            if self.collision.radius == 0: return
            if other.collision.radius == 0: return
            nx, ny = normal
            ppx, ppy = other.position.position
            px, py = self.position.position
            dx = ppx - px
            dy = ppy - py
            d2 = dx ** 2 + dy ** 2
            d = math.sqrt(d2)
            d1 = d - self.collision.radius - other.collision.radius 
            if d1 > 0: d1 += 1
            else: d1 = d / ( self.collision.radius + other.collision.radius )
            if d1 < 0.001: d1 = 0.001
            if (self.world.time - self.states.created < 1.5 or
               self.world.time - other.states.created < 1.5):
                d1 += 0.1
                d1 /= 5.0
                self.position.position[0] += nx / d1 * 0.4 
                self.position.position[1] += ny / d1 * 0.4 
                other.movement.velocity[0] -= nx / d1 * 3.0 
                other.movement.velocity[1] -= ny / d1 * 3.0 
                self.movement.velocity *= 0.5
                other.movement.velocity *= 0.5
            
            self.movement.velocity *= 0.8
            self.movement.velocity[0] += nx / d1 * 3.0 * other.collision.radius / self.collision.radius
            self.movement.velocity[1] += ny / d1 * 3.0 * other.collision.radius / self.collision.radius 
            if self.world.time - self.states.created < 1: return
            if self.world.time - other.states.created < 1: return
            
            px1, py1 = self.position.position
            px2, py2 = other.position.position
            """
            if d < 5:
                total_radius = math.sqrt(self.collision.radius**2 + other.collision.radius**2)
                if total_radius > 60: return
                px = (px1 + px2) / 2.0
                py = (py1 + py2) / 2.0
            
                dx1, dy1 = self.movement.velocity
                dx2, dy2 = other.movement.velocity
                dx = (dx1 + dx2) / 2.0
                dy = (dy1 + dy2) / 2.0
                if total_radius >= 4:
                    Asteroid(self.world, total_radius, (px,py), 
                        (dx,dy), self.award.points / 2)
                    self.collision.radius = 0
                    other.collision.radius = 0
                    self.delete()
                    other.delete()
            """
            return
        if isinstance(other, Shot):
            if other.collision.radius == 0: return
            if self.collision.radius == 0: return
            other.collision.radius = 0
            if self.collision.radius > 6:
                total_area = 3.1415927 * (self.collision.radius ** 2) 
                total_area -= 2
                count = 0
                for i in range(50):
                    if total_area < 4: break
                    min_area = 3.1415927 * 8 * 8
                    if min_area < total_area - 10:
                        chunk_size = random.gauss(self.collision.radius/8,self.collision.radius/8)
                        if chunk_size < 4: continue
                    else:
                        chunk_size = math.sqrt(total_area) / 3.1415927 - 0.5
                        if chunk_size < 4: break
                    chunk_area = 3.1415927 * (chunk_size**2)
            
                    if chunk_area > total_area: continue
                    total_area -= chunk_area
                    px, py = self.position.position
                    angle = random.uniform(0,360)
                    offset = random.uniform(2+chunk_size/2,self.collision.radius/2)
                    dx,dy = math.cos(angle) * offset, math.sin(angle) * offset
                    px += dx
                    py += dy
                    ppos = (px, py) 
                    Asteroid(self.world, chunk_size, ppos, 
                        self.movement.velocity + Vec2d(dx,dy) * 10, self.award.points * 2)
                    count += 1
                if random.gauss(0,total_area) > 8: count += 1
                if random.gauss(0,total_area) > 8: count += 1
                if random.gauss(0,total_area) > 8: count += 1
                for i in range(random.randint(0,int(count/4))):
                    Collectable(self.world,None,self.position.position,self.movement.velocity)
                random.choice(self.HIT_SOUNDS).play()
            else:
                random.choice(self.HIT_SMALL_SOUNDS).play()
            self.collision.radius = 0
                
            self.explode()
            self.delete()    


class Shot(grease.Entity):
    """Pew Pew!
    
    Args:
        `shooter` (Entity): entity that is shooting the shot. Used
        to determine the collision mask, position and velocity 
        so the shot doesn't hit the shooter.
        
        `angle` (float): Angle of the shot trajectory in degrees.
    """

    COLLIDE_INTO_MASK = 0x8
    SPEED = 50
    TIME_TO_LIVE = 2.70 # seconds
    
    def __init__(self, world, shooter, angle, offset2sz = 0, spread = 0):
        offset = geometry.Vec2d(0, shooter.collision.radius)
        offset.rotate(angle)
        offset2 = geometry.Vec2d(0, shooter.collision.radius/2.0)
        offset2.rotate(angle+90)
        spreadn = max(spread, 1.0)
        vertical_sep =  offset2sz/spreadn
        self.collision.radius = 2.0
        
        self.states.exploded = False
        
        self.position.position = shooter.position.position + offset * (2.5-abs(offset2sz/16)) + offset2 * vertical_sep 
        self.movement.velocity = (
            offset.normalized() * self.SPEED + shooter.movement.velocity)
        self.movement.accel = offset2 * -offset2sz / 2.0 + offset.normalized() * (40.0 - abs(offset2sz)*5) / spreadn
        self.shape.verts = [(0, 1.5), (1.5, -1.5), (-1.5, -1.5)]
        self.collision.from_mask = Asteroid.COLLIDE_INTO_MASK
        self.collision.into_mask = Shot.COLLIDE_INTO_MASK
        self.renderable.color = "#ffc"
        if "--cheat1" in sys.argv:
            world.clock.schedule_once(self.expire, self.TIME_TO_LIVE*2 / (1+abs(offset2sz) / 50.0) )
        else:
            world.clock.schedule_once(self.expire, self.TIME_TO_LIVE / (1+abs(offset2sz) / 50.0))

    def on_collide(self, other, point, normal):
        if isinstance(other, Asteroid):
            self.world.systems.game.award_points(other)
            self.delete()
        else:
            print other.__class__.__name__
    
    def expire(self, dt):
        self.delete()

## Define game systems ##

class PositionWrapper(grease.System):
    """Wrap positions around when they go off the edge of the window"""

    def __init__(self):
        self.half_width = window.width / 2
        self.half_height = window.height / 2

    def step(self, dt):
        for entity in self.world[...].collision.aabb.right < -self.half_width:
            entity.position.position.x += window.width + entity.collision.aabb.width
        for entity in self.world[...].collision.aabb.left > self.half_width:
            entity.position.position.x -= window.width + entity.collision.aabb.width
        for entity in self.world[...].collision.aabb.top < -self.half_height:
            entity.position.position.y += window.height + entity.collision.aabb.height 
        for entity in self.world[...].collision.aabb.bottom > self.half_height:
            entity.position.position.y -= window.height + entity.collision.aabb.height


class Gun(grease.System):
    """Fires Shot entities"""

    def step(self, dt):
        for entity in self.world[...].gun.firing == True:
            if self.world.time >= entity.gun.last_fire_time + entity.gun.cool_down:
                precision_angle = random.gauss(0,entity.gun.spread/float(entity.gun.precision))
                shots = int((self.world.time - entity.gun.last_fire_time) / entity.gun.cool_down)
                if shots > entity.gun.shots:
                    shots = entity.gun.shots 
                if shots < 1: shots = 1
                spread = entity.gun.spread
                if spread * (shots) >= 360:
                    spread = 360 / (shots)
                    
                for i in range(shots):
                    Shot(self.world, entity, 
                        entity.position.angle
                        + precision_angle
                        + i * spread - (shots-1) * spread/2, 
                        (i - (shots-1) / 2.0), spread
                        )
                    
                if entity.gun.sound is not None:
                    entity.gun.sound.play()
                entity.gun.last_fire_time = self.world.time


class Sweeper(grease.System):
    """Clears out space debris"""

    SWEEP_TIME = 1.0

    def step(self, dt):
        fade = dt / self.SWEEP_TIME
        for entity in tuple(self.world[Debris].entities):
            color = entity.renderable.color
            if color.a > 0.2:
                color.a = max(color.a - fade, 0)
            else:
                entity.delete()


class GameSystem(KeyControls):
    """Main game logic system

    This subclass KeyControls so that the controls can be bound
    directly to the game logic here
    """

    CHIME_SOUNDS = [
        load_sound('chime1.wav'), 
        load_sound('chime2.wav'),
    ]
    MAX_CHIME_TIME = 2.0
    MIN_CHIME_TIME = 0.6

    def set_world(self, world):
        KeyControls.set_world(self, world)
        self.level = 0       
        self.lives = 3
        self.score = 0
        self.player_ship = PlayerShip(self.world)
        self.start_level()
    
    def start_level(self):
        self.level += 1
        for i in range(self.level * 2 + 1):
            Asteroid(self.world)
            if "--level3" in sys.argv:       
                Asteroid(self.world)
                Asteroid(self.world)
                Asteroid(self.world)
                 
        self.chime_time = self.MAX_CHIME_TIME
        self.chimes = itertools.cycle(self.CHIME_SOUNDS)
        if self.level == 1:
            self.chime()
        else:
            self.player_ship.set_invincible(True)
    
    def chime(self, dt=0):
        """Play tension building chime sounds"""
        if self.lives:
            self.chimes.next().play()
            self.chime_time = max(self.chime_time - dt * 0.01, self.MIN_CHIME_TIME)
            if not self.world[Asteroid].entities:
                self.start_level()
            self.world.clock.schedule_once(self.chime, self.chime_time)

    def award_points(self, entity):
        """Get points for destroying stuff"""
        if entity.award:
            self.score += entity.award.points
    
    def player_died(self):
        self.lives -= 1
        self.player_ship.delete()
        self.world.clock.schedule_once(self.player_respawn, 3.0)
        
    def player_respawn(self, dt=None):
        """Rise to fly again, with temporary invincibility"""
        if self.lives:
            self.player_ship = PlayerShip(self.world, invincible=True)
        if self.world.is_multiplayer:
            # Switch to the next player
            if self.lives:
                window.current_mode.activate_next()
            else:
                window.current_mode.remove_submode()
    
    @KeyControls.key_press(key.LEFT)
    @KeyControls.key_press(key.A)
    def start_turn_left(self):
        if self.player_ship.exists:
            self.player_ship.turn(-1)

    @KeyControls.key_release(key.LEFT)
    @KeyControls.key_release(key.A)
    def stop_turn_left(self):
        if self.player_ship.exists and self.player_ship.movement.rotation < 0:
            self.player_ship.turn(0)

    @KeyControls.key_press(key.RIGHT)
    @KeyControls.key_press(key.D)
    def start_turn_right(self):
        if self.player_ship.exists:
            self.player_ship.turn(1)

    @KeyControls.key_release(key.RIGHT)
    @KeyControls.key_release(key.D)
    def stop_turn_right(self):
        if self.player_ship.exists and self.player_ship.movement.rotation > 0:
            self.player_ship.turn(0)
    
    @KeyControls.key_hold(key.UP)
    @KeyControls.key_hold(key.W)
    def thrust(self, dt):
        if self.player_ship.exists:
            self.player_ship.thrust_on()
        
    @KeyControls.key_release(key.DOWN)
    @KeyControls.key_release(key.S)
    @KeyControls.key_release(key.UP)
    @KeyControls.key_release(key.W)
    def stop_thrust(self):
        if self.player_ship.exists:
            self.player_ship.thrust_off()
            
    @KeyControls.key_hold(key.DOWN)
    @KeyControls.key_hold(key.S)
    def brake(self, dt):
        if self.player_ship.exists:
            self.player_ship.brake()
            
    @KeyControls.key_press(key.SPACE)
    def start_firing(self):
        if self.player_ship.exists:
            self.player_ship.gun.firing = True

    @KeyControls.key_release(key.SPACE)
    def stop_firing(self):
        if self.player_ship.exists:
            self.player_ship.gun.firing = False
    
    @KeyControls.key_press(key.P)
    def pause(self):
        self.world.running = not self.world.running    

    def on_key_press(self, key, modifiers):
        """Start the world with any key if paused"""
        if not self.world.running:
            self.world.running = True
        KeyControls.on_key_press(self, key, modifiers)



class Hud(grease.Renderer):
    """Heads-up display renderer"""
    
    def set_world(self, world):
        self.world = world
        self.last_lives = 0
        self.last_score = None
        self.game_over_label = None
        self.paused_label = None
        self.create_lives_entities()
    
    def create_lives_entities(self):
        """Create entities to represent the remaining lives"""
        self.lives = []
        verts = geometry.Vec2dArray(PlayerShip.SHAPE_VERTS[3:])
        left = -window.width // 2 + 25
        top = window.height // 2 - 25
        for i in range(20):
            entity = grease.Entity(self.world)
            entity.shape.verts = verts.transform(scale=0.75)
            entity.position.position = (i * 20 + left, top)
            self.lives.append((i, entity))

    def draw(self):
        game = self.world.systems.game
        if self.last_lives != game.lives:
            for i, entity in self.lives:
                if game.lives > i:
                    entity.renderable.color = PlayerShip.COLOR
                else:
                    entity.renderable.color = (0,0,0,0)
            self.last_lives = game.lives
        if self.last_score != game.score:
            self.score_label = pyglet.text.Label(
                str(game.score),
                color=(180, 180, 255, 255),
                font_name='Vector Battle', font_size=14, bold=True,
                x=window.width // 2 - 25, y=window.height // 2 - 25, 
                anchor_x='right', anchor_y='center')
            self.last_score = game.score
        self.score_label.draw()
        if game.lives == 0:
            if self.game_over_label is None:
                self.game_over_label = pyglet.text.Label(
                    "GAME OVER",
                    font_name='Vector Battle', font_size=36, bold=True,
                    color=(255, 0, 0, 255),
                    x = 0, y = 0, anchor_x='center', anchor_y='center')
            self.game_over_label.draw()
        if not self.world.running:
            if self.paused_label is None:
                self.player_label = pyglet.text.Label(
                    self.world.player_name,
                    color=(150, 150, 255, 255),
                    font_name='Vector Battle', font_size=18, bold=True,
                    x = 0, y = 20, anchor_x='center', anchor_y='bottom')
                self.paused_label = pyglet.text.Label(
                    "press a key to begin",
                    color=(150, 150, 255, 255),
                    font_name='Vector Battle', font_size=16, bold=True,
                    x = 0, y = -20, anchor_x='center', anchor_y='top')
            self.player_label.draw()
            self.paused_label.draw()


class TitleScreenControls(KeyControls):
    """Title screen key event handler system"""

    @KeyControls.key_press(key._1)
    def start_single_player(self):
        window.push_mode(Game())
    
    @KeyControls.key_press(key._2)
    def start_two_player(self):
        window.push_mode(mode.Multi(
            Game('Player One'), 
            Game('Player Two')))


class BaseWorld(grease.World):

    def configure(self):
        """Configure the game world's components, systems and renderers"""
        self.components.position = component.Position()
        self.components.movement = component.Movement()
        self.components.shape = component.Shape()
        self.components.renderable = component.Renderable()
        self.components.collision = component.Collision()
        self.components.gun = component.Component(
            firing=bool, 
            last_fire_time=float, 
            cool_down=float, 
            spread=float, 
            precision=float, 
            shots=int,
            sound=object)
        self.components.taxonomy = component.Component(
            type1=str,
            subtype1=str)
        self.components.states = component.Component(
            invincible=bool,
            exploded=bool,
            created=float
            )
        self.components.engine = component.Component(
            thrust=float,
            turnrate=float,
            )
        self.components.award = component.Component(points=int)

        self.systems.movement = controller.EulerMovement()
        self.systems.collision = collision.Circular(
            handlers=[collision.dispatch_events])
        self.systems.sweeper = Sweeper()
        self.systems.gun = Gun()
        self.systems.wrapper = PositionWrapper()

        self.renderers.camera = renderer.Camera(
            position=(window.width / 2, window.height / 2))
            
        self.renderers.vector = renderer.Vector(line_width=1.5)


class TitleScreen(BaseWorld):
    """Game title screen world and mode"""
    
    def configure(self):
        BaseWorld.configure(self)
        self.renderers.title = pyglet.text.Label(
            "Blasteroids",
            color=(150, 150, 255, 255),
            font_name='Vector Battle', font_size=32, bold=True,
            x=0, y=50, anchor_x='center', anchor_y='bottom')
        self.renderers.description = pyglet.text.Label(
            "A demo for the Grease game engine",
            color=(150, 150, 255, 255),
            font_name='Vector Battle', font_size=16, bold=True,
            x=0, y=20, anchor_x='center', anchor_y='top')
        self.renderers.one_player = pyglet.text.Label(
            "Press 1 for one player",
            color=(150, 150, 255, 255),
            font_name='Vector Battle', font_size=16, bold=True,
            x=0, y=-100, anchor_x='center', anchor_y='top')
        self.renderers.two_player = pyglet.text.Label(
            "Press 2 for two players",
            color=(150, 150, 255, 255),
            font_name='Vector Battle', font_size=16, bold=True,
            x=0, y=-130, anchor_x='center', anchor_y='top')

        self.systems.controls = TitleScreenControls()
        for i in range(15):
            Asteroid(self, radius=random.randint(12, 45))


class Game(BaseWorld):
    """Main game world and mode"""

    def __init__(self, player_name=""):
        BaseWorld.__init__(self, 10)
        self.player_name = player_name
        self.is_multiplayer = self.player_name != ""
        self.clock.unschedule(self.step)
        self.clock.schedule(self.step)


    def configure(self):
        BaseWorld.configure(self)
        self.systems.game = GameSystem()
        self.renderers.hud = Hud()
    
    def activate(self, manager):
        """Start paused in multiplayer"""
        grease.World.activate(self, manager)
        if self.is_multiplayer:
            self.running = False


def main():
    """Initialize and run the game"""
    global window
    window = mode.ManagerWindow(vsync=False)
    window.set_vsync(False)
    window.push_mode(TitleScreen(20))
    pyglet.app.run()

if __name__ == '__main__':
    main()


# vim: ai ts=4 sts=4 et sw=4

