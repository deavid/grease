#!/usr/bin/env python

# $Id$

import os
import sys
import shutil
from distutils.core import setup, Extension
import platform

from Cython.Distutils import build_ext as cython_build_ext
srcdir = os.path.dirname(__file__)
system_platform = platform.system().lower()
OPENGL_LIBNAME = None
GLUT_LIBNAME = None
GLU_LIBNAME = None

if system_platform == "linux":
    OPENGL_LIBNAME = "GL"
    GLUT_LIBNAME = "GLU"
    GLU_LIBNAME = "glut"
elif system_platform == "darwin":
    OPENGL_LIBNAME = "GL"
    GLUT_LIBNAME = "GLU"
    GLU_LIBNAME = "glut"
elif system_platform == "windows":
    OPENGL_LIBNAME = "opengl32"  # what is glaux ?
    GLUT_LIBNAME = "glut"
    GLU_LIBNAME = "glu32"
else:
    print "PANIC: Unknown platform %s." % repr(system_platform)
    raise ValueError, "Platform unknown!"
    


    

def read(fname):
    return open(os.path.join(srcdir, fname)).read()

# Copy the blasteroids example scripts to the tutorial dir
# Ideally they would just live there, but inflexibility in
# distutils wrt packaging data makes this necessary
#for i in range(1, 4):
#    shutil.copyfile(
#        os.path.join(srcdir, 'examples', 'blasteroids%s.py' % i),
#        os.path.join(srcdir, 'doc', 'tutorial', 'blasteroids%s.py' % i))

ext_modules = []

ext_modules.append(Extension("grease.cython.test.test1", 
    ["grease/cython/test/test1.pyx"]))

ext_modules.append(Extension("grease.cython.renderer.vector", 
    ["grease/cython/renderer/vector.pyx"]))
ext_modules.append(Extension("grease.cython.renderer.camera", 
    ["grease/cython/renderer/camera.pyx"]))


ext_modules.append(Extension("grease.cython.controller.integrator", 
    ["grease/cython/controller/integrator.pyx"]))

ext_modules.append(Extension("grease.cython.collision", 
    ["grease/cython/collision.pyx"]))

ext_modules.append(Extension("grease.cython.arraygl", 
    ["grease/cython/arraygl.pyx"], libraries = [OPENGL_LIBNAME]))

setup(
    name='grease',
    version='0.3', # *** REMEMBER TO UPDATE __init__.py ***
    description='Grease: The highly extensible game engine framework for Python',
    long_description=read('README.txt'),
    author='Casey Duncan',
    author_email='casey.duncan@gmail.com',
    cmdclass = { 'build_ext' : cython_build_ext },
    ext_modules = ext_modules,
    # url='',
    license='MIT',
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Topic :: Multimedia :: Graphics',
        'Topic :: Games/Entertainment',
        'Topic :: Software Development :: Libraries :: Application Frameworks',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 2.6',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
    ],

    package_dir={'grease': 'grease', 
                 'grease.controller': 'grease/controller',
                 'grease.component': 'grease/component',
                 'grease.renderer': 'grease/renderer',
                 'grease.test': 'test',
                 'grease.examples': 'examples'},
    package_data={'grease.examples': ['font/*', 'sfx/*']},
    packages=['grease', 
              'grease.cython', 
              'grease.cython.renderer', 
              'grease.cython.controller', 
              'grease.controller', 
              'grease.component', 
              'grease.renderer', 
              'grease.test',
              'grease.examples'],
)
