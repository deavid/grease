cdef extern from "GL/gl.h":
    ctypedef int           GLint
    ctypedef unsigned int  GLenum
    ctypedef float GLfloat
    int GL_POINTS
    cdef void glBegin(GLenum mode)
    cdef void glEnd()
    cdef void glVertex3f(GLfloat x, GLfloat y, GLfloat z)
    cdef void glLoadIdentity()
    cdef void glTranslatef(GLfloat x, GLfloat y, GLfloat z)
    cdef void glScalef(GLfloat x, GLfloat y, GLfloat z)
    cdef void glCallList(GLenum listidx)
    
import numpy as np
cimport numpy as np

np.import_array()



cpdef int draw_list(GLenum listidx, float x, float y, float sz) except -1:
    glLoadIdentity()
    glTranslatef(x, y, 0)
    glScalef(sz,sz,1)
    glCallList(listidx)
    return 0


cpdef int draw_array(GLenum listidx, object Xl, object Yl, object SZl) except -1:
    cdef object it = np.broadcast(Xl,Yl,SZl)
    cdef double x, y, sz
    while np.PyArray_MultiIter_NOTDONE(it):
        x = (<double*>np.PyArray_MultiIter_DATA(it, 0))[0]
        y = (<double*>np.PyArray_MultiIter_DATA(it, 1))[0]
        sz = (<double*>np.PyArray_MultiIter_DATA(it, 2))[0]
        glLoadIdentity()
        glTranslatef(x, y, 0)
        glScalef(sz,sz,1)
        glCallList(listidx)
        np.PyArray_MultiIter_NEXT(it)
        

def numpylist(object n, object mylist):
    cdef object it = np.broadcast(n,mylist)
    cdef int nvalue
    cdef double lvalue
    cdef double lsum = 0
    while np.PyArray_MultiIter_NOTDONE(it):
        nvalue = (<int*>np.PyArray_MultiIter_DATA(it, 0))[0]
        lvalue = (<double*>np.PyArray_MultiIter_DATA(it, 1))[0]
        lsum += lvalue
        np.PyArray_MultiIter_NEXT(it)
    return lsum