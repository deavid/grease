cimport cython
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
    cdef void glRotatef(GLfloat angle, GLfloat x, GLfloat y, GLfloat z)
    cdef void glCallList(GLenum listidx)
    
import numpy as np
cimport numpy as np

np.import_array()


@cython.boundscheck(False) # turn of bounds-checking for entire function
cdef int draw_list(GLenum listidx, float x, float y, float z, float w, float sz):
    glLoadIdentity()
    glTranslatef(x, y, z)
    glScalef(sz,sz,1)
    glRotatef(w,0,0,1)
    glCallList(listidx)
    return 0


cpdef int draw_array(GLenum listidx, object Xl, object Yl, object Zl, object Wl, object SZl) except -1:
    cdef object it = np.broadcast(Xl,Yl,Zl,Wl,SZl)
    cdef double x, y, z, w, sz
    while np.PyArray_MultiIter_NOTDONE(it):
        x = (<double*>np.PyArray_MultiIter_DATA(it, 0))[0]
        y = (<double*>np.PyArray_MultiIter_DATA(it, 1))[0]
        z = (<double*>np.PyArray_MultiIter_DATA(it, 2))[0]
        w = (<double*>np.PyArray_MultiIter_DATA(it, 3))[0]
        sz = (<double*>np.PyArray_MultiIter_DATA(it, 4))[0]
        draw_list(listidx,x,y,z,w,sz)
        np.PyArray_MultiIter_NEXT(it)
        

@cython.boundscheck(False) # turn of bounds-checking for entire function
cpdef int draw_array2(GLenum listidx, 
        np.ndarray[np.float_t] Xl, np.ndarray[np.float_t] Yl, 
        np.ndarray[np.float_t] Zl, np.ndarray[np.float_t] Wl, 
        np.ndarray[np.float_t] Sl) except -1:
        
    cdef double *X = <double *>Xl.data
    cdef double *Y = <double *>Yl.data
    cdef double *Z = <double *>Zl.data
    cdef double *W = <double *>Wl.data
    cdef double *S = <double *>Sl.data
    cdef double x,y,z,w,s
    cdef int i, array_sz
    array_sz = Xl.shape[0]
    for i in range(array_sz):
        x,y,z,w,s = X[i],Y[i],Z[i],W[i],S[i]
        draw_list(listidx,x,y,z,w,s)
        #glLoadIdentity()
        #glTranslatef(x, y, z)
        #glScalef(s,s,1)
        #glRotatef(w,0,0,1)
        #glCallList(listidx)
    return 0
        
@cython.boundscheck(False) # turn of bounds-checking for entire function
cpdef int draw_array3(GLenum listidx, int entitytype,
        np.ndarray[np.float_t] Tl,
        np.ndarray[np.float_t] Xl, np.ndarray[np.float_t] Yl, 
        np.ndarray[np.float_t] Zl, np.ndarray[np.float_t] Wl, 
        np.ndarray[np.float_t] Sl):
    cdef double *T = <double *>Tl.data
    cdef double *X = <double *>Xl.data
    cdef double *Y = <double *>Yl.data
    cdef double *Z = <double *>Zl.data
    cdef double *W = <double *>Wl.data
    cdef double *S = <double *>Sl.data
    cdef double x,y,z,w,s
    cdef int t, n
    cdef int i, array_sz
    cdef int d_size = sizeof(double)
    cdef int tw = Tl.strides[0] / d_size
    cdef int xw = Xl.strides[0] / d_size
    cdef int yw = Yl.strides[0] / d_size
    cdef int zw = Zl.strides[0] / d_size
    cdef int ww = Wl.strides[0] / d_size
    cdef int sw = Sl.strides[0] / d_size
    array_sz = Xl.shape[0]
    n = 0
    for i in range(array_sz):
        t = <int> T[i*tw]
        if t == entitytype:
            n += 1
            x,y,z,w,s = X[i*xw],Y[i*yw],Z[i*zw],W[i*ww],S[i*sw]
            #if n < 5: print " - x:%.3f, y:%.3f, z:%.3f, w:%.3f, s:%.3f " % (x,y,z,w,s)
            draw_list(listidx,x,y,z,w,s)
    return n
        

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