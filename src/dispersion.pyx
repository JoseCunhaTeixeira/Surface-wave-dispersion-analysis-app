# distutils: language=c++
# distutils: libraries=fftw3

from libcpp.vector cimport vector
cimport cython




cdef class dispersionResult:
    cdef vector[double] freq
    cdef vector[double] trial_vals
    cdef vector[vector[double]] pnorm


cdef extern from "dispersion_src.cpp":    
    cdef struct output:
        vector[double] xs, ys
        vector[vector[double]] arr

    output FK_src(vector[vector[double]] XT, double dt, vector[double] offsets, double f_min, double f_max, double k_min, double k_max)
    
    output phase_shift_src(vector[vector[double]] XT, double dt, vector[double] offsets, double f_min, double f_max, double v_min, double v_max, double dv)


def FK(XT, dt, offsets, f_min, f_max, k_min, k_max):
    cdef output result = FK_src(XT, dt, offsets, f_min, f_max, k_min, k_max)
    return (result.xs, result.ys, result.arr)


def phase_shift(XT, dt, offsets, f_min, f_max, v_min, v_max, dv):
    cdef output result = phase_shift_src(XT, dt, offsets, f_min, f_max, v_min, v_max, dv)
    return (result.xs, result.ys, result.arr)
