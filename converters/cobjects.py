#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sixtracklib as st

def calc_cbuffer_params_for_single_particle_buffer(
    num_particle_sets, max_num_particles_per_set, conf=dict() ):
    slot_size = st.CBufferView.DEFAULT_SLOT_SIZE
    assert num_particle_sets >= 0
    assert max_num_particles_per_set >= 0
    num_particles = num_particle_sets * max_num_particles_per_set
    n_objects  = num_particles
    n_pointers = num_particles * st.st_SingleParticle.COBJ_NUM_DATAPTRS
    n_slots    = num_particles * st.st_SingleParticle.COBJ_REQUIRED_NUM_SLOTS(
        slot_size )
    return n_slots, n_objects, n_pointers

def create_single_particle_cbuffer(
    num_particle_sets, max_num_particles_per_set, conf=dict() ):
    slot_size = st.CBufferView.DEFAULT_SLOT_SIZE
    assert num_particle_sets >= 0
    assert max_num_particles_per_set >= 0
    n_slots, n_objs, n_ptrs = calc_cbuffer_params_for_single_particle_buffer(
        num_particle_sets, max_num_particles_per_set, conf=conf )
    num_single_particles = num_particle_sets * max_num_particles_per_set
    cbuffer = st.CBuffer( n_slots, n_objs, n_ptrs, 0, slot_size )
    for ii in range( 0, num_single_particles ):
        p = st.st_SingleParticle( cbuffer )
    assert n_objs == cbuffer.num_objects
    return cbuffer

def calc_cbuffer_params_for_particles_buffer(
    num_particle_sets, max_num_particles_per_set, conf=dict() ):
    slot_size = st.CBufferView.DEFAULT_SLOT_SIZE
    assert num_particle_sets >= 0
    assert max_num_particles_per_set >= 0
    n_objects  = num_particle_sets
    n_pointers = num_particle_sets * st.st_Particles.COBJ_NUM_DATAPTRS
    n_slots    = num_particle_sets * st.st_Particles.COBJ_REQUIRED_NUM_SLOTS(
        max_num_particles_per_set, slot_size )
    return n_slots, n_objects, n_pointers

def create_particle_set_cbuffer(
    num_particle_sets, max_num_particles_per_set, conf=dict() ):
    slot_size = st.CBufferView.DEFAULT_SLOT_SIZE
    assert num_particle_sets >= 0
    assert max_num_particles_per_set >= 0
    n_slots, n_objs, n_ptrs = calc_cbuffer_params_for_particles_buffer(
        num_particle_sets, max_num_particles_per_set, conf=conf )
    cbuffer = st.CBuffer( n_slots, n_objs, n_ptrs, 0, slot_size )
    for ii in range( 0, num_particle_sets ):
        pset = st.st_Particles( cbuffer, max_num_particles_per_set )
    assert n_objs == cbuffer.num_objects
    return cbuffer
