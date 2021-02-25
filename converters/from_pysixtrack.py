#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import os
import numpy as np

# Tracking is done using pysixtrack
import pysixtrack as pysix

# sixtracklib provides the CObject based beam elements and particle types
import sixtracklib as st

from .pysixtrack_to_cobjects import pysix_line_to_cbuffer
from .pysixtrack_to_cobjects import calc_cbuffer_params_for_pysix_line
from .demotrack import float_to_bytes
from .cobjects import create_particle_set_cbuffer
from .cobjects import create_single_particle_cbuffer

from .pysixtrack_to_cobjects import pysix_particle_to_pset
from .pysixtrack_to_cobjects import pysix_particle_to_single_particle

def generate_lattice_data( input_path, output_path, conf=dict() ):
    print( "**** Generating Lattice Data From pysixtrack Input:" )
    print( "**** -> Reading sixtrack input data from:\r\n" +
          f"****    {input_path}" )
    slot_size = st.CBufferView.DEFAULT_SLOT_SIZE
    path_in_line = os.path.join( input_path, "pysixtrack_line.pickle" )
    with open( path_in_line, "rb" ) as f_in:
        line = pickle.load( f_in )

    n_slots, n_objs, n_ptrs = calc_cbuffer_params_for_pysix_line(
            line, slot_size=slot_size, conf=conf )
    cbuffer = st.CBuffer( n_slots, n_objs, n_ptrs, 0, slot_size )
    pysix_line_to_cbuffer( line, cbuffer )
    path_to_lattice = os.path.join( output_path, "cobj_lattice.bin" )

    if  0 == cbuffer.tofile_normalised( path_to_lattice,
            conf.get( "cbuffer_norm_base_addr", 4096) ):
        print( "**** -> Generated cobjects lattice data at:\r\n" +
              f"****    {path_to_lattice}" )
    else:
        raise RuntimeError( "Problem during creation of lattice data" )

    if conf.get( 'always_use_drift_exact', False ):
        for ii in range( 0, len( line.elements ) ):
            if isinstance( line.elements[ ii ], pysix.elements.Drift ) and \
                not isinstance( line.elements[ ii ], pysix.elements.DriftExact ):
                new_elem = pysix.elements.DriftExact(
                    length=line.elements[ ii ].length )
                line.elements[ ii ] = new_elem
                assert isinstance( line.elements[ ii ], pysix.elements.DriftExact )

        for elem in line.elements:
            assert not isinstance( elem, pysix.elements.Drift ) or \
                   isinstance( elem, pysix.elements.DriftExact )

    path_to_pysix_lattice = os.path.join( output_path, "pysixtrack_lattice.pickle" )

    try:
        pickle.dump( line.elements, open( path_to_pysix_lattice, "wb" ) )
        print( "**** -> Generated pysixtrack lattice as python pickle:\r\n" +
              f"****    {path_to_pysix_lattice}" )
    except:
        raise RuntimeError(
            "Unable to generate pysixtrack lattice data" )

    if conf.get( 'make_demotrack_data', False ) and \
        st.Demotrack_enabled() and st.Demotrack_belems_can_convert( cbuffer ):
        dt_lattice = st.Demotrack_belems_convert( cbuffer )
        if isinstance( dt_lattice, np.ndarray ) and \
            st.Demotrack_belems_num_stored_objects( dt_lattice ) == \
            cbuffer.num_objects:
            path_dt_lattice = os.path.join( output_path, "demotrack_lattice.bin" )
            with open( path_dt_lattice, "wb" ) as fp_out:
                fp_out.write( float_to_bytes( len( dt_lattice ) ) )
                fp_out.write( dt_lattice.tobytes() )
                print( "**** -> Generated demotrack lattice as flat array:\r\n" +
                      f"****    {path_dt_lattice}" )
    return

def generate_particle_data_initial( input_path, output_path, conf=dict() ):
    path_in_particles = os.path.join(
        input_path, "pysixtrack_initial_particles.pickle" )
    with open( path_in_particles, "rb" ) as f_in:
        initial_p_pysix = pickle.load( f_in )
        print( "**** -> Read input data from:\r\n" +
               f"****    {path_in_particles}" )

    num_part = len( initial_p_pysix )
    NORM_ADDR = conf.get( "cbuffer_norm_base_addr", 4096 )
    MAKE_DEMOTRACK = conf.get( "make_demotrack_data", False )
    MAKE_DEMOTRACK &= st.Demotrack_enabled()
    init_particle_idx = 0
    start_at_element  = 0

    initial_p_buffer = create_single_particle_cbuffer( 1, num_part, conf )
    initial_pset_buffer = create_particle_set_cbuffer( 1, num_part, conf )
    path_initial_pset = os.path.join( output_path, "cobj_initial_particles.bin" )

    for jj, in_p in enumerate( initial_p_pysix ):
        assert jj < num_part
        assert isinstance( in_p, pysix.Particles )
        in_p.elemid = start_at_element
        in_p.turn = 0
        in_p.state = 1
        in_p.partid = jj

        p = st.st_SingleParticle.GET( initial_p_buffer, jj )
        pysix_particle_to_single_particle( in_p, p )

        pset = st.st_Particles.GET( initial_pset_buffer, 0 )
        pysix_particle_to_pset( in_p, pset, jj )

    path_init_pset = os.path.join( output_path, "cobj_initial_particles.bin" )
    if  0 == initial_pset_buffer.tofile_normalised( path_init_pset, NORM_ADDR ):
        print( "**** -> Generated initial particle set data at:\r\n"
              f"****    {path_init_pset}" )
    else:
        raise RuntimeError( "Unable to generate initial particle set data" )

    path_init_p = os.path.join( output_path, "cobj_initial_single_particles.bin" )
    if  0 == initial_p_buffer.tofile_normalised( path_init_p, NORM_ADDR ):
        print( "**** -> Generated initial single particle data at:\r\n" +
              f"****    {path_init_p}" )
    else:
        raise RuntimeError( "Unable to generate initial single particle set data" )

    path_init_pysix = os.path.join(
        output_path, "pysixtrack_initial_particles.pickle" )

    with open( path_init_pysix, "wb" ) as f_out:
        pickle.dump( initial_p_pysix, f_out )
        print( "**** -> Generated initial pysixtrack particle data at:\r\n" +
               f"****    {path_init_pysix}" )

    if MAKE_DEMOTRACK:
        dt_particles_buffer = st.st_DemotrackParticle.CREATE_ARRAY( num_part, True )
        assert isinstance( dt_particles_buffer, np.ndarray )
        assert len( dt_particles_buffer ) == num_part
        pset = st.st_Particles.GET( initial_pset_buffer, 0 )
        assert pset.num_particles == num_part
        dt_p = st.st_DemotrackParticle()
        for ii in range( 0, num_part ):
            dt_p.from_cobjects( pset, ii )
            dt_p.to_array( dt_particles_buffer, ii )
            dt_p.clear()
        path_init_dt = os.path.join(
            output_path, "demotrack_initial_particles.pickle" )
        with open( path_init_dt, "wb" ) as f_out:
            f_out.write( float_to_bytes( num_part ) )
            f_out.write( dt_particles_buffer.tobytes() )
            print( "**** -> Generated initial demotrack particle data at:\r\n" +
                  f"****    {path_init_dt}" )
    return

def generate_particle_data_elem_by_elem( input_path, output_path, conf=dict() ):
    path_in_particles = os.path.join(
        output_path, "pysixtrack_initial_particles.pickle" )
    with open( path_in_particles, "rb" ) as f_in:
        initial_p_pysix = pickle.load( f_in )
        print( "**** -> Read input data from:\r\n" +
               f"****    {path_in_particles}" )

    path_in_line = os.path.join( output_path, "pysixtrack_lattice.pickle" )
    with open( path_in_line, "rb" ) as f_in:
        line = pickle.load( f_in )

    num_belem = len( line )
    num_part = len( initial_p_pysix )
    start_at_element = 0

    NORM_ADDR = conf.get( "cbuffer_norm_base_addr", 4096 )
    MAKE_DEMOTRACK = conf.get( "make_demotrack_data", False )
    MAKE_DEMOTRACK &= st.Demotrack_enabled()

    pset_buffer = create_particle_set_cbuffer(
        num_belem + 1, num_part, conf )
    assert pset_buffer.num_objects == num_belem + 1

    if MAKE_DEMOTRACK:
        dt_p = st.st_DemotrackParticle()
        dt_pset_buffer = st.st_DemotrackParticle.CREATE_ARRAY(
            ( num_belem + 1 ) * num_part, True )
        assert isinstance( dt_pset_buffer, np.ndarray )
        assert len( dt_pset_buffer ) <= ( num_belem + 1 ) * num_part

    for ii, in_p in enumerate( initial_p_pysix ):
        assert isinstance( in_p, pysix.Particles )
        assert in_p.elemid == start_at_element
        assert in_p.partid == ii
        assert in_p.turn == 0
        assert in_p.state == 1

        print( f"****    Info :: particle {ii:6d}/{num_part - 1:6d}" )
        for jj, elem in enumerate( line ):
            pset = st.st_Particles.GET( pset_buffer, jj )
            assert pset.num_particles == num_part
            assert in_p.elemid == jj
            assert in_p.partid == ii
            assert in_p.turn == 0
            assert in_p.state == 1
            if conf.get( 'always_use_drift_exact', False ):
                assert not isinstance( elem, pysix.elements.Drift ) or \
                       isinstance( elem, pysix.elements.DriftExact )
            pysix_particle_to_pset( in_p, pset, ii, conf=conf )
            if MAKE_DEMOTRACK:
                dt_p.clear()
                dt_p.from_cobjects( pset, ii )
                kk = jj * num_part + ii
                assert kk < len( dt_pset_buffer )
                dt_p.to_array( dt_pset_buffer, kk )
            if in_p.state == 1:
                elem.track( in_p )

            if isinstance( elem, pysix.elements.Drift ) and \
                in_p.state == 1 and \
                ( in_p.x > 1.0 or in_p.x < -1.0 or
                  in_p.y > 1.0 or in_p.y < -1.0 ):
                in_p.state = 0

            if in_p.state == 1:
                in_p.elemid += 1
            else:
                print( f"lost particle {in_p.partid} at pos {in_p.elemid} : {in_p}" )
                print( f"lost particle {in_p.partid} at elem: {elem}" )
                break
        if in_p.state == 1:
            in_p.turn += 1
            in_p.elemid = start_at_element
        else:
            print( f"lost particle {ii} at pos {in_p.elemid}: {in_p}" )
        pset = st.st_Particles.GET( pset_buffer, num_belem )
        assert pset.num_particles == num_part
        pysix_particle_to_pset( in_p, pset, ii, conf=conf )
        if MAKE_DEMOTRACK:
            dt_p.clear()
            dt_p.from_cobjects( pset, ii )
            kk = num_belem * num_part + ii
            assert len( dt_pset_buffer ) > kk
            dt_p.to_array( dt_pset_buffer, kk )

    path_elem_by_elem = os.path.join(
        output_path, "cobj_particles_elem_by_elem_pysixtrack.bin" )
    if 0 == pset_buffer.tofile_normalised( path_elem_by_elem, NORM_ADDR ):
        print( "**** -> Generated cbuffer of particle elem-by-elem data:\r\n" +
              f"****    {path_elem_by_elem}" )
    else:
        raise RuntimeError(
            "Unable to generate cobjects elem-by-elem data" )

    if MAKE_DEMOTRACK:
        path_elem_by_elem = os.path.join(
            output_path, "demotrack_elem_by_elem_pysixtrack.pickle" )
        with open( path_elem_by_elem, "wb" ) as f_out:
            f_out.write( float_to_bytes( len( dt_pset_buffer ) ) )
            f_out.write( dt_pset_buffer.tobytes() )
            print( "**** -> Generated demotrack particle elem-by-elem data:\r\n" +
                  f"****    {path_elem_by_elem}" )
    return

def generate_particle_data_until_turn( input_path, output_path, until_turn, conf=dict() ):
    path_in_particles = os.path.join(
        output_path, "pysixtrack_initial_particles.pickle" )
    with open( path_in_particles, "rb" ) as f_in:
        initial_p_pysix = pickle.load( f_in )
        print( "**** -> Read input data from:\r\n" +
               f"****    {path_in_particles}" )

    path_in_line = os.path.join( output_path, "pysixtrack_lattice.pickle" )
    with open( path_in_line, "rb" ) as f_in:
        line = pickle.load( f_in )

    num_belem = len( line )
    num_part = len( initial_p_pysix )
    start_at_element = 0

    NORM_ADDR = conf.get( "cbuffer_norm_base_addr", 4096 )
    MAKE_DEMOTRACK = conf.get( "make_demotrack_data", False )
    MAKE_DEMOTRACK &= st.Demotrack_enabled()

    pset_buffer = create_particle_set_cbuffer( 1, num_part, conf )
    assert pset_buffer.num_objects == 1
    pset = st.st_Particles.GET( pset_buffer, 0 )
    assert pset.num_particles == num_part

    if MAKE_DEMOTRACK:
        dt_p = st.st_DemotrackParticle()
        dt_pset_buffer = st.st_DemotrackParticle.CREATE_ARRAY( num_part, True )
        assert isinstance( dt_pset_buffer, np.ndarray )
        assert len( dt_pset_buffer ) <= num_part

    for ii, in_p in enumerate( initial_p_pysix ):
        assert isinstance( in_p, pysix.Particles )
        assert in_p.elemid == start_at_element
        assert in_p.partid == ii
        assert in_p.state == 1
        assert in_p.turn == 0
        print( f"****    Info :: particle {ii:6d}/{num_part - 1:6d}" )
        start_at_turn = in_p.turn
        if start_at_turn < until_turn:
            for jj in range( start_at_turn, until_turn ):
                for kk, elem in enumerate( line ):
                    if conf.get( 'always_use_drift_exact', False ):
                        assert not isinstance( elem, pysix.elements.Drift ) or \
                            isinstance( elem, pysix.elements.DriftExact )
                    if in_p.state == 1:
                        elem.track( in_p )
                    if isinstance( elem, pysix.elements.Drift ) and \
                        in_p.state == 1 and \
                        ( in_p.x > 1.0 or in_p.x < -1.0 or
                          in_p.y > 1.0 or in_p.y < -1.0 ):
                        in_p.state = 0

                    if in_p.state == 1:
                        in_p.elemid += 1
                    else:
                        print( f"lost particle {ii} at pos {jj} : {in_p}" )
                        print( f"lost particle at elem: {elem}" )
                        break
                if in_p.state == 1:
                    in_p.turn += 1
                    in_p.elemid = start_at_element
                else:
                    break
        if in_p.state != 1:
            print( f"lost particle {in_p.partid} at pos {in_p.elem} / turn  {in_p.turn}: {in_p}" )
        pysix_particle_to_pset( in_p, pset, ii, )
        if MAKE_DEMOTRACK:
            dt_p.clear()
            dt_p.from_cobjects( pset, ii )
            dt_p.to_array( dt_pset_buffer, ii )

    path_pset_out = os.path.join(
        output_path, f"cobj_particles_until_turn_{until_turn}.bin" )

    if 0 == pset_buffer.tofile_normalised( path_pset_out, NORM_ADDR ):
        print( "**** -> Generated cbuffer of tracked particle data:\r\n" +
              f"****    {path_pset_out}" )
    else:
        raise ValueError(
            f"Error during tracking particles until turn {until_turn}" )

    if MAKE_DEMOTRACK:
        path_pset_out = os.path.join(
            output_path, f"demotrack_particles_until_turn_{until_turn}.bin" )
        with open( path_pset_out, "wb" ) as f_out:
            f_out.write( float_to_bytes( len( dt_pset_buffer ) ) )
            f_out.write( dt_pset_buffer.tobytes() )
            print( "**** -> Generated demotrack data of tracked particles:\r\n" +
                   f"****    {path_pset_out}" )
    return

def generate_particle_data( input_path, output_path, conf=dict() ):
    print( "**** Generating Particles Data From SixTrack Input:" )
    path_in_particles = os.path.join(
        input_path, "pysixtrack_initial_particles.pickle" )
    with open( path_in_particles, "rb" ) as f_in:
        initial_p_pysix = pickle.load( f_in )
        print( "**** -> Read input data from:\r\n" +
               f"****    {path_in_particles}" )

    path_in_line = os.path.join( input_path, "pysixtrack_line.pickle" )
    with open( path_in_line, "rb" ) as f_in:
        line = pickle.load( f_in )

    num_belem = len( line )
    num_part = len( initial_p_pysix )
    start_at_element = 0

    print( f"****    Info :: num beam elements      : {num_belem}" )
    print( f"****    Info :: num particles          : {num_part}" )
    assert num_part > 0
    assert num_belem > 0

    # =========================================================================
    # Get initial particle distribution:

    # Generate the initial particle disitribution buffers
    print( "**** -> Generating initial particle distributions ..." )
    generate_particle_data_initial( input_path, output_path, conf=conf )

    # =========================================================================
    # Make elem-by-elem data using pysixtrack:

    if conf.get( "make_elem_by_elem_data", False ):
        print( "**** -> Generating elem-by-elem particle data using pysixtrack ..." )
        generate_particle_data_elem_by_elem( input_path, output_path, conf=conf )

    # =========================================================================
    # Make until turn data using pysixtrack:

    if conf.get( "make_until_num_turn_data", False ) and \
        conf.get( "until_num_turns", 1 ) > 0:
        print( "**** -> Generating until_turn tracked data using pysixtrack ..." )
        until_turn = conf.get( "until_num_turns", 1 )
        generate_particle_data_until_turn(
            input_path, output_path, until_turn, conf=conf )


def generate_data( scenario_name, input_path, output_path, conf=dict() ):
    assert scenario_name and len( scenario_name ) > 0
    print( "============================================================" +
           "============================================================" +
           "==============================" )
    print(  "****" )
    print( f"****       Scenario : {scenario_name}" )
    print(  "****" )
    print( f"****         Source : pysixtrack" )
    print( f"****      Input Dir : {input_path}" )
    print( f"****     Output Dir : {output_path}" )
    print(  "****" )
    print(  "------------------------------------------------------------" +
            "------------------------------------------------------------" +
            "------------------------------" )
    print(  "**** " )
    generate_lattice_data( input_path, output_path, conf=conf )
    print(  "**** " )
    print(  "------------------------------------------------------------" +
            "------------------------------------------------------------" +
            "------------------------------" )
    print(  "**** " )
    generate_particle_data( input_path, output_path, conf=conf )
    print(  "**** " )
    print(  "**** " )

