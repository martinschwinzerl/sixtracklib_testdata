#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import os
import numpy as np

# Conversion from SixTrack is done using sixtracktools
import sixtracktools

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
    print( "**** Generating Lattice Data From SixTrack Input:" )
    print( "**** -> Reading sixtrack input data from:\r\n" +
          f"****    {input_path}" )
    six = sixtracktools.SixInput( input_path )
    slot_size = st.CBufferView.DEFAULT_SLOT_SIZE

    line = pysix.Line.from_sixinput( six )
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

def generate_particle_data_initial( output_path, iconv, sixdump, conf=dict() ):
    num_iconv = int( len( iconv ) )
    num_dumps = int( len( sixdump.particles ) )

    assert num_iconv > 0
    assert num_dumps >= num_iconv
    assert ( num_dumps % num_iconv ) == 0
    num_part = num_dumps // num_iconv

    assert num_part  > 0
    assert num_iconv > 0

    NORM_ADDR = conf.get( "cbuffer_norm_base_addr", 4096 )
    MAKE_DEMOTRACK = conf.get( "make_demotrack_data", False )
    MAKE_DEMOTRACK &= st.Demotrack_enabled()
    init_particle_idx = 0
    start_at_element = iconv[ init_particle_idx ]

    initial_p_buffer = create_single_particle_cbuffer( 1, num_part, conf )
    initial_pset_buffer = create_particle_set_cbuffer( 1, num_part, conf )
    initial_p_pysix = []
    path_initial_pset = os.path.join( output_path, "cobj_initial_particles.bin" )

    for jj in range( num_part ):
        kk = num_part * init_particle_idx + jj
        assert kk < len( sixdump.particles )
        initial_p_pysix.append( pysix.Particles(
            **sixdump[ kk ].get_minimal_beam() ) )

        initial_p_pysix[ -1 ].elemid = start_at_element
        initial_p_pysix[ -1 ].turn = 0
        initial_p_pysix[ -1 ].partid = jj
        initial_p_pysix[ -1 ].state = 1

        p = st.st_SingleParticle.GET( initial_p_buffer, kk )
        pysix_particle_to_single_particle( initial_p_pysix[ -1 ], p, conf=conf )

        pset = st.st_Particles.GET( initial_pset_buffer, 0 )
        pysix_particle_to_pset( initial_p_pysix[ -1 ], pset, jj, conf=conf )

    assert len( initial_p_pysix ) == num_part
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


def generate_particle_data_sequ_by_sequ( output_path, line, iconv, sixdump, conf=dict() ):
    num_iconv = int( len( iconv ) )
    num_belem = int( len( line ) )
    num_dumps = int( len( sixdump.particles ) )

    assert num_iconv > 0
    assert num_belem > iconv[num_iconv - 1]
    assert num_dumps >= num_iconv
    assert (num_dumps % num_iconv) == 0
    num_particles = num_dumps // num_iconv

    assert num_particles > 0
    assert num_belem > 0
    assert num_iconv > 0

    NORM_ADDR = conf.get( "cbuffer_norm_base_addr", 4096 )
    pset_buffer = create_particle_set_cbuffer(
        num_iconv + 1, num_particles, conf )

    for ii in range( num_iconv ):
        at_element = iconv[ ii ]
        assert at_element < num_belem
        assert ii < pset_buffer.num_objects
        pset = st.st_Particles.GET( pset_buffer, ii )
        assert pset.num_particles == num_particles
        for jj in range( 0, num_particles ):
            kk = num_particles * ii + jj
            assert kk < num_dumps
            in_p = pysix.Particles( **sixdump[ kk ].get_minimal_beam() )
            in_p.state = 1
            in_p.turn = 0
            in_p.id = jj
            in_p.at_element = at_element
            assert pset.num_particles == num_particles
            pysix_particle_to_pset( in_p, pset, jj, conf=conf )
    path_cobj_pset = os.path.join( output_path, "cobj_particles_sixtrack.bin" )
    if 0 == pset_buffer.tofile_normalised( path_cobj_pset, NORM_ADDR ):
        print( "**** -> Generated cbuffer of sixtrack particle sequ-by-sequ "
               "data:\r\n" + f"****    {path_cobj_pset}" )
    else:
        raise RuntimeError(
            "Unable to generate cobjects sixtrack sequency-by-sequence data" )
    return

def generate_particle_data_elem_by_elem( output_path, line, iconv, sixdump, conf=dict() ):
    num_iconv = int( len( iconv ) )
    num_belem = int( len( line ) )
    num_dumps = int( len( sixdump.particles ) )

    assert num_iconv > 0
    assert num_belem > iconv[num_iconv - 1]
    assert num_dumps >= num_iconv
    assert (num_dumps % num_iconv) == 0
    num_particles = num_dumps // num_iconv
    assert num_particles > 0
    assert num_belem > 0
    assert num_iconv > 0

    start_at_element = iconv[ 0 ]

    NORM_ADDR = conf.get( "cbuffer_norm_base_addr", 4096 )
    MAKE_DEMOTRACK = conf.get( "make_demotrack_data", False )
    MAKE_DEMOTRACK &= st.Demotrack_enabled()

    pset_buffer = create_particle_set_cbuffer(
        num_belem + 1, num_particles, conf )
    assert pset_buffer.num_objects == num_belem + 1

    path_initial_pysix_particles = os.path.join(
        output_path, "pysixtrack_initial_particles.pickle" )
    initial_p_pysix = None
    with open( path_initial_pysix_particles, "rb" ) as f_in:
        initial_p_pysix = pickle.load( f_in )
    assert initial_p_pysix is not None
    assert len( initial_p_pysix ) == num_particles

    if MAKE_DEMOTRACK:
        dt_p = st.st_DemotrackParticle()
        dt_pset_buffer = st.st_DemotrackParticle.CREATE_ARRAY(
            ( num_belem + 1 ) * num_particles, True )
        assert isinstance( dt_pset_buffer, np.ndarray )
        assert len( dt_pset_buffer ) <= ( num_belem + 1 ) * num_particles

    for ii, in_p in enumerate( initial_p_pysix ):
        assert isinstance( in_p, pysix.Particles )
        in_p.elemid = start_at_element
        in_p.turn = 0
        in_p.partid = ii
        in_p.state = 1

        print( f"****    Info :: particle {ii:6d}/{num_particles - 1:6d}" )
        for jj, elem in enumerate( line.elements ):
            pset = st.st_Particles.GET( pset_buffer, jj )
            assert pset.num_particles == num_particles
            kk = ii * num_belem + jj
            pysix_particle_to_pset(
                in_p, pset, ii, particle_id=ii, at_element=jj, conf=conf )
            if MAKE_DEMOTRACK:
                dt_p.clear()
                dt_p.from_cobjects( pset, ii )
                dt_p.to_array( dt_pset_buffer, kk )
            if in_p.state == 1:
                elem.track( in_p )
        pset = st.st_Particles.GET( pset_buffer, num_belem )
        assert pset.num_particles == num_particles
        if in_p.state == 1:
            in_p.turn += 1
            in_p.elemid = start_at_element
        else:
            print( f"lost particle {in_p}" )
        pysix_particle_to_pset( in_p, pset, ii, particle_id=ii,
            at_element=start_at_element, conf=conf )
        if MAKE_DEMOTRACK:
            dt_p.clear()
            dt_p.from_cobjects( pset, ii )
            dt_p.to_array( dt_pset_buffer, ii * num_belem + num_belem )

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

def generate_particle_data_until_turn( output_path, line, iconv, sixdump, until_turn, conf=dict() ):
    num_iconv = int( len( iconv ) )
    num_belem = int( len( line ) )
    num_dumps = int( len( sixdump.particles ) )
    assert num_iconv > 0
    assert num_belem > iconv[num_iconv - 1]
    assert num_dumps >= num_iconv
    assert (num_dumps % num_iconv) == 0
    num_particles = num_dumps // num_iconv
    assert num_particles > 0
    assert num_belem > 0
    assert num_iconv > 0
    assert until_turn > 0
    start_at_element = iconv[ 0 ]

    NORM_ADDR = conf.get( "cbuffer_norm_base_addr", 4096 )
    MAKE_DEMOTRACK = conf.get( "make_demotrack_data", False )
    MAKE_DEMOTRACK &= st.Demotrack_enabled()

    pset_buffer = create_particle_set_cbuffer( 1, num_particles, conf )
    assert pset_buffer.num_objects == 1
    pset = st.st_Particles.GET( pset_buffer, 0 )
    assert pset.num_particles == num_particles

    path_initial_pysix_particles = os.path.join(
        output_path, "pysixtrack_initial_particles.pickle" )
    initial_p_pysix = None
    with open( path_initial_pysix_particles, "rb" ) as f_in:
        initial_p_pysix = pickle.load( f_in )
    assert initial_p_pysix is not None
    assert len( initial_p_pysix ) == num_particles

    if MAKE_DEMOTRACK:
        dt_p = st.st_DemotrackParticle()
        dt_pset_buffer = st.st_DemotrackParticle.CREATE_ARRAY( num_particles, True )
        assert isinstance( dt_pset_buffer, np.ndarray )
        assert len( dt_pset_buffer ) <= num_particles

    for ii, in_p in enumerate( initial_p_pysix ):
        in_p.elemid = start_at_element
        in_p.turn = 0
        in_p.partid = ii
        in_p.state = 1
        print( f"****    Info :: particle {ii:6d}/{num_particles - 1:6d}" )
        for jj in range( in_p.turn, until_turn ):
            for kk, elem in enumerate( line.elements ):
                if in_p.state == 1:
                    elem.track( in_p )
                else:
                    break
            if in_p.state == 1:
                in_p.turn += 1
                in_p.elemid = start_at_element
            else:
                break
        pysix_particle_to_pset( in_p, pset, ii, conf=conf )
        if in_p.state != 1:
            print( f"lost particle {in_p}" )
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
    path_to_dump_file = os.path.join( input_path, "dump3.dat" )

    print( "**** -> Reading sixtrack input data from:\r\n" +
          f"****    {path_to_dump_file}" )
    six = sixtracktools.SixInput( input_path )
    #line, rest, iconv = six.expand_struct( convert=pysixtrack.elements )
    line = pysix.Line.from_sixinput(six)
    iconv = line.other_info["iconv"]
    sixdump = sixtracktools.SixDump101( path_to_dump_file )

    num_iconv = int( len( iconv ) )
    num_belem = int( len( line ) )
    num_dumps = int( len( sixdump.particles ) )

    assert num_iconv > 0
    assert num_belem > iconv[num_iconv - 1]
    assert num_dumps >= num_iconv
    assert (num_dumps % num_iconv) == 0

    num_particles = num_dumps // num_iconv
    print( f"****    Info :: num sixtrack sequences : {num_iconv}" )
    print( f"****    Info :: num beam elements      : {num_belem}" )
    print( f"****    Info :: num particles          : {num_particles}" )
    assert num_particles > 0
    assert num_belem > 0
    assert num_iconv > 0

    # =========================================================================
    # Get initial particle distribution:

    # Generate the initial particle disitribution buffers
    print( "**** -> Generating initial particle distributions ..." )
    generate_particle_data_initial( output_path, iconv, sixdump, conf=conf )

    # =========================================================================
    # Make sixtrack sequency-by-sequence data:

    if conf.get( "make_sixtrack_sequ_by_sequ", False ):
        print( "**** -> Generating SixTrack sequ-by-sequ particle data ..." )
        generate_particle_data_sequ_by_sequ(
            output_path, line, iconv, sixdump, conf=conf )

    # =========================================================================
    # Make elem-by-elem data using pysixtrack:

    if conf.get( "make_elem_by_elem_data", False ):
        print( "**** -> Generating elem-by-elem particle data using pysixtrack ..." )
        generate_particle_data_elem_by_elem(
            output_path, line, iconv, sixdump, conf=conf )

    # =========================================================================
    # Make until turn data using pysixtrack:

    if conf.get( "make_until_num_turn_data", False ) and \
        conf.get( "until_num_turns", 1 ) > 0:
        print( "**** -> Generating until_turn tracked data using pysixtrack ..." )
        until_turn = conf.get( "until_num_turns", 1 )
        generate_particle_data_until_turn(
            output_path, line, iconv, sixdump, until_turn, conf=conf )


def generate_data( scenario_name, input_path, output_path, conf=dict() ):
    assert scenario_name and len( scenario_name ) > 0
    print( "============================================================" +
           "============================================================" +
           "==============================" )
    print(  "****" )
    print( f"****       Scenario : {scenario_name}" )
    print(  "****" )
    print( f"****         Source : sixtrack" )
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
