#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import os
import numpy as np

# Conversion from SixTrack is done using sixtracktools
import sixtracktools

# Tracking is done using pysixtrack
import pysixtrack

# sixtracklib provides the CObject based beam elements and particle types
import sixtracklib as st

from .pysixtrack_to_cobjects import pysixtrack_line_to_cbuffer
from .pysixtrack_to_cobjects import pysixtrack_particle_to_pset
from .pysixtrack_to_cobjects import pysixtrack_particle_to_single_particle
from .helpers import f64_to_bytes

def generate_cobjects_lattice( path_to_testdata_dir, conf=dict() ):
    print( "**** Generating CObjects Lattice Data:" )
    path_to_input = os.path.join( path_to_testdata_dir, "input" )
    print( f"**** -> Reading sixtrack input data from {path_to_input}" )
    six = sixtracktools.SixInput( path_to_input )

    line = pysixtrack.Line.from_sixinput( six )
    cbuffer = st.CBuffer()
    pysixtrack_line_to_cbuffer( line, cbuffer, conf=conf )
    path_to_lattice = os.path.join( path_to_testdata_dir, "cobj_lattice.bin" )

    if  0 == cbuffer.tofile_normalised( path_to_lattice,
            conf.get( "normalised_addr", 0x1000 ) ):
        print( "**** -> Generated cobjects lattice data at:" +
               f"\r\n****    {path_to_lattice}" )
    else:
        raise RuntimeError( "Problem during creation of lattice data" )

    path_to_pysixtrack_line = os.path.join(
        path_to_testdata_dir, "pysixtrack_lattice.pickle" )

    try:
        pickle.dump( line, open( path_to_pysixtrack_line, "wb" ) )
        print( "**** -> Generated pysixtrack lattice data as python pickle: " +
               f"\r\n****    {path_to_pysixtrack_line}" )
    except:
        raise RuntimeError(
            "Unable to generate pysixtrack lattice data" )

    if st.Demotrack_enabled() and st.Demotrack_belems_can_convert( cbuffer ):
        dt_lattice = st.Demotrack_belems_convert( cbuffer )
        if isinstance( dt_lattice, np.ndarray ) and \
            st.Demotrack_belems_num_stored_objects( dt_lattice ) == \
            cbuffer.num_objects:
            path_to_dt_lattice = os.path.join(
                path_to_testdata_dir, "demotrack_lattice.bin" )
            with open( path_to_dt_lattice, "wb" ) as fp:
                fp.write( f64_to_bytes( len( dt_lattice ) ) )
                fp.write( dt_lattice.tobytes() )
            print( f"**** -> Generated demotrack lattice as flat array:" +
                   f"\r\n****    {path_to_dt_lattice}" )
        else:
            raise RuntimeError(
                "Unable to generate demotrack flat array lattice data" )


def generate_cobjects_particles( path_to_testdata_dir, conf=dict() ):
    print( "**** Generating CObjects Particles Data:" )
    path_to_input = os.path.join( path_to_testdata_dir, "input" )
    path_to_dump_file = os.path.join( path_to_input, "dump3.dat" )

    print( f"**** -> Reading sixtrack input data from {path_to_input}" )
    six = sixtracktools.SixInput( path_to_input )
    #line, rest, iconv = six.expand_struct( convert=pysixtrack.elements )
    line = pysixtrack.Line.from_sixinput(six)
    iconv = line.other_info["iconv"]
    sixdump = sixtracktools.SixDump101( path_to_dump_file )

    num_iconv = int(len(iconv))
    num_belem = int(len(line))
    num_dumps = int(len(sixdump.particles))

    assert num_iconv > 0
    assert num_belem > iconv[num_iconv - 1]
    assert num_dumps >= num_iconv
    assert (num_dumps % num_iconv) == 0

    num_particles = int( num_dumps / num_iconv )
    print( f"****    Info :: num sixtrack sequences : {num_iconv}" )
    print( f"****    Info :: num beam elements      : {num_belem}" )
    print( f"****    Info :: num particles          : {num_particles}" )

    # -------------------------------------------------------------------------
    # Get initial particle distribution:

    # Generate the initial particle disitribution buffers
    print( "****\r\n**** Generating initial particle distribution ..." )

    is_demotrack_enabled = st.Demotrack_enabled()

    initial_p_buffer = st.CBuffer()
    initial_pset_buffer = st.CBuffer()
    pset = st.st_Particles( initial_pset_buffer, num_particles )

    num_slots_per_pset = pset.cobj_required_num_bytes(
        initial_pset_buffer.slot_size ) // initial_pset_buffer.slot_size
    num_ptrs_per_pset = st.st_Particles.COBJ_NUM_DATAPTRS

    if is_demotrack_enabled:
        dt_p = st.st_DemotrackParticle()
        dt_initial_particle_data = st.st_DemotrackParticle.CREATE_ARRAY(
            num_particles, True )
        assert isinstance( dt_initial_particle_data, np.ndarray )
        assert len( dt_initial_particle_data ) >= num_particles

    pysix_initial_pset = []

    ii = 0
    at_element = iconv[ ii ]
    assert at_element < num_belem

    for jj in range( num_particles ):
        kk = num_particles * ii + jj
        assert kk < num_dumps
        pysix_initial_pset.append(
            pysixtrack.Particles( **sixdump[ kk ].get_minimal_beam() ) )

        pysix_initial_pset[ -1 ].turn = 0
        pysix_initial_pset[ -1 ].state = 1
        pysix_initial_pset[ -1 ].elemid = at_element
        pysix_initial_pset[ -1 ].partid = jj
        p = st.st_SingleParticle( initial_p_buffer )

        pysixtrack_particle_to_pset(
            pysix_initial_pset[ -1 ], pset, jj, particle_id=jj, conf=conf )

        pysixtrack_particle_to_single_particle(
            pysix_initial_pset[ -1 ], p, particle_id=jj, conf=conf )

        if is_demotrack_enabled:
            assert 0 == dt_p.from_cobjects( pset, jj )
            assert 0 == dt_p.to_array( dt_initial_particle_data, jj )

    path_to_initial_pset = os.path.join(
        path_to_testdata_dir, "cobj_initial_particles.bin" )

    if  0 == initial_pset_buffer.tofile_normalised(
            path_to_initial_pset, conf.get( "normalised_addr", 0x1000 ) ):
        print( "**** -> Generated initial particle set data at:\r\n" +
               f"****    {path_to_initial_pset}" )
    else:
        raise RuntimeError( "Unable to generate initial particle set data" )

    path_to_initial_single_p = os.path.join(
        path_to_testdata_dir, "cobj_initial_single_particles.bin" )

    if  0 == initial_p_buffer.tofile_normalised(
            path_to_initial_single_p, conf.get( "normalised_addr", 0x1000 ) ):
        print( "**** -> Generated initial single particle data at:\r\n" +
              f"****    {path_to_initial_single_p}" )
    else:
        raise RuntimeError( "Unable to generate initial single particle set data" )

    assert len( pysix_initial_pset ) == num_particles

    path_to_initial_pysix = os.path.join(
        path_to_testdata_dir, "pysixtrack_initial_particles.pickle" )

    try:
        pickle.dump( pysix_initial_pset, open( path_to_initial_pysix, "wb" ) )
        print( "**** -> Generated initial pysixtrack particle data at:\r\n" +
               f"****    {path_to_initial_pysix}" )
    except:
        raise RuntimeError( "Unable to generate initial pysixtrack particle data" )

    if is_demotrack_enabled:
        path_to_initial_demotrack = os.path.join(
            path_to_testdata_dir, "demotrack_initial_particles.bin" )
        with open( path_to_initial_demotrack, "wb" ) as fp:
            fp.write( f64_to_bytes( len( dt_initial_particle_data ) ) )
            fp.write( dt_initial_particle_data.tobytes() )
            print( "**** -> Generated initial demotrack particle data at:\r\n" +
                  f"****    {path_to_initial_demotrack}" )

    pset = None
    p = None
    del initial_p_buffer
    initial_p_buffer = None
    del path_to_initial_single_p
    del initial_pset_buffer
    initial_pset_buffer = None
    del path_to_initial_pset

    if is_demotrack_enabled:
        del dt_p
        dt_p = None
        del dt_initial_particle_data
        dt_initial_particle_data = None
        del path_to_initial_demotrack

    # -------------------------------------------------------------------------
    # Get sixtrack sequency-by-sequence data:

    print( "****\r\n**** Generating sixtrack sequence-by-sequence particle data ..." )

    pset_buffer = st.CBuffer(
        num_iconv * num_slots_per_pset, num_iconv * num_ptrs_per_pset,
        num_iconv, 0 )

    for ii in range( num_iconv ):
        at_element = iconv[ ii ]
        assert at_element < num_belem
        pset = st.st_Particles( pset_buffer, num_particles )

        for jj in range( num_particles ):
            kk = num_particles * ii + jj
            assert kk < num_dumps
            in_p = pysixtrack.Particles( **sixdump[ kk ].get_minimal_beam() )
            in_p.state = 1
            in_p.elemid = at_element
            in_p.turn = 0
            in_p.partid = jj
            pysixtrack_particle_to_pset(
                in_p, pset, jj, particle_id=jj, conf=conf )
        pset = None

    path_to_pset_file = os.path.join(
        path_to_testdata_dir, "cobj_particles_sixtrack.bin" )

    if  0 == pset_buffer.tofile_normalised( path_to_pset_file,
            conf.get( "normalised_addr", 0x1000 ) ):
        print( "**** -> Generated sixtrack particle sequence-by-sequence" +
               f"data at:\r\n****    {path_to_pset_file}" )
    else:
        raise RuntimeError(
            "Unable to generate sixtrack sequency-by-sequence data" )

    del pset_buffer
    del path_to_pset_file

    # -------------------------------------------------------------------------
    # Get true elem-by-elem data via pysixtrack:

    print( "****\r\n**** Generating pysixtack elem-by-elem particle data ..." )
    num_elem_by_elem = num_particles * ( num_belem + 1 )
    assert num_slots_per_pset > 0
    elem_by_elem_cbuffer = st.CBuffer(
        num_elem_by_elem * num_slots_per_pset,
        num_elem_by_elem * num_ptrs_per_pset,
        num_elem_by_elem, 0 )

    for ii in range( 0, num_belem + 1 ):
        pset = st.st_Particles( elem_by_elem_cbuffer, num_particles )
        assert pset.num_particles == num_particles

    if is_demotrack_enabled:
        dt_p = st.st_DemotrackParticle()
        dt_elem_by_elem_data = st.st_DemotrackParticle.CREATE_ARRAY(
            num_elem_by_elem, True )
        ll = 0

    for ii, p in enumerate( pysix_initial_pset ):
        p.partid = ii
        p.turn = 0
        print( f"****    tracking particle {ii + 1:7} / {num_particles:7}" )
        for jj, elem in enumerate( line.elements ):
            assert jj < num_belem
            elem_by_elem_pset = st.st_Particles.GET( elem_by_elem_cbuffer, jj )
            assert elem_by_elem_pset.num_particles == num_particles
            p.elemid = jj
            pysixtrack_particle_to_pset( p, elem_by_elem_pset, ii, conf=conf )
            if is_demotrack_enabled:
                assert 0 == dt_p.from_cobjects( elem_by_elem_pset, ii )
                kk = jj * num_particles + ii
                assert kk < len( dt_elem_by_elem_data )
                assert 0 == dt_p.to_array( dt_elem_by_elem_data, kk )
            if p.state == 1:
                elem.track( p )
            if p.state == 1:
                p.elemid = jj + 1
        elem_by_elem_pset = st.st_Particles.GET( elem_by_elem_cbuffer, num_belem )
        assert elem_by_elem_pset.num_particles == num_particles
        if p.state == 1:
            p.turn  += 1
            p.elemid = 0
        pysixtrack_particle_to_pset( p, elem_by_elem_pset, ii, conf=conf )
        if is_demotrack_enabled:
            assert 0 == dt_p.from_cobjects( elem_by_elem_pset, ii )
            kk = num_belem * num_particles + ii
            assert kk < len( dt_elem_by_elem_data )
            assert 0 == dt_p.to_array( dt_elem_by_elem_data, kk )

    path_to_elem_by_elem_file = os.path.join(
        path_to_testdata_dir, "cobj_elem_by_elem_pysixtrack.bin" )

    if  0 == elem_by_elem_cbuffer.tofile_normalised( path_to_elem_by_elem_file,
            conf.get( "normalised_addr", 0x1000 ) ):
        print( "**** -> Generated cobjects elem-by-elem particle data\r\n" +
               "****    Tracking by pysixtrack, data at:\r\n" +
               f"****    {path_to_elem_by_elem_file}" )
    else:
        raise RuntimeError(
            "Unable to generate cobjects elem-by-elem data " +
            "(tracking by pysixtack)" )

    if is_demotrack_enabled:
        path_to_demotrack_elem_by_elem = os.path.join( path_to_testdata_dir,
            "demotrack_elem_by_elem_pysixtrack.bin" )
        with open( path_to_demotrack_elem_by_elem, "wb" ) as fp:
            fp.write( f64_to_bytes( len( dt_elem_by_elem_data ) ) )
            fp.write( dt_elem_by_elem_data.tobytes() )

            print( "**** -> Generated demotrack elem-by-elem particle data\r\n" +
                   "****    Tracking by pysixtrack, data at:\r\n" +
                   f"****    {path_to_demotrack_elem_by_elem}" )

        del dt_elem_by_elem_data
        del dt_p

    del elem_by_elem_pset
    del elem_by_elem_cbuffer



def generate_sixtrack( scenario_name, testdata_dir=None, conf=dict() ):
    assert scenario_name and len( scenario_name ) > 0
    print( "****************************************************************" +
           "***************\r\n****\r\n" +
           f"**** SCENARIO: {scenario_name}\r\n****\r\n****" )

    if testdata_dir is None:
        testdata_dir = os.path.join( os.path.dirname( os.path.dirname(
            os.path.abspath( __file__ ) ) ), scenario_name )

    print( f"**** Base Directory: {testdata_dir}\r\n****" )
    generate_cobjects_lattice( testdata_dir, conf=conf )
    print( "****" );
    generate_cobjects_particles( testdata_dir, conf=conf )
    print( "****\r\n***************" +
           "****************************************************************" )
