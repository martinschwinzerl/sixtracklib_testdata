#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import os

# Conversion from SixTrack is done using sixtracktools
import sixtracktools

# Tracking is done using pysixtrack
import pysixtrack

# sixtracklib provides the CObject based beam elements and particle types
import sixtracklib as st

from .pysixtrack_to_cobjects import pysixtrack_line_to_cbuffer
from .pysixtrack_to_cobjects import pysixtrack_particle_to_pset
from .pysixtrack_to_cobjects import pysixtrack_particle_to_single_particle

def generate_cobjects_lattice( path_to_testdata_dir, conf=dict() ):
    print( "**** Generating CObjects Lattice Data:" )
    path_to_input = os.path.join( path_to_testdata_dir, "input" )
    print( f"**** -> Reading sixtrack input data from {path_to_input}" )
    six = sixtracktools.SixInput( path_to_input )

    line = pysixtrack.Line.from_sixinput( six )
    cbuffer = st.CBuffer()
    pysixtrack_line_to_cbuffer( line, cbuffer )
    path_to_lattice = os.path.join( path_to_testdata_dir, "cobj_lattice.bin" )

    if  0 == cbuffer.tofile_normalised( path_to_lattice,
            conf.get( "normalised_addr", 0x1000 ) ):
        print( f"**** -> Generated cobjects lattice data at:" +
                "\r\n****    {path_to_lattice}" )
    else:
        raise RuntimeError( "Problem during creation of lattice data" )

    path_to_pysixtrack_line = os.path.join(
        path_to_testdata_dir, "pysixtrack_lattice.pickle" )

    try:
        pickle.dump( line, open( path_to_pysixtrack_line, "wb" ) )
        print( "**** -> Generated initial pysixtrack lattice data at:\r\n" +
               f"****    {path_to_pysixtrack_line}" )
    except:
        raise RuntimeError(
            "Unable to generate initial pysixtrack lattice data" )




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
    print( "****\r\n**** -> Generating initial particle distribution ..." )

    initial_p_buffer = st.CBuffer()
    initial_pset_buffer = st.CBuffer()
    pset = st.st_Particles( initial_pset_buffer, num_particles )


    path_to_initial_pset = os.path.join(
        path_to_testdata_dir, "cobj_initial_particles.bin" )

    pysix_initial_pset = []

    ii = 0
    at_element = iconv[ ii ]
    assert at_element < num_belem

    for jj in range( num_particles ):
        kk = num_particles * ii + jj
        assert kk < num_dumps
        pysix_initial_pset.append(
            pysixtrack.Particles( **sixdump[ kk ].get_minimal_beam() ) )

        p = st.st_SingleParticle( initial_p_buffer )

        pysixtrack_particle_to_pset(
            pysix_initial_pset[ -1 ], pset, jj,
            particle_id=jj, at_element=at_element )

        pysixtrack_particle_to_single_particle(
            pysix_initial_pset[ -1 ], p, particle_id=jj, at_element=at_element )

    path_to_initial_pset = os.path.join(
        path_to_testdata_dir, "cobj_initial_particles.bin" )

    if  0 == initial_pset_buffer.tofile_normalised(
            path_to_initial_pset, conf.get( "normalised_addr", 0x1000 ) ):
        print( f"**** -> Generated initial particle set data at: {path_to_initial_pset}" )
    else:
        raise RuntimeError( "Unable to generate initial particle set data" )

    path_to_initial_single_p = os.path.join(
        path_to_testdata_dir, "cobj_initial_single_particles.bin" )

    if  0 == initial_p_buffer.tofile_normalised(
            path_to_initial_single_p, conf.get( "normalised_addr", 0x1000 ) ):
        print( f"**** -> Generated initial single particle data at: {path_to_initial_single_p}" )
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

    pset = None
    p = None
    del initial_p_buffer
    del path_to_initial_single_p
    del initial_pset_buffer
    del path_to_initial_pset

    # -------------------------------------------------------------------------
    # Get sixtrack sequency-by-sequence data:

    # Generate the initial particle disitribution buffers
    print( "****\r\n**** -> Generating sixtrack sequence-by-sequence particle data ..." )

    pset_buffer = st.CBuffer()

    for ii in range( num_iconv ):
        at_element = iconv[ ii ]
        assert at_element < num_belem
        pset = st.st_Particles( pset_buffer, num_particles )

        for jj in range( num_particles ):
            kk = num_particles * ii + jj
            assert kk < num_dumps
            in_p = pysixtrack.Particles( **sixdump[ kk ].get_minimal_beam() )
            pysixtrack_particle_to_pset( in_p, pset, jj,
                particle_id=jj, at_element=at_element )
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
    # Get sixtrack sequency-by-sequence data:

    # Generate the initial particle disitribution buffers
    print( "****\r\n**** -> Generating pysixtack elem-by-elem particle data ..." )

    cbuffer = st.CBuffer()
    output_pset = st.st_Particles( cbuffer, num_particles * num_belem )
    assert output_pset.num_particles == num_particles * num_belem

    for ii, p in enumerate( pysix_initial_pset ):
        print( f"****    tracking particle #{ii + 1} / {num_particles}" )
        for jj, elem in enumerate( line.elements ):
            kk = ii * num_belem + jj
            assert kk < output_pset.num_particles
            pysixtrack_particle_to_pset( p, output_pset, kk,
                particle_id=ii, at_element=jj )
            elem.track( p )

    path_to_elem_by_elem_file = os.path.join(
        path_to_testdata_dir, "cobj_particles_elem_by_elem_pysixtrack.bin" )

    if  0 == cbuffer.tofile_normalised( path_to_elem_by_elem_file,
            conf.get( "normalised_addr", 0x1000 ) ):
        print( "**** -> Generated pysixtrack elem-by-elem particle data" +
               f"data at:\r\n****    {path_to_elem_by_elem_file}" )
    else:
        raise RuntimeError(
            "Unable to generate pysixtrack elem-by-elem data" )


def generate_sixtrack( scenario_name, conf=dict() ):
    assert scenario_name and len( scenario_name ) > 0
    testdata_dir = os.path.join(
        os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ),
            scenario_name )

    generate_cobjects_lattice( testdata_dir, conf )
    generate_cobjects_particles( testdata_dir, conf )
    return
