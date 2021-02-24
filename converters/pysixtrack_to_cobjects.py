import sixtracklib as st
import pysixtrack  as pysix
import numpy as np
from scipy.special import factorial

def pysixtrack_line_to_cbuffer( line, cbuffer, conf=dict() ):
    assert isinstance( cbuffer, st.CBufferView )
    assert isinstance( line, pysix.Line )

    for ii, elem in enumerate( line.elements ):
        if isinstance( elem, pysix.elements.Drift ):
            if not conf.get( 'always_exact_drift', False ):
                st.st_Drift( cbuffer, elem.length )
            else:
                st.st_DriftExact( cbuffer, elem.length )
        elif isinstance( elem, pysix.elements.DriftExact ):
            st.st_DriftExact( cbuffer, elem.length )
        elif isinstance( elem, pysix.elements.Multipole ):
            if elem.order > 0:
                st.st_Multipole( cbuffer, elem.length, elem.hxl,
                    elem.hyl, knl=elem.knl, ksl=elem.ksl, order=elem.order )
            else:
                mp = st.st_Multipole( cbuffer, max_order=0, order=0,
                        length=elem.length, hxl=elem.hxl, hyl=elem.hyl )
                if len( elem.knl ) > 0:
                    mp.set_bal( 0, elem.knl[ 0 ] )
                if len( elem.ksl ) > 0:
                    mp.set_bal( 1, elem.ksl[ 0 ] )
        elif isinstance( elem, pysix.elements.LimitRect ):
            st.st_LimitRect(
                cbuffer, elem.min_x, elem.max_x, elem.min_y, elem.max_y )
        elif isinstance( elem, pysix.elements.LimitEllipse ):
            st.st_LimitEllipse( cbuffer, elem.a * elem.a, elem.b * elem.b )
        elif isinstance( elem, pysix.elements.LimitRectEllipse ):
            st.st_LimitRectEllipse( cbuffer,
                elem.max_x, elem.max_y, elem.a * elem.a, elem.b * elem.b )
        elif isinstance( elem, pysix.elements.DipoleEdge ):
            corr = 2.0 * elem.h * elem.hgap * elem.fint
            r21 = +elem.h * np.tan( elem.e1 )
            r43 = -elem.h * np.tan( elem.e1 -
                ( corr / np.cos( elem.e1 ) ) * ( 1. + np.sin( elem.e1 ) ** 2 ) )
            st.st_DipoleEdge( cbuffer, r21, r43 )
        elif isinstance( elem, pysix.elements.Cavity ):
            st.st_Cavity( cbuffer, elem.voltage, elem.frequency, elem.lag )
        elif isinstance( elem, pysix.elements.SRotation ):
            deg2rad = np.pi / float( 180.0 )
            st.st_SRotation( cbuffer, elem.angle * deg2rad )
        elif isinstance( elem, pysix.elements.XYShift ):
            st.st_XYShift( cbuffer, elem.dx, elem.dy )
        elif isinstance( elem, pysix.be_beamfields.spacecharge.SCCoasting ):
            st.st_SCCoasting( cbuffer, elem.number_of_particles,
                elem.circumference, elem.sigma_x, elem.sigma_y, elem.length,
                    elem.x_co, elem.y_co, elem.min_sigma_diff, elem.enabled )
        elif isinstance( elem, pysix.be_beamfields.spacecharge.SCQGaussProfile ):
            st.st_SCQGaussProfile( cbuffer, elem.number_of_particles,
                elem.bunchlength_rms, elem.sigma_x, elem.sigma_y, elem.length,
                    elem.x_co, elem.y_co, elem.min_sigma_diff, elem.q_param,
                        elem.enabled )
        else:
            print( f"element not converted: {elem}" )
    return

def pysixtrack_particle_to_pset( in_p, pset, index,
    state=None, at_element=None, at_turn=None, particle_id=None, conf=dict() ):
    assert isinstance( in_p, pysix.Particles )
    assert isinstance( pset, st.st_Particles )
    assert index < pset.num_particles

    if particle_id is None:
        if in_p.partid is not None:
            particle_id = in_p.partid
        else:
            particle_id = index

    if state is None:
        if in_p.state is not None:
            state = in_p.state
        else:
            state = 1

    if at_element is None:
        if in_p.elemid is not None:
            at_element = in_p.elemid
        else:
            at_element = 0

    if at_turn is None:
        if in_p.turn is not None:
            at_turn = in_p.turn
        else:
            at_turn = 0

    pset.set_charge0( index, in_p.q0 )
    pset.set_mass0( index, in_p.mass0 )
    pset.set_beta0( index, in_p.beta0 )
    pset.set_gamma0( index, in_p.gamma0 )
    pset.set_p0c( index, in_p.p0c )

    pset.set_x( index, in_p.x  )
    pset.set_y( index, in_p.y )
    pset.set_px( index, in_p.px )
    pset.set_py( index, in_p.py )
    pset.set_zeta( index, in_p.zeta )

    EPS = np.float64( 1e-12 )
    pset.update_delta( index, in_p.delta )

    assert np.allclose( pset.rpp( index ), in_p.rpp, EPS, EPS )
    assert np.allclose( pset.rvv( index ), in_p.rvv, EPS, EPS )
    assert np.allclose( pset.psigma( index ), in_p.psigma, EPS, EPS )

    pset.set_state( index, state )
    pset.set_at_element( index, at_element )
    pset.set_at_turn( index, at_turn )
    pset.set_id( index, particle_id )

    pset.set_chi( index, in_p.chi )
    pset.set_charge_ratio( index, in_p.qratio )
    pset.set_s( index, in_p.s )




def pysixtrack_particle_to_single_particle( in_p, p,
    state=None, at_element=None, at_turn=None, particle_id=None, conf=dict() ):
    assert isinstance( in_p, pysix.Particles )
    assert isinstance( p, st.st_SingleParticle )

    if particle_id is None:
        if in_p.partid is not None:
            particle_id = in_p.partid
        else:
            raise ValueError( "particle_id requires proper value" )

    if state is None:
        if in_p.state is not None:
            state = in_p.state
        else:
            state = 1

    if at_element is None:
        if in_p.elemid is not None:
            at_element = in_p.elemid
        else:
            at_element = 0

    if at_turn is None:
        if in_p.turn is not None:
            at_turn = in_p.turn
        else:
            at_turn = 0

    p.charge0 = in_p.q0
    p.mass0   = in_p.mass0
    p.beta0   = in_p.beta0
    p.gamma0  = in_p.gamma0
    p.p0c     = in_p.p0c

    p.x  = in_p.x
    p.y  = in_p.y
    p.px = in_p.px
    p.py = in_p.py
    p.zeta = in_p.zeta

    EPS = np.float64( 1e-12 )
    p.update_delta( in_p.delta )
    assert np.allclose( p.rpp, in_p.rpp, EPS, EPS )
    assert np.allclose( p.rvv, in_p.rvv, EPS, EPS )
    assert np.allclose( p.psigma, in_p.psigma, EPS, EPS )

    p.state = state
    p.at_element = at_element
    p.at_turn = at_turn
    p.id = particle_id

    p.chi = in_p.chi
    p.charge_ratio = in_p.qratio
    p.s = in_p.s

