import sixtracklib as st
import pysixtrack  as pysix
import numpy as np
from scipy.special import factorial

def calc_cbuffer_params_for_pysix_line( line, slot_size=None, conf=dict() ):
    assert isinstance( line, pysix.Line )
    if slot_size is None:
        slot_size = st.CBufferView.DEFAULT_SLOT_SIZE
    n_slots = 0
    n_objects = 0
    n_pointers = 0
    for ii, elem in enumerate( line.elements ):
        if isinstance( elem, pysix.elements.Drift ) and \
            not( conf.get( 'always_use_drift_exact', False ) ):
            n_objects += 1
            n_slots += st.st_Drift.COBJ_REQUIRED_NUM_SLOTS( slot_size )
            n_pointers += st.st_Drift.COBJ_NUM_DATAPTRS
            continue

        if isinstance( elem, pysix.elements.DriftExact ) or (
            isinstance( elem, pysix.elements.Drift ) and
            conf.get( 'always_use_drift_exact', False ) ):
            n_objects += 1
            n_slots += st.st_DriftExact.COBJ_REQUIRED_NUM_SLOTS( slot_size )
            n_pointers += st.st_DriftExact.COBJ_NUM_DATAPTRS
            continue

        assert not isinstance( elem, pysix.elements.Drift )
        assert not isinstance( elem, pysix.elements.DriftExact )

        if isinstance( elem, pysix.elements.DipoleEdge ):
            n_objects  += 1
            n_slots    += st.st_DipoleEdge.COBJ_REQUIRED_NUM_SLOTS( slot_size )
            n_pointers += st.st_DipoleEdge.COBJ_NUM_DATAPTRS
        elif isinstance( elem, pysix.elements.Cavity ):
            n_objects  += 1
            n_slots    += st.st_Cavity.COBJ_REQUIRED_NUM_SLOTS( slot_size )
            n_pointers += st.st_Cavity.COBJ_NUM_DATAPTRS
        elif isinstance( elem, pysix.elements.Multipole ):
            knl_length = len( elem.knl ) if elem.knl is not None else 0
            ksl_length = len( elem.ksl ) if elem.ksl is not None else 0
            bal_length = 2 * max( knl_length, ksl_length )
            if bal_length < 2:
                raise ValueError( "bal_length < 2" )
            max_order  = ( bal_length - 2 ) // 2
            assert max_order >= 0
            max_order  += max( conf.get( 'multipole_add_max_order', 0 ), 0 )
            n_objects  += 1
            n_pointers += st.st_Multipole.COBJ_NUM_DATAPTRS
            n_slots += st.st_Multipole.COBJ_REQUIRED_NUM_SLOTS(
                max_order, slot_size )
        elif isinstance( elem, pysix.elements.LimitRect ):
            n_objects += 1
            n_pointers += st.st_LimitRect.COBJ_NUM_DATAPTRS
            n_slots += st.st_LimitRect.COBJ_REQUIRED_NUM_SLOTS( slot_size )
        elif isinstance( elem, pysix.elements.LimitEllipse ):
            n_objects += 1
            n_pointers += st.st_LimitEllipse.COBJ_NUM_DATAPTRS
            n_slots += st.st_LimitEllipse.COBJ_REQUIRED_NUM_SLOTS( slot_size )
        elif isinstance( elem, pysix.elements.LimitRectEllipse ):
            n_objects += 1
            n_pointers += st.st_LimitRectEllipse.COBJ_NUM_DATAPTRS
            n_slots += st.st_LimitRectEllipse.COBJ_REQUIRED_NUM_SLOTS( slot_size )
        elif isinstance( elem, pysix.elements.RFMultipole ):
            knl_length = len( elem.knl ) if elem.knl is not None else 0
            ksl_length = len( elem.ksl ) if elem.ksl is not None else 0
            pn_length  = len( elem.pn, ) if elem.pn  is not None else 0
            ps_length  = len( elem.ps, ) if elem.ps  is not None else 0
            assert pn_length == knl_length
            assert ps_length == ksl_length
            bal_length = 2 * max( knl_length, ksl_length )
            if bal_length < 2:
                raise ValueError( "bal_length < 2" )
            max_order  = ( bal_length - 2 ) // 2
            assert max_order >= 0
            max_order  += max( conf.get( 'rf_multipole_add_max_order', 0 ), 0 )
            n_objects  += 1
            n_pointers += st.st_RFMultipole.COBJ_NUM_DATAPTRS
            n_slots += st.st_RFMultipole.COBJ_REQUIRED_NUM_SLOTS(
                max_order, slot_size )
        elif isinstance( elem, pysix.elements.SRotation ):
            n_objects += 1
            n_pointers += st.st_SRotation.COBJ_NUM_DATAPTRS
            n_slots += st.st_SRotation.COBJ_REQUIRED_NUM_SLOTS( slot_size )
        elif isinstance( elem, pysix.elements.XYShift ):
            n_objects += 1
            n_pointers += st.st_XYShift.COBJ_NUM_DATAPTRS
            n_slots += st.st_XYShift.COBJ_REQUIRED_NUM_SLOTS( slot_size )
        elif isinstance( elem, pysix.be_beamfields.spacecharge.SCCoasting ):
            n_objects += 1
            n_pointers += st.st_SCCoasting.COBJ_NUM_DATAPTRS
            n_slots += st.st_SCCoasting.COBJ_REQUIRED_NUM_SLOTS( slot_size )
        elif isinstance( elem, pysix.be_beamfields.spacecharge.SCQGaussProfile ):
            n_objects += 1
            n_pointers += st.st_SCQGaussProfile.COBJ_NUM_DATAPTRS
            n_slots += st.st_SCQGaussProfile.COBJ_REQUIRED_NUM_SLOTS( slot_size )
        else:
            print( f"element not converted at pos = {ii}: {elem}" )
    return n_slots, n_objects, n_pointers

def pysix_line_to_cbuffer( line, cbuffer, conf=dict() ):
    assert isinstance( cbuffer, st.CBufferView )
    assert isinstance( line, pysix.Line )

    for elem in line.elements:
        if isinstance( elem, pysix.elements.Drift ) and \
            not( conf.get( 'always_use_drift_exact', False ) ):
            cobj_elem = st.st_Drift( cbuffer, elem.length )
            continue

        if isinstance( elem, pysix.elements.DriftExact ) or (
            isinstance( elem, pysix.elements.Drift ) and
            conf.get( 'always_use_drift_exact', False ) ):
            cobj_elem = st.st_DriftExact( cbuffer, elem.length )
            continue

        assert not isinstance( elem, pysix.elements.Drift )
        assert not isinstance( elem, pysix.elements.DriftExact )

        if isinstance( elem, pysix.elements.DipoleEdge ):
            cobj_elem = st.st_DipoleEdge(
                cbuffer, elem.h, elem.e1, elem.hgap, elem.fint )
        elif isinstance( elem, pysix.elements.Cavity ):
            cobj_elem = st.st_Cavity(
                cbuffer, elem.voltage, elem.frequency, elem.lag )
        elif isinstance( elem, pysix.elements.LimitRect ):
            cobj_elem = st.st_LimitRect(
                cbuffer, elem.min_x, elem.max_x, elem.min_y, elem.max_y )
        elif isinstance( elem, pysix.elements.LimitEllipse ):
            cobj_elem = st.st_LimitEllipse(
                cbuffer, elem.a * elem.a, elem.b * elem.b )
        elif isinstance( elem, pysix.elements.LimitRectEllipse ):
            cobj_elem = st.st_LimitRectEllipse( cbuffer, elem.max_x, elem.max_y,
                elem.a * elem.a, elem.b * elem.b )
        elif isinstance( elem, pysix.elements.Multipole ):
            knl_length = len( elem.knl ) if elem.knl is not None else 0
            ksl_length = len( elem.ksl ) if elem.ksl is not None else 0
            bal_length = 2 * max( knl_length, ksl_length )
            if bal_length < 2:
                raise ValueError( "bal_length < 2" )

            max_order  = ( bal_length - 2 ) // 2
            assert max_order >= 0
            max_order += max( conf.get( 'multipole_add_max_order', 0 ), 0 )
            bal = np.zeros( bal_length, dtype=np.float64 )
            if knl_length > 0:
                for ii, knl_value in enumerate( elem.knl ):
                    bal[ 2 * ii ] = knl_value / factorial( ii )
            if ksl_length > 0:
                for ii, ksl_value in enumerate( elem.ksl ):
                    bal[ 2 * ii + 1 ] = ksl_value / factorial( ii )
            cobj_elem = st.st_Multipole(
                cbuffer, elem.length, elem.hxl, elem.hyl, bal )
        elif isinstance( elem, pysix.elements.SRotation ):
            angle_rad = elem.angle * np.pi / np.float64( 180.0 )
            cobj_elem = st.st_SRotation( cbuffer, angle_rad )
        elif isinstance( elem, pysix.elements.XYShift ):
            cobj_elem = st.st_XYShift( cbuffer, elem.dx, elem.dy )
        elif isinstance( elem, pysix.be_beamfields.spacecharge.SCCoasting ):
            cobj_elem = st.st_SCCoasting( cbuffer, elem.number_of_particles,
                elem.circumference, elem.sigma_x, elem.sigma_y, elem.length,
                    elem.x_co, elem.y_co, elem.min_sigma_diff, elem.enabled )
        elif isinstance( elem, pysix.be_beamfields.spacecharge.SCQGaussProfile ):
            cobj_elem = st.st_SCQGaussProfile( cbuffer, elem.number_of_particles,
                elem.bunchlength_rms, elem.sigma_x, elem.sigma_y, elem.length,
                    elem.x_co, elem.y_co, elem.min_sigma_diff, elem.q_parameter,
                        elem.enabled )
        else:
            print( f"element at position {ii} in line not converted: {elem}" )
    return

def pysix_particle_to_pset( in_p, pset, index, state=None, at_element=None,
    at_turn=None, particle_id=None, conf=dict() ):
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


def pysix_particle_to_single_particle( in_p, p, state=None, at_element=None,
    at_turn=None, particle_id=None, conf=dict() ):
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
    p.x       = in_p.x
    p.y       = in_p.y
    p.px      = in_p.px
    p.py      = in_p.py
    p.zeta    = in_p.zeta

    EPS = np.float64( 1e-12 )
    p.update_delta( in_p.delta )
    assert np.allclose( p.rpp, in_p.rpp, EPS, EPS )
    assert np.allclose( p.rvv, in_p.rvv, EPS, EPS )
    assert np.allclose( p.psigma, in_p.psigma, EPS, EPS )

    p.state        = state
    p.at_element   = at_element
    p.at_turn      = at_turn
    p.id           = particle_id
    p.chi          = in_p.chi
    p.charge_ratio = in_p.qratio
    p.s            = in_p.s

