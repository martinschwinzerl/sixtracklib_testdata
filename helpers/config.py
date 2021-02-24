import toml
import sixtracklib as st

def build_config( path_to_config_file=None, config_str=None ):
    conf = dict()
    if path_to_config_file is not None:
        with open( path_to_config_file, "r" ) as f_in:
            temp = toml.load( f_in )
    elif config_str is not None and isinstance( config_str, str ):
        temp = toml.loads( config_str )

    if not isinstance( temp, dict ):
        return conf

    default_conf = temp.get( 'default', {} )

    if not 'cbuffer_norm_base_addr' in default_conf:
        default_conf[ 'cbuffer_norm_base_addr' ] = 4096

    if not 'always_use_drift_exact' in default_conf:
        default_conf[ 'always_use_drift_exact' ] = False

    if not 'make_elem_by_elem_data' in default_conf:
        default_conf[ 'make_elem_by_elem_data' ] = True

    if 'make_demotrack_data' in default_conf:
        default_conf[ 'make_demotrack_data' ] &= st.Demotrack_enabled()
    else:
        default_conf[ 'make_demotrack_data' ] = st.Demotrack_enabled()

    if not 'make_sixtrack_sequ_by_sequ' in default_conf:
        default_conf[ 'make_sixtrack_sequ_by_sequ' ] = True

    if 'scenario' in temp:
        for name, subconf in temp[ 'scenario' ].items():
            conf[ name ] = {}
            if not( 'source' in subconf and 'input_dir' in subconf and
                     ( subconf[ 'source' ] == 'sixtrack' or
                       subconf[ 'source' ] == 'pysixtrack' ) ):
                conf[ name ].update( { 'source': None, 'input_dir': None } )
                continue
            conf[ name ].update( default_conf )
            conf[ name ].update( subconf )
    return conf
