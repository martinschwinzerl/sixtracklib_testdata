import os
import toml
import argparse
from converters.from_sixtrack import generate_sixtrack
from converters.from_pysixtrack import generate_pysixtrack

if __name__ == '__main__':
    path_current_dir = os.path.dirname( __file__ )
    path_to_config_file = os.path.join(
        os.path.dirname( __file__ ), "config.toml" )

    default_conf = dict()
    conf = dict()
    if os.path.isfile( path_to_config_file ):
        c = toml.load( path_to_config_file )
        if 'common' in c:
            default_conf[ 'common' ] = dict()
            default_conf[ 'common' ][ 'normalised_base_addr' ] = int(
                c['common'].get( 'normalised_base_addr', 4096 ) )
            default_conf[ 'common' ][ 'always_exact_drift' ] = \
                c[ 'common' ].get( 'always_exact_drift', False )
            default_conf[ 'common' ][ 'source' ] = None

        for scenario_name, subconf in c.items():
            if scenario_name == 'common':
                continue

            path_to_scenario_dir = os.path.join(
                path_current_dir, scenario_name )

            if not os.path.isdir( path_to_scenario_dir ):
                continue

            conf[ scenario_name ] = dict()
            conf[ scenario_name ].update( default_conf )
            if 'source' in subconf:
                if subconf[ 'source' ] == 'sixtrack' or \
                    subconf[ 'source' ] == 'madx' or \
                    subconf[ 'source' ] == 'pysixtrack':
                    conf[ scenario_name ][ 'source' ] = subconf[ 'source' ]

            conf[ scenario_name ][ 'always_exact_drift' ] = subconf.get(
                'always_exact_drift', default_conf.get(
                        'always_exact_drift' ) )

            base_addr = default_conf.get( 'normalised_base_addr', 4096 )
            conf[ scenario_name ][ 'normalised_base_addr' ] = subconf.get(
                'normalised_base_addr', base_addr )


    for scenario_name, subconf in conf.items():
        source = subconf.get( 'source', None )
        if source is None:
            continue
        if source == 'sixtrack':
            generate_sixtrack( scenario_name, conf=subconf )
        elif source == 'pysixtrack':
            generate_pysixtrack( scenario_name, conf=subconf )
