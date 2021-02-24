import argparse
import os
from helpers.config import build_config
from converters.from_sixtrack import generate_data as generate_from_sixtrack
from converters.from_pysixtrack import generate_data as generate_from_pysixtrack

if __name__ == '__main__':
    conf = build_config( "./config.toml" )
    path_to_testdata_dir = os.path.abspath( os.path.dirname( __file__ ) )
    for name, subconf in conf.items():
        if subconf.get( 'source', None ) is None or \
            subconf.get( 'input_dir', None ) is None:
            continue
        input_dir = subconf[ 'input_dir' ]
        scenario_out_dir = os.path.join( path_to_testdata_dir, name )
        scenario_in_dir  = os.path.join( scenario_out_dir, input_dir )
        if subconf[ 'source' ] == 'sixtrack':
            generate_from_sixtrack(
                name, scenario_in_dir, scenario_out_dir, conf=subconf )
        elif subconf[ 'source' ] == 'pysixtrack':
            generate_from_pysixtrack(
                name, scenario_in_dir, scenario_out_dir, conf=subconf )
        else:
            raise ValueError( f"unknown source: {subconf['source']}" )
