import os
import sys
import yaml
import json
import argparse
import pandas as pd
import numpy as np

from route_finders.aizynthfinder import AizRouteFinder
from route_finders.postera import PosRouteFinder

from analysis.aiz_pos_analyser import AizPosTemplateAnalyser

"""
1. Import the actives
2. load the RouteFinders
3. Run the template analysis
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the template analysis")
    parser.add_argument("--targets", help="The path to the actives file")
    parser.add_argument("--config", help="The path to the RetroMix configuration file")
    parser.add_argument("--nproc", help="The number of processes to use", default=1)
    parser.add_argument("--output", help="The path to the output directory")
    args = parser.parse_args()
    
    # load the configuration file
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # load actives
    with open(args.targets, 'r') as f:
        actives = f.readlines()
          
    # load the route finders
    aiz_routes = AizRouteFinder(
        configfile = config['aiz_config'],
        nproc=args.nproc,
        output_dir=args.output,
        smiles=actives
    ).find_routes()
    aiz_routes.to_hdf(os.path.join(args.output, 'aiz_routes.hdf5'), 'table')
    
    pos_routes = PosRouteFinder(
        smiles=actives,
        output=args.output,
    ).find_routes()
    
    # template analysis
    analyser = AizPosTemplateAnalyser(
        template_library=config['template_library']
    )
    
    popular_templates = analyser.find_popular_templates(aiz_routes, config['stock'])
    unused_templates = analyser.find_unused_templates(aiz_routes, pos_routes, config['stock'])
    overlooked_templates, novel_templates = analyser.find_novel_overlooked_templates(unused_templates)
    
    # save the results
    with open(os.path.join(args.output, 'popular_templates.json'), 'w') as f:
        json.dump(popular_templates, f)
        
    with open(os.path.join(args.output, 'unused_templates.json'), 'w') as f:
        json.dump(unused_templates, f)
        
    with open(os.path.join(args.output, 'overlooked_templates.json'), 'w') as f:
        json.dump(overlooked_templates, f)
        
    with open(os.path.join(args.output, 'novel_templates.json'), 'w') as f:
        json.dump(novel_templates, f)
        
    print("Analysis complete.")
    
    

