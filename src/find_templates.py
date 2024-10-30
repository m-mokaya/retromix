import os
import sys
import yaml
import json
import argparse
import pandas as pd
import numpy as np
import dotenv

# load environment variables
dotenv.load_dotenv()

from analysis.aiz_pos_analyser import AizPosTemplateAnalyser
from route_finders.aizynthfinder import AizRouteFinder
from route_finders.postera import PosRouteFinder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the template analysis")
    parser.add_argument("--targets", help="The path to the actives file")
    parser.add_argument("--config", help="The path to the RetroMix configuration file")
    parser.add_argument("--nproc", type=int, help="The number of processes to use", default=1)
    parser.add_argument("--output", help="The path to the output directory")
    args = parser.parse_args()
    
    # load the configuration file
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # load actives
    with open(args.targets, 'r') as f:
        actives = f.readlines()
    actives = [x.strip() for x in actives]
    
    stock = pd.read_hdf(config['stock'], 'table')
    stock_dict = {inchi: price for inchi, price in zip(stock['inchi_key'], stock['price'])}
        
    print("Loaded actives & configuration & stock.")
          
    # # load the route finders
    # aiz_routes = AizRouteFinder(
    #     configfile = config['aizynthfinder_config'],
    #     nproc=args.nproc,
    #     smiles=actives
    # ).find_routes()
    # aiz_routes.to_hdf(os.path.join(args.output, 'aiz_routes.hdf5'), 'table')
    aiz_routes = pd.read_hdf(os.path.join(args.output, 'aiz_routes.hdf5'), 'table')
    
    # pos_routes = PosRouteFinder(
    #     smiles=actives,
    #     output=args.output,
    #     api_key=os.getenv('POSTERA_API_KEY')
    # ).find_routes()
    with open(os.path.join(args.output, 'pos_routes.json'), 'r') as f:
        pos_routes = json.load(f)
    
    # template analysis
    analyser = AizPosTemplateAnalyser(
        template_library=config['template_library']
    )
    
    popular_templates = analyser.find_popular_templates(aiz_routes, stock_dict)
    with open(os.path.join(args.output, 'popular_templates.json'), 'w') as f:
        json.dump(popular_templates, f, indent=4)
    
    unused_templates = analyser.find_unused_templates(aiz_routes, pos_routes, stock_dict)
    with open(os.path.join(args.output, 'unused_templates.json'), 'w') as f:
        json.dump(unused_templates, f, indent=4)
    
    overlooked_templates, novel_templates = analyser.find_novel_overlooked_templates(unused_templates)        
    with open(os.path.join(args.output, 'overlooked_templates.json'), 'w') as f:
        json.dump(overlooked_templates, f, indent=4)
        
    with open(os.path.join(args.output, 'novel_templates.json'), 'w') as f:
        json.dump(novel_templates, f, indent=4)
        
    print("Analysis complete.")
    
    

