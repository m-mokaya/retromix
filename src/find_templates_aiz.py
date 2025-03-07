import os
import sys
import yaml
import json
import argparse
import pandas as pd
import numpy as np
import dotenv
import gzip

# load environment variables
dotenv.load_dotenv()

sys.path.append(os.path.join(os.getcwd(), 'aizynthfinder'))
sys.path.append(os.path.join(os.getcwd(), 'CoPriNet'))

from analysis.aiz_pos_analyser import AizPosTemplateAnalyser
from analysis.aiz_analyser import AizTemplateAnalyser
from route_finders.aizynthfinder import AizRouteFinder
from route_finders.postera import PosRouteFinder
from optimisation.scoring import Scorer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the template analysis")
    parser.add_argument("--targets", help="The path to the actives file")
    parser.add_argument("--target_name", help="The name of the target for output files", default="target", type=str)
    parser.add_argument("--config", help="The path to the RetroMix configuration file")
    parser.add_argument("--nproc", type=int, help="The number of processes to use", default=1)
    parser.add_argument("--output", help="The path to the output directory")
    parser.add_argument("--scoring_type", help="The type of scoring to use", default='state', choices=['state', 'cost', 'coprinet', 'frequency'])
    args = parser.parse_args()
    
    # load the configuration file
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # load actives
    with open(args.targets, 'r') as f:
        actives = f.readlines()
    actives = [x.strip() for x in actives]
   
    # load stock if 'cost' metric used for idenficication
    if args.scoring_type == 'cost': 
        stock = pd.read_hdf(config['stock'], 'table')
        stock_dict = {inchi: price for inchi, price in zip(stock['inchi_key'], stock['price'])}
        
    print("Loaded actives & configuration & stock.")
    
    # check whether the routes to target have already been saved. if not, find them. 
    if os.path.isfile(os.path.join(args.output, f'{args.target_name}_aiz.hdf5')):
        aiz_tb_routes = pd.read_hdf(os.path.join(args.output, f'{args.target_name}_aiz.hdf5'), 'table')
    else:      
        # load the route finders
        aiz_tb_routes = AizRouteFinder(
            configfile = config['aizynthfinder_config'],
            nproc=args.nproc,
            smiles=actives
        ).find_routes()
        aiz_tb_routes.to_hdf(os.path.join(args.output, f'{args.target_name}_aiz.hdf5'), 'table')
    
    if os.path.isfile(os.path.join(args.output, f'{args.target_name}_tf.hdf5')):
        aiz_tf_routes = pd.read_hdf(os.path.join(args.output, f'{args.target_name}_tf.hdf5'), 'table')
    else:      
        # load the route finders
        aiz_tf_routes = AizRouteFinder(
            configfile = config['aizynthfinder_config'],
            nproc=args.nproc,
            smiles=actives
        ).find_routes()
        aiz_tf_routes.to_hdf(os.path.join(args.output, f'{args.target_name}_tf.hdf5'), 'table')
            
    aiz_tb_routes.drop_duplicates(subset=['target'], inplace=True)
    aiz_tf_routes.drop_duplicates(subset=['target'], inplace=True)
    
    with gzip.open(config['template_library'], 'r') as f:
        templates = pd.read_csv(f, sep='\t')
        
    canonical_template_library = templates['canonical_smarts'].tolist()
    
    # template analysis
    analyser = AizTemplateAnalyser(
        template_library=canonical_template_library,
        stock = stock_dict if args.scoring_type == 'cost' else None,
        scoring_type=args.scoring_type,
    )
    
    popular_templates = analyser.find_popular_templates(aiz_tb_routes)
    with open(os.path.join(args.output, f'popular_templates.json'), 'w') as f:
        json.dump(popular_templates, f, indent=4)
    
    unused_templates = analyser.find_unused_templates(aiz_tb_routes, aiz_tf_routes)
    with open(os.path.join(args.output, f'unused_templates.json'), 'w') as f:
        json.dump(unused_templates, f, indent=4)
    
    overlooked_templates, novel_templates = analyser.find_novel_overlooked_templates(unused_templates)        
    with open(os.path.join(args.output, f'overlooked_templates.json'), 'w') as f:
        json.dump(overlooked_templates, f, indent=4)
        
    with open(os.path.join(args.output, f'novel_templates.json'), 'w') as f:
        json.dump(novel_templates, f, indent=4)
        
    print("Analysis complete.")
    
    

