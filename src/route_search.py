import os
import sys
import json
import yaml
import argparse

from route_finders.aizynthfinder import AizRouteFinder

from utils import generate_aiz_configs


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Run Aizynthfinder & optimise templates')
    parser.add_argument('--config', type=str, help='Path to the config file')
    parser.add_argument('--output_dir', type=str, help='Path to the output directory')
    parser.add_argument('--type', type=str, help='Type of templates to optimise', choices=['overlooked', 'popular', 'novel'])
    parser.add_argument('--templates', type=str, help='Path to the file containing the templates')
    parser.add_argument('--optimise', type=bool, default=True, help='Optimise the templates')
    parser.add_argument('--nproc', type=int, default=1, help='Number of processors to use')
    parser.add_argument('--smiles', type=str, help='Path to the file containing the SMILES strings')
    args = parser.parse_args()
    
    with open(args.smiles, 'r') as f:
        smiles = f.readlines()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    if args.optimise:
        new_aiz_config = generate_aiz_configs(config['aizynthfinder_config'], args.output_dir, args.type)
        with open(os.path.join(args.output_dir, f'{args.type}_aiz_config.json'), 'r') as f:
            json.dump(new_aiz_config, f)
            
        optimised_aiz = AizRouteFinder(
            config_file=os.path.join(args.output_dir, f'{args.type}_aiz_config.json'),
            smiles=smiles,
            nproc=args.nproc,
        ).find_routes()
        optimised_aiz.to_hdf(os.path.join(args.output_dir, f'{args.type}_aiz_routes.hdf5'), 'table')
        
    else:
        aiz_routes = AizRouteFinder(
            config_file=config['aizynthfinder_config'],
            smiles=smiles,
            nproc=args.nproc,
        ).find_routes()
        aiz_routes.to_hdf(os.path.join(args.output_dir, 'aiz_routes.hdf5'), 'table')
    
    
    