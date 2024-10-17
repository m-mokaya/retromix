import os
import sys
import pandas as pd
import numpy as np
import yaml

from route_finders import RouteFinder
from route_finders.aizynthfinder import AizRouteFinder
from route_finders.postera import PosRouteFinder
from analysis.template_analyser import TemplateAnalyser
from analysis.aiz_pos_analyser import AizPosTemplateAnalyser
from src import utils


if __name__ == "__main__":
    
    # load data
    with open("data/rf_config.yml") as file:
        config = yaml.safe_load(file)
    
    with open(config['template_library'], 'r') as file:
        template_library = file.readlines()
        
    with open(config['targets'], 'r') as file:
        targets = file.readlines()
    targets = [target.strip() for target in targets]
    print(f"Loaded {len(targets)} targets")
    
    stock_file = config['stock']
    stock = pd.read_hdf(stock_file, 'table')
    stock = dict(zip(stock['inchi_key'], stock['price']))
    
    aiz_data = AizRouteFinder(
        configfile=config['aizynthfinder_config'],
        smiles=targets,
        nproc=config['nproc'],
        output_dir=config['output_dir']
    ).find_routes()
    
    pos_data = PosRouteFinder(
        smiles=targets,
        api_key=os.environ['POSTERA_API_KEY'],
        filename=config['output_dir'] + '/pos_results.json'
    ).find_routes()
    

    # inialize template analyser
    analyser = TemplateAnalyser(template_library)
    aiz_pos_analyser = AizPosTemplateAnalyser(template_library)
    
    # find popular, overlooked and novel templates
    popular_templates = aiz_pos_analyser.find_popular_templates(aiz_data, stock)
    unused_templates = aiz_pos_analyser.find_unused_templates(aiz_data, pos_data, stock)  
    overlooked_templates, novel_templates = analyser.find_novel_overlooked_templates(unused_templates)
      
    print(f"We have found popular: {len(popular_templates)}, overlooked: {len(overlooked_templates)} and novel: {len(novel_templates)} templates")
    
    # generate new aiz config files
    popular_config = utils.generate_aiz_configs(config['aizynthfinder_config'], config['output_dir'], 'popular')
    overlooked_config = utils.generate_aiz_configs(config['aizynthfinder_config'], config['output_dir'], 'overlooked')
    novel_config = utils.generate_aiz_configs(config['aizynthfinder_config'], config['output_dir'], 'novel')
    
    # re-run aizynthfinder with new config files
    
    aiz_popular = AizRouteFinder(
        configfile=config['output_dir'] + '/config_popular.yml',
        smiles=targets,
        nproc=config['nproc'],
        output_dir=config['output_dir']
    ).find_routes()
    
    aiz_overlooked = AizRouteFinder(
        configfile=config['output_dir'] + '/config_overlooked.yml',
        smiles=targets,
        nproc=config['nproc'],
        output_dir=config['output_dir']
    ).find_routes() 
    
    aiz_novel = AizRouteFinder(
        configfile=config['output_dir'] + '/config_novel.yml',
        smiles=targets,
        nproc=config['nproc'],
        output_dir=config['output_dir']
    ).find_routes()
    
    aiz_popular.to_hdf(os.path.join(config['output_dir'], 'aiz_popular.hdf5'), key='table')
    aiz_overlooked.to_hdf(os.path.join(config['output_dir'], 'aiz_overlooked.hdf5'), key='table')
    aiz_novel.to_hdf(os.path.join(config['output_dir'], 'aiz_novel.hdf5'), key='table')
    
    print("Done.")
    
    
    
    
    
    
    
    
    
    

    