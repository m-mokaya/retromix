import os
import sys
import yaml
import json
import argparse
import dotenv
import gzip

import pandas as pd

dotenv.load_dotenv()

retromix_path = os.getenv("RETROMIX_PATH")

sys.path.append(os.path.join(retromix_path, "aizynthfinder"))
sys.path.append(os.path.join(retromix_path, "CoPriNet"))

from pricePrediction.predict.predict import GraphPricePredictor

from route_finders.aizynthfinder import AizRouteFinder
from optimisation.scoring import Scorer
import src.utils as utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the template analysis")
    parser.add_argument("--targets", help="The path to the target molecules file", required=True)
    parser.add_argument("--config", help="The path to the RetroMix configuration file", required=True)
    parser.add_argument("--output_dir", help="The path to the output directory", required=True)
    parser.add_argument("--nproc", type=int, help="The number of processes to use", default=1)
    parser.add_argument("--ngpus", type=int, help="The number of GPUs to use", default=0)
    args = parser.parse_args()
    
    
    """
    Loading neccesary compodent for template identifiaction. 
    """
    
    # load the rm config file
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    if config['tf_model'] == "postera":
        from route_finders.postera import PosRouteFinder
    
    # load the target molecules
    with open(args.targets, 'r') as f:
        targets = f.readlines()
    targets = [x.strip() for x in targets]
    
    # load the stock dictionary
    if config['scoring_type'] == "cost":
        stock = pd.read_hdf(os.path.join(retromix_path, config['stock']), "table")   
        stock_dict = {inchi: price for inchi, price in zip(stock['inchi_key'], stock['price'])} 
    elif config['scoring_type'] == "coprinet":
        predictor = GraphPricePredictor(
            model_path=os.path.join(retromix_path, config['coprinet_model_path']),
            n_cpus=args.nproc,
            n_gpus=args.ngpus,
        )


    # load the templates
    with gzip.open(os.path.join(retromix_path, config['template_library']), 'r') as f:
        templates = pd.read_csv(f, sep='\t')
    canonical_templates = templates['canonical_smarts'].tolist()
    
    print("Loaded configuration, target molecules, stock, and templates.")
    
    
    
    """
    Running the retrosynthesis search using TB and TF models
    """
    
    # load the TB AiZ routes
    if os.path.isfile(os.path.join(args.output_dir, 'aiz_routes.hdf5')):
        tb_routes = pd.read_hdf(os.path.join(args.output_dir, 'aiz_routes.hdf5'), 'table')
    else:      
        tb_routes = AizRouteFinder(
            configfile = os.path.join(retromix_path, config['aizynthfinder_config']),
            nproc=args.nproc,
            smiles=targets
        ).find_routes()
        tb_routes.to_hdf(os.path.join(args.output_dir, 'aiz_routes.hdf5'), 'table')
        
    # load the TF routes
    if config['tf_model'] == "postera":
        if os.path.isfile(os.path.join(args.output_dir, 'pos_routes_aiz_format.json')):
            with open(os.path.join(args.output_dir, 'pos_routes_aiz_format.json'), 'r') as f:
                tf_routes = json.load(f)
        else:      
            tf_routes = PosRouteFinder(
                smiles=targets,
                output=args.output_dir,
                api_key=os.getenv('POSTERA_API_KEY')
            ).find_routes()
    elif config['tf_model'] == "aizynthfinder":
        raise NotImplementedError("The AiZynthFinder model is not yet implemented.")
            
    print("Loaded TB and TF routes.")
    


    """
    Finding the popular, unused, overlooked, and novel templates    
    """
    
    if config['tf_model'] == 'postera':
        from analysis.aiz_pos_analyser import AizPosTemplateAnalyser
        
        analyser = AizPosTemplateAnalyser(
            template_library=canonical_templates,
            stock=stock_dict if config['scoring_type'] == 'cost' else None,
            scoring_type=config['scoring_type'],
            predictor = predictor if config['scoring_type'] == 'coprinet' else None
        )
        
    popular_templates = analyser.find_popular_templates(tb_routes)
    with open(os.path.join(args.output_dir, f'popular_templates.json'), 'w') as f:
        json.dump(popular_templates, f, indent=4)
        
    unused_templates = analyser.find_unused_templates(tb_routes, tf_routes)
    with open(os.path.join(args.output_dir, f'unused_templates.json'), 'w') as f:
        json.dump(unused_templates, f, indent=4)
        
    overlooked_templates, novel_templates = analyser.find_novel_overlooked_templates(unused_templates)        
    with open(os.path.join(args.output_dir, f'overlooked_templates.json'), 'w') as f:
        json.dump(overlooked_templates, f, indent=4)
        
    with open(os.path.join(args.output_dir, f'novel_templates.json'), 'w') as f:
        json.dump(novel_templates, f, indent=4)
        
    print("Template finder complete.")
    
    
    """
    Re-searching with the optimised templates.
    """
    
    novel_aiz_config = utils.generate_aiz_configs(config['aizynthfinder_config'], "novel", os.path.join(args.output_dir, f'novel_templates.json'))
    overlooked_aiz_config = utils.generate_aiz_configs(config['aizynthfinder_config'], "overlooked", os.path.join(args.output_dir, f'overlooked_templates.json'))
    popular_aiz_config = utils.generate_aiz_configs(config['aizynthfinder_config'], "popular", os.path.join(args.output_dir, f'popular_templates.json'))
    
    with open(os.path.join(args.output_dir, 'novel_aiz_config.yml'), 'w') as f:
        yaml.dump(novel_aiz_config, f)
    with open(os.path.join(args.output_dir, 'overlooked_aiz_config.yml'), 'w') as f:
        yaml.dump(overlooked_aiz_config, f)
    with open(os.path.join(args.output_dir, 'popular_aiz_config.yml'), 'w') as f:
        yaml.dump(popular_aiz_config, f)
    
    for opt_type in ["novel", "overlooked", "popular"]:
        print(f"Optimising {opt_type} templates...")
        optimised_aiz = AizRouteFinder(
            configfile=os.path.join(args.output_dir, f'{opt_type}_aiz_config.yml'),
            smiles=targets,
            nproc=args.nproc,
        ).find_routes()
        optimised_aiz.to_hdf(os.path.join(args.output_dir, f'{opt_type}_aiz_routes.hdf5'), 'table')
        

    print("Optimisation complete.")
    
    
    
    
    
        
    
        
    
        
        
        
        
    
    
    
    
    
    
    
    
    