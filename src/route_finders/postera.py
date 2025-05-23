import os
import sys
import json
import requests
import numpy as np
import pandas as pd
from multiprocessing import Pool

from src.route_finders.route_finder import RouteFinder
from rdchiral.template_extractor import extract_from_reaction # type: ignore
from analysis.generate_templates import generate_templates

from rxnutils.chem.reaction import ChemicalReaction
from rxnmapper import RXNMapper



class PosRouteFinder(RouteFinder):
    def __init__(self, smiles, output, maxSearchDepth=4, ignore_zero_steps=False,
                 catalogues=["molport"], api_key=None, verbose=True):   
        
        self.smiles = smiles
        self.url = "https://api.postera.ai/api/v1/retrosynthesis/batch/"
        
        # search parameters
        self.maxSearchDepth = maxSearchDepth
        self.ignore_zero_steps = ignore_zero_steps
        self.catalogues = catalogues
        
        if api_key is None:
            raise ValueError("Please provide a Postera API key")
        
        self.api_key = api_key
        self.output = output
        
        self.rxn_mapper = RXNMapper()
        
    def split_smiles(self):
        """
        Splits the list of SMILES strings into chunks for parallel processing.

        :return: A list of chunks, where each chunk is a list of ten SMILES strings.
        """
        chunks = [self.smiles[i:i + 10] for i in range(0, len(self.smiles), 10)]
        return chunks
    
    def prepateBatchQueryData(self, selected_smiles):
        """
        Prepare the data for a batch query.
        
        :param selected_smiles (list): list of SMILES strings
        :return dict: dictionary with the data
        """
        data = {}
        data["maxSearchDepth"] = self.maxSearchDepth
        data["catalogs"] = self.catalogues
        data['smilesList'] = selected_smiles
        return data

    def retrosynthesis_search(self, smiles):
        """
        Search for retrosynthetic routes for a list of SMILES strings.
        
        :param smiles (list): list of SMILES strings
        :return dict: dictionary with the results
        """
        
        data = self.prepateBatchQueryData(smiles)
        response = requests.post(self.url, json=data, headers={'X-API-KEY': self.api_key})

        if response.status_code == 200: # check if the request was successful
            output = {}
            for smi, result in zip(smiles, response.json()['results']):
                output[smi] = result
            return output
        else:
            print("Error! Status code:", response.status_code, "Message:", response.text)
            return None
        
    def convert_results_to_aiz_format(self, data, target_smiles):
        """ 
        Convert the results to a format that can be saved to a file.
        
        :param data: dictionary with the results
        :retrun pd.DataFrame: dataframe with the results
        """
        molecule_dict = {mol['smiles']: mol for mol in data['molecules']}   # Create a mapping of molecules through the tree
        
        # Function to recursively build the tree
        def build_tree(smiles, reactions):
            node = {
                'type': 'mol',
                'hide': False,
                'smiles': smiles,
                'is_chemical': True,
                'in_stock': molecule_dict[smiles]['isBuildingBlock'] if smiles in molecule_dict else False,
                'children': [],
            }

            for reaction in reactions:
                if reaction['productSmiles'] == smiles:
                    reactants_smi = '.'.join(reaction['reactantSmiles'])
                    rsmi = f"{reactants_smi}>>{reaction['productSmiles']}"
                    mapped_rxn = self.rxn_mapper.get_attention_guided_atom_maps([rsmi])[0]['mapped_rxn']
                    # template_data = generate_templates(reaction)  # Call generate_templates to get the SMARTS string
                    retro_template = ChemicalReaction(smiles=mapped_rxn).generate_reaction_template()[1].smarts
                    reaction_node = {
                        'type': 'reaction',
                        'hide': False,
                        'smiles': f"TEST: {rsmi}",  # No specific SMILES for the reaction itself
                        'is_reaction': True,
                        'postera_reaction': reaction,
                        'metadata': {
                            'name': reaction['name'],
                            'template': retro_template,  # Add SMARTS string to metadata
                            'classification': reaction['name'],
                            'policy_name': 'postera',
                        },
                        'children': [build_tree(reactant, reactions) for reactant in reaction['reactantSmiles']]
                    }
                    node['children'].append(reaction_node)
            return node
        
        # Identify the root molecule (final product)
        root_smiles = target_smiles
        # product_smiles_set = {reaction['productSmiles'] for reaction in data['reactions']}
        # reactant_smiles_set = {reactant for reaction in data['reactions'] for reactant in reaction['reactantSmiles']}
        # root_candidates = product_smiles_set - reactant_smiles_set
        # if len(root_candidates) == 1:
        #     root_smiles = list(root_candidates)[0]
        # else:
        #     raise ValueError("Unable to determine unique root molecule")

        # Build the tree starting from the root molecule
        return build_tree(root_smiles, data['reactions']) 
        
        
    def find_routes(self):
        """
        Finds retrosynthetic routes for the target molecules.

        :param filename: The name of the file to save the results to.
        :return: A pandas DataFrame containing the results.
        """
        
        chunks = self.split_smiles() # split the smiles into chunks for parallel processing
        with Pool(2) as p:
            results = p.map(self.retrosynthesis_search, chunks)
        
        with open(os.path.join(self.output, "pos_routes.json"), 'w') as f:
            json.dump(results, f, indent=4)
        
        # flatten the list of dictionaries to a single dictionary
        flat_results = {k: v for d in results for k, v in d.items()}

        aiz_format_results = {}
        for smiles, routes in flat_results.items():
            route_data = routes['routes']
            pathways = [self.convert_results_to_aiz_format(route, smiles) for route in route_data]
            aiz_format_results[smiles] = pathways
        
        with open(os.path.join(self.output, "pos_routes_aiz_format.json"), 'w') as f:
            json.dump(aiz_format_results, f, indent=4)            
        
        return aiz_format_results
            
                
            
            
            
            
        
        
        