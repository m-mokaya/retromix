import os
import sys
import json
import requests
import numpy as np
import pandas as pd
from multiprocessing import Pool

from src.route_finders.route_finder import RouteFinder
from rdchiral.template_extractor import extract_from_reaction # type: ignore

POSTERA_API_KEY = os.getenv("POSTERA_API_KEY")

class PosRouteFinder(RouteFinder):
    def __init__(self, smiles, output, maxSearchDepth=4, ignore_zero_steps=False,
                 catalogues=["molport"], api_key=POSTERA_API_KEY, verbose=True):   
        
        self.smiles = smiles
        self.url = "https://api.postera.ai/api/v1/retrosynthesis/batch/"
        self.api_key = api_key
        
        # search parameters
        self.maxSearchDepth = maxSearchDepth
        self.ignore_zero_steps = ignore_zero_steps
        self.catalogues = catalogues
        
        self.output = output
        
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
        
    def convert_results_to_aiz_format(self, data):
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
                'in_stock': molecule_dict[smiles]['isBuildingBlock'],
                'children': []
            }

            for reaction in reactions:
                if reaction['productSmiles'] == smiles:
                    template_data = extract_from_reaction(reaction)  # Call generate_templates to get the SMARTS string
                    reaction_node = {
                        'type': 'reaction',
                        'hide': False,
                        'smiles': reaction['reactantSmiles'],  # No specific SMILES for the reaction itself
                        'is_reaction': True,
                        'metadata': {
                            'name': reaction['name'],
                            'template': template_data['reaction_smarts'],  # Add SMARTS string to metadata
                            'classification': reaction['name'],
                            'policy_name': 'postera',
                        },
                        'children': [build_tree(reactant, reactions) for reactant in reaction['reactantSmiles']]
                    }
                    node['children'].append(reaction_node)
            return node
        
        # Identify the root molecule (final product)
        root_smiles = None
        product_smiles_set = {reaction['productSmiles'] for reaction in data['reactions']}
        reactant_smiles_set = {reactant for reaction in data['reactions'] for reactant in reaction['reactantSmiles']}
        root_candidates = product_smiles_set - reactant_smiles_set
        if len(root_candidates) == 1:
            root_smiles = list(root_candidates)[0]
        else:
            raise ValueError("Unable to determine unique root molecule")

        # Build the tree starting from the root molecule
        return build_tree(root_smiles, data['reactions']) 
        
        
    def find_routes(self):
        """
        Finds retrosynthetic routes for the target molecules.

        :param filename: The name of the file to save the results to.
        :return: A pandas DataFrame containing the results.
        """
        
        chunks = self.split_smiles() # split the smiles into chunks for parallel processing
        for chunk in chunks:
            with Pool(2) as p: # use a pool of 2 processes (posterai api has a limit of 2 concurrent requests)
                results = p.map(self.retrosynthesis_search, chunk)
        aiz_format_results = []
        for smiles, routes in results.items():
            route_data = routes['routes']
            pathways = [self.convert_results_to_aiz_format(route) for route in route_data]
            aiz_format_results.append({smiles: pathways})
        
        with open(os.path.join(self.output, "pos_routes.json"), 'w') as f:
            json.dump(aiz_format_results, f)            
        
        return aiz_format_results
            
                
            
            
            
            
        
        
        