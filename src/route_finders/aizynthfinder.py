import os
import sys

import pandas as pd
import numpy as np
import multiprocessing as mp

from src.route_finders.route_finder import RouteFinder

from aizynthfinder.aizynthfinder import AiZynthFinder

class AizRouteFinder(RouteFinder):
    def __init__(self, configfile, smiles, nproc, output_dir):
        self.configfile = configfile
        self.smiles = smiles
        self.nproc = nproc
        self.output = output_dir
        
    def worker(self, chunk):
        """
        Find retrosynthetic routes for a list of SMILES strings
        
        :param smiles (list): list of SMILES strings

        :return dict: dictionary with the results
        """

        finder = AiZynthFinder(configfile = self.configfile)
        finder.stock.select('zinc')  # select an appriopriate stock
        finder.expansion_policy.select('uspto')     # select an appriopriate expansion policy

        results = []

        for smi in chunk:
            finder.target_smiles = smi
            finder.prepare_tree()
            search_time = finder.tree_search()
            finder.build_routes()
            stats = finder.extract_statistics()

            solved_str = 'is_solved' if stats['is_solved'] else 'is not solved'
            print(f'Done with {smi} in {search_time:.3} s and {solved_str}')

            smi_results = {}
            
            for key, value in stats.items():
                if key in smi_results:
                    smi_results[key].append(value)
                else:
                    smi_results[key] = [value]
            
            results.append(pd.DataFrame(smi_results))
        
        return pd.concat(results)
        
    def find_routes(self): 
        """
        Find retrosynthetic routes for the target molecules.
        
        :return pd.DataFrame: A pandas DataFrame containing the results in Aiz Format.
        """
        with mp.Pool(self.nproc) as pool:
            results = pool.map(self.worker, self.split_smiles())
        return pd.concat(results)


    