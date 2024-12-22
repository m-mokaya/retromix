import os
import sys
import pandas as pd
import numpy as np
from joblib import Parallel, delayed


sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'aizynthfinder'))
if '/data/pegasus/not-backed-up/mokaya/aizynthfinder' in sys.path:
    sys.path.remove('/data/pegasus/not-backed-up/mokaya/aizynthfinder')
for path in sys.path:
    print(path)

from route_finders.route_finder import RouteFinder
from aizynthfinder.aizynthfinder import AiZynthFinder

class AizRouteFinder(RouteFinder):
    def __init__(self, configfile, smiles, nproc, configdict=None):
        self.configfile = configfile
        self.smiles = smiles
        self.nproc = nproc
        self.configdict = configdict
       
    def process_smiles(self, smi, finder):
        finder.target_smiles = smi
        finder.prepare_tree()
        search_time = finder.tree_search()
        finder.build_routes()
        stats = finder.extract_statistics()
        stats['trees'] = finder.routes.dicts
        return stats

    def worker(self, chunk):
        if self.configdict is None:
            finder = AiZynthFinder(configfile=self.configfile)
        else:
            finder = AiZynthFinder(configdict=self.configdict)

        finder.stock.select('molport')
        finder.expansion_policy.select('uspto')

        results = []
        for smi in chunk:
            try:
                stats = self.process_smiles(smi, finder)
                results.append(stats)
            except Exception as e:
                print('Error processing %s: %s', smi, e)

        return pd.DataFrame(results)

    def find_routes(self):
        if self.nproc == 1:
            results = self.worker(self.smiles)
            return results
        else:
            chunks = self.split_smiles()
            results = Parallel(n_jobs=self.nproc)(delayed(self.worker)(chunk) for chunk in chunks)
            return pd.concat(results)

