import os
import sys
import json
import argparse
import pandas as pd
import numpy as np

import multiprocessing
from functools import partial

from coprinet.pricePrediction.nets.netsGraph import PricePredictorModule
from coprinet.pricePrediction.predict.predict import GraphPricePredictor

from aizynthfinder.reactiontree import ReactionTree

class OptimisationScorer:
    def __init__(self, predictor, args):
        self.predictor = predictor
        self.args = args
        self.cost_cache = {}
        
        # Set the tree cost function
        if self.use_coprinet == True:
            self.tree_cost = self.coprinet_tree_cost
        else:
            if args.stock:
                self.stock = args.stock
                self.tree_cost = self.stock_library_tree_cost
            else:
                raise ValueError('No stock library specified. Please add compatile stock library')
        
    
    def coprinet_tree_cost(self, tree):
        """
        Calculate the cost of a tree using coprinet model to predict the cost of the molecules
        
        :param tree: the tree to calculate the cost for in AiZ reaction dict format.
        :return: the cost of the tree
        """
        rxn = ReactionTree.from_dict(tree)
        tree_id = rxn.hash_key()
        if tree_id not in self.cost_cache:
            leaf_costs = sum(predictor.predictListOfSmiles([leaf.smiles for leaf in rxn.leafs()]))
            total_cost = 0.7 * leaf_costs + 0.15 * len(list(rxn.leafs())) + 0.15 * len(list(rxn.reactions()))
            self.cost_cache[tree_id] = total_cost
        return self.cost_cache[tree_id]
    
    def stock_library_tree_cost(self, tree):
        """
        Calculate the cost of a tree using MolPort stock library
        
        :param tree: the tree to calculate the cost for
        :return: the cost of the tree
        """
        raise NotImplementedError('Stock library cost calculation not implemented')
    
    def compare_opt_performance(self, routes_1, routes_2):
        """
        calculates the POS terms to assess post optimisation performance. 
        The POS terms are:
        - Difference score: the fraction of targets where the optimised approach found a better solution
        - Extra solved score: the fraction of targets solved by the optimised approach that were not solved by the standard approach
        - Num solved score: a logarithmic curve mapping the number of extra solved targets to a value between 0 and 1
        
        :param routes_1: the standard approach routes
        :param routes_2: the optimised approach routes
        :return: the POS terms
        """
        # Filter solved routes
        r1_solved = self._remove_unsolved_routes(routes_1)
        r2_solved = self._remove_unsolved_routes(routes_2)
        
        # Find targets solved by both approaches
        both_solved_smiles = set(r1_solved['target']).intersection(r2_solved['target'])
        print('Both solved:', len(both_solved_smiles), flush=True)
        
        org_solved_costs = []
        difference_score = 0
        
        r1_costs = {smi: [self.tree_cost(tree) 
                        for tree in r1_solved[r1_solved['target'] == smi]['trees'].values[0][0]]
                    for smi in both_solved_smiles}
        
        r2_costs = {smi: [self.tree_cost(tree) 
                        for tree in r2_solved[r2_solved['target'] == smi]['trees'].values[0][0]]
                    for smi in both_solved_smiles}
        
        # Calculate difference score
        if both_solved_smiles:
            for smi in both_solved_smiles:
                r1_min_cost = min(r1_costs[smi])
                r2_min_cost = min(r2_costs[smi])
                
                if r1_min_cost > r2_min_cost:
                    difference_score += 1
                org_solved_costs.extend(r1_costs[smi])
            difference_score /= len(both_solved_smiles)
        print('Difference score:', difference_score)
        
        # Find targets solved only by the second approach
        extra_solved = [smi for smi in r2_solved['target'] if smi not in both_solved_smiles]
        extra_solved_score = 0
        if extra_solved:
            for smi in extra_solved:
                r2_smi_top_trees = r2_solved[r2_solved['target'] == smi]['trees'].values[0][0]
                r2_cost = [self.tree_cost(tree) for tree in r2_smi_top_trees]
                if min(r2_cost) < np.mean(org_solved_costs):
                    extra_solved_score += 1
            extra_solved_score /= len(extra_solved)
        print('Extra solved score:', extra_solved_score)
        
        # Calculate the number of extra solved targets
        num_solved_score = self._log_curve(len(extra_solved))
        
        return difference_score, extra_solved_score, num_solved_score

    def _log_curve(self, x, total=20):
        """
        Map a number between 0 and 50 to a value between 0 and 1 using a logarithmic curve.
        
        :param x (float): Input value between 0 and 50.
        :return float: Value between 0 and 1 following a logarithmic curve.
        """

        return np.log(1 + x) / np.log(total)

    def _remove_unsolved_routes(self, routes: pd.DataFrame) -> pd.DataFrame:
        """
        Remove unsolved routes from the routes DataFrame

        :param routes: the routes DataFrame
        :return: the routes DataFrame with only solved routes
        """
        solved_routes = routes[routes['is_solved'] == True]

        for i, row in solved_routes.iterrows():
            trees = row['trees']
            solved_trees = []
            for tree in trees[0]:
                if ReactionTree.from_dict(tree).is_solved:
                    solved_trees.append(tree)
            solved_routes.at[i, 'trees'] = [solved_trees]
        return solved_routes




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--standard', type=str, required=True, help='Path to the pre-optimisation results.')
    parser.add_argument('--optimised', type=str, required=True, help='Path to the optimised results.')
    parser.add_argument('--ncpus', type=int, default=2, help='Number of CPUs to use.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output file.')
    args = parser.parse_args()
    
    # import results
    std = pd.read_hdf(args.standard, 'table')
    opt = pd.read_hdf(args.optimised, 'table')
    print('Results imported.')
    
    # load price predictior
    predictor = GraphPricePredictor(
        model_path='/vols/opig/users/mokaya/CoPriNet/data/models/trained_virtual/lightning_logs/version_0/checkpoints/epoch=284-step=5323799.ckpt',
        n_cpus=args.ncpus,
    )
    print('Generated predictor.')
    
    difference, extra, num = compare_opt_performance(std, opt, predictor, args)
    
    with open(args.output, 'w') as f:
        json.dump(
            {
                'difference': difference,
                'extra': extra,
                'num': num
            },
            f
        )
    
    
    
    
    
    
    
    