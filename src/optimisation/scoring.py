import os
import sys
import json
import yaml
import argparse
import pandas as pd
import numpy as np

import multiprocessing
from functools import partial

from rdkit import Chem

# Add the current working directory to sys.path
sys.path.append(os.getcwd())

from aizynthfinder.context.scoring import StateScorer
from aizynthfinder.context.config import Configuration
from aizynthfinder.reactiontree import ReactionTree
from CoPriNet.pricePrediction.predict.predict import GraphPricePredictor



class Scorer:
    def __init__(self, predictor: GraphPricePredictor = None, stock: dict = None, type: str = 'state'):
        self.predictor = predictor
        self.stock = stock
        self.cost_cache = {}
        
        if type == 'state':
            self.scorer = StateScorer(config=Configuration())
            self.tree_cost = self.get_state_score
        if type == 'coprinet':
            if self.predictor is None:
                raise ValueError('No price predictor specified. Please add a compatible price predictor')
            self.tree_cost = self.coprinet_tree_cost
        if type == 'cost':
            if self.stock is None:
                raise ValueError('No stock library specified. Please add a compatible stock library')
            self.tree_cost = self.get_stock_cost
            
    def __call__(self, rxn):
        return self.tree_cost(rxn)
        
    def get_state_score(self, tree):
        """
        Calculate the cost of a tree using the state scorer
        
        :param tree: the tree to calculate the cost for
        :return: the cost of the tree
        """
        rxn = ReactionTree.from_dict(tree)
        tree_id = rxn.hash_key()
        if tree_id not in self.cost_cache:
            # score = self.scorer(rxn)
            score = 0.7 * len(list(rxn.reactions())) + 0.3 * len(list(rxn.leafs()))
            self.cost_cache[tree_id] = score
        return self.cost_cache[tree_id]
    
    def get_stock_cost(self, tree):
        """
        Calculate the cost of a tree using stock library dict[inchi_key: price]
        
        :param tree: the tree to calculate the cost for
        :return: the cost of the tree
        """
        rxn = ReactionTree.from_dict(tree)
        stock_cost = self._calculate_stock_cost(rxn, self.stock)
        cost = 0.7 * stock_cost + 0.15 * len(list(rxn.leafs())) + 0.15 * len(list(rxn.reactions()))
        return cost 
    
    def _calculate_stock_cost(self, route: ReactionTree, stock: dict, not_in_stock=10.0) -> float:
        """
        Calculate the cost of a route using the stock library
        
        :param route: the route to calculate the cost for
        :param stock: the stock library
        """
        leaves = list(route.leafs())
        total_cost = 0
        for leaf in leaves:
            inchi_key = Chem.MolToInchiKey(Chem.MolFromSmiles(leaf.smiles))
            if inchi_key in stock:
                c = stock[inchi_key]
                total_cost += c
            else:
                total_cost += not_in_stock
                print(str(leaf)+' not in stock. Route not solved.')
                return 1000
        return total_cost
        
    
    def get_coprinet_tree_cost(self, tree):
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
        

class OptimisationScorer:
    def __init__(self, stock: dict = None, predictor: GraphPricePredictor = None, use_coprinet: bool = False):
        """
        Initialise the OptimisationScorer to calculate POS score. 
        
        :param args: the arguments to use
        :param predictor: the price predictor to use
        :param use_coprinet: whether to use the coprinet model
        """
        
        self.predictor = predictor
        self.args = args
        self.cost_cache = {}
        self.stock = stock if stock is not None else None
        self.use_coprinet = use_coprinet
        
        
        # Set the tree cost function
        if self.use_coprinet == True and self.predictor is not None:
            self.tree_cost = self.coprinet_tree_cost
            print('Using coprinet model for cost prediction.')
        else:
            if self.stock != None:
                self.stock = self.stock
                self.tree_cost = self.stock_tree_cost
                print('Using stock library for cost prediction.')
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
    
    def stock_tree_cost(self, tree):
        """
        Calculate the cost of a tree using MolPort stock library
        
        :param tree: the tree to calculate the cost for
        :return: the cost of the tree
        """
        stock_dict = {k: v for k, v in self.stock[['inchi_key', 'price']].values}
        rxn = ReactionTree.from_dict(tree)
        stock_cost = self._calculate_stock_cost(rxn, stock_dict)
        cost = 0.7 * stock_cost + 0.15 * len(list(rxn.leafs())) + 0.15 * len(list(rxn.reactions()))
        return cost
    
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
    
    def _calculate_stock_cost(route: ReactionTree, stock: dict, not_in_stock=10.0) -> float:
        """
        Calculate the cost of a route using the stock library
        
        :param route: the route to calculate the cost for
        :param stock: the stock library
        """
        leaves = list(route.leafs())
        total_cost = 0
        not_in_stock_multiplier = 10
        for leaf in leaves:
            inchi = leaf.inchi_key
            if inchi in stock:
                c = stock[inchi]
                total_cost += c
            else:
                total_cost += not_in_stock_multiplier
                print(str(leaf)+' not in stock')
        return total_cost
        




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre', type=str, required=True, help='Path to the pre-optimisation results.')
    parser.add_argument('--post', type=str, required=True, help='Path to the optimised results.')
    parser.add_argument('--ncpus', type=int, default=1, help='Number of CPUs to use.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    parser.add_argument('--ngpus', type=int, default=0, help='Number of GPUs to use.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output file.')
    args = parser.parse_args()
    
    # load yaml config file
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # import results
    std = pd.read_hdf(args.pre, 'table')
    opt = pd.read_hdf(args.post, 'table')
    print('Results imported.')
    
    # load price predictior
    predictor = GraphPricePredictor(
        use_coprinet=True if config['coprinet_model_path'] != None else False,
        model_path=config['coprinet_model_path'],
        n_cpus=args.ncpus,
        n_gpus=args.ngpus
    )
    print('Generated predictor.')
    
    scorer = OptimisationScorer(predictor=predictor, use_coprinet=True)
    cost_difference, extra_price, extra_solved = scorer.compare_opt_performance(std, opt)
    
    with open(args.output, 'w') as f:
        json.dump(
            {
                'cost_difference': cost_difference,
                'extra_price': extra_price,
                'extra_solved': extra_solved
            },
            f
        )
    
    
    
    
    
    
    
    