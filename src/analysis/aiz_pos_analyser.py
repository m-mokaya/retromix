import os
import sys
import pandas as pd
import numpy as np

from src.analysis.template_analyser import TemplateAnalyser
from src import utils

class AizPosTemplateAnalyser(TemplateAnalyser):
    def __init__(self, template_library: list[str]):
        super().__init__(template_library)
        
    def find_popular_templates(self, aiz_data: pd.DataFrame, stock: dict) -> dict[str: float]:
        """
        Find the popular templates in the AiZ synthesis routes.
        
        :param aiz_data: the AiZ data.
        :param stock: the stock data {inchi_key: cost}
        :return: the popular templates in order of popularity
        """
        
        solved_trees = utils.get_solved_trees(aiz_data)
        
        templates = []
        all_templates = []
        for mol in solved_trees:
            costs = [utils.calculate_tree_cost(tree, stock) for tree in mol]
            cheapest_route = mol[np.argmin(costs)]
            used_templates = list(utils.findkeys(cheapest_route, 'template'))
            templates.extend(used_templates)
            
            for route in mol:
                all_templates.extend(list(utils.findkeys(route, 'template')))
        
        # get counts for used templates and all templates
        template_counts = pd.Series(templates).value_counts().to_dict()
        all_template_counts = pd.Series(all_templates).value_counts().to_dict()
        
        # combine duplicate counts in each dictionary
        template_counts = utils.process_duplicate_templates(template_counts)
        all_template_counts = utils.process_duplicate_templates(all_template_counts)
        
        template_scores = {template: used_count/all_template_counts[template] for template, used_count in template_counts.items()}
        sorted_template_scores = {k: v for k, v in sorted(template_scores.items(), key=lambda item: item[1], reverse=True)}
        return sorted_template_scores
    
    def find_unused_templates(self, aiz_data: pd.DataFrame, pos_data: dict[str, dict], stock: dict[str, float]) -> dict[str: float]:
        """
        Find the templates used by postera in cheap routes that are not used by AiZ.
        
        :param aiz_data: the AiZ data.
        :param pos_data: the Postera data.
        :param stock: the stock data {inchi_key: cost}
        :return: the unused templates -> dict{template: score}
        """
        
        aiz_solved_smiles = aiz_data[aiz_data['is_solved'] == True]['target'].tolist()
        aiz_routes = utils.get_solved_trees(aiz_data)
        
        all_aiz_templates = [list(utils.findkeys(route, 'template')) for mol in aiz_routes for route in mol]
        all_pos_templates = [list(utils.findkeys(route, 'template')) for mol, routes in pos_data.items() for route in routes]
        
        templates = []
        for mol, pos_trees in pos_data.items():
            if mol in aiz_solved_smiles:
                aiz_trees = aiz_routes[aiz_solved_smiles.index(mol)]
                pos_costs = [utils.calculate_tree_cost(tree, stock) for tree in pos_trees]
                aiz_costs = [utils.calculate_tree_cost(tree, stock) for tree in aiz_trees]
                
                cheaper_pos_indexes = [i for i, cost in enumerate(pos_costs) if cost < min(aiz_costs)]
                cheaper_routes = [pos_trees[i] for i in cheaper_pos_indexes]
                
                for i in cheaper_routes:
                    templates.extend(list(utils.findkeys(i, 'template')))
            else:
                templates.extend([list(utils.findkeys(i, 'template')) for i in pos_trees])
                
        print(f"Unused templates: {len(templates)} of which {len(set(templates))} are unique")
        
        all_pos_template_counts = pd.Series(all_pos_templates).value_counts().to_dict()
        template_counts = pd.Series(templates).value_counts().to_dict()
        all_pos_template_counts = utils.process_duplicate_templates(all_pos_template_counts)
        template_counts = utils.process_duplicate_templates(template_counts)
        
        pos_template_scores = {template: used_count/all_pos_template_counts[template] for template, used_count in template_counts.items()}
        sorted_template_scores = {k: v for k, v in sorted(pos_template_scores.items(), key=lambda item: item[1], reverse=True)}
        return sorted_template_scores
    
        
        