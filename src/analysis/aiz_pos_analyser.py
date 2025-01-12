import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import norm

from rdcanon import canon_reaction_smarts

from analysis.template_analyser import TemplateAnalyser
from optimisation.scoring import Scorer
import utils

class AizPosTemplateAnalyser(TemplateAnalyser):
    def __init__(self, 
                 template_library: list[str], 
                 stock: dict[str, float] = None, 
                 scoring_type: str = 'state',
                 predictor = None,  
            ):
        
        super().__init__(template_library, scoring_type=scoring_type)
        
        self.scorer = Scorer(
            predictor=predictor,
            type=self.scoring_type,
            stock=stock,
            )
        
    def find_popular_templates(self, aiz_data: pd.DataFrame) -> dict[str: float]:
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
            if self.scoring_type != 'frequency':    
                costs = [self.scorer(tree) for tree in mol]
                cheapest_route = mol[np.argmin(costs)]
                used_templates = list(utils.findkeys(cheapest_route, 'template'))
                templates.extend(used_templates)
            
            for route in mol:
                all_templates.extend(list(utils.findkeys(route, 'template')))
        
        
        if self.scoring_type == 'frequency':
            print(f"Finding popular templates based on frequency")
            template_counts = pd.Series(all_templates).value_counts().to_dict()
            template_scores = {template: count/len(all_templates) for template, count in template_counts.items()}
            template_scores = {template: norm.cdf((score - np.mean(list(template_scores.values()))) / np.std(list(template_scores.values()))) for template, score in template_scores.items()}            
            sorted_template_scores = {k: v for k, v in sorted(template_scores.items(), key=lambda item: item[1], reverse=True)}
            return sorted_template_scores
        else:
            # get counts for used templates and all templates
            template_counts = pd.Series(templates).value_counts().to_dict()
            all_template_counts = pd.Series(all_templates).value_counts().to_dict()
            
            # check all templates are in all_template_counts
            for template in template_counts.keys():
                if template not in all_template_counts:
                    print(f"Template {template} not in all_template_counts")
                
            # # combine duplicate counts in each dictionary
            # template_counts = utils.process_duplicate_templates(template_counts)
            # all_template_counts = utils.process_duplicate_templates(all_template_counts)
            
            template_scores = {template: used_count/all_template_counts[template] for template, used_count in template_counts.items()}
            sorted_template_scores = {k: v for k, v in sorted(template_scores.items(), key=lambda item: item[1], reverse=True)}
            return sorted_template_scores
    
    def find_unused_templates(self, aiz_data: pd.DataFrame, pos_data: dict[str, dict]) -> dict[str: float]:
        """
        Find the templates used by postera in cheap routes that are not used by AiZ.
        
        :param aiz_data: the AiZ data.
        :param pos_data: the Postera data.
        :return: the unused templates -> dict{template: score}
        """
        
        aiz_solved_smiles = aiz_data[aiz_data['is_solved'] == True]['target'].tolist()
        aiz_routes = utils.get_solved_trees(aiz_data)
        
        all_aiz_templates = [list(utils.findkeys(route, 'template')) for mol in aiz_routes for route in mol]
        all_aiz_templates = [item for sublist in all_aiz_templates for item in sublist]
        all_aiz_templates = [canon_reaction_smarts(template) for template in all_aiz_templates]
        # all_pos_templates = [list(utils.findkeys(route, 'template')) for mol, routes in pos_data.items() for route in routes]
        # all_pos_templates = [item for sublist in all_pos_templates for item in sublist]
        
        # collect all pos templates
        all_pos_templates = []
        for mol, routes in pos_data.items():
            for route in routes:
                all_pos_templates.extend(list(utils.findkeys(route, 'template')))
         
        templates = []
        for mol, pos_trees in pos_data.items():
            if mol in aiz_solved_smiles:
                aiz_trees = aiz_routes[aiz_solved_smiles.index(mol)]
                
                if self.scoring_type != 'frequency':
                    pos_costs = [self.scorer(tree) for tree in pos_trees]
                    aiz_costs = [self.scorer(tree) for tree in aiz_trees]
                    
                    cheaper_pos_indexes = [i for i, cost in enumerate(pos_costs) if cost < min(aiz_costs)]
                    cheaper_routes = [pos_trees[i] for i in cheaper_pos_indexes]
                    
                    for i in cheaper_routes:
                        templates.extend(list(utils.findkeys(i, 'template')))
                else:
                    temps = [list(utils.findkeys(route, 'template')) for route in pos_trees]
                    flat_templates = [item for sublist in temps for item in sublist]
                    templates.extend(flat_templates)
            else:
                if self.scoring_type != 'frequency':
                    pos_costs = [self.scorer(tree) for tree in pos_trees]
                    cheapest_route  = pos_trees[np.argmin(pos_costs)]
                    templates.extend(list(utils.findkeys(cheapest_route, 'template')))
                else:
                    temps = [list(utils.findkeys(route, 'template')) for route in pos_trees]
                    flat_templates = [item for sublist in temps for item in sublist]
                    templates.extend(flat_templates)
                       
        print(f"Unused templates: {len(templates)} of which {len(set(templates))} are unique")
      
      
        all_pos_template_counts = pd.Series(all_pos_templates).value_counts().to_dict()
        template_counts = pd.Series(templates).value_counts().to_dict()
        all_pos_template_counts = utils.process_duplicate_templates(all_pos_template_counts, combine=False)
        all_pos_template_counts = utils.process_duplicate_templates(all_pos_template_counts, combine=False)
        template_counts = utils.process_duplicate_templates(template_counts)
        
        template_counts = {template: count for template, count in template_counts.items() if canon_reaction_smarts(template) not in all_aiz_templates}
        
        if self.scoring_type == 'frequency':
            pos_template_scores = {template: (used_count/len(templates)) for template, used_count in template_counts.items()}
            pos_template_scores = {template: norm.cdf((score - np.mean(list(pos_template_scores.values()))) / np.std(list(pos_template_scores.values()))) for template, score in pos_template_scores.items()}
            
        else:
            pos_template_scores = {template: (used_count/all_pos_template_counts[template]) for template, used_count in template_counts.items()}
        
        sorted_template_scores = {k: v for k, v in sorted(pos_template_scores.items(), key=lambda item: item[1], reverse=True)}
        return sorted_template_scores
    
        
        