import pandas as pd
import numpy as np
from scipy.stats import norm

from rdcanon import canon_reaction_smarts

from analysis.template_analyser import TemplateAnalyser
import optimisation.scoring as scoring
import utils

class AizTemplateAnalyser(TemplateAnalyser):
    def __init__(self, 
                 template_library: list[str], 
                 stock: dict[str, float] = None, 
                 scoring_type: str = 'state',
            ):
        super().__init__(template_library, scoring_type=scoring_type)
        
        self.scorer = scoring.Scorer(type=self.scoring_type, stock=stock)
        
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
                    all_template_counts[template] = 0
            
            template_scores = {template: count/all_template_counts[template] for template, count in template_counts.items()}
            sorted_template_scores = {k: v for k, v in sorted(template_scores.items(), key=lambda item: item[1], reverse=True)}
            return sorted_template_scores
        
        
    def find_unused_templates(self, aiz_tb: pd.DataFrame, aiz_tf: pd.DataFrame) -> dict[str: float]:
        """
        Find the unused templates in the AiZ synthesis routes.
        
        :param aiz_tb: the AiZ template-based synthesis routes.
        :param aiz_tf: the AiZ template-free synthesis routes.
        :return: the unused templates in order of popularity
        """
        
        if self.scoring_type == 'frequency':
            raise ValueError("Frequency scoring not implemented for unused templates")
        else:
            tb_solved = aiz_tb[aiz_tb['is_solved'] == True]
            tf_solved = aiz_tf[aiz_tf['is_solved'] == True]
            
            tb_routes = utils.get_solved_trees(tb_solved)
            tf_routes = utils.get_solved_trees(tf_solved)
            
            solved_both = [i for i in tb_solved['target'].tolist() if i in tf_solved['target'].tolist()]
            solved_tf = [i for i in tf_solved['target'].tolist() if i not in solved_both]
            
            all_tb_templates = [list(utils.findkeys(route, 'template')) for mol in tb_routes for route in mol]
            all_tf_templates = [list(utils.findkeys(route, 'template')) for mol in tf_routes for route in mol] 
            all_tf_templates = [item for sublist in all_tf_templates for item in sublist] 
            
            templates = []
            for mol in solved_both:
                tb_results = tb_solved[tb_solved['target'] == mol]
                tf_results = tf_solved[tf_solved['target'] == mol]
                
                tb_r = utils.get_solved_trees(tb_results)
                tf_r = utils.get_solved_trees(tf_results)
                
                tb_r = [item for sublist in tb_r for item in sublist]
                tf_r = [item for sublist in tf_r for item in sublist]
                
                tb_costs = [self.scorer(tree) for tree in tb_r]
                tf_costs = [self.scorer(tree) for tree in tf_r]
                
                # get indexes of tb routes that are cheaper than tf routes
                min_tb_cost = min(tb_costs)
                cheaper_indexes = [i for i, cost in enumerate(tf_costs) if cost > min_tb_cost]
                cheaper_routes = [tf_r[i] for i in cheaper_indexes]
                
                for route in cheaper_routes:
                    templates.extend(list(utils.findkeys(route, 'template')))
                    
            for mol in solved_tf:
                tf_results = tf_solved[tf_solved['target'] == mol]
                tf_r = utils.get_solved_trees(tf_results)
                tf_r = [item for sublist in tf_r for item in sublist]
                
                templates.extend(list(utils.findkeys(tf_r, 'template')))
        
            print(f"Unused templates: {len(templates)} of which {len(set(templates))} are unique")
            
            template_counts = pd.Series(templates).value_counts().to_dict()
            all_tf_template_counts = pd.Series(all_tf_templates).value_counts().to_dict()
            all_tf_template_counts = utils.process_duplicate_templates(all_tf_template_counts, combine=False)
            template_counts = utils.process_duplicate_templates(template_counts, combine=True)
            
            template_counts = {template: count for template, count in template_counts.items() if canon_reaction_smarts(template) not in all_tb_templates}
            
            template_scores = {template: count/all_tf_template_counts[template] for template, count in template_counts.items()}
            sorted_template_scores = {k: v for k, v in sorted(template_scores.items(), key=lambda item: item[1], reverse=True)}
            return sorted_template_scores
                    
                
                
                
                
        
        
        
        