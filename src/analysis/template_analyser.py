import os
import sys
import argparse

from utils import canonicalise_smarts

class TemplateAnalyser:
    def __init__(self, template_library: list[str], scoring_type: str):
        self.template_library = template_library
        self.scoring_type = scoring_type
    
    def find_popular_templates(self, route_data: list, stock: dict) -> list[str]:
        """
        Find the popular templates in the synthesis routes.
        
        :param route_data: the routes data
        :param stock: the stock data
        :param args: the arguments
        :return: the popular templates in order of popularity
        """
        raise NotImplementedError("This method must be implemented in the child class.")
        
    def find_unused_templates(self, route_data_1, route_data_2, stock) -> list[dict[str: float]]:
        """
        Find the templates used in the best scoring route 2 data that are not used in route 1 data.
        
        :param route_data_1: the first routes data
        :param route_data_2: the second routes data
        :param stock: the stock data
        :param args: the arguments
        :return: the unused templates -> dict{template: score}
        """
        raise NotImplementedError("This method must be implemented in the child class.")
    
    def find_novel_overlooked_templates(self, unused_templates: list[dict]) -> list[str]:
        """
        Find unused templates that are in the template library.
        
        :param route_data: the routes data
        :param stock: the stock data
        :param args: the arguments
        :return: the novel templates
        """
        
        # canonicalise the templates
        canon_unused_templates = {template: canonicalise_smarts(template) for template, score in unused_templates.items()}
        
        # find the templates in the library
        overlooked_templates = {template: unused_templates[template] for template, canon_template in canon_unused_templates.items() if canon_template in self.template_library} # to make sure it matches the USPTO template SMARTS. 
        novel_templates = {template: unused_templates[template] for template, canon_template in canon_unused_templates.items() if canon_template not in self.template_library}
        
        return overlooked_templates, novel_templates