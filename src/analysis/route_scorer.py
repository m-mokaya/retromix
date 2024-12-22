import json
import os
import numpy as np
import pandas as pd

from aizynthfinder.aizynthfinder.context.scoring import Scorer
from aizynthfinder.aizynthfinder.reactiontree import ReactionTree

class RouteScorer():
    def __init__(self, type: str = 'state'):
        self.type = type
        
    def __call__(self, route: dict) -> float:
        """
        Score a route.
        
        :param route: the route to score
        :return: the score
        """
        if self.type == 'state':
            return self.route_score(route)
        elif self.type == 'cost':
            return self.cost_score(route)
        elif self.type == 'coprinet':
            return self.coprinet_score(route)
        else:
            raise ValueError(f"Scoring type {self.type} not recognised")
        
    def route_score(self, route: dict) -> float:
        """
        Score a route based on the number of states.
        
        :param route: the route to score
        :return: the score
        """
        rxn = ReactionTree.from_dict(route)
        length = len(rxn.reactions())
        precursors = len(rxn.leafs())
        return 0.7 * length + 0.3 * precursors
    
    def cost_score(self, route: dict) -> float:
        """
        Score a route based on the cost.
        
        :param route: the route to score
        :return: the score
        """
        return route['cost']
    
    def coprinet_score(self, route: dict) -> float:
        """
        Score a route based on the number of Coprinet states.
        
        :param route: the route to score
        :return: the score
        """
        return len(route['coprinet_states'])