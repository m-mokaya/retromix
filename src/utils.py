import os
import sys
import pandas as pd
from rdcanon import canon_reaction_smarts

from rdkit.Chem import rdChemReactions, DataStructs

sys.path.append("...")  # Replace with the path to aizynthfinder

from aizynthfinder.aizynthfinder.reactiontree import ReactionTree

def calculate_molport_cost(route, stock, not_in_stock_multiplier):
    """
    Fucntion to calculate the cost of a route based on the stock of a supplier
    :param route: ReactionTree object
    :param stock: DataFrame with the stock of the supplier
    :return: float with the cost of the route
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

def calculate_tree_cost(tree: dict, stock: dict, not_in_stock_cost=1.08) -> float:
    """
    Calculate the cost of a tree

    :param tree: the tree to calculate the cost for
    :param stock: the stock DataFrame
    :param not_in_stock_cost: the cost to assign if a compound is not in stock
    :return: the cost of the tree
    """
    rxn = ReactionTree.from_dict(tree)
    molport_cost = calculate_molport_cost(rxn, stock, not_in_stock_cost)
    cost = 0.7 * molport_cost + 0.15*float(len(list(rxn.leafs()))) + 0.15*float(len(list(rxn.reactions())))
    return cost

def get_solved_trees(routes: pd.DataFrame) -> list[list[dict]]:
    """
    Get the solved trees from the AiZ results routes DataFrame

    :param routes: the routes DataFrame
    :return: the solved trees
    """
    solved_routes = routes[routes['is_solved'] == True]
    trees = solved_routes['trees'].tolist()
    trees = [i[0] for i in trees]
    
    solved_trees = []
    for mol in trees:
        mol_trees = []
        for route in mol:
            if ReactionTree.from_dict(route).is_solved == True:
                mol_trees.append(route)
        solved_trees.append(mol_trees)        

    return solved_trees

def reaction_template_similarity(template_1, template_2, precomputed: bool = False):
    """
    Calculate the similarity between two reaction templates.
    :param template_1: the first template   
    :param template_2: the second template
    :param precomputed: whether the fingerprints are precomputed
    :return: the similarity score
    """
    
    if precomputed:
        r1 = rdChemReactions.ReactionFromSmarts(template_1)
        fp1 = rdChemReactions.CreateStructuralFingerprintForReaction(r1)

        sim = DataStructs.TanimotoSimilarity(fp1, template_2)
        return sim
    else:
        r1 = rdChemReactions.ReactionFromSmarts(template_1)
        r2 = rdChemReactions.ReactionFromSmarts(template_2)
        
        fp1 = rdChemReactions.CreateStructuralFingerprintForReaction(r1)
        fp2 = rdChemReactions.CreateStructuralFingerprintForReaction(r2)

        sim = DataStructs.TanimotoSimilarity(fp1, fp2)
    return sim

def findkeys(node, kv):
    """
    Find all instances of a key in a nested dictionary or list. 
    (e.g. search reaction tree for all "templates")
    
    :param node: the dictionary or list to search
    :param kv: the key to search for
    """
    
    if isinstance(node, list):
        for i in node:
            for x in findkeys(i, kv):
               yield x
    elif isinstance(node, dict):
        if kv in node:
            yield node[kv]
        for j in node.values():
            for x in findkeys(j, kv):
                yield x

def canonicalise_smarts(smarts):
    """
    Generate a canonicalised SMARTS string for a reaction using rdcanon: https://pubs.acs.org/doi/10.1021/acs.jcim.4c00138
    
    :param smarts: the reaction SMARTS string
    :return: the canonicalised SMARTS string
    """
    try:
        result = canon_reaction_smarts(smarts)
    except Exception as e:
        print('Error canonicalising SMARTS: ', e)
        result = smarts   
    return result

def process_duplicate_templates(template_counts: dict, combine: bool = True) -> dict:
    """
    Processes duplicate templates based on their smarts string or reaction_template_similarity.
    If `combine` is True, it combines the counts for identical templates into a single entry.
    If `combine` is False, it keeps all templates but updates their counts to the total count 
    for all identical templates.

    :param template_counts: A dictionary of template SMILES strings and their counts.
    :type template_counts: dict
    :param combine: Whether to combine duplicate templates into a single entry (True) or 
                    preserve all templates and update their counts (False).
    :type combine: bool
    :return: A modified dictionary with duplicate template counts processed.
    :rtype: dict
    """

    processed_counts = {}
    processed_templates = set()

    for template1, count1 in template_counts.items():
        if template1 in processed_templates:
            continue

        total_count = count1
        processed_templates.add(template1)

        for template2, count2 in template_counts.items():
            if template2 not in processed_templates and (template1 == template2 or reaction_template_similarity(template1, template2) == 1.0):
                total_count += count2
                processed_templates.add(template2)

        if combine:
            processed_counts[template1] = total_count
        else:
            # Update the count for all identical templates
            for template in template_counts:
                if template == template1 or reaction_template_similarity(template1, template) == 1.0:
                    processed_counts[template] = total_count

    return processed_counts 