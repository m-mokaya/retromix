import unittest
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdChemReactions

from aizynthfinder.aizynthfinder.reactiontree import ReactionTree

from aizynthfinder.projects.retrofail.experiments.production.src.utils import (
    calculate_molport_cost,
    calculate_tree_cost,
    get_solved_trees,
    reaction_template_similarity,
    findkeys,
    canonicalise_smarts,
    process_duplicate_templates,
)


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.stock = pd.DataFrame(
            {"InChIKey": ["AAAA", "BBBB"], "stock_price": [10, 20]}
        )
        self.stock = self.stock.set_index("InChIKey")
        self.tree = {
            "children": [
                {
                    "children": [
                        {"smiles": "CC", "children": [], "is_leaf": True},
                        {"smiles": "CCC", "children": [], "is_leaf": True},
                    ],
                    "smiles": "CCCC",
                    "template_score": 0.1,
                },
                {"smiles": "O", "children": [], "is_leaf": True},
            ],
            "smiles": "CCCCC",
            "template_score": 0.2,
        }
        self.reaction1_smarts = "C=CC(=O)O.NCC>>C=CC(O)CNCC"
        self.reaction2_smarts = "C=CC(=O)O.NCC>>C=CC(O)CNCC"
        self.reaction3_smarts = "C=CCO.NCC>>C=CC(O)CNCC"

        self.routes = pd.DataFrame(
            {
                "is_solved": [True, False],
                "trees": [
                    [
                        {
                            "children": [
                                {
                                    "children": [
                                        {
                                            "smiles": "CC",
                                            "children": [],
                                            "is_leaf": True,
                                        },
                                        {
                                            "smiles": "CCC",
                                            "children": [],
                                            "is_leaf": True,
                                        },
                                    ],
                                    "smiles": "CCCC",
                                },
                                {
                                    "smiles": "O",
                                    "children": [],
                                    "is_leaf": True,
                                },
                            ],
                            "smiles": "CCCCC",
                        }
                    ],
                    [],
                ],
            }
        )

        self.template_counts = {
            "CC.OO>>CCO": 2,
            "CC.OO>>CC(O)": 1,
            "CC.OO>>CCO": 3,  # Duplicate with different count
        }
        self.rxn_tree = ReactionTree.from_dict(self.tree)

    def test_calculate_molport_cost(self):
        cost = calculate_molport_cost(self.rxn_tree, self.stock, 10)
        self.assertEqual(cost, 30)

    def test_calculate_tree_cost(self):
        cost = calculate_tree_cost(self.tree, self.stock, 10)
        self.assertAlmostEqual(cost, 23.1, 1)

    def test_get_solved_trees(self):
        trees = get_solved_trees(self.routes)
        self.assertEqual(len(trees), 1)
        self.assertEqual(len(trees[0]), 1)
        self.assertEqual(trees[0][0]["smiles"], "CCCCC")

    def test_reaction_template_similarity(self):
        similarity = reaction_template_similarity(
            self.reaction1_smarts, self.reaction2_smarts
        )
        self.assertEqual(similarity, 1.0)

        similarity = reaction_template_similarity(
            self.reaction1_smarts, self.reaction3_smarts
        )
        self.assertNotEqual(similarity, 1.0)

        # Test precomputed
        rxn = rdChemReactions.ReactionFromSmarts(self.reaction1_smarts)
        fp = rdChemReactions.CreateStructuralFingerprintForReaction(rxn)
        similarity = reaction_template_similarity(
            self.reaction2_smarts, fp, precomputed=True
        )
        self.assertEqual(similarity, 1.0)

    def test_findkeys(self):
        result = list(findkeys(self.tree, "smiles"))
        self.assertEqual(len(result), 5)
        self.assertIn("CCCCC", result)
        self.assertIn("CC", result)

    def test_canonicalise_smarts(self):
        canonical_smarts = canonicalise_smarts(self.reaction1_smarts)
        self.assertEqual(canonical_smarts, "[C:1=[C:2[C:3](=[O:4])[O:5].[N:6][C:7][C:8]>>[C:1]=[C:2][C:3]([O:5])[C:9]([N:6][C:7][C:8)[C:10:4=[O:11")

    def test_process_duplicate_templates_combine(self):
        processed_counts = process_duplicate_templates(self.template_counts, combine=True)
        self.assertEqual(len(processed_counts), 2)
        self.assertEqual(processed_counts["CC.OO>>CCO"], 5)
        self.assertEqual(processed_counts["CC.OO>>CC(O)"], 1)

    def test_process_duplicate_templates_no_combine(self):
        processed_counts = process_duplicate_templates(self.template_counts, combine=False)
        self.assertEqual(len(processed_counts), 3)
        self.assertEqual(processed_counts["CC.OO>>CCO"], 5)
        self.assertEqual(processed_counts["CC.OO>>CC(O)"], 1)


if __name__ == "__main__":
    unittest.main()

