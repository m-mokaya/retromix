import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch
from aizynthfinder.projects.retrofail.experiments.production.src.analysis.aiz_pos_analyser import AizPosTemplateAnalyser
from aizynthfinder.projects.retrofail.experiments.production.src import utils


class TestAizPosTemplateAnalyser(unittest.TestCase):
    def setUp(self):
        self.template_library = ["template1", "template2"]
        self.analyser = AizPosTemplateAnalyser(self.template_library)

    def test_find_popular_templates(self):
        aiz_data = pd.DataFrame(
            {
                "is_solved": [True, True],
                "target": ["mol1", "mol2"],
                "trees": [
                    [
                        {"template": "template1"},
                        {"template": "template2"},
                    ],
                    [
                        {"template": "template1"},
                        {"template": "template1"},
                    ],
                ],
            }
        )
        stock = {"inchikey1": 1.0, "inchikey2": 2.0}

        with patch("src.utils.calculate_tree_cost") as mock_calculate_tree_cost:
            mock_calculate_tree_cost.side_effect = [2.0, 1.0, 1.0, 2.0]
            popular_templates = self.analyser.find_popular_templates(aiz_data, stock)

        self.assertEqual(popular_templates, {"template1": 1.5, "template2": 0.5})

    def test_find_unused_templates(self):
        aiz_data = pd.DataFrame(
            {
                "is_solved": [True, False],
                "target": ["mol1", "mol2"],
                "trees": [
                    [
                        {"template": "template1"},
                    ],
                    [],
                ],
            }
        )
        pos_data = {
            "mol1": [{"template": "template2"}],
            "mol3": [{"template": "template3"}],
        }
        stock = {"inchikey1": 1.0, "inchikey2": 2.0}

        with patch("src.utils.calculate_tree_cost") as mock_calculate_tree_cost:
            mock_calculate_tree_cost.side_effect = [1.0, 2.0]
            unused_templates = self.analyser.find_unused_templates(
                aiz_data, pos_data, stock
            )

        self.assertEqual(unused_templates, {"template2": 1.0, "template3": 1.0})


if __name__ == "__main__":
    unittest.main()
