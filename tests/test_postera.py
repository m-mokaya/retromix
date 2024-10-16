import unittest
from unittest.mock import patch, Mock
import pandas as pd
import os
import sys
import json

sys.path.append("/data/localhost/not-backed-up/mokaya/aizynthfinder/projects/retrofail/experiments/production/src")  # Replace with the path to aizynthfinder
from route_finders.postera import PosRouteFinder
 
class TestPosRouteFinder(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        self.smiles = ["CCO", "CC(=O)O", "CC(=O)OC(=O)C", "CCCC(=O)OC(=O)CCC", "CCCCCCCC(=O)OC(=O)CCCCCCCC"]
        self.maxSearchDepth = 4
        self.ignore_zero_steps = False
        self.catalogues = ["molport"]
        self.api_key = "your_postera_api_key"  # Replace with your actual API key or mock it
        self.output_dir = "test_output"  # Create a test output directory
        os.makedirs(self.output_dir, exist_ok=True)

        self.finder = PosRouteFinder(self.smiles, self.maxSearchDepth, self.ignore_zero_steps, 
                                     self.catalogues, self.api_key, output=self.output_dir)

    def tearDown(self):
        """Clean up after test methods."""
        # Remove the test output directory if you want
        # shutil.rmtree(self.output_dir)
        pass

    def test_split_smiles(self):
        """Test splitting SMILES into chunks."""
        chunks = self.finder.split_smiles()
        self.assertEqual(sum(len(chunk) for chunk in chunks), len(self.smiles))

    def test_prepateBatchQueryData(self):
        """Test preparing batch query data."""
        selected_smiles = ["CCO", "CC(=O)O"]
        data = self.finder.prepateBatchQueryData(selected_smiles)
        self.assertEqual(data["maxSearchDepth"], self.maxSearchDepth)
        self.assertEqual(data["catalogs"], self.catalogues)
        self.assertEqual(data["smilesList"], selected_smiles)

    # @patch('postera.requests.post')
    # def test_retrosynthesis_search_success(self, mock_post):
    #     """Test retrosynthesis_search with successful response."""
    #     mock_response = Mock()
    #     mock_response.status_code = 200
    #     mock_response.json.return_value = {
    #         "results": [{"routes": []}, {"routes": []}]  # Mock some basic response data
    #     }
    #     mock_post.return_value = mock_response

    #     smiles = ["CCO", "CC(=O)O"]
    #     results = self.finder.retrosynthesis_search(smiles)
    #     self.assertIsInstance(results, dict)
    #     self.assertEqual(len(results), len(smiles))

    # @patch('postera.requests.post')
    # def test_retrosynthesis_search_error(self, mock_post):
    #     """Test retrosynthesis_search with error response."""
    #     mock_response = Mock()
    #     mock_response.status_code = 400 
    #     mock_response.text = "Bad Request"
    #     mock_post.return_value = mock_response

    #     smiles = ["CCO", "CC(=O)O"]
    #     results = self.finder.retrosynthesis_search(smiles)
    #     self.assertIsNone(results)  # Expect None in case of error

    def test_convert_results_to_aiz_format(self):
        """Test converting results to AiZynthFinder format."""
        # Create a mock Postera response data structure (simplified for testing)
        postera_data = {
            "molecules": [
                {"smiles": "CCO", "isBuildingBlock": True},
                {"smiles": "C=C", "isBuildingBlock": False},
                {"smiles": "O", "isBuildingBlock": True}
            ],
            "reactions": [
                {"productSmiles": "CCO", "reactantSmiles": ["C=C", "O"], "name": "Some Reaction"},
            ]
        }
        aiz_tree = self.finder.convert_results_to_aiz_format(postera_data)
        self.assertIsInstance(aiz_tree, dict)
        self.assertEqual(aiz_tree['smiles'], "CCO")  # Check root node
        self.assertEqual(len(aiz_tree['children']), 1)  # Check for one reaction child

    def test_find_routes(self):
        """Test the main find_routes function."""
        # We'll need more extensive mocking here, including the Pool and retrosynthesis_search
        with patch('postera.Pool') as mock_pool, \
             patch.object(self.finder, 'retrosynthesis_search') as mock_search:

            # Mock the Pool's map method
            mock_pool_instance = mock_pool.return_value
            mock_pool_instance.map.return_value = [
                {"CCO": [{"routes": []}]},  # Mock simplified return data from retrosynthesis_search
                {"CC(=O)O": [{"routes": []}]} 
            ]
            
            # Mock the retrosynthesis_search to avoid actual API calls
            mock_search.return_value = {"routes": []}

            filename = os.path.join(self.output_dir, "test_routes.json")
            results = self.finder.find_routes(filename)
            
            self.assertIsInstance(results, list)
            # Check if the file was created and contains valid JSON
            self.assertTrue(os.path.exists(filename))
            with open(filename, 'r') as f:
                data = json.load(f)
                self.assertIsInstance(data, list)

    # Add more test methods as needed to cover different aspects of your class


if __name__ == '__main__':
    unittest.main()
