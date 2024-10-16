import shutil
import unittest
import pandas as pd
import os
import sys

sys.path.append("/data/localhost/not-backed-up/mokaya/aizynthfinder/projects/retrofail/experiments/production/src")  # Replace with the path to aizynthfinder

from route_finders.aizynthfinder import AizRouteFinder  # Assuming your class is in aiz_route_finder.py

class TestAizRouteFinder(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        self.configfile = "/data/localhost/not-backed-up/mokaya/aizynthfinder/aizynthfinder/data/config.yml"  # Replace with your actual config file path
        self.smiles = ["CCO", "CC(=O)O"]
        self.nproc = 2
        self.output_dir = "test_output"  # Create a test output directory
        os.makedirs(self.output_dir, exist_ok=True)

        self.finder = AizRouteFinder(self.configfile, self.smiles, self.nproc, self.output_dir)

    def tearDown(self):
        """Clean up after test methods."""
        # Remove the test output directory if you want
        shutil.rmtree(self.output_dir) 
        pass

    def test_split_smiles(self):
        """Test splitting SMILES into chunks."""
        chunks = self.finder.split_smiles()
        self.assertEqual(len(chunks), self.nproc)
        self.assertEqual(sum(len(chunk) for chunk in chunks), len(self.smiles))

    def test_worker(self):
        """Test the worker function."""
        chunk = ["CCO"] 
        result_df = self.finder.worker(chunk)
        self.assertIsInstance(result_df, pd.DataFrame)

    def test_find_routes(self):
        """Test the main find_routes function."""
        result_df = self.finder.find_routes()
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(len(result_df), len(self.smiles))  # Assuming each SMILES produces one row
        self.assertTrue('is_solved' in result_df.columns)



if __name__ == '__main__':
    unittest.main()
