import unittest
import os
import pandas as pd
import yaml
from unittest.mock import patch, MagicMock

from aizynthfinder.projects.retrofail.experiments.production.src.route_finders import RouteFinder
from aizynthfinder.projects.retrofail.experiments.production.src.route_finders.aizynthfinder import AizRouteFinder
from aizynthfinder.projects.retrofail.experiments.production.src.route_finders.postera import PosRouteFinder
from aizynthfinder.projects.retrofail.experiments.production.src.analysis.template_analyser import TemplateAnalyser
from aizynthfinder.projects.retrofail.experiments.production.src.analysis.aiz_pos_analyser import AizPosTemplateAnalyser
from aizynthfinder.projects.retrofail.experiments.production.src import utils

class TestMain(unittest.TestCase):

    def setUp(self):
        self.config = {
            'stock': 'data/stock.hdf5',
            'template_library': 'data/library.txt',
            'targets': 'data/targets.txt',
            'aizynthfinder_config': 'config.yml',
            'output_dir': 'output',
            'nproc': 4
        }

    @patch('builtins.open', new_callable=MagicMock)
    @patch('yaml.safe_load', return_value={})  # Mock yaml loading for simplicity
    @patch('builtins.print')
    @patch('pandas.read_hdf')
    @patch.object(AizRouteFinder, 'find_routes', return_value={'some': 'data'})
    @patch.object(PosRouteFinder, 'find_routes', return_value={'other': 'data'})
    @patch.object(AizPosTemplateAnalyser, 'find_popular_templates', return_value=['popular1', 'popular2'])
    @patch.object(AizPosTemplateAnalyser, 'find_unused_templates', return_value=['unused1', 'unused2'])
    @patch.object(TemplateAnalyser, 'find_novel_overlooked_templates', return_value=(['overlooked1'], ['novel1']))
    def test_main_execution(self, mock_novel_overlooked, mock_unused, mock_popular, mock_pos_find, mock_aiz_find,
                            mock_read_hdf, mock_print, mock_yaml_load, mock_open):
        # Set up environment variable for Postera API key (replace with your actual key if needed)
        os.environ['POSTERA_API_KEY'] = 'dummy_key'

        # Execute the main part of the script
        with patch.dict('sys.modules', {'aizynthfinder.projects.retrofail.experiments.production.src.main': self}):
            import aizynthfinder.projects.retrofail.experiments.production.src.main
            aizynthfinder.projects.retrofail.experiments.production.src.main.main()

        # Assertions to check if functions were called with expected arguments
        mock_aiz_find.assert_called_once_with()
        mock_pos_find.assert_called_once_with()
        mock_popular.assert_called_once()
        mock_unused.assert_called_once()
        mock_novel_overlooked.assert_called_once()

        # Assert print statements (check for specific outputs if needed)
        self.assertEqual(mock_print.call_count, 3)

    def test_template_analyser(self):
        analyser = TemplateAnalyser(["template1", "template2"])
        overlooked, novel = analyser.find_novel_overlooked_templates(["template1", "template3"])
        self.assertEqual(overlooked, ["template1"])
        self.assertEqual(novel, ["template3"])

    def test_aiz_pos_template_analyser(self):
        analyser = AizPosTemplateAnalyser(["template1", "template2"])

        # Mock data for find_popular_templates
        aiz_data = {"template1": {"count": 2}, "template2": {"count": 1}}
        stock = {"reactant1": 10, "reactant2": 20}
        popular = analyser.find_popular_templates(aiz_data, stock)
        self.assertEqual(popular, ["template1"])  # Assuming higher count means more popular

        # Mock data for find_unused_templates
        pos_data = {"template2": {"count": 1}}
        unused = analyser.find_unused_templates(aiz_data, pos_data, stock)
        self.assertEqual(unused, ["template1"])  # Assuming presence in aiz_data but not pos_data means unused


if __name__ == '__main__':
    unittest.main()
