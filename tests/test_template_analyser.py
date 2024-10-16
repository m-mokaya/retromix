import unittest
from unittest.mock import patch, MagicMock
from aizynthfinder.projects.retrofail.experiments.production.src.analysis.template_analyser import TemplateAnalyser

class TestTemplateAnalyser(unittest.TestCase):

    def setUp(self):
        self.template_library = ["[C:1][C:2]>>[C:1][C:2]", "[O:1][C:2]>>[O:1][C:2]"]
        self.analyser = TemplateAnalyser(self.template_library)

    def test_find_popular_templates_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.analyser.find_popular_templates([], {})

    def test_find_unused_templates_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.analyser.find_unused_templates([], [], {})

    @patch("aizynthfinder.projects.retrofail.experiments.production.src.analysis.template_analyser.canonicalise_smarts")
    def test_find_novel_overlooked_templates(self, mock_canonicalise):
        mock_canonicalise.side_effect = lambda x: x.replace(":", "").replace("[", "").replace("]", "")  # Simplify canonicalisation for testing

        unused_templates = {
            "[C:1][C:2]>>[C:1][C:2]": 0.5,
            "[O:1][C:2]>>[O:1][N:2]": 0.2,
            "[N:1][C:2]>>[N:1][O:2]": 0.8,
        }

        overlooked_templates, novel_templates = self.analyser.find_novel_overlooked_templates(unused_templates)

        self.assertEqual(overlooked_templates, {"[C:1][C:2]>>[C:1][C:2]": 0.5})
        self.assertEqual(novel_templates, {
            "OC>>ON": 0.2,
            "NC>>NO": 0.8
        })

if __name__ == '__main__':
    unittest.main()
