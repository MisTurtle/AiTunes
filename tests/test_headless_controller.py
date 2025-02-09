import autoloader
from aitunes.experiments.scenarios._scenario_utils import ScenarioContainer, ScenarioDescriptor
from aitunes.user_controls.headless import HeadlessActionPipeline

import unittest
from unittest.mock import patch, MagicMock


class TestHeadlessController(unittest.TestCase):
    # Writing these tests allowed me to understand how mocks work and why they exist. Props !

    def setUp(self):
        self.controller = HeadlessActionPipeline()
    
    @patch("aitunes.experiments.scenarios.all_scenarios", { "exp1": MagicMock(spec=ScenarioContainer) })
    def test_select_experiment(self):
        experiments = self.controller.list_scripted_experiments()
        self.assertIn("exp1", experiments)
        self.assertFalse(self.controller.select_experiment("exp2"))
        self.assertTrue(self.controller.select_experiment("exp1"))
        self.assertIsNotNone(self.controller.get_selected_experiment())
        self.controller.close_experiment()
        self.assertIsNone(self.controller.get_selected_experiment())

    @patch("aitunes.experiments.scenarios.all_scenarios", {"exp1": MagicMock(spec=ScenarioContainer)})
    def test_select_scenario(self):
        self.assertRaises(RuntimeError, lambda: self.controller.select_scenario("sc1"))
        self.controller.select_experiment("exp1")
        self.controller._selected_experiment.get_scenarios.return_value = {"sc1": MagicMock(spec=ScenarioDescriptor)}
        self.assertFalse(self.controller.select_scenario("sc2"))
        self.assertTrue(self.controller.select_scenario("sc1"))
        self.assertIsNotNone(self.controller.get_selected_scenario(), self.controller.get_selected_scenario())
        self.controller.close_scenario()
        self.assertIsNone(self.controller.get_selected_scenario())

if __name__ == "__main__":
    unittest.main()
