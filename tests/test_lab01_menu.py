"""Console selection and repeat execution without loading GPU frameworks."""

import io
import unittest
from unittest.mock import patch

from deep_learning_course.lab01.__main__ import choose_task, main


class MenuTests(unittest.TestCase):
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_choices_and_invalid_input(self, output):
        for choice, expected in [("1", "1"), ("7", "7"), ("8", "all"), ("0", None)]:
            with self.subTest(choice=choice), patch("builtins.input", side_effect=["invalid", choice]):
                self.assertEqual(choose_task(), expected)
        self.assertIn("Wpisz numer od 0 do 8.", output.getvalue())

    def test_eof_exits(self):
        with patch("builtins.input", side_effect=EOFError), patch("sys.stdout", new_callable=io.StringIO):
            self.assertIsNone(choose_task())

    def test_menu_reuses_environment_and_runs_only_selected_tasks(self):
        environment = {"gpu": "test"}
        with patch("deep_learning_course.lab01.__main__.choose_task", side_effect=["4", "6", None]), \
                patch("deep_learning_course.lab01.__main__.run_tasks", return_value=environment) as run:
            main(["--fast"])
        self.assertEqual([call.args[0] for call in run.call_args_list], ["4", "6"])
        self.assertIsNone(run.call_args_list[0].kwargs["environment"])
        self.assertIs(run.call_args_list[1].kwargs["environment"], environment)

    def test_argument_skips_menu(self):
        with patch("deep_learning_course.lab01.__main__.choose_task") as menu, \
                patch("deep_learning_course.lab01.__main__.run_tasks") as run:
            main(["--task", "5", "--fast"])
        menu.assert_not_called()
        self.assertEqual(run.call_args.args[0], "5")
        self.assertTrue(run.call_args.kwargs["fast"])


if __name__ == "__main__":
    unittest.main()
