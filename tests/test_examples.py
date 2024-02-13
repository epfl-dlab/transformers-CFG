import unittest
import subprocess
import os
import glob


class TestExampleScripts(unittest.TestCase):
    def run_script(self, path_to_script):
        """Utility method to run a script using subprocess and assert that it exits with a 0 status code."""
        result = subprocess.run(
            ["python", path_to_script], capture_output=True, text=True
        )
        self.assertEqual(
            result.returncode,
            0,
            f"Script {path_to_script} failed with output:\n{result.stdout}\n{result.stderr}",
        )

    def test_all_example_scripts(self):
        """Test that all example scripts can be run without error."""
        for script_path in glob.glob("examples/*.py"):
            self.run_script(script_path)
