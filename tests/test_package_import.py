#!/usr/bin/env python3
"""Import tests for the project-owned modeling package."""

from __future__ import annotations

import unittest

from fraud_model import __version__


class PackageImportTest(unittest.TestCase):
    def test_version_is_string(self) -> None:
        self.assertIsInstance(__version__, str)
        self.assertGreater(len(__version__), 0)


if __name__ == "__main__":
    unittest.main()
