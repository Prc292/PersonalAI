import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SOURCE = Path(__file__).with_name("memory.py")


class SearchMemoriesTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.module_path = Path(self.temp_dir.name) / "memory.py"
        shutil.copy2(SOURCE, self.module_path)
        spec = importlib.util.spec_from_file_location("isolated_memory", self.module_path)
        self.memory = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.memory)

    def tearDown(self):
        self.temp_dir.cleanup()

    def search(self, query, rows):
        with patch.object(self.memory, "list_memories", return_value=rows):
            return self.memory.search_memories(query)

    def test_uses_whole_words_not_substrings(self):
        rows = [
            {"id": 1, "fact": "I like party games", "category": "general"},
            {"id": 2, "fact": "I like art museums", "category": "general"},
        ]
        self.assertEqual([2], [row["id"] for row in self.search("art", rows)])

    def test_exact_phrase_ranks_above_separate_keyword_matches(self):
        rows = [
            {"id": 1, "fact": "My favorite color is blue", "category": "general"},
            {"id": 2, "fact": "My favorite room has a color palette", "category": "general"},
        ]
        self.assertEqual([1, 2], [row["id"] for row in self.search("favorite color", rows)])

    def test_category_keyword_can_match(self):
        rows = [{"id": 3, "fact": "Uses a local setup", "category": "preferences"}]
        self.assertEqual(rows, self.search("preferences", rows))

    def test_unrelated_query_returns_empty(self):
        rows = [{"id": 1, "fact": "Likes green tea", "category": "personal"}]
        self.assertEqual([], self.search("ocean", rows))

    def test_result_shape_and_newest_first_tie_break_are_unchanged(self):
        rows = [
            {"id": 1, "fact": "Likes tea", "category": "general", "created_at": "old"},
            {"id": 2, "fact": "Likes tea", "category": "general", "created_at": "new"},
        ]
        result = self.search("tea", rows)
        self.assertEqual([2, 1], [row["id"] for row in result])
        self.assertEqual(rows[1], result[0])


if __name__ == "__main__":
    unittest.main()
