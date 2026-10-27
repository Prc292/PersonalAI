import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

import tools_engine


class LocalKnowledgeSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "project.md").write_text(
            "# PersonalAI\n\nThe local Qwen setup uses LM Studio. "
            "Keep the model context small for faster replies.\n",
            encoding="utf-8",
        )
        (self.root / "notes.txt").write_text(
            "A separate note about weekend gardening.", encoding="utf-8"
        )
        (self.root / "orchid-workshop.md").write_text(
            "The Orange Lantern orchid usually blooms in May.", encoding="utf-8"
        )
        (self.root / "hadley-guide.md").write_text(
            "## Page 2\nHadley uses a d114/f900mm primary mirror set.\n\n"
            "## Page 72\nA.2: Eyepiece diameter. Hadley cannot use 2 inch eyepieces.",
            encoding="utf-8",
        )
        self.pdf_path = self.root / "qwen-notes.pdf"
        writer = PdfWriter()
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})
        })
        content = DecodedStreamObject()
        content.set_data(b"BT /F1 12 Tf 72 720 Td (Qwen PDF extraction phrase) Tj ET")
        page[NameObject("/Contents")] = writer._add_object(content)
        with self.pdf_path.open("wb") as output:
            writer.write(output)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_search_returns_matching_markdown_excerpt_and_file_path(self):
        result = tools_engine.search_local_knowledge(
            "LM Studio setup", knowledge_dir=str(self.root)
        )
        self.assertEqual("knowledge", result["kind"])
        self.assertEqual("project.md", result["results"][0]["file"])
        self.assertIn("LM Studio", result["results"][0]["excerpt"])
        self.assertEqual(1, result["count"])

    def test_specific_orchid_query_ranks_markdown_above_unrelated_pdf(self):
        result = tools_engine.search_local_knowledge(
            "When does the Orange Lantern orchid bloom, and which file says so?",
            knowledge_dir=str(self.root),
        )
        self.assertEqual("orchid-workshop.md", result["results"][0]["file"])
        self.assertIn("blooms in May", result["results"][0]["excerpt"])

    def test_aperture_query_prefers_primary_mirror_spec_page(self):
        result = tools_engine.search_local_knowledge(
            "Can you tell me the aperture size of the Hadley?",
            knowledge_dir=str(self.root),
        )
        self.assertEqual("hadley-guide.md", result["results"][0]["file"])
        self.assertEqual(2, result["results"][0]["page"])
        self.assertIn("d114/f900mm", result["results"][0]["excerpt"])

    def test_pdf_text_is_extracted_and_searchable(self):
        with patch.object(tools_engine, "KNOWLEDGE_MAX_FILE_BYTES", 1), patch.object(
            tools_engine, "KNOWLEDGE_MAX_TOTAL_BYTES", 1
        ):
            result = tools_engine.search_local_knowledge(
                "Qwen PDF extraction", knowledge_dir=str(self.root)
            )
        self.assertEqual("qwen-notes.pdf", result["results"][0]["file"])
        self.assertIn("Qwen PDF extraction phrase", result["results"][0]["excerpt"])
        self.assertEqual(1, result["results"][0]["page"])

    def test_unsupported_files_and_unrelated_documents_are_not_returned(self):
        result = tools_engine.search_local_knowledge(
            "unrelated document terms", knowledge_dir=str(self.root)
        )
        self.assertEqual([], result["results"])

    def test_empty_query_and_empty_folder_are_handled(self):
        empty_query = tools_engine.search_local_knowledge(
            " ", knowledge_dir=str(self.root)
        )
        self.assertEqual("Please provide a search query.", empty_query["message"])
        with tempfile.TemporaryDirectory() as empty:
            result = tools_engine.search_local_knowledge("anything", knowledge_dir=empty)
        self.assertEqual([], result["results"])

    def test_symlinked_documents_are_not_followed(self):
        with tempfile.TemporaryDirectory() as outside:
            secret = Path(outside) / "notes.md"
            secret.write_text("private material", encoding="utf-8")
            (self.root / "linked.md").symlink_to(secret)
            result = tools_engine.search_local_knowledge(
                "private material", knowledge_dir=str(self.root)
            )
        self.assertEqual([], result["results"])

    def test_cloud_mode_does_not_receive_or_run_local_knowledge_tool(self):
        local_names = {
            item["function"]["name"]
            for item in tools_engine.available_tools(allow_local_knowledge=True)
        }
        cloud_names = {
            item["function"]["name"]
            for item in tools_engine.available_tools(allow_local_knowledge=False)
        }
        self.assertIn("search_local_knowledge", local_names)
        self.assertNotIn("search_local_knowledge", cloud_names)
        result = tools_engine.run_tool(
            "search_local_knowledge",
            {"query": "Qwen"},
            allow_local_knowledge=False,
        )
        self.assertIn("error", result)

    def test_direct_headline_search_emits_tool_status_events(self):
        with patch.object(tools_engine, "_headline_response", return_value="Headlines here"):
            events = list(tools_engine.chat_with_tools_stream(
                request_client=None,
                model_name="local-qwen",
                messages=[{"role": "user", "content": "What are today's headlines?"}],
            ))

        self.assertEqual(
            [
                {"tool_status": {"name": "web_search", "status": "running", "source": "direct"}},
                {"tool_status": {"name": "web_search", "status": "complete", "source": "direct"}},
                "Headlines here",
            ],
            events,
        )

    def test_explicit_local_document_search_runs_tool_without_asking_qwen(self):
        search_result = {
            "kind": "knowledge",
            "results": [{
                "file": "hadley-instructions-v1.pdf",
                "excerpt": "Hadley is listed in the instructions.",
            }],
        }
        with patch.object(tools_engine, "search_local_knowledge", return_value=search_result) as search:
            events = list(tools_engine.chat_with_tools_stream(
                request_client=None,
                model_name="local-qwen",
                messages=[{
                    "role": "user",
                    "content": "Search my PDFs for Hadley. Which PDF did you find it in?",
                }],
                allow_local_knowledge=True,
            ))

        search.assert_called_once_with("Hadley")
        self.assertEqual("search_local_knowledge", events[0]["tool_status"]["name"])
        self.assertEqual("running", events[0]["tool_status"]["status"])
        self.assertIn("hadley-instructions-v1.pdf", events[-1])
        self.assertIn("Hadley is listed", events[-1])

    def test_strong_local_match_is_checked_before_qwen_can_choose_web_search(self):
        search_result = {
            "kind": "knowledge",
            "results": [{
                "file": "test-facts.txt",
                "excerpt": "The Cedar Room greenhouse is kept at 21 degrees Celsius.",
                "score": 8,
                "page": 7,
            }],
        }
        fake_chunk = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="It is kept at 21 degrees Celsius."))],
            usage=None,
        )
        create = Mock(return_value=[fake_chunk])
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )
        with patch.object(tools_engine, "search_local_knowledge", return_value=search_result) as search:
            events = list(tools_engine.chat_with_tools_stream(
                request_client=fake_client,
                model_name="local-qwen",
                messages=[{
                    "role": "user",
                    "content": "What temperature is the Cedar Room greenhouse kept at?",
                }],
                allow_local_knowledge=True,
            ))

        search.assert_called_once_with("What temperature is the Cedar Room greenhouse kept at?")
        self.assertEqual("search_local_knowledge", events[0]["tool_status"]["name"])
        self.assertIn("21 degrees Celsius", events[-1])
        self.assertNotIn("tools", create.call_args.kwargs)
        self.assertIn(
            "test-facts.txt, page 7",
            create.call_args.kwargs["messages"][-1]["content"],
        )

    def test_weather_questions_skip_local_knowledge_preflight(self):
        self.assertFalse(tools_engine._should_preflight_local_knowledge([
            {"role": "user", "content": "What's the weather in Boston tomorrow?"}
        ]))

    def test_asking_for_pdf_files_lists_names_without_searching_pdf_text(self):
        with patch.object(
            tools_engine, "list_local_knowledge_files", return_value=["hadley-instructions-v1.pdf"]
        ) as list_files, patch.object(tools_engine, "search_local_knowledge") as search:
            events = list(tools_engine.chat_with_tools_stream(
                request_client=None,
                model_name="local-qwen",
                messages=[{
                    "role": "user",
                    "content": "Search my Documents folder for PDF files",
                }],
                allow_local_knowledge=True,
            ))

        list_files.assert_called_once_with(".pdf")
        search.assert_not_called()
        self.assertEqual("hadley-instructions-v1.pdf", events[-1])

    def test_filename_only_followup_stays_concise(self):
        messages = [
            {"role": "user", "content": "Search my Documents folder for PDF files"},
            {"role": "assistant", "content": "I searched for PDF files."},
            {"role": "user", "content": "Just list the file name"},
        ]
        with patch.object(
            tools_engine, "list_local_knowledge_files", return_value=["hadley-instructions-v1.pdf"]
        ) as list_files:
            events = list(tools_engine.chat_with_tools_stream(
                request_client=None,
                model_name="local-qwen",
                messages=messages,
                allow_local_knowledge=True,
            ))
        list_files.assert_called_once_with(".pdf")
        self.assertEqual("hadley-instructions-v1.pdf", events[-1])


if __name__ == "__main__":
    unittest.main()
