import asyncio
import importlib.util
import json
import shutil
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_ISOLATED_MODULE_DIR = tempfile.TemporaryDirectory()
_ISOLATED_MEMORY_PATH = Path(_ISOLATED_MODULE_DIR.name) / "memory.py"
shutil.copy2(Path(__file__).with_name("memory.py"), _ISOLATED_MEMORY_PATH)
_MEMORY_SPEC = importlib.util.spec_from_file_location("memory", _ISOLATED_MEMORY_PATH)
memory = importlib.util.module_from_spec(_MEMORY_SPEC)
sys.modules["memory"] = memory
_MEMORY_SPEC.loader.exec_module(memory)

import app


class MemoryCategoryTests(unittest.TestCase):
    def test_assistant_name_is_qwen(self):
        self.assertIn("Your name is Qwen.", app.SYSTEM_PROMPT)

    def test_detects_questions_about_assistant_identity_only(self):
        self.assertTrue(app._is_assistant_identity_question("what is your name"))
        self.assertTrue(app._is_assistant_identity_question("Who are you?"))
        self.assertFalse(app._is_assistant_identity_question("what is my name"))

    def test_direct_identity_question_gets_fixed_reply_without_model(self):
        payload = app.ChatRequest(
            messages=[
                {"role": "assistant", "content": "My name is Brian."},
                {"role": "user", "content": "what is your name"},
            ]
        )
        with patch.object(app.tools_engine, "chat_with_tools") as model_call:
            result = app.chat_endpoint(payload)
        self.assertEqual("I'm Qwen, your personal assistant.", result["reply"])
        model_call.assert_not_called()

    def test_streamed_identity_reply_ignores_conflicting_chat_history(self):
        payload = app.ChatRequest(
            messages=[
                {"role": "assistant", "content": "My name is Brian."},
                {"role": "user", "content": "what is your name?"},
            ]
        )
        with patch.object(app, "prepare_chat_request") as prepare:
            response = app.stream_chat_endpoint(payload)

        async def collect_events():
            return [chunk async for chunk in response.body_iterator]

        chunks = asyncio.run(collect_events())
        events = [json.loads(chunk) for chunk in chunks]
        self.assertEqual("I'm Qwen, your personal assistant.", events[0]["delta"])
        self.assertTrue(events[1]["context"]["direct_answer"])
        prepare.assert_not_called()

    def test_chat_mode_controls_local_knowledge_tool_access(self):
        local_payload = app.ChatRequest(
            messages=[{"role": "user", "content": "Search my notes"}],
            mode="local",
        )
        local_context = {"context_length": 4096}
        with (
            patch.object(app, "prepare_chat_request", return_value=(app.client, app.MODEL, [], local_context)),
            patch.object(app.tools_engine, "chat_with_tools", return_value="Local reply") as local_call,
        ):
            app.chat_endpoint(local_payload)
        self.assertTrue(local_call.call_args.kwargs["allow_local_knowledge"])

        cloud_payload = app.ChatRequest(
            messages=[{"role": "user", "content": "Search my notes"}],
            mode="cloud",
        )
        with (
            patch.object(app, "prepare_chat_request", return_value=(app.client, "cloud-model", [], None)),
            patch.object(app.tools_engine, "chat_with_tools", return_value="Cloud reply") as cloud_call,
        ):
            app.chat_endpoint(cloud_payload)
        self.assertFalse(cloud_call.call_args.kwargs["allow_local_knowledge"])

    def test_openrouter_escalation_sends_only_question_and_no_local_knowledge(self):
        question = "Can you explain this more clearly?"
        payload = app.StrongerModelRequest(question=question)
        cloud_client = object()
        with (
            patch.object(
                app,
                "prepare_chat_request",
                return_value=(cloud_client, "configured-cloud-model", [{"role": "system", "content": "Cloud prompt"}], None),
            ) as prepare,
            patch.object(
                app.tools_engine,
                "chat_with_tools_stream",
                return_value=iter(["Cloud explanation"]),
            ) as model_call,
        ):
            response = app.stream_stronger_model_endpoint(payload)
            async def collect_events():
                return [chunk async for chunk in response.body_iterator]

            events = [json.loads(chunk) for chunk in asyncio.run(collect_events())]
        prepared_payload = prepare.call_args.args[0]
        self.assertEqual("cloud", prepared_payload.mode)
        self.assertEqual([{"role": "user", "content": question}], prepared_payload.messages)
        self.assertTrue(any(event.get("delta") == "Cloud explanation" for event in events))
        self.assertFalse(model_call.call_args.kwargs["allow_local_knowledge"])

    def test_identity_question_does_not_retrieve_user_memories(self):
        payload = app.ChatRequest(
            messages=[{"role": "user", "content": "what is your name"}]
        )
        with (
            patch.object(app.memory, "search_memories") as search,
            patch.object(app, "_local_context_length", return_value=(4096, True)),
        ):
            _, _, messages, _ = app.prepare_chat_request(payload)
        search.assert_not_called()
        self.assertIn("Your name is Qwen", messages[0]["content"])

    def test_short_followup_retries_memory_search_with_recent_user_turns(self):
        payload = app.ChatRequest(messages=[
            {"role": "user", "content": "Remember I prefer concise answers."},
            {"role": "assistant", "content": "Got it."},
            {"role": "user", "content": "What about that?"},
        ])
        saved_fact = {
            "id": 50,
            "fact": "I prefer concise answers.",
            "category": "preferences",
        }
        with (
            patch.object(memory, "search_memories", side_effect=[[], [saved_fact]]) as search,
            patch.object(app, "_local_context_length", return_value=(4096, True)),
        ):
            _, _, messages, _ = app.prepare_chat_request(payload)

        self.assertEqual("What about that?", search.call_args_list[0].args[0])
        self.assertIn("Remember I prefer concise answers", search.call_args_list[1].args[0])
        self.assertIn("I prefer concise answers", messages[0]["content"])

    def test_headline_failure_surfaces_search_diagnostic(self):
        with patch.object(
            app.tools_engine,
            "web_search",
            return_value={
                "error": "Web search is temporarily unavailable.",
                "detail": "TimeoutException: timed out",
            },
        ):
            reply = app.tools_engine._headline_response([
                {"role": "user", "content": "What are today's headlines?"}
            ])
        self.assertIn("TimeoutException", reply)

    def test_headlines_drop_undated_and_stale_items_and_sort_newest_first(self):
        now = datetime(2026, 10, 27, 20, 0, tzinfo=timezone.utc)
        rows = [
            {"title": "yesterday", "date": "2026-10-26T21:00:00+00:00"},
            {"title": "older", "date": "2026-10-26T19:00:00+00:00"},
            {"title": "today", "date": "2026-10-27T19:00:00Z"},
            {"title": "undated", "date": ""},
            {"title": "future", "date": "2026-10-28T01:00:00+00:00"},
        ]
        result = app.tools_engine._recent_news_results(rows, now=now)
        self.assertEqual(["today", "yesterday"], [item["title"] for item in result])

    def test_web_search_returns_only_recent_dated_headlines(self):
        now = datetime.now(timezone.utc)
        fresh = (now - timedelta(hours=1)).isoformat()
        stale = (now - timedelta(days=3)).isoformat()

        class FakeDDGS:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def news(self, *args, **kwargs):
                return [
                    {"title": "Fresh story", "url": "https://example.com/fresh", "date": fresh, "source": "Example"},
                    {"title": "Old story", "url": "https://example.com/old", "date": stale, "source": "Example"},
                ]

        with patch.object(app.tools_engine, "DDGS", return_value=FakeDDGS()):
            result = app.tools_engine.web_search("today's headlines")

        self.assertEqual(1, result["count"])
        self.assertEqual("Fresh story", result["results"][0]["title"])

    def test_memory_context_marks_first_person_facts_as_the_users(self):
        context = app._format_memory_context([
            {"category": "personal", "fact": "My birthday is December 17."}
        ])
        self.assertEqual(
            "- [User's personal memory] My birthday is December 17.", context
        )
        self.assertIn('"my"', app.SYSTEM_PROMPT)

    def test_existing_and_new_categories_are_valid(self):
        for category in (
            "general", "preferences", "personal", "projects", "people",
            "setup", "decisions", "lessons", "temporary",
        ):
            with self.subTest(category=category):
                self.assertEqual(category, memory.validate_category(category))

    def test_category_suggestion_uses_local_model_and_returns_valid_choice(self):
        model_reply = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="projects\n"))]
        )
        with patch.object(
            app.client.chat.completions, "create", return_value=model_reply
        ) as create:
            result = app.suggest_memory_category(
                app.SuggestMemoryCategoryRequest(fact="I am building PersonalAI")
            )

        self.assertEqual({"category": "projects"}, result)
        self.assertEqual(app.MODEL, create.call_args.kwargs["model"])
        self.assertEqual(0, create.call_args.kwargs["temperature"])

    def test_invalid_fact_is_rejected_before_model_call(self):
        with patch.object(app.client.chat.completions, "create") as create:
            with self.assertRaises(app.HTTPException) as raised:
                app.suggest_memory_category(
                    app.SuggestMemoryCategoryRequest(fact=" ")
                )
        self.assertEqual(400, raised.exception.status_code)
        create.assert_not_called()

    def test_unknown_model_category_is_not_accepted(self):
        model_reply = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="unknown"))]
        )
        with patch.object(
            app.client.chat.completions, "create", return_value=model_reply
        ):
            with self.assertRaises(app.HTTPException) as raised:
                app.suggest_memory_category(
                    app.SuggestMemoryCategoryRequest(fact="Some fact")
                )
        self.assertEqual(502, raised.exception.status_code)


class ChatSessionStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "memory.db"
        self.database_path_patch = patch.object(memory, "DB_PATH", self.database_path)
        self.database_path_patch.start()
        memory.init_db()

    def tearDown(self):
        self.database_path_patch.stop()
        self.temp_dir.cleanup()

    def test_sessions_save_reopen_and_delete_independently(self):
        local_id = memory.create_chat_session("local")
        cloud_id = memory.create_chat_session("cloud")
        local_messages = [
            {"role": "user", "content": "Plan my weekend"},
            {"role": "assistant", "content": "Sure."},
        ]
        self.assertEqual(2, memory.replace_session_history(local_id, local_messages))
        self.assertEqual(1, memory.replace_session_history(cloud_id, [local_messages[0]]))

        local = memory.get_chat_session(local_id)
        self.assertEqual("Plan my weekend", local["title"])
        self.assertEqual(local_messages, memory.list_session_history(local_id))
        self.assertEqual([local_messages[0]], memory.list_session_history(cloud_id))

        self.assertTrue(memory.delete_chat_session(local_id))
        self.assertIsNone(memory.get_chat_session(local_id))
        self.assertEqual([local_messages[0]], memory.list_session_history(cloud_id))

    def test_legacy_mode_histories_migrate_once_without_losing_messages(self):
        legacy_path = Path(self.temp_dir.name) / "legacy.db"
        connection = sqlite3.connect(legacy_path)
        connection.execute("""
            CREATE TABLE chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mode TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        connection.executemany("""
            INSERT INTO chat_messages (mode, role, content, created_at)
            VALUES (?, ?, ?, ?)
        """, [
            ("local", "user", "Local old chat", "2026-01-01T00:00:00+00:00"),
            ("local", "assistant", "Local reply", "2026-01-01T00:01:00+00:00"),
            ("cloud", "user", "Cloud old chat", "2026-01-02T00:00:00+00:00"),
            ("cloud", "assistant", "Cloud reply", "2026-01-02T00:01:00+00:00"),
        ])
        connection.commit()
        connection.close()

        with patch.object(memory, "DB_PATH", legacy_path):
            memory.init_db()
            sessions = memory.list_chat_sessions()
            self.assertEqual(2, len(sessions))
            self.assertEqual({"local", "cloud"}, {item["mode"] for item in sessions})
            local_session = next(item for item in sessions if item["mode"] == "local")
            cloud_session = next(item for item in sessions if item["mode"] == "cloud")
            self.assertEqual("Local old chat", local_session["title"])
            self.assertEqual("Cloud old chat", cloud_session["title"])
            self.assertEqual(2, len(memory.list_session_history(local_session["id"])))
            self.assertEqual(2, len(memory.list_session_history(cloud_session["id"])))
            memory.init_db()
            self.assertEqual(2, len(memory.list_chat_sessions()))


if __name__ == "__main__":
    unittest.main()
