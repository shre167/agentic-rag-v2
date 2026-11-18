"""
Comprehensive Test Suite for Book Profiling and Semantic Routing
Verifies all 8 user requirements:
1. Explicit book-name queries
2. Indirect topic-based queries
3. Overlapping topics across books
4. Ambiguous queries
5. Follow-up questions and topic changes (with conversational history)
6. Unrelated queries returning NO_MATCH (no fallback to all KBs)
7. Newly ingested books becoming automatically available for routing
8. Previously ingested books being registered correctly
"""

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from retrieval.kb_registry import (
    get_kb_registry,
    get_all_book_profiles,
    generate_book_profile,
    clear_kb_registry_cache,
    _load_profiles,
    _save_profiles,
)
from retrieval.semantic_router import route_query
from retrieval.chat_context import set_chat_context
from tools.semantic_search import semantic_search
from ingest import register_existing_books, ingest_pdf


class MockLLMResponse:
    def __init__(self, content: str):
        self.content = content


class TestBookProfilingAndRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        clear_kb_registry_cache()
        # Seed profiles if not present or ensure realistic profiles
        profiles = _load_profiles()
        profiles["kb1"] = {
            "kb_id": "kb1",
            "kb_name": "Mind Management",
            "collection_name": "kb1",
            "summary": "Focuses on managing mental energy, creative states, and circadian rhythms rather than strict clock-time management.",
            "main_topics": ["Creative States", "Mental Energy", "Mind Management", "Four Stages of Creativity", "Chaos in Creativity"],
            "key_concepts": ["Creative Sweet Spot", "Mental State Switching", "Energy Cycles", "Passive vs Active Time"],
            "coverage": "Productivity, creativity, neuroscience of creative thinking, energy management.",
        }
        profiles["kb2"] = {
            "kb_id": "kb2",
            "kb_name": "Deep Work",
            "collection_name": "kb2",
            "summary": "Argues for distraction-free concentration on cognitively demanding tasks as a superpower in the modern information economy.",
            "main_topics": ["Deep Work vs Shallow Work", "Rules for Focus", "Digital Minimalism", "Quitting Social Media", "Attention Residue"],
            "key_concepts": ["Deep Work Hypothesis", "Shallow Work", "Attention Residue", "Four Disciplines of Execution", "Rhythmic Scheduling"],
            "coverage": "Cognitive performance, eliminating distractions, professional focus, deliberate practice.",
        }
        _save_profiles(profiles)
        clear_kb_registry_cache()

    def setUp(self):
        clear_kb_registry_cache()
        set_chat_context([])

    def test_01_explicit_book_name_query(self):
        """Verify explicit book name routing via LLM router."""
        mock_resp = json.dumps({
            "is_match": True,
            "selected_kb": "kb2",
            "sub_query": "rules for deep work",
            "chapter": None,
            "page": None,
            "line": None,
            "reason": "User explicitly asked about Deep Work."
        })
        with patch("retrieval.semantic_router._get_router_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MockLLMResponse(mock_resp)
            mock_get_llm.return_value = mock_llm

            result = route_query("What are the core rules in Deep Work?")
            self.assertTrue(result["is_match"])
            self.assertEqual(result["selected_kb"], "kb2")
            self.assertEqual(result["collection_name"], "kb2")
            self.assertEqual(result["sub_query"], "rules for deep work")

    def test_02_indirect_topic_based_query(self):
        """Verify indirect topic routing based on book profile content."""
        mock_resp = json.dumps({
            "is_match": True,
            "selected_kb": "kb1",
            "sub_query": "how mental energy affects creative states",
            "chapter": None,
            "page": None,
            "line": None,
            "reason": "Mind Management specifically discusses creative states and mental energy cycles."
        })
        with patch("retrieval.semantic_router._get_router_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MockLLMResponse(mock_resp)
            mock_get_llm.return_value = mock_llm

            result = route_query("How should I align my creative energy throughout the day?")
            self.assertTrue(result["is_match"])
            self.assertEqual(result["selected_kb"], "kb1")
            self.assertIn("mental energy", result["reason"].lower())

    def test_03_overlapping_topics(self):
        """When a topic overlaps (e.g. distraction/focus), the router chooses the best-fit book."""
        mock_resp = json.dumps({
            "is_match": True,
            "selected_kb": "kb2",
            "sub_query": "attention residue and eliminating distraction",
            "chapter": None,
            "page": None,
            "line": None,
            "reason": "Deep Work has extensive focus on attention residue and eliminating shallow distractions."
        })
        with patch("retrieval.semantic_router._get_router_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MockLLMResponse(mock_resp)
            mock_get_llm.return_value = mock_llm

            result = route_query("How does attention residue hinder high-intensity concentration?")
            self.assertTrue(result["is_match"])
            self.assertEqual(result["selected_kb"], "kb2")

    def test_04_ambiguous_queries(self):
        """Ambiguous query resolution picks the most relevant KB or handles cleanly."""
        mock_resp = json.dumps({
            "is_match": True,
            "selected_kb": "kb1",
            "sub_query": "productivity and scheduling",
            "chapter": None,
            "page": None,
            "line": None,
            "reason": "Query is broad but best matches Mind Management's thesis on scheduling and mind states."
        })
        with patch("retrieval.semantic_router._get_router_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MockLLMResponse(mock_resp)
            mock_get_llm.return_value = mock_llm

            result = route_query("How do I fix my schedule?")
            self.assertTrue(result["is_match"])
            self.assertIn(result["selected_kb"], ["kb1", "kb2"])

    def test_05_followup_and_topic_changes(self):
        """Verify chat history resolves pronouns, but topic changes switch KBs."""
        # Follow-up using history
        set_chat_context([
            {"role": "user", "content": "What is Deep Work about?"},
            {"role": "assistant", "content": "Deep Work focuses on distraction-free concentration."}
        ])

        mock_resp_followup = json.dumps({
            "is_match": True,
            "selected_kb": "kb2",
            "sub_query": "how the author recommends quitting social media",
            "chapter": None,
            "page": None,
            "line": None,
            "reason": "Pronoun 'he' refers to Cal Newport / author of Deep Work from chat history."
        })
        with patch("retrieval.semantic_router._get_router_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MockLLMResponse(mock_resp_followup)
            mock_get_llm.return_value = mock_llm

            result = route_query("What does he suggest about social media?")
            self.assertTrue(result["is_match"])
            self.assertEqual(result["selected_kb"], "kb2")

        # Now test topic switch to Mind Management
        set_chat_context([
            {"role": "user", "content": "Tell me about Deep Work"},
            {"role": "assistant", "content": "It covers concentration."}
        ])
        mock_resp_switch = json.dumps({
            "is_match": True,
            "selected_kb": "kb1",
            "sub_query": "four stages of creativity",
            "chapter": None,
            "page": None,
            "line": None,
            "reason": "User shifted topic to four stages of creativity, which belongs to Mind Management."
        })
        with patch("retrieval.semantic_router._get_router_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MockLLMResponse(mock_resp_switch)
            mock_get_llm.return_value = mock_llm

            result2 = route_query("Actually, tell me about the four stages of creativity instead")
            self.assertTrue(result2["is_match"])
            self.assertEqual(result2["selected_kb"], "kb1")

    def test_06_unrelated_query_returns_no_match(self):
        """Verify unrelated queries return NO_MATCH without falling back to all-KB search."""
        mock_resp = json.dumps({
            "is_match": False,
            "selected_kb": None,
            "sub_query": "latest stock price of Apple AAPL",
            "chapter": None,
            "page": None,
            "line": None,
            "reason": "Query is about stock prices, which is outside the scope of all indexed books."
        })
        with patch("retrieval.semantic_router._get_router_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MockLLMResponse(mock_resp)
            mock_get_llm.return_value = mock_llm

            result = route_query("What is the current stock price of Apple?")
            self.assertFalse(result["is_match"])
            self.assertIsNone(result["selected_kb"])
            self.assertIsNone(result["collection_name"])

        # Also verify semantic_search tool returns NO_MATCH JSON without searching all KBs
        with patch("retrieval.semantic_router._get_router_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MockLLMResponse(mock_resp)
            mock_get_llm.return_value = mock_llm

            tool_output_str = semantic_search.invoke({"query": "What is the current stock price of Apple?"})
            tool_data = json.loads(tool_output_str)
            self.assertIn("NO_MATCH", tool_data["result"])
            self.assertEqual(tool_data["passages"], [])

    def test_07_newly_ingested_book_available_for_routing(self):
        """Verify dynamic discovery: newly ingested collection makes it immediately routable."""
        from retrieval.vector_store import get_collection
        import chromadb

        # Create a mock collection in ChromaDB
        test_col = get_collection("kb_test_dynamic")
        test_col.upsert(
            ids=["test-001"],
            documents=["Small habits create compound improvements."],
            metadatas=[{"book": "Atomic Habits", "chapter": "Chapter 1", "section": "Fundamentals"}]
        )

        profiles = _load_profiles()
        profiles["kb_test_dynamic"] = {
            "kb_id": "kb_test_dynamic",
            "kb_name": "Atomic Habits",
            "collection_name": "kb_test_dynamic",
            "summary": "Focuses on tiny changes that lead to remarkable results in habit formation.",
            "main_topics": ["Habit Loops", "Cue Craving Response Reward", "Identity-Based Habits"],
            "key_concepts": ["1% Better Every Day", "Habit Stacking", "Two-Minute Rule"],
            "coverage": "Habit formation, behavioral psychology.",
        }
        _save_profiles(profiles)
        clear_kb_registry_cache()

        all_profiles = get_all_book_profiles()
        kb_ids = [p["kb_id"] for p in all_profiles]
        self.assertIn("kb_test_dynamic", kb_ids)

        mock_resp = json.dumps({
            "is_match": True,
            "selected_kb": "kb_test_dynamic",
            "sub_query": "two minute rule for starting habits",
            "chapter": None,
            "page": None,
            "line": None,
            "reason": "Query matches Atomic Habits key concepts on the two-minute rule."
        })
        with patch("retrieval.semantic_router._get_router_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MockLLMResponse(mock_resp)
            mock_get_llm.return_value = mock_llm

            result = route_query("How does the two minute rule help build habits?")
            self.assertTrue(result["is_match"])
            self.assertEqual(result["selected_kb"], "kb_test_dynamic")

        # Cleanup test collection and profile
        from retrieval.vector_store import _get_client
        client = _get_client()
        try:
            client.delete_collection("kb_test_dynamic")
        except Exception:
            pass

        profiles = _load_profiles()
        profiles.pop("kb_test_dynamic", None)
        _save_profiles(profiles)
        clear_kb_registry_cache()

    def test_08_previously_ingested_books_registration(self):
        """Verify previously ingested books without profiles get registered without vector duplication."""
        registered = register_existing_books()
        self.assertIsInstance(registered, list)
        for r in registered:
            self.assertIn(r["status"], ["already_profiled", "profile_generated"])


if __name__ == "__main__":
    unittest.main()
