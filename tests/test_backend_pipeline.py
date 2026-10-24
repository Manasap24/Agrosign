"""Regression checks for the sentence-to-process/sign backend path.

Run from the repository root with: venv\\Scripts\\python.exe -m unittest discover -s tests
"""

import csv
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from starlette.requests import Request
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from manual_translator import translate_manual
from main import app, translate, TextRequest, translate_text_endpoint, TranslateTextRequest
from process_detector import detect_process, initialize_process_detector, _bart_verify_candidates
from sign_sequence_generator import generate_video_sequence


class BackendPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        initialize_process_detector()

    def test_bart_is_only_called_for_uncertain_shortlisted_candidates(self):
        strong = [{"score": 0.91, "profile": {"process_name": "Irrigation"}}]
        with patch("process_detector.classify_processes") as classify:
            chosen, verified, bart_score = _bart_verify_candidates("The farmer irrigates the field.", strong)
        classify.assert_not_called()
        self.assertIs(chosen, strong[0])
        self.assertIsNone(verified)
        self.assertIsNone(bart_score)

        close = [
            {"score": 0.69, "profile": {"process_name": "Irrigation"}},
            {"score": 0.67, "profile": {"process_name": "Water Management"}},
            {"score": 0.40, "profile": {"process_name": "Fertilization"}},
            {"score": 0.39, "profile": {"process_name": "Soil Preparation"}},
        ]
        with patch("process_detector.classify_processes", return_value={
            "Irrigation": 0.62,
            "Water Management": 0.83,
            "Fertilization": 0.12,
        }) as classify:
            chosen, verified, bart_score = _bart_verify_candidates(
                "The farmer supplies water to the crops.", close
            )
        self.assertEqual([call.args[1] for call in classify.call_args_list], [[
            "Irrigation", "Water Management", "Fertilization"
        ]])
        # BART verifies the first-stage choice but does not substitute its own
        # general-domain label for a process selected by action/object evidence.
        self.assertEqual(chosen["profile"]["process_name"], "Irrigation")
        self.assertFalse(verified)
        self.assertEqual(bart_score, 0.62)

    def test_bart_uncertain_result_does_not_force_a_process(self):
        close = [
            {"score": 0.60, "profile": {"process_name": "Irrigation"}},
            {"score": 0.59, "profile": {"process_name": "Water Management"}},
        ]
        with patch("process_detector.classify_processes", return_value={
            "Irrigation": 0.54, "Water Management": 0.60
        }):
            chosen, verified, bart_score = _bart_verify_candidates("ambiguous clause", close)
        self.assertIs(chosen, close[0])
        self.assertFalse(verified)
        self.assertEqual(bart_score, 0.54)

    def test_benchmark_processes_context_and_ideal_concepts(self):
        benchmark = ROOT / "dataset" / "process_recognition_benchmark.csv"
        with benchmark.open(encoding="utf-8", newline="") as handle:
            cases = list(csv.DictReader(handle))
        self.assertEqual(len(cases), 16)
        for case in cases:
            with self.subTest(case=case["case_id"]):
                result = detect_process(case["text"])
                events = result["events"]
                ids = [event["process_id"] for event in events]
                expected_ids = case["expected_process_ids"].split("|") if case["expected_process_ids"] else []
                self.assertEqual(ids, expected_ids)
                if events:
                    self.assertEqual(events[0]["status"], case["expected_status"])
                    concepts = [concept for event in events for concept in event["sign_concepts"]]
                    for concept in case["expected_ideal_concepts"].split("|"):
                        self.assertIn(concept, concepts)

    def test_sequence_order_and_api_compatible_response(self):
        result = translate_manual("First prepare the soil, then sow the seeds and finally irrigate the field.")
        names = result["process_sequence"]
        self.assertEqual(names, ["Soil Preparation", "Seed Sowing", "Irrigation"])
        self.assertEqual([item["concept_id"] for item in result["ideal_sign_sequence"] if item["type"] == "process"],
                         ["SOIL_PREPARATION", "SEED_SOWING", "IRRIGATION"])
        self.assertTrue(result["complete_video_sequence"])
        self.assertEqual(result["translations"][0]["sentence"].startswith("First"), True)

    def test_page_length_passages_preserve_events_status_and_global_order(self):
        benchmark = ROOT / "dataset" / "page_level_benchmark.json"
        passages = json.loads(benchmark.read_text(encoding="utf-8"))
        self.assertEqual(len(passages), 2)
        for passage in passages:
            with self.subTest(case=passage["case_id"]):
                self.assertGreaterEqual(len(passage["text"].split()), 250)
                result = translate_manual(passage["text"])
                events = [event for sentence in result["translations"] for event in sentence["events"]]
                self.assertEqual([event["process_id"] for event in events], passage["expected_process_sequence"])
                self.assertEqual([event["status"] for event in events], passage["expected_statuses"])
                self.assertEqual([event["sequence_index"] for event in events], list(range(1, len(events) + 1)))
                for event in events:
                    for link in event["temporal_links"]:
                        self.assertGreaterEqual(link["to_sequence_index"], 1)
                        self.assertLessEqual(link["to_sequence_index"], len(events))
                        self.assertNotEqual(link["to_sequence_index"], event["sequence_index"])
                if "may irrigate" in passage["text"].lower():
                    carried = next(event for event in events if "may irrigate" in event["evidence"].lower())
                    self.assertEqual(carried["agent"].lower(), "farmer")

    def test_missing_negation_sign_withholds_positive_process_playlist(self):
        result = translate_manual("Do not water the crop.")
        event = result["translations"][0]["events"][0]
        self.assertEqual(event["process_id"], "IRRIGATION")
        self.assertEqual(event["polarity"], "negative")
        self.assertEqual(event["status"], "prohibited")
        self.assertIn("NEGATION", [item["concept_id"] for item in result["missing_sign_concepts"]])
        self.assertEqual(result["complete_video_sequence"], [])
        self.assertEqual(result["translation_diagnostics"][0]["code"], "UNREPRESENTED_NEGATION")

    def test_action_scoped_negation_regressions(self):
        cases = [
            ("Apply fertilizer to the crops.", [("FERTILIZATION", "positive")]),
            ("Do not apply fertilizer to the crops.", [("FERTILIZATION", "negative")]),
            ("The farmer does not irrigate the field.", [("IRRIGATION", "negative")]),
            ("Do not prepare the soil, but sow the seeds.",
             [("SOIL_PREPARATION", "negative"), ("SEED_SOWING", "positive")]),
            ("The farmer applies fertilizer, not pesticide.", [("FERTILIZATION", "positive")]),
            ("The farmer applies fertilizer.", [("FERTILIZATION", "positive")]),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                events = translate_manual(text)["translations"][0]["events"]
                self.assertEqual([(event["process_id"], event["polarity"]) for event in events], expected)
        negative = translate_manual("Do not apply fertilizer to the crops.")
        self.assertIn("NEGATION", [item["concept_id"] for item in negative["ideal_sign_sequence"]])
        self.assertTrue(any(item["concept_id"] == "NEGATION" and not item["video_available"]
                            for item in negative["ideal_sign_sequence"]))
        self.assertTrue(negative["translations"][0]["processes"][0]["translation_diagnostics"])

    def test_common_negation_forms_are_detected_near_the_action(self):
        for text in (
            "The farmer doesn't irrigate the field.",
            "The farmer doesn’t irrigate the field.",
            "The farmer did not irrigate the field.",
            "The farmer never irrigates the field.",
            "The farmer should not irrigate the field.",
            "The farmer must not irrigate the field.",
        ):
            with self.subTest(text=text):
                events = translate_manual(text)["translations"][0]["events"]
                self.assertEqual(events[0]["process_id"], "IRRIGATION")
                self.assertEqual(events[0]["polarity"], "negative")

    def test_tomato_tutorial_covers_actions_in_order_and_reports_process_video_gaps(self):
        tutorial = (
            "First, select a well-drained field for the tomato crop. "
            "Clear the field of weeds and stones. "
            "Add compost to the soil and mix it well. "
            "Prepare raised beds to improve drainage. "
            "Sow tomato seeds in a nursery and transplant healthy seedlings to the beds. "
            "Irrigate the plants after transplanting and maintain soil moisture during dry weather. "
            "Apply fertilizer when the plants begin flowering."
        )
        result = translate_manual(tutorial)
        events = [event for sentence in result["translations"] for event in sentence["events"]]
        expected = [
            "FIELD_SELECTION", "LAND_CLEARING", "SOIL_AMENDMENT", "BED_PREPARATION",
            "SEED_SOWING", "TRANSPLANTING", "IRRIGATION",
            "SOIL_MOISTURE_CONSERVATION", "FERTILIZATION",
        ]
        ids = [event["process_id"] for event in events]
        self.assertEqual(ids, expected)
        self.assertEqual([event["sequence_index"] for event in events], list(range(1, len(events) + 1)))
        self.assertEqual(len(ids), len(set(ids)))  # same-process add/mix actions are merged
        self.assertIn("Add compost", next(event for event in events if event["process_id"] == "SOIL_AMENDMENT")["evidence"])
        missing_ids = {item["concept_id"] for item in result["missing_sign_concepts"]}
        for process_id in {
            "FIELD_SELECTION", "LAND_CLEARING", "SOIL_AMENDMENT", "BED_PREPARATION",
            "TRANSPLANTING", "SOIL_MOISTURE_CONSERVATION",
        }:
            self.assertIn(process_id, missing_ids)
        self.assertTrue(result["renderable_sign_sequence"])

    def test_empty_text_is_a_valid_empty_translation(self):
        result = translate_manual("   ")
        self.assertEqual(result["total_sentences"], 0)
        self.assertEqual(result["translations"], [])
        self.assertEqual(result["complete_video_sequence"], [])

    def test_parent_video_fallback_is_visible_and_keeps_ideal_concept(self):
        fake_metadata = {
            "TEST_SPECIFIC_METHOD": {"name": "Test Specific Method", "parent_id": "IRRIGATION", "sign_concepts": ["missing_method_video"]},
            "IRRIGATION": {"name": "Irrigation", "parent_id": "CROP_WATER_MANAGEMENT", "sign_concepts": ["irrigation", "water"]},
        }
        event = {"process_id": "IRRIGATION", "process_name": "Irrigation", "sequence_index": 1,
                 "evidence": "test sentence", "sign_concepts": ["TEST_SPECIFIC_METHOD"]}
        with patch("sign_sequence_generator.load_concept_metadata", return_value=fake_metadata):
            result = generate_video_sequence({"event": event})
        self.assertEqual(result["ideal_sign_sequence"][0]["concept_id"], "TEST_SPECIFIC_METHOD")
        self.assertEqual(result["missing_sign_concepts"][0]["fallback"], "IRRIGATION")
        self.assertTrue(result["fallbacks"])
        self.assertEqual(Path(result["video_sequence"][0]).name, "irrigation.mp4")

    def test_fastapi_startup_and_backward_compatible_translate_response(self):
        request = Request({
            "type": "http", "http_version": "1.1", "method": "POST",
            "scheme": "http", "path": "/translate", "raw_path": b"/translate",
            "query_string": b"", "root_path": "", "headers": [],
            "server": ("test", 80), "client": ("test", 1),
        })
        payload = translate(TextRequest(
            text="The farmer uses drip irrigation to water the crops.", language="english"
        ), request)
        self.assertEqual(payload["process_sequence"], ["Irrigation"])
        self.assertTrue(payload["complete_video_sequence"])
        self.assertEqual([item["concept_id"] for item in payload["ideal_sign_sequence"]],
                         ["IRRIGATION", "DRIP_IRRIGATION"])
        self.assertTrue(all("video_url" in item for item in payload["renderable_sign_sequence"]))
        with self.assertRaises(HTTPException) as invalid_language:
            translate_text_endpoint(TranslateTextRequest(text="x", language="english"))
        self.assertEqual(invalid_language.exception.status_code, 400)
        route_paths = {getattr(route, "path", "") for route in app.routes}
        self.assertTrue({"/translate", "/translate-text", "/speech-to-text", "/live-hindi", "/live-kannada", "/live-english"}.issubset(route_paths))


if __name__ == "__main__":
    unittest.main()
