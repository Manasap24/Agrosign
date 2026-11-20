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
from process_detector import detect_process, initialize_process_detector, _bart_verify_candidates, _clause_spans
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

    def test_generic_apply_requires_process_specific_irrigation_evidence(self):
        # The crop recipient alone must not make a generic application verb
        # irrigation. Keep the applied material as separate evidence.
        false_positive = detect_process(
            "The farmer applies pressure to the crops."
        )
        self.assertEqual(false_positive["events"], [])

        # A material in the next coordinated clause must not leak backward
        # and validate the unsupported pressure action.
        coordinated = detect_process(
            "The farmer applies pressure to the crops, and applies fertilizer to the field."
        )["events"]
        self.assertEqual([event["process_id"] for event in coordinated], ["FERTILIZATION"])
        self.assertIn("applies fertilizer", coordinated[0]["evidence"].lower())

        # Explicitly applying water remains a valid irrigation event even if
        # the general-domain verifier returns a low score. BART is not a
        # calibrated universal rejection threshold.
        with patch.dict("process_detector.SCORING_CONFIG", {"bart_trigger_score": 1.0}), \
                patch("process_detector.classify_processes", return_value={"Irrigation": 0.20}):
            explicit_water = detect_process("The farmer applies water to the crops.")
        self.assertEqual([event["process_id"] for event in explicit_water["events"]], ["IRRIGATION"])
        self.assertFalse(explicit_water["events"][0]["bart_verified"])
        self.assertEqual(explicit_water["events"][0]["bart_score"], 0.2)

        valid_examples = {
            "The farmer irrigates the field.": "IRRIGATION",
            "The farmer waters the crops.": "IRRIGATION",
            "The farmer applies fertilizer to the crop.": "FERTILIZATION",
            "The farmer sows seeds.": "SEED_SOWING",
        }
        for text, process_id in valid_examples.items():
            with self.subTest(text=text):
                events = detect_process(text)["events"]
                self.assertEqual([event["process_id"] for event in events], [process_id])

    def test_rejected_process_does_not_reach_translate_video_playlist(self):
        request = Request({
            "type": "http", "http_version": "1.1", "method": "POST",
            "scheme": "http", "path": "/translate", "raw_path": b"/translate",
            "query_string": b"", "root_path": "", "headers": [],
            "server": ("test", 80), "client": ("test", 1),
        })
        response = translate(TextRequest(
            text="The farmer applies pressure to the crops.", language="english"
        ), request)
        self.assertEqual(response["process_sequence"], [])
        self.assertEqual(response["ideal_event_sequence"], [])
        self.assertEqual(response["ideal_sign_sequence"], [])
        self.assertEqual(response["complete_video_sequence"], [])
        self.assertEqual(response["translations"][0]["events"], [])
        self.assertEqual(response["translations"][0]["video_sequence"], [])

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

    def test_coordinated_actions_are_separate_without_splitting_subordinate_clauses(self):
        cases = [
            (
                "The farmer clears weeds, removes stones.",
                ["FIELD_SANITATION", "LAND_CLEARING"],
                ["clears weeds", "removes stones"],
            ),
            (
                "The farmer clears weeds, removes stones, and levels the field.",
                ["FIELD_SANITATION", "LAND_CLEARING", "SOIL_PREPARATION"],
                ["clears weeds", "removes stones", "levels the field"],
            ),
        ]
        for text, expected_ids, evidence_parts in cases:
            with self.subTest(text=text):
                events = detect_process(text)["events"]
                self.assertEqual([event["process_id"] for event in events], expected_ids)
                for event, evidence in zip(events, evidence_parts):
                    self.assertIn(evidence, event["evidence"].lower())
                self.assertEqual([event["sequence_index"] for event in events],
                                 list(range(1, len(events) + 1)))

    def test_shared_object_compound_action_and_subordinate_clause_are_preserved(self):
        shared_object = [clause.lower() for clause, _ in _clause_spans(
            "The farmer ploughs and levels the field."
        )]
        self.assertEqual(len(shared_object), 2)
        self.assertIn("ploughs", shared_object[0])
        self.assertIn("levels the field", shared_object[1])

        subordinate = detect_process(
            "The farmer clears weeds, which are growing near the field, and levels the field."
        )["events"]
        self.assertEqual([event["process_id"] for event in subordinate],
                         ["FIELD_SANITATION", "SOIL_PREPARATION"])
        self.assertIn("which are growing", subordinate[0]["evidence"].lower())

    def test_sequence_order_and_api_compatible_response(self):
        result = translate_manual("First prepare the soil, then sow the seeds and finally irrigate the field.")
        names = result["process_sequence"]
        self.assertEqual(names, ["Soil Preparation", "Seed Sowing", "Irrigation"])
        self.assertEqual([item["concept_id"] for item in result["ideal_sign_sequence"] if item["type"] == "process"],
                         ["SOIL_PREPARATION", "SEED_SOWING", "IRRIGATION"])
        self.assertTrue(result["complete_video_sequence"])
        self.assertEqual(result["translations"][0]["sentence"].startswith("First"), True)

    def test_cross_sentence_after_reference_reuses_prior_event_and_sign(self):
        result = translate_manual(
            "The farmer sows seeds. After sowing the seeds, irrigate the field."
        )
        events = result["ideal_event_sequence"]
        self.assertEqual([event["process_id"] for event in events], ["SEED_SOWING", "IRRIGATION"])
        self.assertEqual(events[1]["temporal_links"], [{"relation": "after", "to_sequence_index": 1}])
        second_sentence = result["translations"][1]
        self.assertEqual([event["process_id"] for event in second_sentence["events"]], ["IRRIGATION"])
        self.assertEqual([item["concept_id"] for item in result["ideal_sign_sequence"]
                          if item["type"] == "process"], ["SEED_SOWING", "IRRIGATION"])
        video_names = [Path(path).name for path in result["complete_video_sequence"]]
        self.assertEqual(video_names.count("seed.mp4"), 1)
        self.assertIn("irrigation.mp4", video_names)

    def test_distinct_repeated_action_is_not_removed_by_reference_resolution(self):
        result = translate_manual(
            "The farmer sows seeds. The farmer sows seeds in another field."
        )
        events = result["ideal_event_sequence"]
        self.assertEqual([event["process_id"] for event in events], ["SEED_SOWING", "SEED_SOWING"])
        self.assertEqual([event["sequence_index"] for event in events], [1, 2])
        self.assertIn("another field", events[1]["evidence"].lower())

    def test_explicit_before_and_implicit_then_relations_are_preserved(self):
        explicit_before = translate_manual("Before sowing the seeds, prepare the soil.")["ideal_event_sequence"]
        soil_preparation = next(event for event in explicit_before
                                if event["process_id"] == "SOIL_PREPARATION")
        self.assertEqual(soil_preparation["temporal_links"],
                         [{"relation": "before", "to_sequence_index": 2}])

        sequential = translate_manual(
            "The farmer prepares the soil. The farmer sows seeds."
        )["ideal_event_sequence"]
        self.assertEqual([event["process_id"] for event in sequential],
                         ["SOIL_PREPARATION", "SEED_SOWING"])
        self.assertEqual(sequential[1]["temporal_links"],
                         [{"relation": "then", "to_sequence_index": 1}])

    def test_postposed_temporal_action_clause_is_detected_in_order(self):
        result = translate_manual("Prepare the soil before planting seeds.")
        events = result["ideal_event_sequence"]
        self.assertEqual([event["process_id"] for event in events],
                         ["SOIL_PREPARATION", "SEED_SOWING"])
        self.assertEqual(events[0]["temporal_links"],
                         [{"relation": "before", "to_sequence_index": 2}])
        names = [Path(path).name for path in result["complete_video_sequence"]]
        self.assertIn("soil.mp4", names)
        self.assertIn("plough.mp4", names)
        self.assertIn("seed.mp4", names)
        self.assertIn("planting.mp4", names)
        self.assertNotIn("crops.mp4", names)

        fronted = translate_manual("Before sowing seeds, prepare the soil.")
        self.assertEqual([event["process_id"] for event in fronted["ideal_event_sequence"]],
                         ["SOIL_PREPARATION", "SEED_SOWING"])
        self.assertEqual(fronted["ideal_event_sequence"][0]["temporal_links"],
                         [{"relation": "before", "to_sequence_index": 2}])
        no_action = translate_manual("Prepare the soil before the rain.")
        self.assertEqual(len(no_action["ideal_event_sequence"]), 1)

        postposed_after = translate_manual("Plant seeds after the farmer prepares the soil.")
        after_events = postposed_after["ideal_event_sequence"]
        self.assertEqual([event["process_id"] for event in after_events],
                         ["SOIL_PREPARATION", "SEED_SOWING"])
        self.assertEqual(after_events[1]["temporal_links"],
                         [{"relation": "after", "to_sequence_index": 1}])

    def test_csv_term_matching_is_separate_from_process_detection(self):
        cases = [
            ("The farmer ploughs the soil.", ["SOIL_PREPARATION"],
             {"farmer.mp4", "soil.mp4", "plough.mp4"}),
            ("tractor", [], {"tractor.mp4"}),
            ("soil", [], {"soil.mp4"}),
            ("The farmer uses a tractor.", [], {"farmer.mp4", "tractor.mp4"}),
        ]
        for text, process_ids, expected_videos in cases:
            with self.subTest(text=text):
                result = translate_manual(text)
                self.assertEqual([event["process_id"] for event in result["ideal_event_sequence"]],
                                 process_ids)
                video_names = [Path(path).name for path in result["complete_video_sequence"]]
                self.assertEqual(len(video_names), len(set(video_names)))
                self.assertTrue(expected_videos.issubset(set(video_names)))

        unsupported = translate_manual("biochar")
        self.assertEqual(unsupported["complete_video_sequence"], [])
        self.assertTrue(any(item["code"] == "UNMAPPED_TERM"
                            for item in unsupported["translation_diagnostics"]))

        request = Request({
            "type": "http", "http_version": "1.1", "method": "POST",
            "scheme": "http", "path": "/translate", "raw_path": b"/translate",
            "query_string": b"", "root_path": "", "headers": [],
            "server": ("test", 80), "client": ("test", 1),
        })
        payload = translate(TextRequest(text="tractor", language="english"), request)
        self.assertEqual(payload["process_sequence"], [])
        self.assertEqual(payload["complete_video_sequence"], ["http://test/videos/tractor.mp4"])
        self.assertEqual(payload["translations"][0]["video_sequence"],
                         ["http://test/videos/tractor.mp4"])
        self.assertEqual(payload["translations"][0]["renderable_sign_sequence"][0]["video_url"],
                         "http://test/videos/tractor.mp4")

    def test_term_matching_preserves_negation_and_condition_safety(self):
        negative = translate_manual("The farmer does not irrigate the field.")
        self.assertEqual(negative["ideal_event_sequence"][0]["polarity"], "negative")
        self.assertEqual(negative["complete_video_sequence"], [])
        self.assertTrue(any(item["code"] == "UNREPRESENTED_NEGATION"
                            for item in negative["translation_diagnostics"]))

        conditional = translate_manual("If the soil is dry, irrigate the field.")
        event = conditional["ideal_event_sequence"][0]
        self.assertEqual((event["condition_relation"], event["condition_text"]),
                         ("if", "the soil is dry"))
        self.assertTrue(any(item["code"] == "UNREPRESENTED_CONDITION"
                            for item in conditional["translation_diagnostics"]))
        self.assertNotIn("soil.mp4", [Path(path).name
                                      for path in conditional["complete_video_sequence"]])

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

    def test_translate_does_not_report_material_clips_as_action_video(self):
        request = Request({
            "type": "http", "http_version": "1.1", "method": "POST",
            "scheme": "http", "path": "/translate", "raw_path": b"/translate",
            "query_string": b"", "root_path": "", "headers": [],
            "server": ("test", 80), "client": ("test", 1),
        })

        fertilizer = translate(TextRequest(
            text="Apply fertilizer to the crops.", language="english"
        ), request)
        self.assertEqual(fertilizer["process_sequence"], ["Fertilization"])
        fertilizer_action = next(item for item in fertilizer["ideal_sign_sequence"]
                                 if item["concept_id"] == "FERTILIZATION")
        self.assertFalse(fertilizer_action["video_available"])
        self.assertEqual(
            [Path(url).name for url in fertilizer["complete_video_sequence"]],
            ["fertilizer.mp4", "compost.mp4"],
        )

        irrigation = translate(TextRequest(
            text="The farmer irrigates the field.", language="english"
        ), request)
        irrigation_action = next(item for item in irrigation["ideal_sign_sequence"]
                                 if item["concept_id"] == "IRRIGATION")
        self.assertTrue(irrigation_action["video_available"])
        self.assertTrue(any(url.endswith("/irrigation.mp4")
                            for url in irrigation["complete_video_sequence"]))

    def test_translate_withholds_positive_clips_for_requested_negative_events(self):
        request = Request({
            "type": "http", "http_version": "1.1", "method": "POST",
            "scheme": "http", "path": "/translate", "raw_path": b"/translate",
            "query_string": b"", "root_path": "", "headers": [],
            "server": ("test", 80), "client": ("test", 1),
        })
        cases = [
            ("The farmer does not irrigate the field.", "IRRIGATION"),
            ("The farmer does not apply fertilizer to the crops.", "FERTILIZATION"),
        ]
        for text, process_id in cases:
            with self.subTest(text=text):
                payload = translate(TextRequest(text=text, language="english"), request)
                event = payload["ideal_event_sequence"][0]
                self.assertEqual(event["process_id"], process_id)
                self.assertEqual(event["polarity"], "negative")
                self.assertIn("NEGATION", event["sign_concepts"])
                self.assertEqual(payload["complete_video_sequence"], [])
                self.assertTrue(any(
                    item["code"] == "UNREPRESENTED_NEGATION"
                    for item in payload["translation_diagnostics"]
                ))

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

    def test_if_unless_until_conditions_are_preserved_and_action_scoped(self):
        cases = [
            ("The farmer irrigates the field if the soil is dry.",
             "if", "the soil is dry", "positive"),
            ("The farmer does not irrigate the field unless the soil is dry.",
             "unless", "the soil is dry", "negative"),
            ("The farmer does not irrigate the field.", "", "", "negative"),
            ("The farmer irrigates the field until the soil is moist.",
             "until", "the soil is moist", "positive"),
            ("The farmer irrigates the field.", "", "", "positive"),
        ]
        for text, relation, condition, polarity in cases:
            with self.subTest(text=text):
                event = translate_manual(text)["translations"][0]["events"][0]
                self.assertEqual(event["process_id"], "IRRIGATION")
                self.assertEqual(event["condition_relation"], relation)
                self.assertEqual(event["condition_text"], condition)
                self.assertEqual(event["polarity"], polarity)
                if relation == "unless":
                    self.assertNotEqual(event["status"], "prohibited")

        multi_action = translate_manual(
            "The farmer irrigates the field if the soil is dry, and applies fertilizer unless rain is expected."
        )["translations"][0]["events"]
        self.assertEqual([event["process_id"] for event in multi_action],
                         ["IRRIGATION", "FERTILIZATION"])
        self.assertEqual([(event["condition_relation"], event["condition_text"]) for event in multi_action],
                         [("if", "the soil is dry"), ("unless", "rain is expected")])

    def test_translate_reports_unrepresented_conditions(self):
        request = Request({
            "type": "http", "http_version": "1.1", "method": "POST",
            "scheme": "http", "path": "/translate", "raw_path": b"/translate",
            "query_string": b"", "root_path": "", "headers": [],
            "server": ("test", 80), "client": ("test", 1),
        })
        cases = [
            ("If the soil is dry, irrigate the field.", "if", "the soil is dry", "positive"),
            ("The farmer does not irrigate the field unless the soil is dry.",
             "unless", "the soil is dry", "negative"),
        ]
        for text, relation, condition, polarity in cases:
            with self.subTest(text=text):
                payload = translate(TextRequest(text=text, language="english"), request)
                event = payload["ideal_event_sequence"][0]
                self.assertEqual(event["condition_relation"], relation)
                self.assertEqual(event["condition_text"], condition)
                self.assertEqual(event["polarity"], polarity)
                diagnostic = next(item for item in payload["translation_diagnostics"]
                                  if item["code"] == "UNREPRESENTED_CONDITION")
                self.assertEqual(diagnostic["condition_relation"], relation)
                self.assertEqual(diagnostic["condition_text"], condition)
                self.assertEqual(bool(payload["complete_video_sequence"]), polarity == "positive")
                if polarity == "negative":
                    self.assertTrue(any(item["code"] == "UNREPRESENTED_NEGATION"
                                        for item in payload["translation_diagnostics"]))

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
        conditional_payload = translate(TextRequest(
            text="The farmer irrigates the field if the soil is dry.", language="english"
        ), request)
        conditional_event = conditional_payload["ideal_event_sequence"][0]
        self.assertEqual(conditional_event["condition_relation"], "if")
        self.assertEqual(conditional_event["condition_text"], "the soil is dry")
        temporal_payload = translate(TextRequest(
            text="The farmer sows seeds. After sowing the seeds, irrigate the field.",
            language="english",
        ), request)
        self.assertEqual(temporal_payload["process_sequence"], ["Seed Sowing", "Irrigation"])
        self.assertEqual(temporal_payload["ideal_event_sequence"][1]["temporal_links"],
                         [{"relation": "after", "to_sequence_index": 1}])
        self.assertTrue(any(url.endswith("/seed.mp4") for url in temporal_payload["complete_video_sequence"]))
        self.assertTrue(any(url.endswith("/irrigation.mp4") for url in temporal_payload["complete_video_sequence"]))
        with self.assertRaises(HTTPException) as invalid_language:
            translate_text_endpoint(TranslateTextRequest(text="x", language="english"))
        self.assertEqual(invalid_language.exception.status_code, 400)
        route_paths = {getattr(route, "path", "") for route in app.routes}
        self.assertTrue({"/translate", "/translate-text", "/speech-to-text", "/live-hindi", "/live-kannada", "/live-english"}.issubset(route_paths))


if __name__ == "__main__":
    unittest.main()
