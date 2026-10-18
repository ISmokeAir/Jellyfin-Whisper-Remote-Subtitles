"""
Unit and integration tests for Jellyfin Whisper Worker.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to sys.path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import config
import idle_monitor
import jellyfin_client
import transcriber


class TestConfig(unittest.TestCase):
    def test_summary_redacts_key(self):
        summary = config.summary()
        self.assertIn("Jellyfin URL", summary)
        self.assertIn("****", summary)
        self.assertNotIn(config.JELLYFIN_API_KEY, summary)

    def test_subtitle_langs_parsed(self):
        self.assertIsInstance(config.SUBTITLE_LANGS, list)
        self.assertGreater(len(config.SUBTITLE_LANGS), 0)


class TestIdleMonitor(unittest.TestCase):
    def test_idle_duration_non_negative(self):
        idle = idle_monitor.get_idle_duration_seconds()
        self.assertGreaterEqual(idle, 0.0)

    def test_gpu_utilization_or_none(self):
        util = idle_monitor.get_gpu_utilization()
        if util is not None:
            self.assertIsInstance(util, int)
            self.assertGreaterEqual(util, 0)
            self.assertLessEqual(util, 100)

    def test_is_gpu_busy_returns_bool(self):
        busy = idle_monitor.is_gpu_busy()
        self.assertIsInstance(busy, bool)


class TestTranscriber(unittest.TestCase):
    def test_format_timestamp(self):
        self.assertEqual(transcriber._format_timestamp(0.0), "00:00:00,000")
        self.assertEqual(transcriber._format_timestamp(65.123), "00:01:05,123")
        self.assertEqual(transcriber._format_timestamp(3661.05), "01:01:01,050")

    def test_segments_to_srt_with_watermark(self):
        # Starts at 5.0s, so watermark badge should be inserted in the first 3.5s
        segments = [(5.0, 8.0, "Hello world"), (8.5, 12.0, "Second line")]
        srt = transcriber._segments_to_srt(segments, watermark=True)
        self.assertIn("[Subtitles: Whisper AI]", srt)
        self.assertIn("00:00:00,500 --> 00:00:03,500", srt)
        self.assertIn("Hello world", srt)
        self.assertIn("Second line", srt)

    def test_segments_to_srt_without_watermark_when_speech_starts_early(self):
        # Starts at 0.5s, so watermark badge should NOT be inserted to avoid overlap
        segments = [(0.5, 3.0, "Immediate dialogue")]
        srt = transcriber._segments_to_srt(segments, watermark=True)
        self.assertNotIn("[Subtitles: Whisper AI]", srt)
        self.assertIn("Immediate dialogue", srt)

    def test_ffmpeg_binary_detection(self):
        binary = transcriber.get_ffmpeg_binary()
        self.assertIsInstance(binary, str)
        self.assertTrue(len(binary) > 0)


class TestJellyfinClient(unittest.TestCase):
    def setUp(self):
        # Create an isolated temporary state file for testing
        self.tmp_fd, self.tmp_path = tempfile.mkstemp(suffix=".json")
        os.close(self.tmp_fd)
        self._orig_state_file = jellyfin_client._STATE_FILE
        jellyfin_client._STATE_FILE = Path(self.tmp_path)

    def tearDown(self):
        jellyfin_client._STATE_FILE = self._orig_state_file
        if os.path.exists(self.tmp_path):
            os.unlink(self.tmp_path)

    def test_state_load_save(self):
        initial = jellyfin_client._load_state()
        self.assertEqual(initial, {"processed": {}})

        jellyfin_client.mark_processed(
            item_id="item-123",
            status="success",
            name="Test Movie",
            item_type="Movie",
            completed_langs=["eng", "rum"],
            detail="2 tracks complete",
        )

        state = jellyfin_client._load_state()
        self.assertIn("item-123", state["processed"])
        self.assertEqual(state["processed"]["item-123"]["status"], "success")
        self.assertEqual(state["processed"]["item-123"]["completed_langs"], ["eng", "rum"])

    def test_pending_languages_calculation(self):
        # Fresh item needs all configured languages (canonicalized to 3-letter)
        pending = jellyfin_client.get_item_pending_languages("item-fresh")
        self.assertEqual(pending, [jellyfin_client.normalize_to_3letter(l) for l in config.SUBTITLE_LANGS])

        # Partially completed item with ISO-639-2 variant (rum -> ron)
        with patch.object(config, "SUBTITLE_LANGS", ["eng", "rum"]):
            jellyfin_client.mark_processed(
                item_id="item-partial",
                status="in_progress",
                completed_langs=["eng"],
            )
            pending_partial = jellyfin_client.get_item_pending_languages("item-partial")
            self.assertEqual(pending_partial, ["ron"])
            self.assertFalse(jellyfin_client.is_already_processed("item-partial"))

            # Fully completed item
            jellyfin_client.mark_processed(
                item_id="item-partial",
                status="success",
                completed_langs=["ron"],
            )
            self.assertTrue(jellyfin_client.is_already_processed("item-partial"))

    def test_get_item_display_name(self):
        episode = {
            "Type": "Episode",
            "SeriesName": "Breaking Bad",
            "ParentIndexNumber": 1,
            "IndexNumber": 2,
            "Name": "Cat's in the Bag...",
        }
        name = jellyfin_client.get_item_display_name(episode)
        self.assertEqual(name, "Breaking Bad S01E02 - Cat's in the Bag...")

        movie = {"Type": "Movie", "Name": "Inception"}
        self.assertEqual(jellyfin_client.get_item_display_name(movie), "Inception")

    @patch.object(jellyfin_client, "_SESSION")
    def test_upload_subtitle_payload_tags_whisper(self, mock_session):
        mock_resp = MagicMock()
        mock_resp.status_code = 204
        mock_session.post.return_value = mock_resp

        jellyfin_client.upload_subtitle("test-id", "1\n00:00:01,000 --> 00:00:02,000\nHi\n", language="eng")
        mock_session.post.assert_called_once()
        args, kwargs = mock_session.post.call_args
        payload = kwargs["json"]
        # Must tag .whisper in Language so Jellyfin formats it as 'whisper - English'
        self.assertEqual(payload["Language"], "eng.whisper")
        self.assertEqual(payload["Format"], "srt")

    def test_language_normalization(self):
        self.assertEqual(jellyfin_client.normalize_to_3letter("en"), "eng")
        self.assertEqual(jellyfin_client.normalize_to_3letter("eng"), "eng")
        self.assertEqual(jellyfin_client.normalize_to_3letter("ro"), "ron")
        self.assertEqual(jellyfin_client.normalize_to_3letter("rum"), "ron")
        self.assertEqual(jellyfin_client.normalize_to_3letter("ron"), "ron")
        self.assertEqual(jellyfin_client.normalize_to_3letter("ja"), "jpn")
        self.assertEqual(jellyfin_client.normalize_to_2letter("eng"), "en")
        self.assertEqual(jellyfin_client.normalize_to_2letter("ron"), "ro")
        self.assertEqual(jellyfin_client.normalize_to_2letter("rum"), "ro")


class TestTranslator(unittest.TestCase):
    def test_nllb_lang_mapping(self):
        import translator
        self.assertEqual(translator._LANG_TO_NLLB["eng"], "eng_Latn")
        self.assertEqual(translator._LANG_TO_NLLB["en"], "eng_Latn")
        self.assertEqual(translator._LANG_TO_NLLB["ron"], "ron_Latn")
        self.assertEqual(translator._LANG_TO_NLLB["rum"], "ron_Latn")
        self.assertEqual(translator._LANG_TO_NLLB["ro"], "ron_Latn")

    def test_empty_srt_handling(self):
        import translator
        res = translator.translate_srt("", src_lang="eng", tgt_lang="ron")
        self.assertTrue(res.success)
        self.assertEqual(res.srt_text, "")

    def test_same_language_bypass(self):
        import translator
        sample = "1\n00:00:01,000 --> 00:00:02,000\nHello\n"
        res = translator.translate_srt(sample, src_lang="eng", tgt_lang="eng")
        self.assertTrue(res.success)
        self.assertEqual(res.srt_text, sample)

    def test_srt_parsing_regex(self):
        import translator
        sample = (
            "1\n00:00:01,000 --> 00:00:03,000\nLine one\n\n"
            "2\n00:00:04,500 --> 00:00:07,000\nLine two with multiple\nwords\n"
        )
        matches = list(translator._SRT_BLOCK_REGEX.finditer(sample.strip()))
        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0].group(1), "1")
        self.assertEqual(matches[0].group(3), "Line one")
        self.assertEqual(matches[1].group(1), "2")
        self.assertEqual(matches[1].group(3), "Line two with multiple\nwords")


if __name__ == "__main__":
    unittest.main()
