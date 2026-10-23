"""Voice interface with scripted speech-to-text and a recording text-to-speech. Nothing here uses a microphone or speaker,
so these tests prove the control flow and the safety rules, not recognition accuracy."""
from __future__ import annotations

import sys

import pytest

from app import voice
from app.config import get_settings
from app.services import execution_service as ex
from app.services.llm import Providers, set_providers_override
from app.tools.email_provider import set_email_provider_override

from .helpers import FakeLLM
from .test_email_tools import FakeEmail, email_plan, handler

RESEARCH = ("Research three competitors in the AI automation market, compare their features, use publicly available pricing information, "
            "create an Excel comparison and a PowerPoint summary with source references.")
EMAIL_OBJ = "Write a short note and email it to anita@example.com with the subject Hello and the note attached"


class ScriptedStt:
    name = "scripted"

    def __init__(self, *heard: str):
        self.heard, self.calls = list(heard), []

    async def listen(self, timeout=8, phrases=None):
        self.calls.append(phrases)
        return self.heard.pop(0) if self.heard else ""


class RecTts:
    name = "rec"

    def __init__(self):
        self.said = []

    async def speak(self, text, wav_path=None):
        self.said.append(text)


@pytest.fixture(autouse=True)
def demo_mode(monkeypatch):
    monkeypatch.setenv("JOSE_MODE", "demo")
    get_settings.cache_clear()


def session(*heard):
    return voice.VoiceSession(ScriptedStt(*heard), RecTts(), wake="hey jose", retries=2)


def test_choice_matching_is_exact_and_never_guesses_consent():
    m = voice.match_choice
    assert m("accept", voice.PLAN_CHOICES) == "accept" and m("Please ACCEPT it.", voice.PLAN_CHOICES) == "accept"
    assert m("yes", ("approve", "deny")) is None and m("ok sure go ahead", ("approve", "deny")) is None and m("", ("approve", "deny")) is None
    assert m("approve or deny", ("approve", "deny")) is None  # two options heard: ambiguous, so no consent


async def test_wake_phrase_detection():
    s = session("", "random chatter", "hey jose")
    assert await s.wait_for_wake(max_cycles=5) is True and len(s.stt.calls) == 3
    assert s.stt.calls[0] == ["hey jose", "goodbye jose"]  # the recogniser is constrained to the wake phrase
    assert await session("goodbye jose").wait_for_wake(5) is False
    assert await session("", "").wait_for_wake(2) is False


async def test_spoken_flow_plan_accept_progress_and_report():
    s = session(RESEARCH, "accept")
    status = await s.run_once()
    said = " ".join(s.tts.said)
    assert status == "completed"
    assert "The plan has 11 steps" in said and "Say accept to run it, relisten" in said and "simulated" in said.lower()
    assert "verified." in said and "Completed: all 11 steps passed" in said and "competitors.xlsx" in said
    assert s.stt.calls[1] == list(voice.PLAN_CHOICES)


async def test_relisten_repeats_the_plan_and_change_makes_a_new_plan():
    s = session(RESEARCH, "relisten", "change", "also compare two more vendors", "accept")
    status = await s.run_once()
    assert sum("The plan has" in x for x in s.tts.said) == 3  # read, relistened, then the changed plan
    assert any(x.startswith("What should change") for x in s.tts.said)
    runs = ex.list_runs(10)
    assert any(r["status"] == "cancelled" for r in runs) and any("Change: also compare two more vendors" in r["objective"] for r in runs)
    assert status in ("completed", "partial", "failed", "rejected")


async def test_cancel_and_silence_never_run_anything():
    s = session(RESEARCH, "cancel")
    assert await s.run_once() == "cancelled" and "Nothing was run" in s.tts.said[-1]
    s2 = session(RESEARCH, "", "")  # the user says nothing when asked to accept
    assert await s2.run_once() == "cancelled"
    runs = ex.list_runs(5)
    assert all(r["status"] == "cancelled" and r["task_counts"].get("completed", 0) == 0 for r in runs[:2])
    assert "did not get a clear answer" in s2.tts.said[-1]


async def test_unclear_objective_is_explained_not_attempted():
    s = session("make it better")  # too vague for the engine
    assert await s.run_once() == "rejected" and s.tts.said[-1].startswith("I cannot do that")
    assert await session("", "").run_once() == "no_input"


async def test_email_approval_needs_the_exact_word_and_content_is_read_aloud(monkeypatch):
    monkeypatch.setenv("JOSE_MODE", "real")
    monkeypatch.setenv("GEMINI_API_KEY", "x" * 12)
    get_settings.cache_clear()
    fake = FakeEmail()
    set_email_provider_override(fake)
    prov = Providers(llm=FakeLLM(lambda n, p, pr: handler(email_plan())(n, p, pr)), search=None, fetcher=None, browser=None, simulated=False)
    set_providers_override(lambda s, i: prov)
    try:
        s = session(EMAIL_OBJ, "accept", "yes please", "sure")  # "yes" and "sure" are not "approve"
        assert await s.run_once() == "halted" and fake.sent == []
        said = " ".join(s.tts.said)
        assert "To anita@example.com" in said and "Subject: Hello" in said and "Message: Hi Anita" in said and "note.txt" in said
        assert "I will not do it" in said and "Please say one of: approve, deny." in said
        s2 = session(EMAIL_OBJ, "accept", "approve")
        assert await s2.run_once() == "completed" and len(fake.sent) == 1
        assert "accepted by the service" in " ".join(s2.tts.said) and "not proof of delivery" in s2.tts.said[-1]
    finally:
        set_email_provider_override(None)


async def test_serve_once_with_wake_phrase():
    s = session("hey jose", RESEARCH, "accept")
    await s.serve(once=True)
    assert s.tts.said[0].startswith("JOSE voice is ready. Say hey jose") and any("Completed" in x for x in s.tts.said)


def test_speech_text_is_sanitised_and_bounded():
    assert voice.clean_for_speech("a\x00b\n\n c\x07") == "a b c"
    assert len(voice.clean_for_speech("x " * 2000)) <= voice.MAX_SPOKEN


@pytest.mark.skipif(sys.platform != "win32", reason="Windows speech only")
async def test_real_windows_tts_renders_audio_to_a_file(tmp_path):
    wav = tmp_path / "plan.wav"
    await voice.SapiTts().speak("The plan has three steps.", wav_path=str(wav))
    assert wav.exists() and wav.stat().st_size > 5000  # real synthesized audio, written to a file instead of played
