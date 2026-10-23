"""Voice interface (Phase 4): wake phrase -> speech-to-text -> spoken plan -> Accept / Relisten / Change / Cancel -> spoken progress -> spoken report.

It is a thin layer over the same engine as the website and the terminal (cli.follow, RunManager, approval and ledger services).
Safety rules that live here, not in the model:
  * silence, noise or an unrecognised answer is NEVER an approval; after a few tries the plan is cancelled / the action is denied
  * an outward action (email, Telegram, moves, commands) is approved only by the exact word "approve" after its content was read aloud
  * speech recognition can mishear: the plan, including the understood objective, is read back before anything runs

Backends (Windows, no extra packages): System.Speech through PowerShell for text-to-speech and offline speech recognition.
Backends are plain objects with async `speak` / `listen`, so they can be replaced (or faked in tests).
"""
from __future__ import annotations

import asyncio
import difflib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from . import cli
from .schemas.domain import ObjectiveCreate
from .services import execution_service as ex
from .services import report, store

DEFAULT_WAKE = "hi jose"
DEFAULT_WAKE_ALIASES = ("hey jose",)
PLAN_CHOICES = ("accept", "relisten", "change", "cancel")
MAX_SPOKEN = 1200

_TTS_PS = r'''
param([string]$WavPath = "")
Add-Type -AssemblyName System.Speech
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
$text = [Console]::In.ReadToEnd()
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
if ($WavPath) { $s.SetOutputToWaveFile($WavPath) }
$s.Speak($text)
$s.Dispose()
'''

_STT_PS = r'''
param([int]$Timeout = 8, [string]$Phrases = "", [double]$MinConfidence = 0.55)
Add-Type -AssemblyName System.Speech
$r = New-Object System.Speech.Recognition.SpeechRecognitionEngine
if ($Phrases) {
  $c = New-Object System.Speech.Recognition.Choices
  $Phrases.Split("|") | ForEach-Object { $c.Add($_) }
  $gb = New-Object System.Speech.Recognition.GrammarBuilder
  $gb.Append($c)
  $r.LoadGrammar((New-Object System.Speech.Recognition.Grammar $gb))
} else {
  $r.LoadGrammar((New-Object System.Speech.Recognition.DictationGrammar))
}
$r.SetInputToDefaultAudioDevice()
$r.InitialSilenceTimeout = [TimeSpan]::FromSeconds($Timeout)
$r.EndSilenceTimeout = [TimeSpan]::FromSeconds(1.5)
$r.EndSilenceTimeoutAmbiguous = [TimeSpan]::FromSeconds(2)
$res = $r.Recognize([TimeSpan]::FromSeconds($Timeout))
if ($res -and $res.Confidence -ge $MinConfidence) { [Console]::Out.Write($res.Text) }
$r.Dispose()
'''


def _script(name: str, body: str) -> Path:
    d = Path(tempfile.gettempdir()) / "jose_voice"
    d.mkdir(exist_ok=True)
    p = d / name
    if not p.exists() or p.read_text(encoding="utf-8") != body:
        p.write_text(body, encoding="utf-8")
    return p


def _ps(script: Path, args: list[str], stdin: str | None, timeout: float) -> str:
    cmd = ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script), *args]
    p = subprocess.run(cmd, input=(stdin or "").encode("utf-8") if stdin is not None else None, capture_output=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", "replace").strip()[:300] or f"powershell exited {p.returncode}")
    return p.stdout.decode("utf-8", "replace").strip()


_PROVIDERS = r"(?:gmail|googlemail|yahoo|outlook|hotmail|icloud|proton|protonmail|rediffmail)"
_LOCAL_STOP = {"to", "mail", "email", "address", "at", "and", "send", "cc", "bcc", "the", "a", "name", "named", "called", "id", "on", "for", "with"}


def normalize_spoken_emails(text: str) -> str:
    """Speech-to-text writes 'name 123 gmail com' / 'name at gmail dot com'; rebuild real addresses so the planner gets exactly what was meant."""
    pat = re.compile(rf"((?:[\w.\-]+[ ,]+){{0,5}}?)(?:at[ ]+)?({_PROVIDERS})(?:[ ]+dot[ ]+|[ ]+|\.)(com|in|org|net|co[ ]+in|co[ ]+uk)(?![a-z])", re.I)

    def fix(m: re.Match) -> str:
        words = [w for w in re.split(r"[ ,]+", m.group(1).strip()) if w]
        local: list[str] = []
        for w in reversed(words):
            if w.lower() in _LOCAL_STOP:
                break
            local.insert(0, w.lower().replace("underscore", "_").replace("dash", "-"))
        if not local:
            return m.group(0)
        tld = re.sub(r"\s+", ".", m.group(3).lower())
        head = " ".join(words[: len(words) - len(local)])
        return (head + " " if head else "") + "".join(local) + "@" + m.group(2).lower() + "." + tld

    return pat.sub(fix, text)


def clean_for_speech(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b-\x1f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_SPOKEN]


class SapiTts:
    name = "windows-sapi"

    async def speak(self, text: str, wav_path: str | None = None) -> None:
        text = clean_for_speech(text)
        if not text:
            return
        await asyncio.to_thread(_ps, _script("tts.ps1", _TTS_PS), ["-WavPath", wav_path] if wav_path else [], text, 120)


class SapiStt:
    name = "windows-sapi"

    async def listen(self, timeout: int = 8, phrases: list[str] | None = None) -> str:
        # free dictation scores low even when transcribed correctly, so only the fixed-phrase grammar is confidence-gated
        args = ["-Timeout", str(timeout)] + (["-Phrases", "|".join(phrases)] if phrases else ["-MinConfidence", "0.0"])
        try:
            return await asyncio.to_thread(_ps, _script("stt.ps1", _STT_PS), args, None, timeout + 20)
        except Exception as e:  # noqa: BLE001 - no microphone / recogniser: say so instead of pretending to hear
            raise VoiceUnavailable(f"speech recognition failed: {e}") from e


class MicStt:
    """Records from the default microphone (stops after a pause) and transcribes with Gemini: far more accurate than offline SAPI dictation."""

    name = "mic+gemini"
    RATE = 16000

    def __init__(self):
        import sounddevice  # noqa: F401 - fail early so build_session can fall back
        from .config import get_settings

        s = get_settings()
        if not s.gemini_api_key.strip():
            raise VoiceUnavailable("no GEMINI_API_KEY configured for speech-to-text")
        from google import genai

        self.client, self.model = genai.Client(api_key=s.gemini_api_key), s.gemini_model

    def _record(self, timeout: int) -> bytes | None:
        import numpy as np
        import sounddevice as sd

        block, quiet_end, max_s = int(self.RATE * 0.1), 1.2, max(timeout, 3) + 20
        frames, started, silent, waited = [], False, 0.0, 0.0
        floor = None
        with sd.InputStream(samplerate=self.RATE, channels=1, dtype="int16", blocksize=block) as stream:
            while True:
                data, _ = stream.read(block)
                level = float(np.abs(data).mean())
                if floor is None:
                    floor = level  # first block calibrates to the room noise
                    continue
                loud = level > max(floor * 2.5, 180)
                if not started:
                    waited += 0.1
                    if loud:
                        started = True
                        frames.append(data.copy())
                    elif waited >= timeout:
                        return None
                    continue
                frames.append(data.copy())
                silent = 0.0 if loud else silent + 0.1
                if silent >= quiet_end or len(frames) * 0.1 >= max_s:
                    break
        return np.concatenate(frames).tobytes()

    def _transcribe(self, pcm: bytes) -> str:
        import io
        import wave

        from google.genai import types

        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.RATE)
            w.writeframes(pcm)
        r = self.client.models.generate_content(model=self.model, config=types.GenerateContentConfig(
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)), contents=[
            types.Part.from_bytes(data=buf.getvalue(), mime_type="audio/wav"),
            "Transcribe this speech exactly. Output only the words spoken, no commentary. If there is no intelligible speech, output nothing. "
            "The speaker may address an assistant named Jose (pronounced HO-say), e.g. 'Hi Jose' or 'Hey Jose'; write that name as 'Jose'."])
        return re.sub(r"[^\w\s']", " ", r.text or "").strip()

    async def listen(self, timeout: int = 8, phrases: list[str] | None = None) -> str:
        try:
            pcm = await asyncio.to_thread(self._record, timeout)
            return await asyncio.to_thread(self._transcribe, pcm) if pcm else ""
        except Exception as e:  # noqa: BLE001
            raise VoiceUnavailable(f"speech recognition failed: {e}") from e


class ConsoleTts:
    """--text mode: print what would be spoken."""

    name = "console"

    async def speak(self, text: str, wav_path: str | None = None) -> None:
        print(f"JOSE says: {clean_for_speech(text)}", flush=True)


class ConsoleStt:
    """--text mode: type what you would say."""

    name = "console"

    async def listen(self, timeout: int = 8, phrases: list[str] | None = None) -> str:
        hint = f" ({'/'.join(phrases)})" if phrases and len(phrases) < 6 else ""
        return (await asyncio.to_thread(cli.ask, f"you say{hint}> ")).strip()


class VoiceUnavailable(RuntimeError):
    pass


def match_choice(heard: str, options: tuple[str, ...]) -> str | None:
    """Exact word match only (case-insensitive). 'yes', 'ok', 'sure' are deliberately NOT synonyms for approval."""
    words = re.findall(r"[a-z']+", heard.lower())
    hits = [o for o in options if o in words]
    return hits[0] if len(hits) == 1 else None


def plan_script(run: dict) -> str:
    tasks = run["tasks"]
    steps = " ".join(f"Step {i}: {t['title'].rstrip('.')}." for i, t in enumerate(tasks, 1))
    risky = [t["title"] for t in tasks if t["risk_level"] in ("HIGH", "MEDIUM")]
    parts = [f"I understood: {run['objective'].rstrip('.')}.", f"The plan has {len(tasks)} steps. {steps}"]
    if risky:
        parts.append(f"{len(risky)} step{'s' if len(risky) != 1 else ''} will ask for your approval before running: " + "; ".join(r.rstrip('.') for r in risky) + ".")
    assumptions = (run.get("intent") or {}).get("assumptions") or []
    if assumptions:
        parts.append("I assumed: " + " ".join(assumptions[:3]))
    if run.get("simulated"):
        parts.append("This is a simulated run. Nothing real will be sent.")
    parts.append("Say accept to run it, relisten to hear it again, change to adjust it, or cancel.")
    return " ".join(parts)


def approval_script(a: dict) -> str:
    d = a["details"]
    sim = " This is a simulation; nothing will really be sent." if d.get("dry_run") else ""
    if a["action"] == "tool.email.send":
        att = d.get("attachments") or []
        body = str(d.get("body", ""))[:500]
        text = (f"Approval needed to send an email. From {d.get('from')}. To {d.get('to')}." + (f" Copy to {d['cc']}." if d.get("cc") else "")
                + f" Subject: {d.get('subject')}. Message: {body}. " + (f"Attachments: {', '.join(att)}. " if att else "No attachments. "))
    else:
        text = f"Approval needed. {a['consequences']} Target: {a['target']}. "
    return text + sim + " Say approve to go ahead, or deny."


_PICKER_CODE = """
import tkinter as tk
from tkinter import filedialog
r = tk.Tk()
r.attributes('-topmost', True)
r.geometry('1x1+200+200')
r.update()
r.focus_force()
print(filedialog.askopenfilename(parent=r, title='JOSE: choose a file to attach') or '', end='')
"""


def pick_file_dialog() -> str:
    """Native file picker in its own process (so it gets a fresh UI thread and can be forced to the front). Returns '' when cancelled."""
    print("JOSE: opening the file picker (check your taskbar if you do not see it)...", flush=True)
    try:
        p = subprocess.run([sys.executable, "-c", _PICKER_CODE], capture_output=True, timeout=300)
        if p.returncode == 0:
            return p.stdout.decode("utf-8", "replace").strip()
        print("file picker failed: " + p.stderr.decode("utf-8", "replace").strip()[-200:], flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"file picker failed: {e}", flush=True)
    return cli.ask("type the file path instead (blank to skip)> ").strip().strip('"')


class VoiceSession:
    def __init__(self, stt, tts, wake: str | None = None, retries: int = 3, listen_seconds: int = 8, simulate: bool = False, files: list[str] | None = None, picker=None):
        self.stt, self.tts, self.simulate = stt, tts, simulate
        self.files = list(files or [])  # paths attached to the next objective
        self.picker = picker or (lambda: asyncio.to_thread(pick_file_dialog))
        custom = wake or os.environ.get("JOSE_WAKE_PHRASE")
        self.wake = (custom or DEFAULT_WAKE).lower()
        self.wake_aliases = () if custom else DEFAULT_WAKE_ALIASES  # a phrase you chose is matched on its own
        self.retries, self.listen_seconds = retries, listen_seconds
        self.said: list[str] = []

    async def say(self, text: str) -> None:
        self.said.append(text)
        if self.tts.name != "console":
            print(f"JOSE says: {clean_for_speech(text)}", flush=True)
        await self.tts.speak(text)

    # ---------------------------------------------------------------- wake word
    GREETINGS = {"hi", "hey", "hay", "hai", "high", "hello", "hie"}

    def is_wake(self, heard: str) -> bool:
        """Exact wake phrase, or a greeting followed by something that sounds like the name (speech-to-text writes 'Jose' as josh, george, gausah...)."""
        words = re.findall(r"[a-z']+", heard.lower())
        if any(re.findall(r"[a-z']+", w) == words[i:i + len(re.findall(r"[a-z']+", w))] for w in (self.wake, *self.wake_aliases) for i in range(len(words))):
            return True
        if self.wake_aliases != DEFAULT_WAKE_ALIASES:
            return False  # a custom phrase is matched exactly
        name = DEFAULT_WAKE.split()[1]
        for i in range(len(words) - 1):
            if words[i] in self.GREETINGS and (words[i + 1][:2] in ("jo", "ge", "ga", "ha", "ho") or difflib.SequenceMatcher(None, words[i + 1], name).ratio() >= 0.5):
                return True
        return False

    async def wait_for_wake(self, max_cycles: int | None = None) -> bool:
        """Listen for the wake phrase. Returns False on 'quit'/'goodbye' or when max_cycles is exhausted (used by tests)."""
        n = 0
        while max_cycles is None or n < max_cycles:
            n += 1
            heard = (await self.stt.listen(6, [self.wake, *self.wake_aliases, "goodbye jose"])).lower()
            if self.stt.name != "console":
                print(f"heard: {heard or '(nothing)'}", flush=True)
            if self.is_wake(heard):
                return True
            if "goodbye" in heard:
                return False
        return False

    # ---------------------------------------------------------------- one objective
    async def capture(self, prompt: str, seconds: int = 15) -> str | None:
        await self.say(prompt)
        for _ in range(self.retries):
            heard = (await self.stt.listen(seconds)).strip()
            if heard and self.stt.name != "console":
                print(f"heard: {heard}", flush=True)
            if len(heard.split()) >= 2:
                fixed = normalize_spoken_emails(heard)
                if fixed != heard:
                    print(f"address understood: {fixed}", flush=True)
                return fixed
            await self.say("I did not catch that. Please say it again.")
        return None

    async def choose(self, prompt: str, options: tuple[str, ...]) -> str | None:
        await self.say(prompt)
        for _ in range(self.retries):
            choice = match_choice(await self.stt.listen(self.listen_seconds, list(options)), options)
            if choice:
                return choice
            await self.say(f"Please say one of: {', '.join(options)}.")
        return None  # no clear answer is never treated as consent

    async def approver(self, a: dict) -> bool:
        choice = await self.choose(approval_script(a), ("approve", "deny"))
        if choice != "approve":
            await self.say("I did not get a clear approval, so I will not do it." if choice is None else "Denied. I will not do it.")
            return False
        return True

    async def narrate(self, e: dict) -> None:
        d, key = e["details"], (e["task_id"] or "").split(":")[-1]
        ty = e["type"]
        title = ""
        if key and ty in ("task.completed", "task.failed"):
            title = next((t.title for t in store.get_tasks(e["run_id"]) if t.key == key), "")
        if ty == "task.completed":
            await self.say(f"{title or key} verified.")
        elif ty == "task.failed":
            await self.say(f"{title or key} failed: {str(d.get('error', ''))[:160]}")
        elif ty == "recovery.started":
            await self.say("Something failed. I am trying to recover.")
        elif ty == "external.action" and d.get("status") == "accepted":
            await self.say(f"{d.get('kind')} was accepted by the service.")
        elif ty == "external.action" and d.get("status") == "ambiguous":
            await self.say(f"I cannot tell whether {d.get('kind')} happened. Please check, and I will not repeat it by myself.")

    async def completion(self, run_id: str) -> None:
        r = report.build(run_id)
        parts = [r["headline"]]
        if r["artifacts"]:
            parts.append("Files: " + ", ".join(a["filename"] for a in r["artifacts"][:5]) + ".")
        lv = r["levels"]
        if lv["external_actions_accepted"]:
            parts.append(f"{lv['external_actions_accepted']} outward action accepted by the service. Acceptance is not proof of delivery.")
        for c in r["caveats"][:2]:
            parts.append(c)
        if not r["workflow_completed"]:
            failed = [t for t in r["tasks"] if t["status"] in ("failed", "blocked") and t["error"]]
            if failed:
                parts.append(f"{failed[0]['title']}: {failed[0]['error'][:200]}")
        await self.say(" ".join(parts))

    async def run_once(self, objective: str | None = None) -> str:
        """One spoken objective, from capture to spoken report. Returns the final run status (or 'cancelled' / 'no_input')."""
        objective = objective or await self.capture("How can I help you, sir? You can also say attach a file.")
        while objective and re.search(r"(?<![a-z])(attach|upload)(?![a-z])", objective.lower()):
            path = await self.picker()
            if path:
                self.files.append(path)
                await self.say(f"Attached {Path(path).name}.")
            else:
                await self.say("No file chosen.")
            if len(objective.split()) > 6:  # "attach this file and make a report": the sentence is also the objective
                break
            objective = await self.capture("What would you like me to do?")
        if not objective:
            await self.say("I did not hear an objective.")
            return "no_input"
        while True:
            run = await ex.manager.create_run(ObjectiveCreate(objective=objective, mode="guided", simulate=self.simulate, upload_ids=[cli.register_upload(f) for f in self.files]))
            if run["status"] == "rejected":
                await self.say("I cannot do that. " + (run["error"] or ""))
                self.files.clear()
                return "rejected"
            while run["status"] == "awaiting_plan_approval":
                choice = await self.choose(plan_script(run), PLAN_CHOICES)
                if choice == "relisten":
                    continue
                if choice == "accept":
                    run = ex.manager.approve_plan(run["id"], run["plan_hash"])  # bound to exactly the plan that was read out
                    break
                if choice == "change":
                    extra = await self.capture("What should change?")
                    ex.manager.cancel(run["id"])
                    if extra:
                        objective = f"{objective.rstrip('.')}. Change: {extra}"
                    run = None
                    break
                ex.manager.cancel(run["id"])  # cancel, or no clear answer
                await self.say("Cancelled. Nothing was run." if choice == "cancel" else "I did not get a clear answer, so I cancelled. Nothing was run.")
                return "cancelled"
            if run is None:
                continue
            final = await cli.follow(run["id"], approver=self.approver, on_event=self.narrate)
            await self.completion(final["id"])
            return final["status"]

    # ---------------------------------------------------------------- forever
    async def serve(self, once: bool = False, wake: bool = True) -> None:
        await self.say(f"JOSE voice is ready. Say {self.wake} to begin." if wake else "JOSE voice is ready.")
        while True:
            if wake and not await self.wait_for_wake():
                await self.say("Goodbye.")
                return
            await self.run_once()
            self.files.clear()  # attached files belong to one objective
            if once:
                return


async def _typed_path() -> str:
    return (await asyncio.to_thread(cli.ask, "file path (blank to skip)> ")).strip().strip('"')


def build_session(text_mode: bool, wake: str | None, simulate: bool = False, files: list[str] | None = None) -> VoiceSession:
    if text_mode or sys.platform != "win32":
        return VoiceSession(ConsoleStt(), ConsoleTts(), wake, simulate=simulate, files=files, picker=_typed_path if text_mode else None)
    try:
        stt = MicStt()
    except Exception as e:  # noqa: BLE001 - missing sounddevice / key: use the offline recogniser
        print(f"(using offline Windows speech recognition: {e})", flush=True)
        stt = SapiStt()
    return VoiceSession(stt, SapiTts(), wake, simulate=simulate, files=files)
