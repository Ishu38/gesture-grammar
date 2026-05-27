"""TTS provider abstraction for MLAF.

Why this exists
---------------
Web Speech API in the browser depends entirely on OS-installed voices.
On stock Windows 10/11 and most Linux distros, **no Bengali voice ships
by default**. The user has to manually install one through Settings →
Time & language → Speech → Add voices. That's friction we don't want
to push onto motor-impaired learners or rural-school therapists.

This module gives the FastAPI app a single ``synthesize(text, lang)``
entrypoint that returns MP3 bytes. Today it routes through Google's
public translate TTS via ``gTTS`` — free, no API key, no signup, decent
quality for English / Hindi / Bengali. Tomorrow, if Neil registers at
bhashini.gov.in and supplies BHASHINI_USER_ID + BHASHINI_API_KEY env
vars, the same call routes through Bhashini's Dhruva models without
any frontend change.

Design choices
--------------
* **Pure stdlib + one third-party (gTTS)** — keeps the HF Space image
  small and the request path predictable.
* **Provider chain** — try Bhashini first if env vars present, fall
  back to gTTS. Either way the route always returns audio or 503.
* **In-memory cache** keyed by (text, lang) — common phrases like
  "I grab apple" should not hit Google twice in a session.
"""

from __future__ import annotations

import io
import logging
import os
from collections import OrderedDict
from typing import Protocol

logger = logging.getLogger(__name__)

# Languages MLAF's TTS UI currently exposes. Adding a new one is one line.
SUPPORTED_LANGS = ("en", "hi", "bn")

# Max characters per request — guards against runaway lesson text.
MAX_TEXT_CHARS = 400


class TTSProvider(Protocol):
    """Anything that can turn (text, lang) into MP3 bytes."""
    name: str

    def synthesize(self, text: str, lang: str) -> bytes: ...

    def available(self) -> bool: ...


# ---------------------------------------------------------------------------
# Google Translate TTS via gTTS — works today, no API key
# ---------------------------------------------------------------------------

class GTTSProvider:
    name = "gtts"

    def __init__(self) -> None:
        try:
            from gtts import gTTS  # noqa: F401  (deferred to runtime)
            self._ok = True
        except Exception as exc:
            logger.warning("gTTS not importable: %s", exc)
            self._ok = False

    def available(self) -> bool:
        return self._ok

    def synthesize(self, text: str, lang: str) -> bytes:
        from gtts import gTTS

        # gTTS uses "en", "hi", "bn" — match Web Speech API codes 1:1.
        tts = gTTS(text=text, lang=lang, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        return buf.getvalue()


# ---------------------------------------------------------------------------
# Bhashini (Indian Govt — Dhruva models) — stub until BHASHINI_API_KEY set
# ---------------------------------------------------------------------------

class BhashiniProvider:
    """Bhashini ULCA inference for Indian languages.

    Requires two env vars (set in the HF Space settings, NOT in repo):
        BHASHINI_USER_ID   — your user-id from ulcacontrib.org dashboard
        BHASHINI_API_KEY   — the ULCA_API_KEY shown alongside

    Pipeline endpoint and service-id are looked up once per language at
    first call, then cached. The actual inference goes to whatever URL
    the pipeline-config returns — Bhashini may rotate model providers
    behind the scenes, but the API contract is stable.
    """

    name = "bhashini"
    PIPELINE_URL = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline"
    PIPELINE_REQUEST_ID = "64392f96daac500b55c543cd"  # ULCA's standard pipeline ID for TTS

    def __init__(self) -> None:
        self._user_id = os.environ.get("BHASHINI_USER_ID")
        self._api_key = os.environ.get("BHASHINI_API_KEY")
        # Per-language cached (endpoint, service_id, headers)
        self._pipeline_cache: dict[str, tuple[str, str, dict[str, str]]] = {}

    def available(self) -> bool:
        return bool(self._user_id and self._api_key)

    def synthesize(self, text: str, lang: str) -> bytes:
        import base64
        import json
        import urllib.request

        endpoint, service_id, headers = self._get_pipeline(lang)
        body = json.dumps({
            "pipelineTasks": [{
                "taskType": "tts",
                "config": {
                    "language": {"sourceLanguage": lang},
                    "serviceId": service_id,
                    "gender": "female",
                },
            }],
            "inputData": {"input": [{"source": text}]},
        }).encode("utf-8")

        req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read())

        audio_b64 = payload["pipelineResponse"][0]["audio"][0]["audioContent"]
        return base64.b64decode(audio_b64)

    def _get_pipeline(self, lang: str) -> tuple[str, str, dict[str, str]]:
        if lang in self._pipeline_cache:
            return self._pipeline_cache[lang]

        import json
        import urllib.request

        body = json.dumps({
            "pipelineTasks": [{
                "taskType": "tts",
                "config": {"language": {"sourceLanguage": lang}},
            }],
            "pipelineRequestConfig": {"pipelineId": self.PIPELINE_REQUEST_ID},
        }).encode("utf-8")
        req = urllib.request.Request(
            self.PIPELINE_URL,
            data=body,
            headers={
                "Content-Type": "application/json",
                "userID": self._user_id,
                "ulcaApiKey": self._api_key,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        cfg = data["pipelineResponseConfig"][0]["config"][0]
        service_id = cfg["serviceId"]
        callback_cfg = data["pipelineInferenceAPIEndPoint"]
        endpoint = callback_cfg["callbackUrl"]
        auth = callback_cfg["inferenceApiKey"]
        headers = {
            "Content-Type": "application/json",
            auth["name"]: auth["value"],
        }
        self._pipeline_cache[lang] = (endpoint, service_id, headers)
        return self._pipeline_cache[lang]


# ---------------------------------------------------------------------------
# Provider chain + simple LRU cache
# ---------------------------------------------------------------------------

_CACHE: "OrderedDict[tuple[str, str], bytes]" = OrderedDict()
_CACHE_MAX_ENTRIES = 64

_providers: list[TTSProvider] = []


def _init_providers() -> list[TTSProvider]:
    """Lazy provider construction — Bhashini first when configured,
    gTTS as the always-on fallback."""
    chain: list[TTSProvider] = []
    bh = BhashiniProvider()
    if bh.available():
        chain.append(bh)
        logger.info("TTS chain: bhashini (primary) → gtts (fallback)")
    g = GTTSProvider()
    if g.available():
        chain.append(g)
    if not chain:
        logger.error("No TTS provider available. Install gtts or set BHASHINI_* env.")
    elif len(chain) == 1 and chain[0].name == "gtts":
        logger.info("TTS chain: gtts (only). Set BHASHINI_USER_ID + BHASHINI_API_KEY to add Bhashini.")
    return chain


def synthesize(text: str, lang: str) -> tuple[bytes, str]:
    """Synthesize ``text`` in ``lang`` to MP3 bytes.

    Returns (audio_bytes, provider_name). Raises ValueError on bad input,
    RuntimeError if no provider succeeds.
    """
    global _providers
    if not _providers:
        _providers = _init_providers()

    text = (text or "").strip()
    if not text:
        raise ValueError("text is empty")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"text exceeds {MAX_TEXT_CHARS} char limit")
    if lang not in SUPPORTED_LANGS:
        raise ValueError(f"lang {lang!r} not in {SUPPORTED_LANGS}")

    cache_key = (lang, text)
    if cache_key in _CACHE:
        _CACHE.move_to_end(cache_key)
        return _CACHE[cache_key], "cache"

    last_err: Exception | None = None
    for provider in _providers:
        try:
            audio = provider.synthesize(text, lang)
            if audio:
                _CACHE[cache_key] = audio
                _CACHE.move_to_end(cache_key)
                while len(_CACHE) > _CACHE_MAX_ENTRIES:
                    _CACHE.popitem(last=False)
                return audio, provider.name
        except Exception as exc:
            last_err = exc
            logger.warning("TTS provider %s failed: %s", provider.name, exc)
            continue

    raise RuntimeError(f"all TTS providers failed: {last_err}")


def provider_status() -> dict:
    """For the /tts/health endpoint — reports what's wired up right now."""
    global _providers
    if not _providers:
        _providers = _init_providers()
    return {
        "active_chain": [p.name for p in _providers],
        "supported_languages": list(SUPPORTED_LANGS),
        "cache_entries": len(_CACHE),
        "bhashini_configured": BhashiniProvider().available(),
    }
