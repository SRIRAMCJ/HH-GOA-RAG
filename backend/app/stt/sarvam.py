import httpx

from app.config import Settings


class SarvamSTT:
    endpoint = "https://api.sarvam.ai/speech-to-text"

    def __init__(self, settings: Settings):
        if not settings.sarvam_api_key:
            raise RuntimeError("SARVAM_API_KEY is required")
        self.settings = settings

    async def transcribe(self, audio: bytes, filename: str = "audio.webm", language_code: str | None = None) -> dict:
        data = {"model": "saaras:v3", "mode": "transcribe"}
        if language_code:
            data["language_code"] = language_code
        files = {"file": (filename, audio, "audio/webm")}
        headers = {"api-subscription-key": self.settings.sarvam_api_key}
        async with httpx.AsyncClient(timeout=35.0) as client:
            response = await client.post(self.endpoint, headers=headers, data=data, files=files)
            response.raise_for_status()
            return response.json()
