const API_BASE = window.API_BASE || new URLSearchParams(location.search).get("api") || "";
const query = document.getElementById("query");
const ask = document.getElementById("ask");
const record = document.getElementById("record");
const recordState = document.getElementById("recordState");
const answerCard = document.getElementById("answerCard");
const answer = document.getElementById("answer");
const grounded = document.getElementById("grounded");
const latency = document.getElementById("latency");
const sources = document.getElementById("sources");

async function runQuery(text) {
  if (!text.trim()) return;
  ask.disabled = true;
  ask.textContent = "Retrieving...";
  try {
    const res = await fetch(`${API_BASE}/api/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: text, debug: true })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "RAG request failed");
    answerCard.classList.remove("hidden");
    answer.textContent = data.answer;
    grounded.textContent = data.grounded ? "✓ Grounded" : "⚠ Refused";
    latency.textContent = `${Number(data.latency_ms || 0).toFixed(1)} ms`;
    sources.innerHTML = (data.sources || []).map((s) =>
      `<div class="source"><strong>${escapeHtml(s.chunk_id)}</strong> · ${Number(s.score).toFixed(4)} · ${escapeHtml(s.strategy)}<br>${escapeHtml(s.text)}</div>`
    ).join("");
  } catch (err) {
    answerCard.classList.remove("hidden");
    answer.textContent = err.message || "The RAG service is unavailable.";
    grounded.textContent = "⚠ Error";
    latency.textContent = "—";
  } finally {
    ask.disabled = false;
    ask.textContent = "Ask RAG →";
  }
}

ask.addEventListener("click", () => runQuery(query.value));

let recorder;
let chunks = [];
record.addEventListener("click", async () => {
  if (recorder?.state === "recording") {
    recorder.stop();
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recorder = new MediaRecorder(stream);
    chunks = [];
    recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    recorder.onstart = () => {
      record.classList.add("active");
      recordState.textContent = "Listening... press again to stop";
    };
    recorder.onstop = async () => {
      record.classList.remove("active");
      stream.getTracks().forEach((track) => track.stop());
      recordState.textContent = "Transcribing...";
      const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
      const form = new FormData();
      form.append("file", blob, "voice.webm");
      try {
        const response = await fetch(`${API_BASE}/api/voice/transcribe`, { method: "POST", body: form });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "STT failed");
        query.value = data.transcript || "";
        recordState.textContent = data.language_code ? `Detected ${data.language_code}` : "Transcribed";
        await runQuery(query.value);
      } catch (err) {
        recordState.textContent = err.message || "Transcription failed";
      }
    };
    recorder.start();
  } catch (err) {
    recordState.textContent = "Microphone permission is required.";
  }
});

function escapeHtml(value) {
  return String(value).replace(/[&<>\"]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"}[c]));
}
