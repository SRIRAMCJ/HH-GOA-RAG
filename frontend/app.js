const API_BASE = window.API_BASE || "http://localhost:8000";
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
    answerCard.classList.remove("hidden");
    answer.textContent = data.answer;
    grounded.textContent = data.grounded ? "✓ Grounded" : "⚠ Refused";
    latency.textContent = `${Number(data.latency_ms || 0).toFixed(1)} ms`;
    sources.innerHTML = (data.sources || []).map((s) =>
      `<div class="source"><strong>${s.chunk_id}</strong> · ${Number(s.score).toFixed(4)}<br>${escapeHtml(s.text)}</div>`
    ).join("");
  } catch (err) {
    answerCard.classList.remove("hidden");
    answer.textContent = "The RAG service is unavailable.";
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
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  recorder = new MediaRecorder(stream);
  chunks = [];
  recorder.ondataavailable = (e) => chunks.push(e.data);
  recorder.onstart = () => {
    record.classList.add("active");
    recordState.textContent = "Listening... press again to stop";
  };
  recorder.onstop = async () => {
    record.classList.remove("active");
    recordState.textContent = "Voice captured — STT endpoint will be connected next";
    stream.getTracks().forEach((track) => track.stop());
    // The audio route is intentionally kept separate from /api/query so the
    // mandated Sarvam/ElevenLabs provider can be swapped without touching RAG.
  };
  recorder.start();
});

function escapeHtml(value) {
  return String(value).replace(/[&<>\"]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"}[c]));
}
