# Voice UI — Design Spec

## Project Context

Sub-projects 1–4 (local chat loop, MCP tool-calling, files/apps, web access) all
run as a terminal application today. This spec covers a new, separate layer: a
real desktop application with voice interaction and an animated visual
character, sitting on top of everything already built rather than replacing
any of it.

## Goal

A voice-first desktop app: speak to the assistant, it speaks back, with an
animated on-screen character reflecting what it's doing (listening, thinking,
speaking, idle) — the full "Jarvis" experience, built entirely with local,
private components (no cloud speech APIs), consistent with the rest of this
project.

## Architecture

Four components:

1. **Python backend (FastAPI)** — wraps the existing `Chat` class, both model
   backends (`transformers`, `ollama`), and the MCP tool clients unchanged.
   Exposes one WebSocket endpoint for real-time, bidirectional communication:
   text/audio in, streamed text + tool-call status + audio out.
2. **Local voice pipeline**, inside the Python backend — `faster-whisper` for
   speech-to-text, **Kokoro** for text-to-speech (both fully local, no cloud
   calls — matching the rest of the project, and the same choices already
   flagged as good fits in the `pipeline-copilot-when` project's stack). Mic
   audio arrives over the websocket, gets transcribed, and is fed into the
   existing `Chat.chat()` flow exactly as if typed; the reply is spoken back
   through Kokoro as it's generated.
3. **Tauri shell** — the native desktop app wrapper (Rust). Window, packaging,
   eventual installable build. Mostly plumbing.
4. **Frontend (React, inside Tauri)** — renders the character and UI state
   coming from the backend (idle / listening / thinking / speaking /
   disconnected) and sends mic audio + typed text up over the websocket. Holds
   almost no logic of its own.

**Data flow for one voice turn:** speak → mic audio streams to the backend →
`faster-whisper` transcribes it → transcript enters `Chat.chat()` (all existing
tool-calling, multi-hop logic, and confirmation gates work unchanged) → the
reply streams back as text and is simultaneously spoken via Kokoro → the
frontend animates through listening → thinking → speaking as this happens.

## Layout

Full-bleed, minimal chrome: the character is centered and dominates the
window; there is no persistent visible transcript or toolbar. A single small
status word appears beneath the character during active states. This was
validated during brainstorming against three layout options (full-bleed
minimal / side transcript panel / bottom status bar) — full-bleed minimal was
preferred.

## Character Design: the Lemon-Mirchi Charm

The on-screen character is a nimbu-mirchi (lemon and chili) protective charm,
hanging freely on a thread — a deliberate pivot away from an earlier
"silver/sharp/geometric AI head" direction that was tried and rejected during
brainstorming (see design history below). Construction: one lemon (with a
small leaf) at the top of a thread, a horizontal bundle of four dried red
chilies layered beneath it. Full visual reference (confirmed via iterative
mockups, including exact SVG shapes/colors/animation keyframes) is preserved
in `.superpowers/brainstorm/803-1790108288/content/` — `lemon-mirchi-v3.html`
(base rest/idle/speaking), `listening-state-v3.html`, `thinking-state.html`
(fly-buzz), and `disconnected-state-v8.html` — these are session-local,
gitignored files, meant as an implementation reference, not final production
code.

**Five states, each a distinct animation:**

- **Idle** — the whole charm sways gently side to side on its thread (a slow
  pendulum motion), no other change.
- **Listening** — the lemon's leaf twitches on a quick, uneven rhythm (like an
  ear reacting to sound), and the lemon gets a soft pulsing glow. Rest of the
  charm continues its idle sway underneath.
- **Thinking** — a fly buzzes erratically near the chili bundle, then gets
  "eaten" when the mouth (see Speaking) briefly snaps open, on a ~4s loop.
  Chosen deliberately over a more literal "processing" animation (a twist on
  the thread was tried and rejected in favor of this, once prototyped).
- **Speaking** — the front-center chili in the bundle has a mouth (a dark slit
  drawn on it) that opens and closes irregularly, timed to suggest speech
  rather than a steady beat.
- **Disconnected** (backend/websocket lost) — not a generic dimmed state.
  The lemon hangs low and tilted at a precarious angle from a long, clearly
  visible thread (swinging through a wide arc, not stuck leaning to one
  side), while all four chilies have fallen and lie scattered, motionless, at
  the bottom of the frame, and several flies drift in loose independent
  orbits around the now-bare lemon. Colors throughout are desaturated
  relative to the normal states, to reinforce "neglected," while remaining
  clearly visible against the dark background (an earlier version over-
  darkened everything to the point of being unreadable — fixed).

**Design history worth preserving:** the original brief was "sleek, silver,
sharp, aggressive-edge, Jarvis-style" — two geometric directions (a faceted
low-poly "head" shape, then a sharper "blade mask" with a visor) were built
and rejected as not matching what the user actually pictured, despite fitting
the literal brief. The pivot to the lemon-mirchi charm came from the user
directly, and turned out to be the right direction — worth remembering that
"matches the literal brief" and "is actually liked" aren't the same thing, and
that abandoning a technically-on-brief direction after real negative feedback
was the correct call, not a failure to nail the brief.

## Error Handling

- **Mic permission denied** — character shows a brief distinct visual cue
  (not yet designed in detail) and the UI falls back to a text input box so
  the app remains usable without voice.
- **STT/TTS failure** — falls back to text-only for that turn (shows the
  transcript, skips speaking it aloud) rather than freezing or crashing.
- **Backend/websocket disconnected** — the "Disconnected" character state
  above. This is deliberately a strong, hard-to-miss visual (not a subtle
  dimming), since a broken connection with no clear signal was judged worse
  than an obvious, slightly startling one.
- **Ollama not running / model not loaded** — same pattern already built into
  `main.py`: fail with a clear message before the window even opens, don't
  let the UI come up in a broken state.

## Testing

Manual, consistent with the rest of this project (no automated suite exists
anywhere in it): a checklist per state — does idle sway continuously, does
listening visibly react when you actually speak, does the mouth roughly track
real speech timing during Speaking, does Thinking show during real tool calls
(not just a fixed delay), does Disconnected trigger correctly when the
backend is killed and clear when it reconnects.

## Out of Scope

- Exact visual treatment for the mic-permission-denied cue (flagged above,
  not designed).
- Packaging/distribution (installer, auto-update) — Tauri supports this, not
  addressed here.
- Wake-word / always-listening detection — assumed push-to-talk or an
  explicit "start listening" trigger for v1, not specified further.
- Multiple simultaneous conversations/windows.
- Any visual accommodation for the geometric "blade mask" direction — fully
  superseded by the charm, not kept as a fallback or alternate theme.
