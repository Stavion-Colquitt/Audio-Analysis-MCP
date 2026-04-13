# audio-analysis-mcp

An MCP (Model Context Protocol) server for real-time audio analysis, key detection, and session context building in music production environments. Designed to work alongside Ableton Live via the ableton-mcp Remote Script.

## What it does

- **Mix analysis** — captures system audio and returns frequency band energy, RMS, peak, headroom, stereo width, mono compatibility, and brightness
- **Key detection** — multi-source voting system using bass track MIDI, melody track MIDI, and audio pitch detection (pyin). Classifies primary notes separately from passing tones and accidentals so chromatic lines don't corrupt key reads. Detects major, minor, and modal keys
- **Per-section key detection** — runs key detection per arrangement section rather than globally, supporting songs that modulate between sections
- **Section structure inference** — infers song structure from clip positions, detects repeated sections by MIDI fingerprinting, and auto-labels verse/chorus patterns
- **Session context builder** — builds a unified pre-session payload containing key, tempo, time signature, harmonic rhythm, melodic range, song structure, and pitch distribution — everything a model needs before a recording starts
- **Live vocal capture** — STT listener (Whisper) and pitch tracker (pyin) that write transcribed lyrics and detected notes to a per-section performed layer in real time
- **Polyphonic instrument analysis** — Basic-Pitch integration for analyzing piano, guitar, violin, and other instruments that play multiple notes simultaneously
- **Qdrant integration** — writes session context to a vector database so downstream models can query it independently. Event-driven auto-sync keeps the database current when Ableton session properties change
- **Manual overrides** — any detected value (key, section label, section key) can be overridden at any granularity with an `is_override` flag that all downstream models respect

## Requirements

```
pip install mcp sounddevice librosa scipy numpy qdrant-client
```

Optional, for live vocal capture:
```
pip install openai-whisper soundfile
```

Optional, for polyphonic instrument analysis:
```
pip install basic-pitch
```

## Setup

### Windows
Enable Stereo Mix (Realtek) in your audio settings to capture system output. Use `list_audio_devices` to find the device index.

### Mac
Install BlackHole or Loopback for system audio capture.

### Ableton integration
Install the ableton-mcp Remote Script and select it as a Control Surface in Ableton Preferences → MIDI. The server connects to it over localhost:9877 for MIDI data and session info.

## Running the server

```bash
python server.py
```

Then connect Claude Desktop or any MCP-compatible client to the server.

## Available tools

### Audio capture
- `list_audio_devices` — list available input devices
- `start_capture` / `stop_capture` — start/stop continuous audio analysis
- `get_mix_analysis` — full spectral analysis of captured audio
- `get_mix_report` — plain English mixing notes
- `get_frequency_report` — focused frequency balance report with EQ suggestions
- `get_stereo_analysis` — stereo field, mono compatibility, L/R balance
- `capture_and_analyze_file` — analyze any audio file directly

### Key detection
- `get_key_with_voting` — primary key detection tool, three-source voting
- `get_key_from_midi` — detect key from specific MIDI tracks
- `analyze_bounced_instrumental` — detect key from a session with only a bounced WAV/MP3
- `set_key_override` / `get_key_override` — manual key override
- `set_section_key_override` — override key for a specific section
- `build_per_section_keys` — run key detection per section

### Session context
- `build_session_context` — build full pre-session context payload
- `get_current_session_context` — retrieve current context with all updates
- `get_song_context` — musical context for ML model initialization
- `detect_section_repetitions` — find verse/chorus patterns by MIDI fingerprinting
- `update_section_heard` — mark a section complete, updates prior confidence
- `set_recording_section` — tell live listeners which section is active

### Live capture
- `start_stt_listener` / `stop_stt_listener` — Whisper vocal transcription
- `start_pitch_listener` / `stop_pitch_listener` — pyin vocal pitch tracking
- `analyze_instrument_audio` — Basic-Pitch polyphonic instrument analysis
- `get_performed_context` — view captured lyrics, notes, instrument data

### Vector database
- `write_context_to_qdrant` — write full session context to Qdrant
- `update_section_in_qdrant` — update a single section after new data arrives
- `start_auto_sync` / `stop_auto_sync` / `get_auto_sync_status` — event-driven sync

## Key detection approach

Standard key detection libraries run Krumhansl-Schmuckler directly on raw chroma. This server takes a different approach:

1. **Note classification** — each pitch class is classified as primary, secondary, passing, or accidental based on duration, frequency of occurrence, and rhythmic context. Short fast notes (melisma, passing tones) are weighted low. Sustained notes on strong beats are weighted high.

2. **Tonic detection** — the tonic is found from primary notes only, weighted by beat position (beat 1 of bar is strongest), note duration, first/last note position, and velocity.

3. **Scale matching** — primary notes are matched against 11 scale templates: major, natural minor, dorian, phrygian, lydian, mixolydian, locrian, harmonic minor, blues, pentatonic minor, pentatonic major.

4. **Multi-source voting** — three independent sources vote on the key. Bass track MIDI and melody track MIDI are compared. If they agree, result is high confidence. If they disagree, a third source (vocal audio or additional MIDI track) acts as tiebreaker. All-disagree cases return the highest confidence result with a manual verification flag.

## Qdrant schema

Each section is stored as a point with a 12-dimensional pitch class vector and a payload containing full section context:

```
song (point 0)  — global context, key, tempo, structure summary
section_0       — key, mode, scale degrees, performed layer, overrides
section_1       — ...
section_N       — ...
```

The auto-sync feature holds an open connection to the Ableton Remote Script and pushes Qdrant updates automatically when tempo, time signature, or clip content changes.

## Extending

The `orchestrator_client.py` in the companion repo provides a Python HTTP client for calling these tools from your own orchestrator model or plugin without going through an MCP-compatible client.
