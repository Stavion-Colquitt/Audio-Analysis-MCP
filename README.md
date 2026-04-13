# audio-analysis-mcp

An MCP (Model Context Protocol) server for real-time audio analysis, key detection, and pre-session context building in music production environments. Connects to Ableton Live via the ableton-mcp Remote Script for MIDI-based analysis and session data.

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/audio-analysis-mcp
cd audio-analysis-mcp
pip install -r requirements.txt
```

Install optional dependencies for the features you need:

```bash
pip install -r requirements-optional.txt
```

| Package | Required for |
|---|---|
| `qdrant-client` | Vector database write tools, auto-sync |
| `openai-whisper` + `soundfile` | Live vocal transcription (`start_stt_listener`) |
| `basic-pitch` + `soundfile` | Polyphonic instrument analysis (`analyze_instrument_audio`, `start_polyphonic_listener`)

## Claude Desktop setup

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "audio-analysis": {
      "command": "python",
      "args": ["/absolute/path/to/server.py"]
    }
  }
}
```

Config file locations:
- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

Restart Claude Desktop after editing.

## Audio device setup

### Windows
Enable Stereo Mix (Realtek) in your sound settings to capture system output without additional routing. Use `list_audio_devices` to find the device index.

### Mac
Install [BlackHole](https://existential.audio/blackhole/) or [Loopback](https://rogueamoeba.com/loopback/) for system audio capture.

## Ableton Live integration

Install the [ableton-mcp](https://github.com/ahujasid/ableton-mcp) Remote Script and select it as a Control Surface in Ableton Preferences → MIDI. This server connects to it over localhost:9877 for MIDI data, session info, and change events.

Without Ableton connected, all mix analysis and audio file tools still work. MIDI-dependent tools (key detection from MIDI, session context builder, auto-sync) will return connection errors.

## Running the server

```bash
python server.py
```

The server starts and waits for MCP client connections.

---

## Tools

### Audio devices and capture

| Tool | Description |
|---|---|
| `list_audio_devices` | List available audio input devices with index numbers |
| `start_capture(device_index, window_seconds)` | Start continuous audio capture for analysis. Default window is 8s. Use 30-60s for better key/BPM detection. |
| `stop_capture` | Stop the capture loop |
| `get_mix_analysis` | Full spectral analysis — frequency bands, RMS, peak, headroom, stereo field, key candidates, brightness |
| `get_mix_report` | Plain English mixing notes with actionable suggestions |
| `get_frequency_report` | Focused frequency balance report with EQ suggestions |
| `get_stereo_analysis` | Stereo width, L/R balance, mono compatibility, correlation |
| `capture_and_analyze_file(file_path)` | Analyze any audio file directly (WAV, MP3, FLAC) |

### Key detection

| Tool | Description |
|---|---|
| `get_key_with_voting(vocal_file_path)` | Primary key detection tool. Finds bass and melody tracks automatically, runs three-source voting. |
| `get_key_from_midi(melodic_track_indices, vocal_file_path)` | Detect key from specific MIDI track indices |
| `analyze_bounced_instrumental` | Detect key from a session containing only a bounced WAV/MP3 with no MIDI |
| `set_key_override(key)` | Set global key manually. Pass empty string to clear. Format: `"E minor"`, `"C major"`, `"A dorian"` |
| `get_key_override` | Check if a manual override is active |
| `build_per_section_keys` | Run key detection per section — supports songs that modulate between sections |
| `set_section_key_override(section_index, key, label)` | Override the key for one section. Sets `is_override` flag so all models trust it. |

### Session context

| Tool | Description |
|---|---|
| `build_session_context(vocal_file_path)` | Build full pre-session context payload — key, tempo, time sig, structure, harmonic rhythm, melodic range, priors. Run once before recording starts. |
| `get_current_session_context` | Retrieve current context with all progressive updates |
| `get_song_context` | Simplified musical context object for ML model initialization |
| `detect_section_repetitions` | Fingerprint MIDI content per section to identify verse/chorus patterns and auto-label sections |
| `update_section_heard(section_index, section_label)` | Mark a section complete. Updates prior confidence level from genre-only toward full-song context. |
| `set_recording_section(section_index)` | Tell the STT and pitch listeners which section is currently being recorded |

### Live capture

| Tool | Description |
|---|---|
| `start_stt_listener(device_index, section_index)` | Start Whisper transcription on a mic. Writes lyrics with timestamps to the performed layer. Requires `openai-whisper` and `soundfile`. |
| `stop_stt_listener` | Stop STT listener |
| `start_pitch_listener(device_index, section_index)` | Start pyin vocal pitch tracking. Writes detected notes to the performed layer. |
| `stop_pitch_listener` | Stop pitch listener |
| `analyze_instrument_audio(file_path, track_name, section_index)` | Run Basic-Pitch on a saved polyphonic audio file (WAV, MP3, FLAC). Extracts chord and note events and writes to the performed layer. Use this for recorded takes. Requires `basic-pitch` and `soundfile`. |
| `start_polyphonic_listener(device_index, track_name, section_index, window_seconds)` | Live polyphonic chord and harmony detection via Basic-Pitch. Works for piano, guitar, violin, or any instrument playing multiple notes simultaneously. Captures audio in windows (default 2.5s) and writes detected notes to the performed layer in near-real time. There is inherent latency equal to the window size — use Ableton MIDI routing when the instrument can output MIDI. Requires `basic-pitch` and `soundfile`. |
| `stop_polyphonic_listener` | Stop the live polyphonic listener. |
| `get_performed_context(section_index)` | View what has been captured — lyrics, notes sung, instrument note data. Pass -1 for all sections. |

### Vector database (Qdrant)

| Tool | Description |
|---|---|
| `write_context_to_qdrant(qdrant_url, collection_name)` | Write full session context to Qdrant. One point per section plus a global point. Creates the collection if it doesn't exist. |
| `update_section_in_qdrant(section_index, qdrant_url, collection_name)` | Update a single section after new performed data arrives. Faster than rewriting everything. |
| `start_auto_sync(qdrant_url, collection_name)` | Open a persistent connection to Ableton and listen for change events. Tempo, time signature, and clip changes automatically trigger re-analysis and Qdrant updates. |
| `stop_auto_sync` | Stop auto-sync |
| `get_auto_sync_status` | Check whether auto-sync is running |

---

## Key detection approach

Standard libraries run Krumhansl-Schmuckler directly on raw chroma. This server uses a four-stage approach:

**Stage 1 — Note classification.** Each pitch class is classified as primary, secondary, passing, or accidental based on duration and rhythmic context. Short fast notes (melisma, passing tones) are weighted low. Notes that only appear in fast contexts are flagged as likely accidentals and excluded from key detection. This prevents chromatic lines and modal color notes from corrupting the key read.

**Stage 2 — Tonic detection.** The tonic is found from primary notes only, weighted by beat position (beat 1 of bar weighted highest), note duration, first/last note position, and velocity.

**Stage 3 — Scale matching.** Primary note distribution is matched against 11 scale templates: major, natural minor, dorian, phrygian, lydian, mixolydian, locrian, harmonic minor, blues, pentatonic minor, pentatonic major. The tonic detection result receives a confidence bonus.

**Stage 4 — Multi-source voting.** Bass track MIDI and melody track MIDI are compared independently. Agreement = high confidence result. Disagreement = third source tiebreaker (vocal audio if provided, otherwise a bounced WAV or additional MIDI track). All-disagree cases return the highest confidence result with a `manual_verification_needed` flag.

Per-section key detection follows the same process applied to notes within each section's time range, supporting songs that modulate mid-arrangement.

---

## Qdrant schema

```
point 0    — global context: key, tempo, time sig, section count, priors
point 1    — section 0: key, mode, scale degrees, performed layer, override flags
point 2    — section 1: ...
point N+1  — section N: ...
```

Each section point vector is the 12-dimensional pitch class distribution for that section. Sections with manual key overrides carry an `is_override: true` flag in the payload.

---

## Orchestrator integration

The companion repo `backing-vocalist-orchestrator` provides a Python HTTP client for calling these tools from your own orchestrator model or plugin without going through Claude Desktop.
