"""
Audio Analysis MCP Server
Captures audio from a virtual loopback device, extracts spectral features,
and exposes them as MCP tools so Claude can give mixing feedback.

Requirements:
    pip install mcp sounddevice librosa scipy numpy

Setup:
    - Stereo Mix (Realtek) captures your system output automatically
    - Claude calls the MCP tools to get spectral data and give mixing feedback
"""

from mcp.server.fastmcp import FastMCP
import numpy as np
import sounddevice as sd
import librosa
import scipy.signal
import json
import threading
import time
import threading as _threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AudioAnalysisMCP")

mcp = FastMCP("AudioAnalysisMCP")

# Shared state
_latest_analysis = {}
_latest_stereo = {}
_capture_thread = None
_capturing = False
_key_override = None  # manually set key bypasses detection
_auto_sync_thread = None
_auto_sync_running = False
_auto_sync_qdrant_url = "http://localhost:6333"
_auto_sync_collection = "backing_vocalist_context"
SAMPLE_RATE = 44100
CAPTURE_DURATION = 8.0


def analyze_audio(audio: np.ndarray, sr: int) -> dict:
    """Extract spectral features from a numpy audio buffer."""
    # Keep stereo for stereo analysis before mono mixdown
    stereo = audio.copy() if audio.ndim > 1 else None

    if audio.ndim > 1:
        mono = np.mean(audio, axis=1)
    else:
        mono = audio.copy()

    mono = mono.astype(np.float32)
    mono = mono / (np.max(np.abs(mono)) + 1e-9)

    # --- RMS Energy ---
    rms = float(np.sqrt(np.mean(mono ** 2)))
    rms_db = float(20 * np.log10(rms + 1e-9))

    # --- Peak amplitude ---
    peak = float(np.max(np.abs(mono)))
    peak_db = float(20 * np.log10(peak + 1e-9))

    # --- Dynamic range ---
    dynamic_range_db = float(peak_db - rms_db)

    # --- Headroom ---
    headroom_db = round(0.0 - peak_db, 2)

    # --- FFT spectral analysis ---
    fft = np.fft.rfft(mono)
    fft_magnitude = np.abs(fft)
    freqs = np.fft.rfftfreq(len(mono), 1 / sr)

    def band_energy_db(low, high):
        mask = (freqs >= low) & (freqs < high)
        energy = np.mean(fft_magnitude[mask] ** 2) if mask.any() else 1e-9
        return float(10 * np.log10(energy + 1e-9))

    sub_bass_db   = band_energy_db(20, 80)
    bass_db       = band_energy_db(80, 250)
    low_mid_db    = band_energy_db(250, 800)
    mid_db        = band_energy_db(800, 2500)
    high_mid_db   = band_energy_db(2500, 6000)
    presence_db   = band_energy_db(6000, 12000)
    air_db        = band_energy_db(12000, 20000)

    # --- Spectral balance score vs reference curve ---
    # Reference: typical well-balanced mix has sub>bass>low_mid>mid>high_mid>presence>air
    # Score each band deviation from expected relative slope
    bands = [sub_bass_db, bass_db, low_mid_db, mid_db, high_mid_db, presence_db, air_db]
    band_names = ["sub_bass", "bass", "low_mid", "mid", "high_mid", "presence", "air"]
    balance_issues = []
    for i in range(1, len(bands)):
        diff = bands[i] - bands[i-1]
        if diff > 3:
            balance_issues.append(f"{band_names[i]} is {round(diff,1)}dB louder than {band_names[i-1]} — unexpected rise")
        if diff < -20:
            balance_issues.append(f"{band_names[i]} drops {round(abs(diff),1)}dB from {band_names[i-1]} — steep rolloff")

    # --- Spectral centroid (brightness) ---
    centroid = librosa.feature.spectral_centroid(y=mono, sr=sr)
    centroid_mean = float(np.mean(centroid))

    # Brightness descriptor
    if centroid_mean < 2000:
        brightness = "dark/warm"
    elif centroid_mean < 4000:
        brightness = "balanced"
    elif centroid_mean < 7000:
        brightness = "present/bright"
    else:
        brightness = "very bright/airy"

    # --- Spectral flatness (how noise-like vs tonal) ---
    flatness = librosa.feature.spectral_flatness(y=mono)
    flatness_mean = float(np.mean(flatness))

    # --- MFCCs ---
    mfccs = librosa.feature.mfcc(y=mono, sr=sr, n_mfcc=13)
    mfcc_means = [float(x) for x in np.mean(mfccs, axis=1)]

    # --- BPM ---
    try:
        tempo, _ = librosa.beat.beat_track(y=mono, sr=sr, hop_length=512)
        bpm = float(np.atleast_1d(tempo)[0])
    except Exception:
        bpm = 0.0

    # --- Key detection via HPSS + Krumhansl-Schmuckler ---
    try:
        harmonic, _ = librosa.effects.hpss(mono, margin=3.0)
        chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_mean = chroma_mean / (np.sum(chroma_mean) + 1e-9)

        major_profile = np.array([6.35,2.23,3.48,2.33,4.38,4.09,
                                   2.52,5.19,2.39,3.66,2.29,2.88])
        minor_profile = np.array([6.33,2.68,3.52,5.38,2.60,3.53,
                                   2.54,4.75,3.98,2.69,3.34,3.17])
        key_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

        best_score = -np.inf
        estimated_key = "Unknown"
        key_confidence_scores = {}

        for i in range(12):
            maj  = np.roll(major_profile, i) / (np.sum(major_profile) + 1e-9)
            min_ = np.roll(minor_profile, i) / (np.sum(minor_profile) + 1e-9)
            score_maj = float(np.dot(chroma_mean, maj))
            score_min = float(np.dot(chroma_mean, min_))
            key_confidence_scores[f"{key_names[i]} major"] = round(score_maj, 4)
            key_confidence_scores[f"{key_names[i]} minor"] = round(score_min, 4)
            if score_maj > best_score:
                best_score = score_maj
                estimated_key = f"{key_names[i]} major"
            if score_min > best_score:
                best_score = score_min
                estimated_key = f"{key_names[i]} minor"

        # Top 3 key candidates for context
        top_keys = sorted(key_confidence_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    except Exception:
        estimated_key = "Unknown"
        top_keys = []

    # --- Clipping detection ---
    clip_threshold = 0.999
    clip_percent = float(np.mean(np.abs(mono) > clip_threshold) * 100)
    clipping = clip_percent > 0.5

    # --- Stereo field analysis ---
    stereo_data = {}
    if stereo is not None and stereo.ndim > 1 and stereo.shape[1] >= 2:
        left  = stereo[:, 0].astype(np.float32)
        right = stereo[:, 1].astype(np.float32)
        left  = left  / (np.max(np.abs(left))  + 1e-9)
        right = right / (np.max(np.abs(right)) + 1e-9)

        # L/R balance
        rms_left  = float(np.sqrt(np.mean(left  ** 2)))
        rms_right = float(np.sqrt(np.mean(right ** 2)))
        balance_db = float(20 * np.log10((rms_left + 1e-9) / (rms_right + 1e-9)))

        # Stereo width via mid-side
        mid  = (left + right) / 2
        side = (left - right) / 2
        rms_mid  = float(np.sqrt(np.mean(mid  ** 2)))
        rms_side = float(np.sqrt(np.mean(side ** 2)))
        width = float(rms_side / (rms_mid + 1e-9))

        # Mono compatibility — correlation between L and R
        correlation = float(np.corrcoef(left, right)[0, 1])

        if correlation > 0.9:
            mono_compat = "excellent — nearly mono"
        elif correlation > 0.6:
            mono_compat = "good"
        elif correlation > 0.3:
            mono_compat = "moderate — check on mono"
        else:
            mono_compat = "poor — may lose elements in mono"

        stereo_data = {
            "balance_db": round(balance_db, 2),
            "balance_note": "centered" if abs(balance_db) < 0.5 else ("left-heavy" if balance_db > 0 else "right-heavy"),
            "stereo_width": round(width, 3),
            "width_note": "narrow" if width < 0.2 else ("moderate" if width < 0.5 else "wide"),
            "mono_compatibility": mono_compat,
            "lr_correlation": round(correlation, 3),
        }

    return {
        "rms_db": round(rms_db, 2),
        "peak_db": round(peak_db, 2),
        "dynamic_range_db": round(dynamic_range_db, 2),
        "headroom_db": headroom_db,
        "clipping": clipping,
        "clip_percent": round(clip_percent, 3),
        "frequency_bands": {
            "sub_bass_20_80hz":    round(sub_bass_db, 2),
            "bass_80_250hz":       round(bass_db, 2),
            "low_mid_250_800hz":   round(low_mid_db, 2),
            "mid_800_2500hz":      round(mid_db, 2),
            "high_mid_2500_6khz":  round(high_mid_db, 2),
            "presence_6_12khz":    round(presence_db, 2),
            "air_12_20khz":        round(air_db, 2),
        },
        "balance_issues": balance_issues,
        "spectral_centroid_hz": round(centroid_mean, 1),
        "brightness": brightness,
        "spectral_flatness": round(flatness_mean, 4),
        "stereo": stereo_data,
        "mfcc_means": mfcc_means,
        "estimated_bpm": round(bpm, 1),
        "bpm_note": "Use Ableton session BPM for accuracy",
        "estimated_key": estimated_key,
        "key_candidates": [{"key": k, "score": s} for k, s in top_keys],
    }


def _capture_loop(device_index: int, window_seconds: float):
    """Background thread — continuously captures and analyzes audio."""
    global _latest_analysis, _capturing
    logger.info(f"Starting capture loop on device index {device_index}, window={window_seconds}s")
    while _capturing:
        try:
            audio = sd.rec(
                int(window_seconds * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=2,
                dtype="float32",
                device=device_index,
            )
            sd.wait()
            result = analyze_audio(audio, SAMPLE_RATE)
            result["captured_at"] = time.strftime("%H:%M:%S")
            result["window_seconds"] = window_seconds
            _latest_analysis = result
            logger.info(f"Analysis updated: RMS={result['rms_db']}dB brightness={result['brightness']}")
        except Exception as e:
            logger.error(f"Capture error: {e}")
            time.sleep(1)


# ─────────────────────────────────────────
# MCP Tools
# ─────────────────────────────────────────

@mcp.tool()
def list_audio_devices() -> str:
    """List all available audio input devices so you can pick the right one."""
    devices = sd.query_devices()
    result = []
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            result.append(f"[{i}] {d['name']}  (inputs: {d['max_input_channels']})")
    return "\n".join(result) if result else "No input devices found."


@mcp.tool()
def start_capture(device_index: int, window_seconds: float = 8.0) -> str:
    """
    Start capturing audio from a device for continuous analysis.
    Use list_audio_devices first to find the right device index.
    On Windows, Stereo Mix (Realtek) captures system output without routing changes.
    On Mac, use BlackHole or Loopback as the device.

    Parameters:
    - device_index: The audio device to capture from
    - window_seconds: How many seconds to capture per analysis window.
      8 seconds (default) is good for mix feedback.
      30-60 seconds gives much better key and BPM detection.
      Use longer windows when analyzing a full arrangement or checking key.
    """
    global _capture_thread, _capturing
    if _capturing:
        return "Capture already running. Call stop_capture first to restart."
    _capturing = True
    _capture_thread = threading.Thread(
        target=_capture_loop, args=(device_index, window_seconds), daemon=True
    )
    _capture_thread.start()
    return f"Capture started on device {device_index} with {window_seconds}s window. First analysis ready in ~{window_seconds}s."


@mcp.tool()
def stop_capture() -> str:
    """Stop the audio capture loop."""
    global _capturing
    _capturing = False
    return "Capture stopped."


@mcp.tool()
def get_mix_analysis() -> str:
    """
    Get the latest full spectral analysis of the captured audio.
    Returns frequency bands, RMS, peak, headroom, stereo field, key candidates,
    brightness, balance issues, and mono compatibility.
    """
    if not _latest_analysis:
        return "No analysis available yet. Call start_capture first and wait a few seconds."
    return json.dumps(_latest_analysis, indent=2)


@mcp.tool()
def get_mix_report() -> str:
    """
    Get a plain English mixing report based on the latest analysis.
    Interprets the spectral data and gives specific actionable mixing notes.
    Use this for human-readable feedback instead of raw numbers.
    """
    if not _latest_analysis:
        return "No analysis available yet. Call start_capture first."

    a = _latest_analysis
    notes = []

    # Levels
    rms = a.get("rms_db", 0)
    peak = a.get("peak_db", 0)
    headroom = a.get("headroom_db", 0)
    clipping = a.get("clipping", False)

    if clipping:
        notes.append(f"CLIPPING DETECTED — {a.get('clip_percent', 0)}% of samples are saturating. Reduce track or master levels immediately.")
    elif headroom < 1:
        notes.append(f"Very little headroom ({headroom}dB). Consider pulling the master down slightly before adding more elements.")
    elif headroom > 12:
        notes.append(f"Plenty of headroom ({headroom}dB). Mix is running quiet — levels could come up.")

    if rms < -20:
        notes.append(f"Mix RMS is low at {rms}dB. This is fine for a work-in-progress but will need to come up in mastering.")
    elif rms > -6:
        notes.append(f"Mix RMS is hot at {rms}dB. Leave more room for mastering — aim for -14 to -10 RMS at mix stage.")

    # Frequency balance
    bands = a.get("frequency_bands", {})
    sub  = bands.get("sub_bass_20_80hz", 0)
    bass = bands.get("bass_80_250hz", 0)
    lmid = bands.get("low_mid_250_800hz", 0)
    mid  = bands.get("mid_800_2500hz", 0)
    hmid = bands.get("high_mid_2500_6khz", 0)
    pres = bands.get("presence_6_12khz", 0)

    if sub > bass + 5:
        notes.append(f"Sub bass ({sub}dB) is significantly louder than bass ({bass}dB). The low end may feel heavy on small speakers. Consider a high-pass filter on non-bass elements around 40-60Hz.")
    if bass > lmid + 8:
        notes.append(f"Bass range is very dominant. Low mids ({lmid}dB) are much quieter. Mix may feel bottom-heavy.")
    if lmid > mid + 3:
        notes.append(f"Low mids ({lmid}dB) are elevated relative to the mids ({mid}dB). This can make a mix sound boxy or muddy. Try cutting 300-500Hz on any instruments that don't need that body.")
    if hmid > pres + 6:
        notes.append(f"High mids ({hmid}dB) are dominating the presence range ({pres}dB). Mix may feel harsh or fatiguing. Check for resonances around 3-5kHz.")

    # Balance issues from analyze_audio
    for issue in a.get("balance_issues", []):
        notes.append(f"Frequency shape: {issue}")

    # Brightness
    brightness = a.get("brightness", "")
    centroid = a.get("spectral_centroid_hz", 0)
    notes.append(f"Overall tone: {brightness} (spectral centroid {centroid}Hz). {'Vocals will need presence boost to cut through.' if centroid > 6000 else 'Good space for vocals to sit in the high mids.' if centroid < 4000 else 'Balanced brightness — vocals should sit naturally.'}")

    # Stereo field
    stereo = a.get("stereo", {})
    if stereo:
        balance = stereo.get("balance_db", 0)
        width_note = stereo.get("width_note", "")
        mono_compat = stereo.get("mono_compatibility", "")
        balance_note = stereo.get("balance_note", "")

        if abs(balance) > 1.0:
            notes.append(f"Stereo balance is {balance_note} by {abs(round(balance, 1))}dB. Check if this is intentional.")
        notes.append(f"Stereo width: {width_note}. Mono compatibility: {mono_compat}.")
        if "poor" in mono_compat:
            notes.append("Warning: Some elements may disappear when played in mono (phone speakers, club PA mono check). Check which panned elements are causing the low correlation.")

    # Key
    key = a.get("estimated_key", "Unknown")
    candidates = a.get("key_candidates", [])
    if key != "Unknown":
        candidate_str = ", ".join([f"{c['key']}" for c in candidates[1:3]]) if len(candidates) > 1 else ""
        notes.append(f"Detected key: {key}. {'Close candidates: ' + candidate_str + ' — verify against your MIDI data.' if candidate_str else 'Use MIDI note data for confirmation.'}")

    captured_at = a.get("captured_at", "")
    report = f"MIX REPORT — captured at {captured_at}\n\n"
    report += "\n\n".join([f"• {n}" for n in notes])
    return report


@mcp.tool()
def get_stereo_analysis() -> str:
    """
    Get detailed stereo field analysis from the latest capture.
    Returns L/R balance, stereo width, mono compatibility, and correlation.
    Use this when checking panning decisions or before bouncing.
    """
    if not _latest_analysis:
        return "No analysis available yet. Call start_capture first."
    stereo = _latest_analysis.get("stereo", {})
    if not stereo:
        return "No stereo data available. Make sure audio is playing in stereo."
    return json.dumps(stereo, indent=2)


@mcp.tool()
def get_frequency_report() -> str:
    """
    Get a focused frequency balance report with specific EQ suggestions.
    More detailed than the full mix report — use this when troubleshooting
    a specific frequency problem.
    """
    if not _latest_analysis:
        return "No analysis available yet. Call start_capture first."

    bands = _latest_analysis.get("frequency_bands", {})
    issues = _latest_analysis.get("balance_issues", [])
    brightness = _latest_analysis.get("brightness", "")
    centroid = _latest_analysis.get("spectral_centroid_hz", 0)

    lines = ["FREQUENCY REPORT", ""]
    lines.append("Band energy levels:")
    for name, val in bands.items():
        bar = "█" * max(0, int((val + 60) / 5))
        lines.append(f"  {name:<25} {val:>7.1f}dB  {bar}")

    lines.append("")
    lines.append(f"Spectral centroid: {centroid}Hz ({brightness})")

    if issues:
        lines.append("")
        lines.append("Balance issues detected:")
        for issue in issues:
            lines.append(f"  • {issue}")
    else:
        lines.append("")
        lines.append("No major balance issues detected.")

    return "\n".join(lines)


@mcp.tool()
def capture_and_analyze_file(file_path: str) -> str:
    """
    Analyze an audio file directly (WAV, MP3, FLAC, etc.) instead of live capture.
    Useful for analyzing a bounced stem, exported mix, or reference track.
    Also returns the plain English mix report for the file.
    """
    try:
        audio, sr = librosa.load(file_path, sr=None, mono=False)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        audio = audio.T
        result = analyze_audio(audio, sr)
        result["source"] = file_path
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error analyzing file: {e}"


def _get_ableton_midi(track_index: int, clip_index: int) -> list:
    """Pull MIDI notes from Ableton Remote Script directly via socket."""
    import socket as sock_module
    try:
        s = sock_module.socket(sock_module.AF_INET, sock_module.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect(("localhost", 9877))
        import json as json_module
        cmd = json_module.dumps({
            "type": "get_arrangement_clip_notes",
            "params": {"track_index": track_index, "clip_index": clip_index}
        })
        s.sendall((cmd + "\n").encode())
        response = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            response += chunk
            try:
                data = json_module.loads(response.decode())
                if "result" in data or "error" in data:
                    break
            except Exception:
                continue
        s.close()
        data = json_module.loads(response.decode())
        if "result" in data:
            return data["result"].get("notes", [])
        return []
    except Exception as e:
        logger.warning(f"Could not fetch MIDI from Ableton for track {track_index} clip {clip_index}: {e}")
        return []


def _get_ableton_arrangement_clips(track_index: int) -> list:
    """Get list of arrangement clips for a track from Ableton Remote Script."""
    import socket as sock_module
    import json as json_module
    try:
        s = sock_module.socket(sock_module.AF_INET, sock_module.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect(("localhost", 9877))
        cmd = json_module.dumps({
            "type": "get_arrangement_clips",
            "params": {"track_index": track_index}
        })
        s.sendall((cmd + "\n").encode())
        response = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            response += chunk
            try:
                data = json_module.loads(response.decode())
                if "result" in data or "error" in data:
                    break
            except Exception:
                continue
        s.close()
        data = json_module.loads(response.decode())
        if "result" in data:
            return data["result"].get("clips", [])
        return []
    except Exception as e:
        logger.warning(f"Could not fetch arrangement clips for track {track_index}: {e}")
        return []


def _get_all_track_info() -> list:
    """Get name, type, and clip info for all tracks from Ableton."""
    import socket as sock_module
    import json as json_module
    tracks = []
    try:
        # Get session info to find track count
        s = sock_module.socket(sock_module.AF_INET, sock_module.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect(("localhost", 9877))
        cmd = json_module.dumps({"type": "get_session_info", "params": {}})
        s.sendall((cmd + "\n").encode())
        response = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            response += chunk
            try:
                data = json_module.loads(response.decode())
                if "result" in data or "error" in data:
                    break
            except Exception:
                continue
        s.close()
        session = json_module.loads(response.decode()).get("result", {})
        track_count = session.get("track_count", 0)

        for i in range(track_count):
            s = sock_module.socket(sock_module.AF_INET, sock_module.SOCK_STREAM)
            s.settimeout(5.0)
            s.connect(("localhost", 9877))
            cmd = json_module.dumps({"type": "get_track_info", "params": {"track_index": i}})
            s.sendall((cmd + "\n").encode())
            response = b""
            while True:
                chunk = s.recv(65536)
                if not chunk:
                    break
                response += chunk
                try:
                    data = json_module.loads(response.decode())
                    if "result" in data or "error" in data:
                        break
                except Exception:
                    continue
            s.close()
            info = json_module.loads(response.decode()).get("result", {})
            if info:
                tracks.append({
                    "index": i,
                    "name": info.get("name", ""),
                    "is_midi": info.get("is_midi_track", False),
                    "is_audio": info.get("is_audio_track", False),
                })
    except Exception as e:
        logger.warning(f"Could not get track info: {e}")
    return tracks


def _get_audio_clip_file_paths(track_index: int) -> list:
    """Get file paths of audio clips in a track's arrangement."""
    import socket as sock_module
    import json as json_module
    paths = []
    clips = _get_ableton_arrangement_clips(track_index)
    for clip in clips:
        if not clip.get("is_midi_clip", True):
            try:
                s = sock_module.socket(sock_module.AF_INET, sock_module.SOCK_STREAM)
                s.settimeout(5.0)
                s.connect(("localhost", 9877))
                cmd = json_module.dumps({
                    "type": "get_audio_clip_file_path",
                    "params": {"track_index": track_index, "clip_index": clip["index"]}
                })
                s.sendall((cmd + "\n").encode())
                response = b""
                while True:
                    chunk = s.recv(65536)
                    if not chunk:
                        break
                    response += chunk
                    try:
                        data = json_module.loads(response.decode())
                        if "result" in data or "error" in data:
                            break
                    except Exception:
                        continue
                s.close()
                result = json_module.loads(response.decode()).get("result", {})
                fp = result.get("file_path", "")
                if fp:
                    paths.append(fp)
            except Exception:
                continue
    return paths


def _pitch_detect_audio_file(file_path: str) -> dict:
    """
    Run pyin pitch detection on an audio file (WAV, MP3, etc).
    Runs twice and compares results. If the two runs disagree on root note,
    runs a third time as tiebreaker. No playback required — reads directly
    from disk.

    Each run analyzes a different segment of the file to get independent
    samples of the pitch content — beginning, middle, and end thirds.
    """
    try:
        audio, sr = librosa.load(file_path, sr=None, mono=True)
        total_samples = len(audio)
        third = total_samples // 3

        def analyze_segment(segment, label):
            try:
                f0, voiced_flag, _ = librosa.pyin(
                    segment,
                    fmin=librosa.note_to_hz("C1"),
                    fmax=librosa.note_to_hz("C7"),
                    sr=sr
                )
                voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
                if len(voiced_f0) == 0:
                    return None
                hop = librosa.frames_to_time(1, sr=sr)
                notes = [{"pitch": int(round(p)), "start_time": i * hop,
                          "duration": hop, "velocity": 80}
                         for i, p in enumerate(voiced_f0) if not np.isnan(p)]
                result = _detect_key_from_notes(notes)
                result["segment"] = label
                result["voiced_frames"] = len(voiced_f0)
                return result
            except Exception as e:
                logger.warning(f"Segment analysis failed ({label}): {e}")
                return None

        # Run 1 — first third (intro, establishes key strongly)
        run1 = analyze_segment(audio[:third], "first_third")

        # Run 2 — middle third (body of song, most harmonic content)
        run2 = analyze_segment(audio[third:2*third], "middle_third")

        if not run1 and not run2:
            return {"key": "Unknown", "tonic": "Unknown",
                    "source": file_path, "error": "No pitched content detected in audio"}

        if not run1:
            run2["source"] = file_path
            run2["runs"] = 1
            return run2
        if not run2:
            run1["source"] = file_path
            run1["runs"] = 1
            return run1

        # Compare run 1 and run 2 root notes
        root1 = run1.get("key", "").split()[0]
        root2 = run2.get("key", "").split()[0]

        if root1 == root2 and root1:
            # Agreement — use the one with higher confidence
            winner = max([run1, run2], key=lambda r: r.get("key_confidence", 0))
            winner["source"] = file_path
            winner["runs"] = 2
            winner["run_agreement"] = True
            winner["run_keys"] = [run1.get("key"), run2.get("key")]
            return winner

        # Disagreement — run third time on final third as tiebreaker
        run3 = analyze_segment(audio[2*third:], "final_third")

        all_runs = [r for r in [run1, run2, run3] if r is not None]
        roots = [r.get("key", "").split()[0] for r in all_runs]

        # Count votes per root
        from collections import Counter
        root_counts = Counter(roots)
        most_common_root, count = root_counts.most_common(1)[0]

        if count >= 2:
            # Majority wins — pick the run with that root and highest confidence
            matching = [r for r in all_runs if r.get("key", "").split()[0] == most_common_root]
            winner = max(matching, key=lambda r: r.get("key_confidence", 0))
            winner["source"] = file_path
            winner["runs"] = len(all_runs)
            winner["run_agreement"] = True
            winner["run_keys"] = [r.get("key") for r in all_runs]
            winner["tiebreaker_used"] = True
            return winner
        else:
            # All three disagree — return highest confidence
            winner = max(all_runs, key=lambda r: r.get("key_confidence", 0))
            winner["source"] = file_path
            winner["runs"] = len(all_runs)
            winner["run_agreement"] = False
            winner["run_keys"] = [r.get("key") for r in all_runs]
            winner["manual_verification_needed"] = True
            winner["note"] = "All three analysis segments disagreed. Returning highest confidence. Verify by ear."
            return winner

    except Exception as e:
        return {"key": "Unknown", "tonic": "Unknown", "source": file_path, "error": str(e)}


def _auto_find_bass_track(tracks: list) -> dict | None:
    """Find bass or 808 track by name."""
    keywords = ["bass", "808", "sub"]
    for t in tracks:
        name_lower = t["name"].lower()
        if any(k in name_lower for k in keywords) and t["is_midi"]:
            return t
    return None


def _auto_find_melody_track(tracks: list, exclude_indices: list) -> dict | None:
    """
    Find the melody track — either the MIDI track with the most notes
    or the one with the most consistent arrangement coverage.
    Excludes drum racks and specified indices.
    """
    drum_keywords = ["drum", "perc", "hat", "kick", "snare"]
    candidates = []
    for t in tracks:
        if not t["is_midi"]:
            continue
        if t["index"] in exclude_indices:
            continue
        name_lower = t["name"].lower()
        if any(k in name_lower for k in drum_keywords):
            continue
        # Count total notes
        clips = _get_ableton_arrangement_clips(t["index"])
        total_notes = 0
        total_coverage = 0.0
        for clip in clips:
            if clip.get("is_midi_clip", False):
                notes = _get_ableton_midi(t["index"], clip["index"])
                total_notes += len(notes)
                total_coverage += float(clip.get("length", 0))
        candidates.append({**t, "total_notes": total_notes, "coverage": total_coverage})

    if not candidates:
        return None
    # Score: notes * coverage — consistent presence with lots of notes wins
    candidates.sort(key=lambda x: x["total_notes"] * x["coverage"], reverse=True)
    return candidates[0] if candidates else None


def _auto_find_audio_source(tracks: list, exclude_indices: list) -> tuple:
    """
    Find an audio source for key detection.
    Prefers tracks with 'vocal' in name, then any audio track with clips.
    Returns (track_info, file_path) or (None, None).
    """
    # First pass — look for vocal track
    vocal_keywords = ["vocal", "vox", "voice", "lead", "bg"]
    for t in tracks:
        if t["index"] in exclude_indices:
            continue
        name_lower = t["name"].lower()
        if any(k in name_lower for k in vocal_keywords):
            paths = _get_audio_clip_file_paths(t["index"])
            if paths:
                return t, paths[0]

    # Second pass — any audio track with clips
    for t in tracks:
        if not t["is_audio"] or t["index"] in exclude_indices:
            continue
        paths = _get_audio_clip_file_paths(t["index"])
        if paths:
            return t, paths[0]

    # Third pass — random MIDI track not already used
    remaining = [t for t in tracks if t["is_midi"] and t["index"] not in exclude_indices]
    if remaining:
        import random
        return random.choice(remaining), None

    return None, None


def _vote_on_key(results: list) -> dict:
    """
    Given a list of key detection results from different sources,
    vote on the correct key. Two agreeing sources win.
    If all disagree, return all results with a flag.
    Agreement is based on root note matching (not mode).
    """
    if not results:
        return {"key": "Unknown", "vote": "no_sources"}

    if len(results) == 1:
        return {**results[0], "vote": "single_source"}

    # Extract root notes
    def root(key_str):
        return key_str.split()[0] if key_str and key_str != "Unknown" else ""

    keys = [r.get("key", "Unknown") for r in results]
    roots = [root(k) for k in keys]

    # Check for agreement between first two
    if len(results) >= 2 and roots[0] == roots[1] and roots[0]:
        # Bass and melody agree — use the result with higher confidence
        winner = max(results[:2], key=lambda r: r.get("key_confidence", 0))
        return {
            **winner,
            "vote": "bass_melody_agree",
            "sources_used": [r.get("source_label", "unknown") for r in results[:2]],
            "vote_explanation": f"Bass and melody both point to {keys[0]}. High confidence."
        }

    # They disagree — use tiebreaker if available
    if len(results) >= 3:
        tb_root = roots[2]
        if tb_root == roots[0]:
            winner = max([results[0], results[2]], key=lambda r: r.get("key_confidence", 0))
            return {
                **winner,
                "vote": "tiebreaker_sides_with_bass",
                "sources_used": [r.get("source_label", "unknown") for r in results],
                "vote_explanation": f"Bass and tiebreaker agree on {keys[0]}. Melody suggested {keys[1]}."
            }
        elif tb_root == roots[1]:
            winner = max([results[1], results[2]], key=lambda r: r.get("key_confidence", 0))
            return {
                **winner,
                "vote": "tiebreaker_sides_with_melody",
                "sources_used": [r.get("source_label", "unknown") for r in results],
                "vote_explanation": f"Melody and tiebreaker agree on {keys[1]}. Bass suggested {keys[0]}."
            }
        else:
            # All three disagree
            best = max(results, key=lambda r: r.get("key_confidence", 0))
            return {
                **best,
                "vote": "all_disagree_highest_confidence_wins",
                "all_results": [{"source": r.get("source_label"), "key": r.get("key")} for r in results],
                "vote_explanation": "All three sources disagree. Returning highest confidence result. Manual verification recommended.",
                "manual_verification_needed": True
            }

    # Only two sources and they disagree
    best = max(results, key=lambda r: r.get("key_confidence", 0))
    return {
        **best,
        "vote": "two_sources_disagree_highest_confidence_wins",
        "all_results": [{"source": r.get("source_label"), "key": r.get("key")} for r in results],
        "vote_explanation": f"Bass suggests {keys[0]}, melody suggests {keys[1]}. Returning highest confidence. Manual verification recommended.",
        "manual_verification_needed": True
    }


def _detect_key_from_notes(all_notes: list) -> dict:
    """
    Detect key from MIDI notes using a multi-stage approach:

    Stage 1 — Classify each note as primary, secondary, or accidental based on
    how it is used. Short fast notes (melisma, passing tones) are weighted low.
    Sustained notes on strong beats are weighted high. Notes that only appear in
    fast contexts are flagged as likely accidentals.

    Stage 2 — Find the tonic from primary notes only using duration, beat
    position, first/last note weighting, and repetition.

    Stage 3 — Match primary notes against scale templates including major,
    natural minor, Dorian, Phrygian, Lydian, Mixolydian, Locrian, and blues.
    Best fit wins. Accidentals are reported separately so they don't corrupt
    the key detection.

    Stage 4 — Return full context including accidentals, scale degrees, mode,
    tonic confidence, and a human-readable explanation of the decision.
    """
    if not all_notes:
        return {"key": "Unknown", "tonic": "Unknown", "confidence": 0.0}

    note_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

    # ── Stage 1: Classify notes ───────────────────────────────────────────────
    # For each pitch class collect all its note durations and contexts
    # A note is "primary" if it appears with significant sustained duration
    # A note is "accidental" if it only appears in very short bursts

    pitch_data = {i: {"durations": [], "beat_positions": [], "velocities": [], "total_time": 0.0}
                  for i in range(12)}

    for n in all_notes:
        pc = int(n["pitch"]) % 12
        dur = float(n.get("duration", 0.25))
        start = float(n.get("start_time", 0.0))
        vel = int(n.get("velocity", 64)) / 127.0
        beat_in_bar = start % 4.0

        pitch_data[pc]["durations"].append(dur)
        pitch_data[pc]["beat_positions"].append(beat_in_bar)
        pitch_data[pc]["velocities"].append(vel)
        pitch_data[pc]["total_time"] += dur

    # Classify each pitch class
    # Threshold: if median duration < 0.25 beats (16th note) it is likely passing
    # If total_time is very small relative to the song it is likely accidental
    max_total = max(d["total_time"] for d in pitch_data.values()) + 1e-9

    note_classification = {}
    for pc, data in pitch_data.items():
        if not data["durations"]:
            note_classification[pc] = "absent"
            continue
        median_dur = float(np.median(data["durations"]))
        total_frac = data["total_time"] / max_total

        # Check if it mostly appears in fast contexts
        fast_count = sum(1 for d in data["durations"] if d < 0.2)
        fast_ratio = fast_count / len(data["durations"])

        if total_frac < 0.02 and median_dur < 0.3:
            note_classification[pc] = "accidental"
        elif fast_ratio > 0.8 and total_frac < 0.05:
            note_classification[pc] = "passing"
        elif total_frac > 0.05 or median_dur >= 0.5:
            note_classification[pc] = "primary"
        else:
            note_classification[pc] = "secondary"

    # ── Stage 2: Find tonic from primary notes ────────────────────────────────
    tonic_scores = np.zeros(12)

    for n in all_notes:
        pc = int(n["pitch"]) % 12
        classification = note_classification[pc]

        # Skip accidentals and passing tones for tonic detection
        if classification in ("accidental", "passing", "absent"):
            continue

        dur = float(n.get("duration", 0.25))
        start = float(n.get("start_time", 0.0))
        vel = int(n.get("velocity", 64)) / 127.0
        beat_in_bar = start % 4.0

        # Duration weight
        dur_weight = min(dur, 4.0) / 4.0 + 0.1

        # Strong beat weight — beat 1 of bar is strongest tonic indicator
        if beat_in_bar < 0.25:
            beat_weight = 3.0
        elif beat_in_bar < 1.0:
            beat_weight = 1.5
        else:
            beat_weight = 1.0

        # First note of the piece
        first_weight = 4.0 if start < 0.5 else 1.0

        tonic_scores[pc] += dur_weight * beat_weight * first_weight * vel

    # Last note — strong tonic indicator
    if all_notes:
        last_note = max(all_notes, key=lambda n: float(n.get("start_time", 0)))
        last_pc = int(last_note["pitch"]) % 12
        if note_classification[last_pc] not in ("accidental", "passing"):
            tonic_scores[last_pc] += 3.0

    tonic_scores = tonic_scores / (np.sum(tonic_scores) + 1e-9)
    tonic_index = int(np.argmax(tonic_scores))
    tonic = note_names[tonic_index]

    # ── Stage 3: Match against scale templates ────────────────────────────────
    # Build primary pitch weight vector — exclude accidentals and passing tones
    primary_weights = np.zeros(12)
    for n in all_notes:
        pc = int(n["pitch"]) % 12
        if note_classification[pc] in ("primary", "secondary"):
            dur = float(n.get("duration", 0.25))
            vel = int(n.get("velocity", 64)) / 127.0
            primary_weights[pc] += (min(dur, 4.0) / 4.0 + 0.1) * vel

    primary_weights = primary_weights / (np.sum(primary_weights) + 1e-9)

    # Scale templates — intervals from root
    scale_templates = {
        "major":           [0, 2, 4, 5, 7, 9, 11],
        "natural minor":   [0, 2, 3, 5, 7, 8, 10],
        "dorian":          [0, 2, 3, 5, 7, 9, 10],
        "phrygian":        [0, 1, 3, 5, 7, 8, 10],
        "lydian":          [0, 2, 4, 6, 7, 9, 11],
        "mixolydian":      [0, 2, 4, 5, 7, 9, 10],
        "locrian":         [0, 1, 3, 5, 6, 8, 10],
        "harmonic minor":  [0, 2, 3, 5, 7, 8, 11],
        "blues":           [0, 3, 5, 6, 7, 10],
        "pentatonic minor":[0, 3, 5, 7, 10],
        "pentatonic major":[0, 2, 4, 7, 9],
    }

    # Score each scale rooted at the detected tonic
    # Also check all 12 roots to catch cases where tonic detection was imperfect
    best_score = -np.inf
    estimated_key = "Unknown"
    estimated_mode = "Unknown"
    key_scores = {}

    for root_idx in range(12):
        for mode_name, intervals in scale_templates.items():
            # Build a weight vector for this scale
            scale_weight = np.zeros(12)
            for interval in intervals:
                scale_weight[(root_idx + interval) % 12] = 1.0
            scale_weight = scale_weight / (np.sum(scale_weight) + 1e-9)

            score = float(np.dot(primary_weights, scale_weight))
            key_str = f"{note_names[root_idx]} {mode_name}"
            key_scores[key_str] = round(score, 4)

            # Give a bonus if root matches our tonic detection
            if root_idx == tonic_index:
                score *= 1.3

            if score > best_score:
                best_score = score
                estimated_key = key_str
                estimated_mode = mode_name

    # Top 5 candidates
    top_keys = sorted(key_scores.items(), key=lambda x: x[1], reverse=True)[:5]

    # ── Stage 4: Build output ─────────────────────────────────────────────────
    root_of_key = estimated_key.split()[0]
    root_idx = note_names.index(root_of_key)
    intervals_of_key = scale_templates.get(estimated_mode, [0, 2, 4, 5, 7, 9, 11])

    # Scale degree names
    degree_labels = {
        "major":            ["1","2","3","4","5","6","7"],
        "natural minor":    ["1","2","b3","4","5","b6","b7"],
        "dorian":           ["1","2","b3","4","5","6","b7"],
        "phrygian":         ["1","b2","b3","4","5","b6","b7"],
        "lydian":           ["1","2","3","#4","5","6","7"],
        "mixolydian":       ["1","2","3","4","5","6","b7"],
        "locrian":          ["1","b2","b3","4","b5","b6","b7"],
        "harmonic minor":   ["1","2","b3","4","5","b6","7"],
        "blues":            ["1","b3","4","b5","5","b7"],
        "pentatonic minor": ["1","b3","4","5","b7"],
        "pentatonic major": ["1","2","3","5","6"],
    }
    labels = degree_labels.get(estimated_mode, ["1","2","3","4","5","6","7"])

    scale_degrees_present = []
    for degree_idx, interval in enumerate(intervals_of_key):
        pc = (root_idx + interval) % 12
        if primary_weights[pc] > 0.01:
            scale_degrees_present.append({
                "degree": labels[degree_idx] if degree_idx < len(labels) else str(degree_idx + 1),
                "note": note_names[pc],
                "weight": round(float(primary_weights[pc]), 4),
                "classification": note_classification[pc]
            })

    # Collect accidentals — notes that are present but outside the detected scale
    scale_pcs = set((root_idx + interval) % 12 for interval in intervals_of_key)
    accidentals = []
    for pc in range(12):
        if pc not in scale_pcs and note_classification[pc] in ("primary", "secondary", "accidental", "passing"):
            if primary_weights[pc] > 0.005 or note_classification[pc] == "primary":
                accidentals.append({
                    "note": note_names[pc],
                    "classification": note_classification[pc],
                    "weight": round(float(primary_weights[pc]), 4),
                    "note": note_names[pc]
                })

    # Human readable explanation
    explanation_parts = [f"Tonic detected as {tonic} based on beat position and duration weighting."]
    if accidentals:
        acc_notes = [a["note"] for a in accidentals if a["classification"] in ("primary", "secondary")]
        passing_notes = [a["note"] for a in accidentals if a["classification"] == "passing"]
        if acc_notes:
            explanation_parts.append(f"Notes outside scale: {', '.join(acc_notes)} — likely chromatic color or modal mixture.")
        if passing_notes:
            explanation_parts.append(f"Passing/melisma tones excluded from key detection: {', '.join(passing_notes)}.")
    if estimated_mode not in ("major", "natural minor"):
        explanation_parts.append(f"Mode detected as {estimated_mode} — not standard major/minor. Verify by ear.")

    return {
        "key": estimated_key,
        "tonic": tonic,
        "mode": estimated_mode,
        "tonic_confidence": round(float(tonic_scores[tonic_index]), 4),
        "key_confidence": round(best_score / 1.3, 4),  # normalize back
        "key_candidates": [{"key": k, "score": s} for k, s in top_keys],
        "scale_degrees_present": scale_degrees_present,
        "accidentals": accidentals,
        "note_classifications": {note_names[i]: note_classification[i] for i in range(12)},
        "pitch_class_weights": {note_names[i]: round(float(primary_weights[i]), 4) for i in range(12)},
        "explanation": " ".join(explanation_parts)
    }


@mcp.tool()
def analyze_bounced_instrumental() -> str:
    """
    Detect the key from a session that contains only a bounced WAV or MP3
    instrumental (no MIDI). This is common when someone has bounced their
    beat into a session for vocal recording.

    Automatically finds audio tracks with WAV/MP3 files, reads them
    directly from disk without requiring playback, and runs the multi-run
    pitch detection system (two passes + tiebreaker if needed).

    No playback required — reads audio files directly from disk.
    """
    if _key_override:
        return json.dumps({
            "override_active": True,
            "key": _key_override,
            "note": "Manual key override active."
        }, indent=2)

    tracks = _get_all_track_info()
    if not tracks:
        return json.dumps({"error": "Could not connect to Ableton."}, indent=2)

    # Find all audio tracks with files
    audio_sources = []
    for t in tracks:
        if t["is_audio"]:
            paths = _get_audio_clip_file_paths(t["index"])
            for p in paths:
                if p.lower().endswith((".wav", ".mp3", ".flac", ".aiff", ".aif")):
                    audio_sources.append({"track": t, "path": p})

    if not audio_sources:
        return json.dumps({
            "error": "No audio files found in session. This tool is for bounced instrumental sessions.",
            "suggestion": "If you have MIDI tracks, use get_key_with_voting instead."
        }, indent=2)

    # Use the longest audio file — likely the full instrumental
    # Get duration of each
    best_source = None
    best_duration = 0
    for source in audio_sources:
        try:
            duration = librosa.get_duration(path=source["path"])
            if duration > best_duration:
                best_duration = duration
                best_source = source
        except Exception:
            continue

    if not best_source:
        best_source = audio_sources[0]

    logger.info(f"Analyzing bounced instrumental: {best_source['path']} ({round(best_duration, 1)}s)")

    result = _pitch_detect_audio_file(best_source["path"])

    output = {
        "key": result.get("key", "Unknown"),
        "tonic": result.get("tonic", "Unknown"),
        "mode": result.get("mode", "Unknown"),
        "source_track": best_source["track"]["name"],
        "source_file": best_source["path"],
        "file_duration_seconds": round(best_duration, 1),
        "runs_completed": result.get("runs", 1),
        "run_agreement": result.get("run_agreement", False),
        "run_keys": result.get("run_keys", []),
        "tiebreaker_used": result.get("tiebreaker_used", False),
        "manual_verification_needed": result.get("manual_verification_needed", False),
        "accidentals": result.get("accidentals", []),
        "scale_degrees_present": result.get("scale_degrees_present", []),
        "explanation": result.get("explanation", ""),
        "note": result.get("note", ""),
        "playback_required": False,
        "ml_priors": {
            "key": result.get("key", "Unknown"),
            "tonic": result.get("tonic", "Unknown"),
            "is_minor": "minor" in result.get("mode", ""),
            "mode": result.get("mode", "Unknown"),
            "scale_degrees": [d["note"] for d in result.get("scale_degrees_present", [])],
            "pitch_distribution": result.get("pitch_class_weights", {}),
        }
    }

    return json.dumps(output, indent=2)


@mcp.tool()
def get_key_with_voting(vocal_file_path: str = "") -> str:
    """
    Detect the key of the song using a three-source voting system.

    Automatically finds:
    1. Bass source — track with 'bass', '808', or 'sub' in the name
    2. Melody source — MIDI track with the most notes and consistent coverage
    3. Tiebreaker — vocal audio file if provided, otherwise scans audio tracks
       for a bounced WAV/MP3 (common when someone has bounced their instrumental),
       otherwise picks a random remaining MIDI track

    Voting logic:
    - If bass and melody agree on root → confirmed, high confidence
    - If they disagree → tiebreaker runs, majority wins
    - If all three disagree → returns highest confidence result with a
      manual verification flag

    This is the primary key detection tool. Use get_key_from_midi for
    manual track selection or get_song_context for full ML context.

    Parameters:
    - vocal_file_path: Optional path to isolated vocal audio (WAV/MP3).
      If not provided the system will find its own tiebreaker source.
    """
    if _key_override:
        return json.dumps({
            "override_active": True,
            "key": _key_override,
            "note": "Manual key override active. Call set_key_override('') to clear."
        }, indent=2)

    # Get all tracks
    tracks = _get_all_track_info()
    if not tracks:
        return json.dumps({"error": "Could not connect to Ableton or no tracks found."}, indent=2)

    used_indices = []
    sources = []

    # ── Source 1: Bass ────────────────────────────────────────────────────────
    bass_track = _auto_find_bass_track(tracks)
    if bass_track:
        used_indices.append(bass_track["index"])
        clips = _get_ableton_arrangement_clips(bass_track["index"])
        bass_notes = []
        for clip in clips:
            if clip.get("is_midi_clip", False):
                bass_notes.extend(_get_ableton_midi(bass_track["index"], clip["index"]))
        if bass_notes:
            result = _detect_key_from_notes(bass_notes)
            result["source_label"] = f"bass:{bass_track['name']}"
            result["note_count"] = len(bass_notes)
            sources.append(result)
            logger.info(f"Bass source ({bass_track['name']}): {result['key']}")
    else:
        logger.warning("No bass track found by name.")

    # ── Source 2: Melody ──────────────────────────────────────────────────────
    melody_track = _auto_find_melody_track(tracks, used_indices)
    if melody_track:
        used_indices.append(melody_track["index"])
        clips = _get_ableton_arrangement_clips(melody_track["index"])
        melody_notes = []
        for clip in clips:
            if clip.get("is_midi_clip", False):
                melody_notes.extend(_get_ableton_midi(melody_track["index"], clip["index"]))
        if melody_notes:
            result = _detect_key_from_notes(melody_notes)
            result["source_label"] = f"melody:{melody_track['name']}"
            result["note_count"] = len(melody_notes)
            sources.append(result)
            logger.info(f"Melody source ({melody_track['name']}): {result['key']}")
    else:
        logger.warning("No melody track found.")

    # ── Check if we need tiebreaker ───────────────────────────────────────────
    need_tiebreaker = True
    if len(sources) >= 2:
        root0 = sources[0].get("key", "").split()[0]
        root1 = sources[1].get("key", "").split()[0]
        need_tiebreaker = root0 != root1 or not root0

    # ── Source 3: Tiebreaker ──────────────────────────────────────────────────
    tiebreaker_used = None
    if need_tiebreaker or len(sources) < 2:
        if vocal_file_path:
            # Use provided vocal file
            result = _pitch_detect_audio_file(vocal_file_path)
            result["source_label"] = f"vocal:{vocal_file_path}"
            sources.append(result)
            tiebreaker_used = f"vocal file: {vocal_file_path}"
            logger.info(f"Vocal tiebreaker: {result.get('key', 'Unknown')}")
        else:
            # Auto-find audio source
            audio_track, file_path = _auto_find_audio_source(tracks, used_indices)
            if file_path:
                # Bounced audio file — run pitch detection
                result = _pitch_detect_audio_file(file_path)
                result["source_label"] = f"audio:{audio_track['name'] if audio_track else 'unknown'}"
                sources.append(result)
                tiebreaker_used = f"bounced audio: {file_path}"
                logger.info(f"Audio tiebreaker ({file_path}): {result.get('key', 'Unknown')}")
            elif audio_track:
                # Random MIDI track
                clips = _get_ableton_arrangement_clips(audio_track["index"])
                tb_notes = []
                for clip in clips:
                    if clip.get("is_midi_clip", False):
                        tb_notes.extend(_get_ableton_midi(audio_track["index"], clip["index"]))
                if tb_notes:
                    result = _detect_key_from_notes(tb_notes)
                    result["source_label"] = f"random_midi:{audio_track['name']}"
                    result["note_count"] = len(tb_notes)
                    sources.append(result)
                    tiebreaker_used = f"random MIDI track: {audio_track['name']}"
                    logger.info(f"Random MIDI tiebreaker ({audio_track['name']}): {result.get('key', 'Unknown')}")

    if not sources:
        return json.dumps({"error": "Could not find any usable sources for key detection."}, indent=2)

    # ── Vote ──────────────────────────────────────────────────────────────────
    vote_result = _vote_on_key(sources)

    output = {
        "key": vote_result.get("key", "Unknown"),
        "tonic": vote_result.get("tonic", "Unknown"),
        "mode": vote_result.get("mode", "Unknown"),
        "vote": vote_result.get("vote"),
        "vote_explanation": vote_result.get("vote_explanation", ""),
        "sources": [
            {
                "label": s.get("source_label"),
                "key": s.get("key"),
                "tonic": s.get("tonic"),
                "confidence": s.get("key_confidence", 0),
                "note_count": s.get("note_count", s.get("voiced_frames", 0))
            }
            for s in sources
        ],
        "tiebreaker_used": tiebreaker_used,
        "manual_verification_needed": vote_result.get("manual_verification_needed", False),
        "accidentals": vote_result.get("accidentals", []),
        "scale_degrees_present": vote_result.get("scale_degrees_present", []),
        "explanation": vote_result.get("explanation", ""),
        "ml_priors": {
            "key": vote_result.get("key", "Unknown"),
            "tonic": vote_result.get("tonic", "Unknown"),
            "is_minor": "minor" in vote_result.get("mode", ""),
            "mode": vote_result.get("mode", "Unknown"),
            "scale_degrees": [d["note"] for d in vote_result.get("scale_degrees_present", [])],
            "pitch_distribution": vote_result.get("pitch_class_weights", {}),
        }
    }

    return json.dumps(output, indent=2)


@mcp.tool()
def get_key_from_midi(
    melodic_track_indices: list,
    vocal_file_path: str = ""
) -> str:
    """
    Detect the key of the song from MIDI data and optional vocal audio.
    Much more reliable than audio-based key detection.

    Pulls MIDI from all arrangement clips on the specified melodic tracks,
    weights notes by duration and beat position to find the tonic,
    then runs Krumhansl-Schmuckler key matching on the weighted pitch distribution.

    If a vocal audio file path is provided, also runs pitch detection on the
    vocal to confirm the key from a second independent source.

    Parameters:
    - melodic_track_indices: List of track indices to analyze (skip drum racks).
      Example: [0, 1, 2] for bells, bass, lyre.
    - vocal_file_path: Optional path to an isolated vocal audio file (WAV/MP3).
      Leave empty to skip vocal pitch analysis.

    Returns key, tonic, scale degrees present, confidence scores, and
    a full context object suitable for initializing ML model priors.
    """
    # Respect manual key override
    if _key_override:
        return json.dumps({
            "override_active": True,
            "key": _key_override,
            "note": "Manual key override is active. Call set_key_override('') to clear."
        }, indent=2)

    all_notes = []
    track_note_counts = {}

    # Pull MIDI from all specified tracks
    for track_idx in melodic_track_indices:
        clips = _get_ableton_arrangement_clips(track_idx)
        track_notes = []
        for clip in clips:
            if clip.get("is_midi_clip", False):
                notes = _get_ableton_midi(track_idx, clip["index"])
                track_notes.extend(notes)
        all_notes.extend(track_notes)
        track_note_counts[f"track_{track_idx}"] = len(track_notes)

    if not all_notes:
        return json.dumps({
            "error": "No MIDI notes found on specified tracks. Check track indices and make sure arrangement clips exist."
        }, indent=2)

    # Key detection from MIDI
    midi_key_result = _detect_key_from_notes(all_notes)

    result = {
        "source": "MIDI",
        "tracks_analyzed": melodic_track_indices,
        "total_notes_analyzed": len(all_notes),
        "notes_per_track": track_note_counts,
        "key": midi_key_result["key"],
        "tonic": midi_key_result["tonic"],
        "tonic_confidence": midi_key_result["tonic_confidence"],
        "key_confidence": midi_key_result["key_confidence"],
        "key_candidates": midi_key_result["key_candidates"],
        "scale_degrees_present": midi_key_result["scale_degrees_present"],
        "pitch_class_weights": midi_key_result["pitch_class_weights"],
    }

    # Optional vocal pitch analysis
    if vocal_file_path:
        try:
            audio, sr = librosa.load(vocal_file_path, sr=None, mono=True)
            # pyin is the most accurate monophonic pitch detector
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr
            )
            # Only use voiced frames
            voiced_f0 = f0[voiced_flag]
            if len(voiced_f0) > 0:
                # Convert Hz to MIDI pitch classes
                midi_pitches = librosa.hz_to_midi(voiced_f0)
                vocal_notes = [{"pitch": int(round(p)), "start_time": i * 0.01,
                                 "duration": 0.01, "velocity": 80}
                                for i, p in enumerate(midi_pitches)
                                if not np.isnan(p)]
                vocal_key_result = _detect_key_from_notes(vocal_notes)
                result["vocal_analysis"] = {
                    "source": vocal_file_path,
                    "voiced_frames": len(voiced_f0),
                    "key": vocal_key_result["key"],
                    "tonic": vocal_key_result["tonic"],
                    "key_confidence": vocal_key_result["key_confidence"],
                    "key_candidates": vocal_key_result["key_candidates"],
                }

                # Agreement check between MIDI and vocal
                midi_root = midi_key_result["key"].split()[0]
                vocal_root = vocal_key_result["key"].split()[0]
                result["midi_vocal_agreement"] = midi_root == vocal_root
                if not result["midi_vocal_agreement"]:
                    result["agreement_note"] = f"MIDI suggests {midi_key_result['key']} but vocal suggests {vocal_key_result['key']} — check tuning or transposition."
            else:
                result["vocal_analysis"] = {"error": "No voiced frames detected in vocal file."}
        except Exception as e:
            result["vocal_analysis"] = {"error": f"Could not analyze vocal file: {e}"}

    # Pre-song context object for ML model initialization
    result["ml_context"] = {
        "key": midi_key_result["key"],
        "tonic": midi_key_result["tonic"],
        "tonic_midi": librosa.note_to_midi(midi_key_result["tonic"] + "3") if midi_key_result["tonic"] != "Unknown" else None,
        "is_minor": "minor" in midi_key_result["key"],
        "scale_degrees": [d["note"] for d in midi_key_result["scale_degrees_present"]],
        "pitch_distribution": midi_key_result["pitch_class_weights"],
        "total_notes": len(all_notes),
        "source_tracks": melodic_track_indices,
    }

    return json.dumps(result, indent=2)


@mcp.tool()
def get_song_context() -> str:
    """
    Get full pre-song musical context for ML model initialization.
    Analyzes tracks 0-2 (bells, bass, lyre) automatically — skips drum rack.
    Returns key, tonic, scale degrees, pitch distribution, and session info.

    Use this before recording or before starting the backing vocalist system
    to initialize priors without playing a single note.
    Combines MIDI analysis with current Ableton session state.
    """
    # Respect manual key override
    if _key_override:
        return json.dumps({
            "override_active": True,
            "key": _key_override,
            "note": "Manual key override is active. Call set_key_override('') to clear."
        }, indent=2)

    import socket as sock_module
    import json as json_module

    # Get session info from Ableton
    try:
        s = sock_module.socket(sock_module.AF_INET, sock_module.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect(("localhost", 9877))
        cmd = json_module.dumps({"type": "get_session_info", "params": {}})
        s.sendall((cmd + "\n").encode())
        response = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            response += chunk
            try:
                data = json_module.loads(response.decode())
                if "result" in data or "error" in data:
                    break
            except Exception:
                continue
        s.close()
        session = json_module.loads(response.decode()).get("result", {})
    except Exception as e:
        session = {"error": str(e)}

    # Auto-detect melodic tracks — skip track 3 (drum rack) and audio tracks
    # Analyze tracks 0, 1, 2 by default
    melodic_indices = [0, 1, 2]
    all_notes = []
    for track_idx in melodic_indices:
        clips = _get_ableton_arrangement_clips(track_idx)
        for clip in clips:
            if clip.get("is_midi_clip", False):
                notes = _get_ableton_midi(track_idx, clip["index"])
                all_notes.extend(notes)

    key_result = _detect_key_from_notes(all_notes) if all_notes else {"key": "Unknown", "tonic": "Unknown"}

    context = {
        "session": {
            "tempo": session.get("tempo", 0),
            "time_signature": f"{session.get('signature_numerator', 4)}/{session.get('signature_denominator', 4)}",
            "track_count": session.get("track_count", 0),
        },
        "key": key_result.get("key", "Unknown"),
        "tonic": key_result.get("tonic", "Unknown"),
        "tonic_confidence": key_result.get("tonic_confidence", 0),
        "key_confidence": key_result.get("key_confidence", 0),
        "key_candidates": key_result.get("key_candidates", []),
        "scale_degrees_present": key_result.get("scale_degrees_present", []),
        "pitch_distribution": key_result.get("pitch_class_weights", {}),
        "total_notes_analyzed": len(all_notes),
        "ml_priors": {
            "key": key_result.get("key", "Unknown"),
            "tonic": key_result.get("tonic", "Unknown"),
            "is_minor": "minor" in key_result.get("key", ""),
            "tempo": session.get("tempo", 0),
            "scale_degrees": [d["note"] for d in key_result.get("scale_degrees_present", [])],
            "pitch_distribution": key_result.get("pitch_class_weights", {}),
        }
    }

    return json.dumps(context, indent=2)


# ─────────────────────────────────────────
# Session context state — persists in memory for model use
# ─────────────────────────────────────────
_session_context = {}           # Full pre-run context object
_section_templates = {}         # Stores heard section audio/MIDI fingerprints (V2)
_sections_heard = []            # Which sections have completed (V2 progressive confidence)


def _infer_song_structure(tracks: list) -> dict:
    """
    Infer song structure from arrangement clip positions.
    Groups clip start positions into likely sections based on
    clustering — sections tend to start at the same positions
    across multiple tracks simultaneously.
    """
    import socket as sock_module
    import json as json_module

    # Collect all clip start times across MIDI tracks
    all_starts = []
    for t in tracks:
        if not t["is_midi"]:
            continue
        clips = _get_ableton_arrangement_clips(t["index"])
        for clip in clips:
            start = float(clip.get("start_time", 0))
            if start not in all_starts:
                all_starts.append(start)

    if not all_starts:
        return {"sections": [], "total_beats": 0}

    all_starts.sort()

    # Get arrangement length
    try:
        s = sock_module.socket(sock_module.AF_INET, sock_module.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect(("localhost", 9877))
        cmd = json_module.dumps({"type": "get_arrangement_length", "params": {}})
        s.sendall((cmd + "\n").encode())
        response = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            response += chunk
            try:
                data = json_module.loads(response.decode())
                if "result" in data or "error" in data:
                    break
            except Exception:
                continue
        s.close()
        arr_length = json_module.loads(response.decode()).get("result", {}).get("length", 0)
    except Exception:
        arr_length = max(all_starts) + 32 if all_starts else 0

    # Find section boundaries — positions where multiple tracks change simultaneously
    # Count how many tracks have a clip starting at each position
    from collections import Counter
    start_counts = Counter()
    for t in tracks:
        if not t["is_midi"]:
            continue
        clips = _get_ableton_arrangement_clips(t["index"])
        for clip in clips:
            start = float(clip.get("start_time", 0))
            start_counts[start] += 1

    # Boundaries where 2+ tracks change simultaneously are likely section starts
    section_starts = sorted([s for s, c in start_counts.items() if c >= 2])

    # Label sections generically — we don't know verse/chorus without more context
    # but we can label them by position
    sections = []
    for i, start in enumerate(section_starts):
        end = section_starts[i + 1] if i + 1 < len(section_starts) else arr_length
        sections.append({
            "index": i,
            "start_beat": start,
            "end_beat": end,
            "length_beats": end - start,
            "label": f"section_{i}",
            "heard": False,       # V2 progressive confidence tracking
            "template": None      # V2 section template — filled after section completes
        })

    return {
        "sections": sections,
        "total_beats": arr_length,
        "section_count": len(sections)
    }


@mcp.tool()
def build_session_context(vocal_file_path: str = "") -> str:
    """
    Build the complete pre-session context object for V1 and V2 model initialization.
    Run this ONCE before recording starts. Returns a single unified payload
    containing everything the models need — key, tonic, mode, scale degrees,
    song structure, tempo, harmonic rhythm, and confidence scores for each.

    For V2 real-time use: this initializes the hierarchical prior system.
    Genre priors are built in. Artist priors use the session's own MIDI.
    Song priors start empty and fill in as sections complete via update_section_heard.

    No playback required — reads all data directly from the session.

    Parameters:
    - vocal_file_path: Optional path to isolated vocal audio for key confirmation.
    """
    global _session_context, _section_templates, _sections_heard

    if _key_override:
        key_data = {
            "key": _key_override,
            "tonic": _key_override.split()[0],
            "mode": _key_override.split()[1] if len(_key_override.split()) > 1 else "unknown",
            "source": "manual_override",
            "confidence": 1.0,
            "manual_verification_needed": False
        }
    else:
        # Run key detection with voting
        import json as json_module
        try:
            raw = get_key_with_voting(vocal_file_path)
            key_data = json_module.loads(raw)
        except Exception as e:
            key_data = {"key": "Unknown", "tonic": "Unknown", "mode": "Unknown",
                        "error": str(e)}

    # Get session info
    tracks = _get_all_track_info()
    import socket as sock_module
    import json as json_module
    session_info = {}
    try:
        s = sock_module.socket(sock_module.AF_INET, sock_module.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect(("localhost", 9877))
        cmd = json_module.dumps({"type": "get_session_info", "params": {}})
        s.sendall((cmd + "\n").encode())
        response = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            response += chunk
            try:
                data = json_module.loads(response.decode())
                if "result" in data or "error" in data:
                    break
            except Exception:
                continue
        s.close()
        session_info = json_module.loads(response.decode()).get("result", {})
    except Exception as e:
        session_info = {"error": str(e)}

    # Infer song structure
    structure = _infer_song_structure(tracks)

    # Analyze harmonic rhythm from bass — how many beats per chord change on average
    bass_track = _auto_find_bass_track(tracks)
    harmonic_rhythm = None
    if bass_track:
        clips = _get_ableton_arrangement_clips(bass_track["index"])
        if clips:
            first_clip_notes = _get_ableton_midi(bass_track["index"], 0)
            if len(first_clip_notes) >= 2:
                # Average gap between consecutive bass notes = rough harmonic rhythm
                starts = sorted([float(n["start_time"]) for n in first_clip_notes])
                gaps = [starts[i+1] - starts[i] for i in range(len(starts)-1)]
                harmonic_rhythm = round(float(np.mean(gaps)), 2) if gaps else None

    # Get melodic range from melody track
    melody_range = None
    melody_track = _auto_find_melody_track(tracks, [bass_track["index"]] if bass_track else [])
    if melody_track:
        clips = _get_ableton_arrangement_clips(melody_track["index"])
        all_melody_notes = []
        for clip in clips[:3]:  # Sample first 3 clips
            if clip.get("is_midi_clip", False):
                all_melody_notes.extend(_get_ableton_midi(melody_track["index"], clip["index"]))
        if all_melody_notes:
            pitches = [int(n["pitch"]) for n in all_melody_notes]
            melody_range = {
                "lowest_midi": min(pitches),
                "highest_midi": max(pitches),
                "lowest_note": librosa.midi_to_note(min(pitches)),
                "highest_note": librosa.midi_to_note(max(pitches)),
                "span_semitones": max(pitches) - min(pitches)
            }

    # Build the unified context object
    context = {
        # ── Identity ──────────────────────────────────────────────────────────
        "version": "1.0",
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "playback_required": False,

        # ── Session ───────────────────────────────────────────────────────────
        "session": {
            "tempo": session_info.get("tempo", 0),
            "time_signature": f"{session_info.get('signature_numerator', 4)}/{session_info.get('signature_denominator', 4)}",
            "beats_per_bar": session_info.get("signature_numerator", 4),
            "total_beats": structure.get("total_beats", 0),
            "total_seconds": round(structure.get("total_beats", 0) / max(session_info.get("tempo", 120) / 60, 0.01), 1),
            "track_count": session_info.get("track_count", 0),
        },

        # ── Key and Harmony ───────────────────────────────────────────────────
        "harmony": {
            "key": key_data.get("key", "Unknown"),
            "tonic": key_data.get("tonic", "Unknown"),
            "mode": key_data.get("mode", "Unknown"),
            "is_minor": "minor" in key_data.get("mode", ""),
            "scale_degrees": [d["note"] for d in key_data.get("scale_degrees_present", [])],
            "accidentals": key_data.get("accidentals", []),
            "pitch_distribution": key_data.get("ml_priors", {}).get("pitch_distribution", {}),
            "key_confidence": key_data.get("key_confidence", 0),
            "tonic_confidence": key_data.get("tonic_confidence", 0),
            "manual_verification_needed": key_data.get("manual_verification_needed", False),
            "key_detection_vote": key_data.get("vote", "unknown"),
            "harmonic_rhythm_beats": harmonic_rhythm,
        },

        # ── Song Structure ────────────────────────────────────────────────────
        "structure": {
            "sections": structure.get("sections", []),
            "section_count": structure.get("section_count", 0),
            "sections_heard": [],        # Fills in V2 via update_section_heard
            "sections_remaining": list(range(structure.get("section_count", 0))),
        },

        # ── Vocal Range ───────────────────────────────────────────────────────
        "vocal_range": melody_range,

        # ── Prior System (V2) ─────────────────────────────────────────────────
        "priors": {
            "genre": {
                "note": "Genre priors not yet trained. Defaults to hip-hop/R&B structure patterns.",
                "expected_structure": "intro → verse → chorus → verse → chorus → bridge → chorus",
                "ad_lib_density_by_section": {
                    "intro": "none",
                    "verse_1": "ambient_only",
                    "verse_2": "low",
                    "chorus": "medium_high",
                    "bridge": "low",
                    "final_chorus": "maximum"
                }
            },
            "artist": {
                "note": "Artist priors built from this session's MIDI.",
                "tonic": key_data.get("tonic", "Unknown"),
                "mode": key_data.get("mode", "Unknown"),
                "melodic_range": melody_range,
                "harmonic_rhythm_beats": harmonic_rhythm,
            },
            "song": {
                "note": "Song priors empty — fill in via update_section_heard as recording progresses.",
                "sections_with_templates": [],
                "confidence_level": "genre_only"   # Upgrades as sections complete
            }
        },

        # ── Confidence Tracker (V2) ───────────────────────────────────────────
        "confidence": {
            "key_detection": key_data.get("key_confidence", 0),
            "structure_detection": min(1.0, structure.get("section_count", 0) / 8),
            "overall": "low",   # Upgrades as song plays
            "note": "Confidence increases as sections complete. Call update_section_heard after each section."
        }
    }

    # Store in memory for V2 streaming updates
    _session_context = context
    _sections_heard = []

    return json.dumps(context, indent=2)


@mcp.tool()
def update_section_heard(section_index: int, section_label: str = "") -> str:
    """
    Mark a section as completed during V2 real-time recording.
    Call this when a section finishes playing to update the song priors
    and increase the system's confidence for subsequent sections.

    This is how V2 collapses uncertainty progressively — each completed
    section teaches the system what to expect when that section repeats.

    Parameters:
    - section_index: The index of the section that just completed (0-based)
    - section_label: Optional human label like 'verse_1', 'chorus_1' etc.
    """
    global _session_context, _sections_heard

    if not _session_context:
        return json.dumps({"error": "No session context built yet. Run build_session_context first."})

    if section_index not in _sections_heard:
        _sections_heard.append(section_index)

    # Update structure state
    structure = _session_context.get("structure", {})
    sections = structure.get("sections", [])
    if section_index < len(sections):
        sections[section_index]["heard"] = True
        if section_label:
            sections[section_index]["label"] = section_label

    # Update confidence level
    heard_count = len(_sections_heard)
    total = structure.get("section_count", 1)
    fraction_heard = heard_count / max(total, 1)

    if fraction_heard < 0.25:
        confidence_level = "genre_only"
        overall = "low"
    elif fraction_heard < 0.5:
        confidence_level = "genre_plus_artist"
        overall = "medium"
    elif fraction_heard < 0.75:
        confidence_level = "genre_artist_partial_song"
        overall = "high"
    else:
        confidence_level = "full_song_context"
        overall = "very_high"

    _session_context["structure"]["sections_heard"] = _sections_heard
    _session_context["structure"]["sections_remaining"] = [
        i for i in range(total) if i not in _sections_heard
    ]
    _session_context["priors"]["song"]["sections_with_templates"] = _sections_heard
    _session_context["priors"]["song"]["confidence_level"] = confidence_level
    _session_context["confidence"]["overall"] = overall

    return json.dumps({
        "section_marked": section_index,
        "label": section_label or sections[section_index].get("label", f"section_{section_index}") if section_index < len(sections) else "",
        "sections_heard": _sections_heard,
        "sections_remaining": _session_context["structure"]["sections_remaining"],
        "confidence_level": confidence_level,
        "overall_confidence": overall,
        "note": f"{heard_count}/{total} sections heard. Prior confidence: {overall}."
    }, indent=2)


@mcp.tool()
def get_current_session_context() -> str:
    """
    Get the current session context including any progressive updates
    from update_section_heard. Use this to retrieve the latest state
    of the prior system during a V2 recording session.
    """
    if not _session_context:
        return json.dumps({
            "error": "No session context built yet. Run build_session_context first."
        })
    return json.dumps(_session_context, indent=2)


@mcp.tool()
def set_key_override(key: str = "") -> str:
    """
    Manually set the key of the song, bypassing automatic detection.
    Use this when you know the key and want the system to use it directly.
    This overrides get_key_from_midi and get_song_context.

    Format: "E minor", "C major", "A dorian", "G blues" etc.
    Call with empty string to clear the override and return to auto detection.

    Parameters:
    - key: The key string. Leave empty to clear override.
    """
    global _key_override
    if not key:
        _key_override = None
        return "Key override cleared. Auto detection will be used."
    _key_override = key
    return f"Key override set to: {key}. All key detection tools will return this value."


@mcp.tool()
def get_key_override() -> str:
    """Get the current manually set key override, if any."""
    if _key_override:
        return json.dumps({"override_active": True, "key": _key_override})
    return json.dumps({"override_active": False, "key": None,
                       "note": "Auto detection is active."})


# ─────────────────────────────────────────
# NEW TOOLS — Session context expansion
# Per-section keys, STT, pitch tracking,
# polyphonic instrument analysis, Qdrant write
# ─────────────────────────────────────────

import threading as _threading
import queue as _queue


# ─────────────────────────────────────────
# Section-level key detection
# ─────────────────────────────────────────

@mcp.tool()
def build_per_section_keys() -> str:
    """
    Run key detection per section rather than globally.
    Uses MIDI data from each section's clip range to detect the local key.
    Supports songs that modulate between sections.

    Updates the session context harmony data with per-section keys.
    Call this after build_session_context to add section-level key resolution.
    """
    if not _session_context:
        return json.dumps({"error": "No session context. Run build_session_context first."})

    import json as json_module
    import socket as sock_module

    sections = _session_context.get("structure", {}).get("sections", [])
    if not sections:
        return json.dumps({"error": "No sections found. Check song structure."})

    tracks = _get_all_track_info()
    if not tracks:
        return json.dumps({"error": "Could not connect to Ableton."})

    results = []

    for section in sections:
        section_idx = section["index"]
        start_beat = section["start_beat"]
        end_beat = section["end_beat"]

        # Collect MIDI notes that fall within this section's time range
        section_notes = []
        for t in tracks:
            if not t["is_midi"]:
                continue
            # Skip drum-like tracks
            name_lower = t["name"].lower()
            if any(k in name_lower for k in ["drum", "perc", "hat", "kick", "snare"]):
                continue
            clips = _get_ableton_arrangement_clips(t["index"])
            for clip in clips:
                clip_start = float(clip.get("start_time", 0))
                clip_end = clip_start + float(clip.get("length", 0))
                # Only process clips that overlap this section
                if clip_end <= start_beat or clip_start >= end_beat:
                    continue
                if clip.get("is_midi_clip", False):
                    notes = _get_ableton_midi(t["index"], clip["index"])
                    # Filter notes to those within the section window
                    for n in notes:
                        note_start = float(n.get("start_time", 0)) + clip_start
                        if start_beat <= note_start < end_beat:
                            section_notes.append(n)

        if not section_notes:
            result = {
                "section_index": section_idx,
                "label": section.get("label", f"section_{section_idx}"),
                "key": _session_context.get("harmony", {}).get("key", "Unknown"),
                "source": "fallback_to_global",
                "note_count": 0
            }
        else:
            key_result = _detect_key_from_notes(section_notes)
            result = {
                "section_index": section_idx,
                "label": section.get("label", f"section_{section_idx}"),
                "key": key_result.get("key", "Unknown"),
                "tonic": key_result.get("tonic", "Unknown"),
                "mode": key_result.get("mode", "Unknown"),
                "key_confidence": key_result.get("key_confidence", 0),
                "accidentals": key_result.get("accidentals", []),
                "scale_degrees_present": key_result.get("scale_degrees_present", []),
                "note_count": len(section_notes),
                "is_override": False,
                "source": "detected"
            }

        # Write back to session context
        sections[section_idx]["key_data"] = result
        results.append(result)

    _session_context["structure"]["sections"] = sections
    _session_context["per_section_keys_built"] = True

    return json.dumps({
        "sections_analyzed": len(results),
        "results": results
    }, indent=2)


@mcp.tool()
def set_section_key_override(section_index: int, key: str, label: str = "") -> str:
    """
    Override the key for a specific section.
    Use this for sections that modulate to a different key, or when
    detection was incorrect due to heavy accidentals that look like a key change.

    The is_override flag is set to True so all models trust this over detection.

    Parameters:
    - section_index: The section to override (0-based)
    - key: The key string e.g. "E minor", "C major", "A dorian"
    - label: Optional human label for this section e.g. "bridge", "chorus_2"
    """
    if not _session_context:
        return json.dumps({"error": "No session context. Run build_session_context first."})

    sections = _session_context.get("structure", {}).get("sections", [])
    if section_index >= len(sections):
        return json.dumps({"error": f"Section index {section_index} out of range. "
                                     f"Only {len(sections)} sections found."})

    note_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    parts = key.split()
    tonic = parts[0] if parts else "Unknown"
    mode = " ".join(parts[1:]) if len(parts) > 1 else "unknown"

    key_data = {
        "section_index": section_index,
        "label": label or sections[section_index].get("label", f"section_{section_index}"),
        "key": key,
        "tonic": tonic,
        "mode": mode,
        "is_override": True,
        "source": "manual_override"
    }

    sections[section_index]["key_data"] = key_data
    if label:
        sections[section_index]["label"] = label

    _session_context["structure"]["sections"] = sections

    return json.dumps({
        "section_index": section_index,
        "key_set": key,
        "label": key_data["label"],
        "is_override": True,
        "note": f"Section {section_index} key overridden to {key}. "
                f"All models will use this value for this section."
    }, indent=2)


@mcp.tool()
def detect_section_repetitions() -> str:
    """
    Detect which sections are repetitions of each other by comparing
    MIDI content fingerprints. Helps identify verse/chorus patterns
    and label repeated sections automatically.

    A section is considered a repetition if its pitch class distribution
    closely matches another section's distribution (cosine similarity > 0.92).

    Updates section labels in the session context with repetition info.
    """
    if not _session_context:
        return json.dumps({"error": "No session context. Run build_session_context first."})

    sections = _session_context.get("structure", {}).get("sections", [])
    if len(sections) < 2:
        return json.dumps({"note": "Need at least 2 sections to detect repetitions."})

    tracks = _get_all_track_info()
    section_fingerprints = []

    for section in sections:
        start_beat = section["start_beat"]
        end_beat = section["end_beat"]
        pitch_weights = np.zeros(12)

        for t in tracks:
            if not t["is_midi"]:
                continue
            name_lower = t["name"].lower()
            if any(k in name_lower for k in ["drum", "perc", "hat", "kick", "snare"]):
                continue
            clips = _get_ableton_arrangement_clips(t["index"])
            for clip in clips:
                clip_start = float(clip.get("start_time", 0))
                clip_end = clip_start + float(clip.get("length", 0))
                if clip_end <= start_beat or clip_start >= end_beat:
                    continue
                if clip.get("is_midi_clip", False):
                    notes = _get_ableton_midi(t["index"], clip["index"])
                    for n in notes:
                        pc = int(n["pitch"]) % 12
                        dur = float(n.get("duration", 0.25))
                        pitch_weights[pc] += dur

        total = np.sum(pitch_weights)
        if total > 0:
            pitch_weights = pitch_weights / total
        section_fingerprints.append(pitch_weights)

    # Compare all pairs
    repetition_groups = {}
    similarity_matrix = []

    for i in range(len(sections)):
        row = []
        for j in range(len(sections)):
            if i == j:
                row.append(1.0)
                continue
            a = section_fingerprints[i]
            b = section_fingerprints[j]
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                row.append(0.0)
            else:
                similarity = float(np.dot(a, b) / (norm_a * norm_b))
                row.append(round(similarity, 3))
        similarity_matrix.append(row)

    # Group similar sections
    threshold = 0.92
    visited = set()
    groups = []

    for i in range(len(sections)):
        if i in visited:
            continue
        group = [i]
        visited.add(i)
        for j in range(i + 1, len(sections)):
            if j not in visited and similarity_matrix[i][j] >= threshold:
                group.append(j)
                visited.add(j)
        groups.append(group)

    # Auto-label groups if they look like verse/chorus patterns
    # Largest repeated group = likely chorus, second = likely verse
    repeated_groups = [g for g in groups if len(g) > 1]
    repeated_groups.sort(key=lambda g: -len(g))

    section_labels = {}
    label_names = ["chorus", "verse", "bridge", "outro", "intro"]

    for group_idx, group in enumerate(repeated_groups):
        base_label = label_names[group_idx] if group_idx < len(label_names) else f"section_type_{group_idx}"
        for occurrence_idx, section_idx in enumerate(sorted(group)):
            label = f"{base_label}_{occurrence_idx + 1}"
            section_labels[section_idx] = label

    # Sections not in any repeated group
    all_grouped = {s for g in repeated_groups for s in g}
    unique_sections = [i for i in range(len(sections)) if i not in all_grouped]
    for i, section_idx in enumerate(unique_sections):
        section_labels[section_idx] = f"unique_section_{i}"

    # Update session context
    for section_idx, label in section_labels.items():
        if section_idx < len(sections):
            if not sections[section_idx].get("key_data", {}).get("is_override"):
                sections[section_idx]["label"] = label

    _session_context["structure"]["sections"] = sections
    _session_context["structure"]["repetition_groups"] = [
        {"group_index": i, "sections": g, "label_base": repeated_groups[i][0]}
        for i, g in enumerate(repeated_groups)
    ]

    return json.dumps({
        "sections_analyzed": len(sections),
        "repetition_groups": [
            {
                "sections": g,
                "labels": [section_labels.get(s, f"section_{s}") for s in g],
                "similarity_to_first": [similarity_matrix[g[0]][s] for s in g]
            }
            for g in repeated_groups
        ],
        "unique_sections": unique_sections,
        "all_labels": section_labels
    }, indent=2)


# ─────────────────────────────────────────
# STT Listener — Live vocal transcription
# ─────────────────────────────────────────

_stt_thread = None
_stt_running = False

def _stt_loop(device_index: int, section_getter):
    """
    Background thread — records short audio windows and transcribes
    with Whisper, writing results to the session context performed layer.
    """
    global _stt_running
    try:
        import whisper as whisper_module
        import soundfile as sf_module
        import tempfile, os

        model = whisper_module.load_model("base")
        logger.info("Whisper model loaded. STT listener running.")
        window = 5.0  # seconds per transcription window

        while _stt_running:
            try:
                audio = sd.rec(
                    int(window * SAMPLE_RATE),
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                    device=device_index
                )
                sd.wait()

                # Save to temp file for Whisper
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                sf_module.write(tmp_path, audio, SAMPLE_RATE)

                result = model.transcribe(tmp_path, language="en", fp16=False)
                os.unlink(tmp_path)

                text = result.get("text", "").strip()
                confidence = result.get("segments", [{}])[0].get("no_speech_prob", 1.0)

                if text and confidence < 0.5:  # low no-speech probability = actual speech
                    current_section = section_getter()
                    timestamp = time.time()

                    if _session_context:
                        sections = _session_context.get("structure", {}).get("sections", [])
                        if current_section < len(sections):
                            performed = sections[current_section].get("performed", {})
                            lyrics = performed.get("lyrics", [])
                            lyrics.append({
                                "text": text,
                                "timestamp": timestamp,
                                "confidence": round(1.0 - confidence, 3)
                            })
                            performed["lyrics"] = lyrics
                            sections[current_section]["performed"] = performed
                            _session_context["structure"]["sections"] = sections
                            logger.info(f"STT [{current_section}]: {text}")

            except Exception as e:
                logger.error(f"STT window error: {e}")
                time.sleep(1)

    except ImportError:
        logger.error("Whisper not installed. Run: pip install openai-whisper")
        _stt_running = False


@mcp.tool()
def start_stt_listener(device_index: int, section_index: int = 0) -> str:
    """
    Start listening to a microphone and transcribing vocals in real time.
    Transcribed lyrics are written to the session context performed layer
    for the current section so the language model knows what was sung.

    Requires: pip install openai-whisper soundfile

    Parameters:
    - device_index: Microphone device index (use list_audio_devices to find it)
    - section_index: Which section is currently being recorded (0-based)
    """
    global _stt_thread, _stt_running, _current_section_index

    if _stt_running:
        return "STT listener already running. Call stop_stt_listener first."

    if not _session_context:
        return "No session context. Run build_session_context first."

    _current_section_index = section_index
    _stt_running = True
    _stt_thread = _threading.Thread(
        target=_stt_loop,
        args=(device_index, lambda: _current_section_index),
        daemon=True
    )
    _stt_thread.start()
    return (f"STT listener started on device {device_index}. "
            f"Recording section {section_index}. "
            f"Lyrics will update in session context performed layer.")


@mcp.tool()
def stop_stt_listener() -> str:
    """Stop the STT vocal transcription listener."""
    global _stt_running
    _stt_running = False
    return "STT listener stopped."


@mcp.tool()
def set_recording_section(section_index: int) -> str:
    """
    Update which section is currently being recorded.
    Call this when you move to a new section during a live recording session
    so the STT and pitch listeners write to the correct section.

    Parameters:
    - section_index: The section now being recorded (0-based)
    """
    global _current_section_index
    _current_section_index = section_index

    label = ""
    if _session_context:
        sections = _session_context.get("structure", {}).get("sections", [])
        if section_index < len(sections):
            label = sections[section_index].get("label", f"section_{section_index}")

    return json.dumps({
        "recording_section": section_index,
        "label": label,
        "note": "STT and pitch listeners now writing to this section."
    })


# ─────────────────────────────────────────
# Pitch Listener — Live vocal pitch tracking
# ─────────────────────────────────────────

def _pitch_listener_loop(device_index: int, section_getter):
    """
    Background thread — captures short audio windows and runs pyin
    pitch detection, writing detected notes to the session context
    performed layer for the current section.
    """
    global _pitch_listener_running
    window = 2.0  # seconds per pitch window
    hop_length = 512

    while _pitch_listener_running:
        try:
            audio = sd.rec(
                int(window * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=device_index
            )
            sd.wait()

            mono = audio.flatten().astype(np.float32)
            rms = float(np.sqrt(np.mean(mono ** 2)))
            if rms < 0.005:
                continue  # Skip silent windows

            f0, voiced_flag, _ = librosa.pyin(
                mono,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=SAMPLE_RATE,
                hop_length=hop_length
            )

            voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
            if len(voiced_f0) == 0:
                continue

            hop_time = librosa.frames_to_time(1, sr=SAMPLE_RATE, hop_length=hop_length)
            notes = []
            for i, freq in enumerate(voiced_f0):
                midi_pitch = float(librosa.hz_to_midi(freq))
                pitch_class = int(round(midi_pitch)) % 12
                note_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
                notes.append({
                    "midi_pitch": round(midi_pitch, 2),
                    "pitch_class": pitch_class,
                    "note_name": note_names[pitch_class],
                    "frequency_hz": round(float(freq), 2),
                    "timestamp": time.time() + (i * hop_time),
                    "duration": hop_time
                })

            current_section = section_getter()
            if _session_context and notes:
                sections = _session_context.get("structure", {}).get("sections", [])
                if current_section < len(sections):
                    performed = sections[current_section].get("performed", {})
                    existing = performed.get("notes_sung", [])
                    existing.extend(notes)
                    performed["notes_sung"] = existing
                    sections[current_section]["performed"] = performed
                    _session_context["structure"]["sections"] = sections
                    logger.info(f"Pitch [{current_section}]: {len(notes)} voiced frames captured")

        except Exception as e:
            logger.error(f"Pitch listener error: {e}")
            time.sleep(1)


@mcp.tool()
def start_pitch_listener(device_index: int, section_index: int = 0) -> str:
    """
    Start listening to a microphone and detecting vocal pitch in real time.
    Detected notes are written to the session context performed layer
    so the voice synthesis model knows what notes were actually sung.

    Works alongside start_stt_listener — run both simultaneously for
    full live vocal context capture.

    Parameters:
    - device_index: Microphone device index (use list_audio_devices to find it)
    - section_index: Which section is currently being recorded (0-based)
    """
    global _pitch_listener_thread, _pitch_listener_running, _current_section_index

    if _pitch_listener_running:
        return "Pitch listener already running. Call stop_pitch_listener first."

    if not _session_context:
        return "No session context. Run build_session_context first."

    _current_section_index = section_index
    _pitch_listener_running = True
    _pitch_listener_thread = _threading.Thread(
        target=_pitch_listener_loop,
        args=(device_index, lambda: _current_section_index),
        daemon=True
    )
    _pitch_listener_thread.start()
    return (f"Pitch listener started on device {device_index}. "
            f"Recording section {section_index}. "
            f"Vocal notes will update in session context performed layer.")


@mcp.tool()
def stop_pitch_listener() -> str:
    """Stop the vocal pitch tracking listener."""
    global _pitch_listener_running
    _pitch_listener_running = False
    return "Pitch listener stopped."


# ─────────────────────────────────────────
# Polyphonic instrument analysis — Basic-Pitch
# ─────────────────────────────────────────

@mcp.tool()
def analyze_instrument_audio(
    file_path: str,
    track_name: str = "unknown",
    section_index: int = -1
) -> str:
    """
    Analyze a polyphonic audio file (piano, guitar, violin, etc.)
    using Basic-Pitch to extract note events.

    For MIDI instruments use the Ableton MCP connection directly — it's exact.
    Use this only for audio recordings of polyphonic instruments where
    MIDI is not available.

    Writes the detected notes to the session context instrument layer
    for the specified section so all models know what was played.

    Requires: pip install basic-pitch

    Parameters:
    - file_path: Path to the audio file (WAV, MP3, FLAC)
    - track_name: Name of the instrument track (e.g. "piano", "guitar")
    - section_index: Which section to write notes to (-1 = don't write to context)
    """
    try:
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH

        logger.info(f"Running Basic-Pitch on {file_path}")
        model_output, midi_data, note_events = predict(
            file_path,
            ICASSP_2022_MODEL_PATH
        )

        notes = []
        for note in note_events:
            start_time, end_time, pitch, velocity, _ = note
            pitch_class = int(pitch) % 12
            note_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
            notes.append({
                "midi_pitch": int(pitch),
                "pitch_class": pitch_class,
                "note_name": note_names[pitch_class],
                "start_time": round(float(start_time), 4),
                "end_time": round(float(end_time), 4),
                "duration": round(float(end_time) - float(start_time), 4),
                "velocity": int(velocity * 127),
            })

        # Write to session context if section specified
        if section_index >= 0 and _session_context:
            sections = _session_context.get("structure", {}).get("sections", [])
            if section_index < len(sections):
                performed = sections[section_index].get("performed", {})
                instruments = performed.get("instrument_notes", {})
                instruments[track_name] = notes
                performed["instrument_notes"] = instruments
                sections[section_index]["performed"] = performed
                _session_context["structure"]["sections"] = sections

        return json.dumps({
            "file": file_path,
            "track_name": track_name,
            "note_count": len(notes),
            "written_to_section": section_index if section_index >= 0 else None,
            "notes": notes[:20],  # First 20 for preview
            "note": f"Full {len(notes)} notes written to context. Showing first 20."
                    if len(notes) > 20 else None
        }, indent=2)

    except ImportError:
        return json.dumps({
            "error": "Basic-Pitch not installed.",
            "install": "pip install basic-pitch",
            "note": "For MIDI instruments, use Ableton MCP connection instead — it's exact and requires no extra install."
        })
    except Exception as e:
        return json.dumps({"error": f"Analysis failed: {str(e)}"})


@mcp.tool()
def get_performed_context(section_index: int = -1) -> str:
    """
    Get the performed layer of the session context — what has actually
    been captured so far during recording (lyrics, notes sung, instrument notes).

    Parameters:
    - section_index: Which section to retrieve (-1 = all sections)
    """
    if not _session_context:
        return json.dumps({"error": "No session context. Run build_session_context first."})

    sections = _session_context.get("structure", {}).get("sections", [])

    if section_index >= 0:
        if section_index >= len(sections):
            return json.dumps({"error": f"Section {section_index} not found."})
        section = sections[section_index]
        return json.dumps({
            "section_index": section_index,
            "label": section.get("label", f"section_{section_index}"),
            "performed": section.get("performed", {}),
            "key_data": section.get("key_data", {})
        }, indent=2)
    else:
        result = []
        for s in sections:
            performed = s.get("performed", {})
            result.append({
                "section_index": s["index"],
                "label": s.get("label", f"section_{s['index']}"),
                "lyrics_count": len(performed.get("lyrics", [])),
                "notes_captured": len(performed.get("notes_sung", [])),
                "instruments_captured": list(performed.get("instrument_notes", {}).keys()),
                "has_data": bool(performed)
            })
        return json.dumps({"sections": result}, indent=2)


# ─────────────────────────────────────────
# Qdrant vector database integration
# ─────────────────────────────────────────

@mcp.tool()
def write_context_to_qdrant(
    qdrant_url: str = "http://localhost:6333",
    collection_name: str = "backing_vocalist_context"
) -> str:
    """
    Write the full session context to a Qdrant vector database.
    All models (language model, voice synthesis, orchestrator) read
    from this database rather than receiving context directly.

    Creates the collection if it doesn't exist.
    Overwrites previous session data for the same song.

    Each section is stored as a separate point with:
    - Vector: pitch class distribution (12-dimensional)
    - Payload: full section context (key, structure, performed data, overrides)

    Requires: pip install qdrant-client

    Parameters:
    - qdrant_url: URL of the Qdrant instance (default: localhost)
    - collection_name: Collection to write to
    """
    if not _session_context:
        return json.dumps({"error": "No session context. Run build_session_context first."})

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import (
            Distance, VectorParams, PointStruct, UpdateStatus
        )

        client = QdrantClient(url=qdrant_url)

        # Create collection if needed (12-dim pitch class vectors)
        existing = [c.name for c in client.get_collections().collections]
        if collection_name not in existing:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=12, distance=Distance.COSINE)
            )
            logger.info(f"Created Qdrant collection: {collection_name}")

        points = []
        sections = _session_context.get("structure", {}).get("sections", [])
        harmony = _session_context.get("harmony", {})
        session_info = _session_context.get("session", {})

        # Point 0 — global session context
        global_pitch_dist = harmony.get("pitch_distribution", {})
        note_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        global_vector = [float(global_pitch_dist.get(n, 0.0)) for n in note_names]
        if sum(global_vector) == 0:
            global_vector = [1/12.0] * 12

        points.append(PointStruct(
            id=0,
            vector=global_vector,
            payload={
                "type": "global",
                "key": harmony.get("key", "Unknown"),
                "tonic": harmony.get("tonic", "Unknown"),
                "mode": harmony.get("mode", "Unknown"),
                "is_minor": harmony.get("is_minor", False),
                "scale_degrees": harmony.get("scale_degrees", []),
                "tempo": session_info.get("tempo", 0),
                "time_signature": session_info.get("time_signature", "4/4"),
                "total_beats": session_info.get("total_beats", 0),
                "section_count": len(sections),
                "priors": _session_context.get("priors", {}),
                "built_at": _session_context.get("built_at", ""),
                "version": _session_context.get("version", "1.0")
            }
        ))

        # One point per section
        for section in sections:
            section_idx = section["index"]
            key_data = section.get("key_data", harmony)
            performed = section.get("performed", {})

            # Build pitch vector for this section
            pitch_dist = key_data.get("pitch_distribution", global_pitch_dist)
            if isinstance(pitch_dist, dict):
                section_vector = [float(pitch_dist.get(n, 0.0)) for n in note_names]
            else:
                section_vector = global_vector
            if sum(section_vector) == 0:
                section_vector = global_vector

            points.append(PointStruct(
                id=section_idx + 1,  # offset by 1 since 0 is global
                vector=section_vector,
                payload={
                    "type": "section",
                    "section_index": section_idx,
                    "label": section.get("label", f"section_{section_idx}"),
                    "start_beat": section.get("start_beat", 0),
                    "end_beat": section.get("end_beat", 0),
                    "length_beats": section.get("length_beats", 0),
                    "key": key_data.get("key", harmony.get("key", "Unknown")),
                    "tonic": key_data.get("tonic", harmony.get("tonic", "Unknown")),
                    "mode": key_data.get("mode", harmony.get("mode", "Unknown")),
                    "is_override": key_data.get("is_override", False),
                    "accidentals": key_data.get("accidentals", []),
                    "scale_degrees": key_data.get("scale_degrees_present",
                                                    harmony.get("scale_degrees", [])),
                    "heard": section.get("heard", False),
                    "performed_lyrics": performed.get("lyrics", []),
                    "performed_notes": performed.get("notes_sung", [])[-50:],  # last 50
                    "instrument_notes": {
                        k: v[-20:] for k, v in
                        performed.get("instrument_notes", {}).items()
                    },
                    "tempo": session_info.get("tempo", 0),
                    "time_signature": session_info.get("time_signature", "4/4"),
                }
            ))

        # Upsert all points
        result = client.upsert(collection_name=collection_name, points=points)
        success = result.status == UpdateStatus.COMPLETED

        return json.dumps({
            "success": success,
            "collection": collection_name,
            "qdrant_url": qdrant_url,
            "points_written": len(points),
            "global_point": 1,
            "section_points": len(sections),
            "note": "All models can now query this collection for session context."
        }, indent=2)

    except ImportError:
        return json.dumps({
            "error": "qdrant-client not installed.",
            "install": "pip install qdrant-client"
        })
    except Exception as e:
        return json.dumps({"error": f"Qdrant write failed: {str(e)}"})


@mcp.tool()
def update_section_in_qdrant(
    section_index: int,
    qdrant_url: str = "http://localhost:6333",
    collection_name: str = "backing_vocalist_context"
) -> str:
    """
    Update a single section's point in Qdrant after new data has been
    captured (lyrics, notes, or a key override). Faster than rewriting
    the full context — use this during live recording to keep the database
    current as each section completes.

    Parameters:
    - section_index: The section to update
    - qdrant_url: URL of the Qdrant instance
    - collection_name: Collection to update
    """
    if not _session_context:
        return json.dumps({"error": "No session context."})

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct, UpdateStatus

        client = QdrantClient(url=qdrant_url)
        sections = _session_context.get("structure", {}).get("sections", [])
        harmony = _session_context.get("harmony", {})
        session_info = _session_context.get("session", {})

        if section_index >= len(sections):
            return json.dumps({"error": f"Section {section_index} not found."})

        section = sections[section_index]
        key_data = section.get("key_data", harmony)
        performed = section.get("performed", {})

        note_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        pitch_dist = key_data.get("pitch_distribution",
                                   harmony.get("pitch_distribution", {}))
        if isinstance(pitch_dist, dict):
            vector = [float(pitch_dist.get(n, 0.0)) for n in note_names]
        else:
            vector = [1/12.0] * 12
        if sum(vector) == 0:
            vector = [1/12.0] * 12

        point = PointStruct(
            id=section_index + 1,
            vector=vector,
            payload={
                "type": "section",
                "section_index": section_index,
                "label": section.get("label", f"section_{section_index}"),
                "start_beat": section.get("start_beat", 0),
                "end_beat": section.get("end_beat", 0),
                "key": key_data.get("key", harmony.get("key", "Unknown")),
                "tonic": key_data.get("tonic", harmony.get("tonic", "Unknown")),
                "mode": key_data.get("mode", harmony.get("mode", "Unknown")),
                "is_override": key_data.get("is_override", False),
                "heard": section.get("heard", False),
                "performed_lyrics": performed.get("lyrics", []),
                "performed_notes": performed.get("notes_sung", [])[-50:],
                "instrument_notes": {
                    k: v[-20:] for k, v in
                    performed.get("instrument_notes", {}).items()
                },
                "tempo": session_info.get("tempo", 0),
            }
        )

        result = client.upsert(collection_name=collection_name, points=[point])
        success = result.status == UpdateStatus.COMPLETED

        return json.dumps({
            "success": success,
            "section_index": section_index,
            "label": section.get("label", f"section_{section_index}"),
            "updated": True
        }, indent=2)

    except ImportError:
        return json.dumps({"error": "qdrant-client not installed. pip install qdrant-client"})
    except Exception as e:
        return json.dumps({"error": f"Update failed: {str(e)}"})




# ─────────────────────────────────────────
# Auto-sync — event-driven Qdrant updates
# Listens to Ableton change events and
# re-runs analysis + updates DB automatically
# ─────────────────────────────────────────


def _auto_sync_loop(qdrant_url: str, collection_name: str):
    """
    Background thread that holds an open socket connection to the
    Ableton Remote Script and listens for change events.

    When a change arrives (tempo, time sig, clip notes) it:
    1. Re-runs the relevant analysis on the session context
    2. Calls update_section_in_qdrant or write_context_to_qdrant
       to keep the database current without polling.

    This means the orchestrator and all models always have fresh data
    without having to ask for it — it just appears in Qdrant.
    """
    global _auto_sync_running
    import socket as sock_module
    import json as json_module

    logger.info("Auto-sync thread starting...")

    while _auto_sync_running:
        try:
            s = sock_module.socket(sock_module.AF_INET, sock_module.SOCK_STREAM)
            s.settimeout(5.0)
            s.connect(("localhost", 9877))

            # Register as a change subscriber
            cmd = json_module.dumps({"type": "subscribe_changes", "params": {}})
            s.sendall((cmd + "\n").encode())

            # Read the subscription confirmation
            response = b""
            while b"\n" not in response and len(response) < 4096:
                chunk = s.recv(1024)
                if not chunk:
                    break
                response += chunk

            s.settimeout(None)  # Block indefinitely waiting for events
            logger.info("Auto-sync subscribed to Ableton change events")

            buffer = ""
            while _auto_sync_running:
                try:
                    data = s.recv(4096)
                    if not data:
                        logger.warning("Auto-sync: Ableton disconnected")
                        break

                    try:
                        buffer += data.decode("utf-8")
                    except Exception:
                        buffer += data

                    # Parse complete JSON messages (newline delimited)
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            event = json_module.loads(line)
                            if event.get("type") == "change_event":
                                _handle_change_event(
                                    event.get("event", ""),
                                    event.get("data", {}),
                                    qdrant_url,
                                    collection_name
                                )
                        except json_module.JSONDecodeError:
                            pass

                except sock_module.error as e:
                    logger.warning(f"Auto-sync socket error: {e}")
                    break

            s.close()

        except Exception as e:
            logger.warning(f"Auto-sync connection failed: {e}")

        if _auto_sync_running:
            logger.info("Auto-sync reconnecting in 5s...")
            time.sleep(5)

    logger.info("Auto-sync thread stopped")


def _handle_change_event(event_type: str, data: dict,
                          qdrant_url: str, collection_name: str):
    """
    Handle an incoming change event from Ableton.
    Re-runs relevant analysis and updates Qdrant.
    """
    logger.info(f"Change event: {event_type} | {data}")

    if not _session_context:
        return

    try:
        if event_type == "tempo_changed":
            new_tempo = data.get("tempo", 0)
            if new_tempo:
                _session_context["session"]["tempo"] = new_tempo
                _session_context["priors"]["artist"]["harmonic_rhythm_beats"] = None
                logger.info(f"Auto-sync: tempo updated to {new_tempo}")
                _write_global_to_qdrant(qdrant_url, collection_name)

        elif event_type == "time_sig_changed":
            num = data.get("numerator", 4)
            den = data.get("denominator", 4)
            _session_context["session"]["time_signature"] = f"{num}/{den}"
            _session_context["session"]["beats_per_bar"] = num
            logger.info(f"Auto-sync: time sig updated to {num}/{den}")
            _write_global_to_qdrant(qdrant_url, collection_name)

        elif event_type in ("clip_notes_changed", "clip_added", "clip_deleted"):
            # Re-run section structure and key detection then rewrite all sections
            logger.info(f"Auto-sync: clip change detected, rebuilding context...")
            tracks = _get_all_track_info()
            structure = _infer_song_structure(tracks)
            _session_context["structure"]["sections"] = structure.get("sections", [])
            _session_context["structure"]["section_count"] = structure.get("section_count", 0)
            # Write all sections back to Qdrant
            _write_all_sections_to_qdrant(qdrant_url, collection_name)

    except Exception as e:
        logger.error(f"Auto-sync change handler error: {e}")


def _write_global_to_qdrant(qdrant_url: str, collection_name: str):
    """Write just the global context point to Qdrant."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct

        client = QdrantClient(url=qdrant_url)
        harmony = _session_context.get("harmony", {})
        session_info = _session_context.get("session", {})
        note_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        pitch_dist = harmony.get("pitch_distribution", {})
        vector = [float(pitch_dist.get(n, 0.0)) for n in note_names]
        if sum(vector) == 0:
            vector = [1/12.0] * 12

        client.upsert(collection_name=collection_name, points=[
            PointStruct(
                id=0,
                vector=vector,
                payload={
                    "type": "global",
                    "key": harmony.get("key", "Unknown"),
                    "tempo": session_info.get("tempo", 0),
                    "time_signature": session_info.get("time_signature", "4/4"),
                    "section_count": len(
                        _session_context.get("structure", {}).get("sections", [])
                    ),
                }
            )
        ])
        logger.info("Auto-sync: global Qdrant point updated")
    except Exception as e:
        logger.error(f"Auto-sync Qdrant write error: {e}")


def _write_all_sections_to_qdrant(qdrant_url: str, collection_name: str):
    """Re-write all section points after a structural change."""
    try:
        import json as j
        result = write_context_to_qdrant(qdrant_url, collection_name)
        data = j.loads(result)
        if data.get("success"):
            logger.info(f"Auto-sync: all sections rewritten to Qdrant")
        else:
            logger.warning(f"Auto-sync: Qdrant rewrite issue: {data}")
    except Exception as e:
        logger.error(f"Auto-sync full rewrite error: {e}")


@mcp.tool()
def start_auto_sync(
    qdrant_url: str = "http://localhost:6333",
    collection_name: str = "backing_vocalist_context"
) -> str:
    """
    Start the auto-sync background thread.

    This opens a persistent connection to the Ableton Remote Script
    and listens for change events (tempo, time sig, clip changes).
    When a change occurs the relevant analysis re-runs automatically
    and Qdrant is updated without any manual tool calls.

    Your orchestrator model and voice synthesis model always see
    fresh data — they just query Qdrant and it's already current.

    Parameters:
    - qdrant_url: URL of your Qdrant instance
    - collection_name: Collection to keep in sync
    """
    global _auto_sync_thread, _auto_sync_running
    global _auto_sync_qdrant_url, _auto_sync_collection

    if _auto_sync_running:
        return "Auto-sync already running. Call stop_auto_sync to restart."

    _auto_sync_qdrant_url = qdrant_url
    _auto_sync_collection = collection_name
    _auto_sync_running = True

    _auto_sync_thread = _threading.Thread(
        target=_auto_sync_loop,
        args=(qdrant_url, collection_name),
        daemon=True
    )
    _auto_sync_thread.start()

    return json.dumps({
        "auto_sync": "started",
        "qdrant_url": qdrant_url,
        "collection": collection_name,
        "note": ("Ableton changes now trigger automatic Qdrant updates. "
                 "Orchestrator and models always have current data.")
    }, indent=2)


@mcp.tool()
def stop_auto_sync() -> str:
    """Stop the auto-sync background thread."""
    global _auto_sync_running
    _auto_sync_running = False
    return "Auto-sync stopped."


@mcp.tool()
def get_auto_sync_status() -> str:
    """Get the current status of the auto-sync background thread."""
    return json.dumps({
        "running": _auto_sync_running,
        "qdrant_url": _auto_sync_qdrant_url,
        "collection": _auto_sync_collection
    })




# ─────────────────────────────────────────
# Live polyphonic instrument listener
# Windowed Basic-Pitch for real-time chord/harmony capture
# ─────────────────────────────────────────

_poly_listener_thread = None
_poly_listener_running = False


def _poly_listener_loop(device_index, track_name, section_getter, window_seconds):
    global _poly_listener_running
    import tempfile, os
    try:
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH
        import soundfile as sf_module
    except ImportError as e:
        logger.error(f"Polyphonic listener requires basic-pitch and soundfile: {e}")
        _poly_listener_running = False
        return

    logger.info(f"Polyphonic listener started on device {device_index}, track: {track_name}")

    while _poly_listener_running:
        try:
            audio = sd.rec(
                int(window_seconds * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=device_index
            )
            sd.wait()

            mono = audio.flatten().astype("float32")
            if float(np.sqrt(np.mean(mono ** 2))) < 0.003:
                continue

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            sf_module.write(tmp_path, mono, SAMPLE_RATE)
            _, _, note_events = predict(tmp_path, ICASSP_2022_MODEL_PATH)
            os.unlink(tmp_path)

            if not note_events:
                continue

            note_names_list = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
            window_timestamp = time.time()
            notes = []
            for event in note_events:
                start_t, end_t, pitch, velocity, _ = event
                pc = int(pitch) % 12
                notes.append({
                    "midi_pitch": int(pitch),
                    "pitch_class": pc,
                    "note_name": note_names_list[pc],
                    "start_time": round(float(start_t), 4),
                    "end_time": round(float(end_t), 4),
                    "duration": round(float(end_t) - float(start_t), 4),
                    "velocity": int(velocity * 127),
                    "captured_at": window_timestamp,
                })

            current_section = section_getter()
            if _session_context and notes:
                sections = _session_context.get("structure", {}).get("sections", [])
                if current_section < len(sections):
                    performed = sections[current_section].get("performed", {})
                    instruments = performed.get("instrument_notes", {})
                    existing = instruments.get(track_name, [])
                    recent_pitches = {
                        n["midi_pitch"] for n in existing
                        if window_timestamp - n.get("captured_at", 0) < 3.0
                    }
                    new_notes = [n for n in notes if n["midi_pitch"] not in recent_pitches]
                    existing.extend(new_notes)
                    instruments[track_name] = existing
                    performed["instrument_notes"] = instruments
                    sections[current_section]["performed"] = performed
                    _session_context["structure"]["sections"] = sections
                    if new_notes:
                        pitches = [n["note_name"] for n in new_notes]
                        logger.info(f"Poly [{current_section}] {track_name}: {pitches[:8]}")

        except Exception as e:
            logger.error(f"Polyphonic listener error: {e}")
            time.sleep(1)


@mcp.tool()
def start_polyphonic_listener(
    device_index: int,
    track_name: str = "instrument",
    section_index: int = 0,
    window_seconds: float = 2.5
) -> str:
    """
    Start a live polyphonic audio listener that detects chords and harmonies
    in real time using Basic-Pitch. Works for piano, guitar, violin, or any
    instrument playing multiple notes simultaneously.

    There is inherent latency equal to window_seconds because Basic-Pitch
    processes complete audio windows. Default 2.5 seconds balances accuracy
    against latency. For instruments with MIDI output, routing through Ableton
    as a MIDI track and reading it via the MCP connection is more accurate
    and has zero latency — use this only when MIDI is not available.

    Detected notes are written to the session context performed layer under
    instrument_notes[track_name]. Language model and voice synthesis model
    can query Qdrant to see harmonic context for this instrument.

    Requires: pip install basic-pitch soundfile

    Parameters:
    - device_index: Audio device index (use list_audio_devices to find it)
    - track_name: Label for this instrument e.g. 'piano', 'guitar', 'strings'
    - section_index: Which section is currently being recorded (0-based)
    - window_seconds: Audio window size. Smaller = more responsive, less accurate.
      Larger = better chord detection, more latency. 2-4 seconds recommended.
    """
    global _poly_listener_thread, _poly_listener_running, _current_section_index

    if _poly_listener_running:
        return "Polyphonic listener already running. Call stop_polyphonic_listener first."
    if not _session_context:
        return "No session context. Run build_session_context first."

    _current_section_index = section_index
    _poly_listener_running = True
    _poly_listener_thread = _threading.Thread(
        target=_poly_listener_loop,
        args=(device_index, track_name, lambda: _current_section_index, window_seconds),
        daemon=True
    )
    _poly_listener_thread.start()

    return json.dumps({
        "started": True,
        "device_index": device_index,
        "track_name": track_name,
        "section_index": section_index,
        "window_seconds": window_seconds,
        "latency_note": f"~{window_seconds}s latency per window. Use Ableton MIDI routing for zero-latency polyphonic capture."
    }, indent=2)


@mcp.tool()
def stop_polyphonic_listener() -> str:
    """Stop the live polyphonic instrument listener."""
    global _poly_listener_running
    _poly_listener_running = False
    return "Polyphonic listener stopped."


if __name__ == "__main__":
    logger.info("Audio Analysis MCP Server starting...")
    mcp.run()
