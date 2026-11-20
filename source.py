from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import cv2
import requests

try:
    import yt_dlp
except ImportError:  # verified at startup; kept optional for clearer errors
    yt_dlp = None

NETWORK_SCHEMES = ("rtsp://", "rtsps://", "http://", "https://")


def save_uploaded_video(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    uploaded_file.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        while True:
            chunk = uploaded_file.read(1024 * 1024)
            if not chunk:
                break
            temp_file.write(chunk)
        return Path(temp_file.name)


def _find_stream_url(payload: Any) -> Optional[str]:
    """Find a direct stream URL in common API response shapes."""
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, list):
        for item in payload:
            found = _find_stream_url(item)
            if found:
                return found
        return None
    if not isinstance(payload, dict):
        return None

    preferred = (
        "rtsp_url", "rtsps_url", "stream_url", "playback_url", "hls_url",
        "video_url", "url", "uri", "src",
    )
    for key in preferred:
        value = payload.get(key)
        if isinstance(value, str) and value.strip().lower().startswith(NETWORK_SCHEMES):
            return value.strip()

    for key in ("data", "result", "stream", "camera", "payload"):
        found = _find_stream_url(payload.get(key))
        if found:
            return found
    return None


def resolve_stream_from_api(api_url: str, bearer_token: str = "", timeout: int = 20) -> str:
    api_url = api_url.strip()
    if not api_url:
        raise ValueError("Enter the API endpoint.")
    headers = {"Accept": "application/json"}
    if bearer_token.strip():
        headers["Authorization"] = f"Bearer {bearer_token.strip()}"
    response = requests.get(api_url, headers=headers, timeout=timeout)
    response.raise_for_status()
    try:
        payload: Any = response.json()
    except ValueError as exc:
        raise ValueError("The API did not return JSON.") from exc
    url = _find_stream_url(payload)
    if not url:
        raise ValueError(
            "API JSON did not contain a direct RTSP/RTSPS/HTTP/HLS stream URL "
            "under a recognised field."
        )
    return url



def validate_youtube_url(url: str) -> str:
    value = url.strip()
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()
    valid_hosts = {
        "youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
        "youtu.be", "www.youtu.be", "youtube-nocookie.com", "www.youtube-nocookie.com",
    }
    if parsed.scheme.lower() not in {"http", "https"} or host not in valid_hosts:
        raise ValueError("Enter a valid YouTube video, Shorts, live-stream, or youtu.be URL.")
    return value


def resolve_youtube_stream(youtube_url: str) -> tuple[str, str]:
    """Resolve a YouTube page URL to a direct video stream URL for OpenCV.

    A single-file format is selected because OpenCV cannot merge separate
    video-only and audio-only DASH streams. Audio is not needed by this app.
    """
    value = validate_youtube_url(youtube_url)
    if yt_dlp is None:
        raise RuntimeError(
            "YouTube support is not installed. Run run_app.bat again so yt-dlp is installed."
        )

    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "format": "best[height<=1080][protocol^=http]/best[protocol^=http]/best",
        "socket_timeout": 20,
        "retries": 3,
        "extractor_retries": 3,
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(value, download=False)
    except Exception as exc:
        raise RuntimeError(
            "YouTube could not be opened. Confirm the video is public or unlisted, "
            "not age-restricted, and accessible on this network. "
            f"yt-dlp message: {exc}"
        ) from exc

    if info is None:
        raise RuntimeError("YouTube returned no video information.")
    if "entries" in info:
        entries = [entry for entry in (info.get("entries") or []) if entry]
        if not entries:
            raise RuntimeError("The YouTube URL did not contain a playable video.")
        info = entries[0]

    direct_url = str(info.get("url") or "").strip()
    if not direct_url:
        formats = info.get("formats") or []
        candidates = [
            f for f in formats
            if f.get("url") and f.get("vcodec") not in (None, "none")
            and str(f.get("protocol", "")).startswith("http")
        ]
        candidates.sort(
            key=lambda f: (
                int(f.get("height") or 0) <= 1080,
                int(f.get("height") or 0),
                float(f.get("tbr") or 0),
            ),
            reverse=True,
        )
        if candidates:
            direct_url = str(candidates[0]["url"])

    if not direct_url:
        raise RuntimeError("No OpenCV-compatible YouTube video stream was found.")

    title = str(info.get("title") or "YouTube video")
    return direct_url, title


def _is_network_stream(source: object) -> bool:
    return isinstance(source, str) and source.lower().startswith(NETWORK_SCHEMES)


def _open_capture_once(source: object, backend: int, transport: str) -> cv2.VideoCapture:
    if isinstance(source, str) and source.lower().startswith(("rtsp://", "rtsps://")):
        # Must be set before VideoCapture is constructed. TCP is normally more
        # reliable through company networks; UDP remains selectable in the UI.
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            f"rtsp_transport;{transport}|stimeout;10000000|rw_timeout;10000000"
        )
    capture = cv2.VideoCapture(source, backend) if backend else cv2.VideoCapture(source)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    try:
        capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
        capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
    except Exception:
        pass
    return capture


def open_capture_with_fallback(source: object, transport: str = "tcp") -> tuple[cv2.VideoCapture, str]:
    """Open a source using FFmpeg first and the default backend second."""
    errors: list[str] = []
    backends = [(cv2.CAP_FFMPEG, "FFmpeg"), (0, "OpenCV default")]
    for backend, label in backends:
        capture = _open_capture_once(source, backend, transport)
        if capture.isOpened():
            return capture, label
        capture.release()
        errors.append(label)
        time.sleep(0.15)
    raise RuntimeError(
        "Could not open the video source with FFmpeg or the default OpenCV backend. "
        "For RTSP, verify the exact direct URL, username/password, camera reachability, "
        "VPN/LAN access, and firewall port. A normal website page URL is not a video stream. "
        f"Attempted backends: {', '.join(errors)}."
    )


def validate_network_url(url: str) -> str:
    value = url.strip()
    parsed = urlparse(value)
    if parsed.scheme.lower() not in {"rtsp", "rtsps", "http", "https"}:
        raise ValueError("Enter a direct RTSP, RTSPS, HTTP video, or HLS (.m3u8) URL.")
    if not parsed.netloc:
        raise ValueError("The stream URL is incomplete or invalid.")
    return value


def open_source(
    mode: str,
    uploaded_file=None,
    rtsp_url: str = "",
    youtube_url: str = "",
    api_url: str = "",
    bearer_token: str = "",
    webcam_index: int = 0,
    transport: str = "tcp",
):
    temp_path: Optional[Path] = None
    if mode == "Uploaded video":
        if uploaded_file is None:
            raise ValueError("Upload a video first.")
        temp_path = save_uploaded_video(uploaded_file)
        source: object = str(temp_path)
        name = uploaded_file.name
    elif mode == "Network stream URL":
        source = validate_network_url(rtsp_url)
        name = "network_stream"
    elif mode == "YouTube URL":
        source, youtube_title = resolve_youtube_stream(youtube_url)
        name = f"youtube: {youtube_title}"
    elif mode == "API → stream URL":
        source = resolve_stream_from_api(api_url, bearer_token)
        name = "api_stream"
    else:
        source = int(webcam_index)
        name = f"webcam_{webcam_index}"

    try:
        capture, backend_name = open_capture_with_fallback(source, transport)
    except Exception:
        if temp_path:
            temp_path.unlink(missing_ok=True)
        raise
    return capture, temp_path, name, source, backend_name
