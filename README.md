# Visitor Tracking and Entry Counting System

Python-based computer vision application for detecting and tracking visitors, validating entry through multiple gates, and linking counter observations with later gate entries. Built with YOLO11s, BoT-SORT tracking, OSNet appearance-based re-identification, and Streamlit.

## Features

- Detect people in video frames using YOLO11s and maintain tracker IDs using **BoT-SORT**.
- Draw a counter region and up to three gate corridors using a click-point region editor.
- Define an outer (yellow) line and inner (green) line for each gate.
- Validate an entry only after a directionally correct two-stage crossing; reject certain partial/reversed crossings.
- Associate counter and gate observations into visitor journeys with appearance-based matching.
- Review event logs and export structured CSV outputs.
- Accept locally uploaded video, camera/webcam, direct RTSP/HTTP/HLS sources, API-resolved streams, or public YouTube sources, subject to supported formats and access.

## Getting started (Windows)

1. Install **64-bit Python 3.10** and ensure the launcher `py -3.10` works.
2. Download or clone this repository into a local folder.
3. Double-click **`run_app.bat`**. This creates `.venv`, installs `requirements.txt`, downloads model weights, runs deterministic checks, and opens Streamlit. Internet access is needed for initial setup.
4. In the application, choose a permitted video source. Draw the counter/gate polygons and their crossing lines, then run the analysis.

For manual setup, use Python 3.10 to install `requirements.txt`, run `python download_models.py`, then `python verify_setup.py`, and finally `python -m streamlit run app.py`.

## How the entry logic works

Each gate has its own corridor polygon and two user-drawn boundary lines. The system checks tracked foot positions and movement direction to detect an **outer → inner** crossing. It uses tracking and matching state to reduce double counting across repeated observations. The exact line placement matters and must be configured for each camera.

## Tests and current limitations

`verify_setup.py` includes deterministic checks of crossing and journey logic. These synthetic tests are **not** evidence of accuracy on unseen real-world CCTV footage. No independently validated accuracy figure is provided. Crowding, occlusion, poor lighting, tracker ID changes, similar appearances, and unsuitable camera views can cause incorrect detections or identity matches. Appearance-based re-identification cannot guarantee identity across time.

Model files aren't included; `download_models.py` retrieves them from their respective sources. Compatibility and full end-to-end execution should be verified on a clean supported machine before production use.

## Outputs, security, and privacy

Analysis files are generated locally in the `output/` folder and excluded from Git. **Saving identifiable visitor crops is disabled by default** (`SAVE_CROPS = False` in `vision.py`). Do not commit video footage, visitor images, camera passwords, private API endpoints, tokens, or visitor records. Only process streams and share demonstrations when you have authorization. RTSP and API credentials entered through the interface should be handled as sensitive information.

## Third-party software

See `THIRD_PARTY_NOTICES.txt` for attribution to OSNet/Torchreid. YOLO/Ultralytics and the pretrained weights have their own license and usage requirements; verify those before redistribution, public deployment, or commercial use. This repository does not include model weights or grant rights over external software.

## Additional help

- `RTSP_HELP.txt`: connecting a direct network video stream.
- `YOUTUBE_HELP.txt`: supported public video URLs.
- `models/README.txt`: model files.
