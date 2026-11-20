from __future__ import annotations

from datetime import date, datetime, time as dt_time, timedelta
import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from download_models import ensure_model_files
from journey import JourneyManager
from regions import (
    RegionConfig,
    denormalize_points,
    normalize_points,
    point_in_polygon,
)
from source import open_source
from vision import OUTPUT_DIR, REID_MODEL_PATH, YOLO_MODEL_PATH, Gate, VideoProcessor

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "project_regions.json"
DRAFT_PATH = BASE_DIR / "project_regions_draft.json"
ITEMS = [
    "Counter polygon",
    "Gate 1 polygon", "Gate 1 yellow line", "Gate 1 green line",
    "Gate 2 polygon", "Gate 2 yellow line", "Gate 2 green line",
    "Gate 3 polygon", "Gate 3 yellow line", "Gate 3 green line",
]

st.set_page_config(page_title="Visitor Tracking Journey Counter", page_icon="🎡", layout="wide")
st.title("🎡 Visitor Journey and Gate Counting")
st.caption(
    "User-drawn counter and gate regions, tandem-safe two-line entry confirmation, "
    "and full-session counter-to-gate correction."
)


def get_processor() -> VideoProcessor:
    if "processor" not in st.session_state:
        st.session_state.processor = VideoProcessor()
    return st.session_state.processor


def ensure_models(processor: VideoProcessor) -> None:
    if processor.is_models_loaded():
        return
    if not YOLO_MODEL_PATH.is_file() or not REID_MODEL_PATH.is_file():
        ensure_model_files()
    processor.load_models()


def source_controls(prefix: str):
    mode = st.selectbox(
        "Source type",
        ["Uploaded video", "Network stream URL", "YouTube URL", "API → stream URL", "Webcam"],
        key=f"{prefix}_mode",
    )
    upload = rtsp = youtube = api = token = None
    webcam = 0
    if mode == "Uploaded video":
        upload = st.file_uploader(
            "Video", type=["mp4", "avi", "mov", "mkv"], key=f"{prefix}_upload"
        )
    elif mode == "Network stream URL":
        rtsp = st.text_input("Direct RTSP/RTSPS/HTTP/HLS URL", type="password", key=f"{prefix}_rtsp")
        st.caption("Paste the direct media stream URL, not a normal camera website page.")
    elif mode == "YouTube URL":
        youtube = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            key=f"{prefix}_youtube",
        )
        st.caption("Supports standard videos, Shorts, youtu.be links, and public live streams.")
    elif mode == "API → stream URL":
        api = st.text_input("API endpoint", key=f"{prefix}_api")
        token = st.text_input("Bearer token (optional)", type="password", key=f"{prefix}_token")
        st.caption("The API may return an RTSP, RTSPS, HTTP, or HLS URL in common nested fields.")
    else:
        webcam = int(st.number_input("Webcam index", 0, 10, 0, key=f"{prefix}_webcam"))
    transport = "tcp"
    if mode in ("Network stream URL", "API → stream URL"):
        transport = st.selectbox("RTSP transport", ["tcp", "udp"], key=f"{prefix}_transport", help="TCP is usually more reliable on company networks.")
    return mode, upload, rtsp or "", youtube or "", api or "", token or "", webcam, transport


def read_first_frame(args):
    cap, temp_path, _, _, _ = open_source(*args)
    try:
        if not cap.isOpened():
            raise RuntimeError("Could not open the selected source.")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("The source opened, but its first frame could not be read.")
        return frame
    finally:
        cap.release()
        if temp_path:
            temp_path.unlink(missing_ok=True)


def shapes_from_config(config: RegionConfig, width: int, height: int) -> dict[str, list[tuple[int, int]]]:
    shapes = {"Counter polygon": denormalize_points(config.counter_polygon, width, height)}
    for i in (1, 2, 3):
        gate = f"Gate {i}"
        shapes[f"{gate} polygon"] = denormalize_points(config.gate_polygons.get(gate, []), width, height)
        shapes[f"{gate} yellow line"] = denormalize_points(config.outer_lines.get(gate, []), width, height)
        shapes[f"{gate} green line"] = denormalize_points(config.confirmation_lines.get(gate, []), width, height)
    return shapes


def config_from_shapes(shapes: dict, width: int, height: int) -> RegionConfig:
    return RegionConfig(
        frame_width=width,
        frame_height=height,
        counter_polygon=normalize_points(shapes["Counter polygon"], width, height),
        gate_polygons={
            f"Gate {i}": normalize_points(shapes[f"Gate {i} polygon"], width, height)
            for i in (1, 2, 3)
        },
        outer_lines={
            f"Gate {i}": normalize_points(shapes[f"Gate {i} yellow line"], width, height)
            for i in (1, 2, 3)
        },
        confirmation_lines={
            f"Gate {i}": normalize_points(shapes[f"Gate {i} green line"], width, height)
            for i in (1, 2, 3)
        },
        ignore_polygons=[],
        version=2,
    )


def save_draft_shapes(shapes: dict[str, list[tuple[int, int]]], width: int, height: int) -> None:
    """Persist every individually saved polygon/line immediately.

    Coordinates are normalized so the same JSON can be loaded against the
    source frame even when the preview is displayed at a different size.
    """
    payload = {
        "version": 1,
        "frame_width": int(width),
        "frame_height": int(height),
        "items": {
            name: [[float(x) / width, float(y) / height] for x, y in points]
            for name, points in shapes.items()
            if points
        },
    }
    DRAFT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_draft_shapes(width: int, height: int) -> dict[str, list[tuple[int, int]]]:
    if not DRAFT_PATH.is_file():
        return {}
    payload = json.loads(DRAFT_PATH.read_text(encoding="utf-8"))
    result: dict[str, list[tuple[int, int]]] = {}
    for name, points in (payload.get("items") or {}).items():
        if name not in ITEMS or not isinstance(points, list):
            continue
        converted: list[tuple[int, int]] = []
        for point in points:
            if isinstance(point, (list, tuple)) and len(point) == 2:
                x = max(0.0, min(1.0, float(point[0])))
                y = max(0.0, min(1.0, float(point[1])))
                converted.append((int(round(x * width)), int(round(y * height))))
        if converted:
            result[name] = converted
    return result


def validate_shapes(shapes: dict) -> list[str]:
    errors: list[str] = []
    for item in ITEMS:
        pts = shapes.get(item, [])
        is_line = item.endswith("line")
        if is_line and len(pts) != 2:
            errors.append(f"{item}: draw exactly one line")
        elif not is_line and len(pts) < 3:
            errors.append(f"{item}: draw a polygon or rectangle")
    return errors


def build_gates(config: RegionConfig, width: int, height: int) -> list[Gate]:
    gates: list[Gate] = []
    for name in ("Gate 1", "Gate 2", "Gate 3"):
        polygon = denormalize_points(config.gate_polygons.get(name, []), width, height)
        yellow = denormalize_points(config.outer_lines.get(name, []), width, height)
        green = denormalize_points(config.confirmation_lines.get(name, []), width, height)
        if len(polygon) < 3 or len(yellow) != 2 or len(green) != 2:
            continue
        gates.append(
            Gate(
                name=name,
                corridor_polygon=polygon,
                count_line=(yellow[0], yellow[1]),
                confirmation_line=(green[0], green[1]),
                # Direction is derived from yellow toward green, so line drawing
                # order cannot accidentally reverse entry direction.
                entry_from_bottom=False,
            )
        )
    return gates


def draw_saved_shapes(frame: np.ndarray, shapes: dict[str, list[tuple[int, int]]]) -> np.ndarray:
    output = frame.copy()
    for item, points in shapes.items():
        if not points:
            continue
        if item == "Counter polygon":
            color = (255, 0, 255)
        elif "yellow" in item:
            color = (0, 255, 255)
        elif "green" in item:
            color = (0, 255, 0)
        else:
            color = (255, 180, 0)
        contour = np.asarray(points, dtype=np.int32)
        if item.endswith("line") and len(points) == 2:
            cv2.line(output, points[0], points[1], color, 3, cv2.LINE_AA)
        elif len(points) >= 3:
            cv2.polylines(output, [contour], True, color, 3, cv2.LINE_AA)
        cv2.putText(output, item, points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    return output


def draw_runtime_regions(frame: np.ndarray, config: RegionConfig) -> np.ndarray:
    h, w = frame.shape[:2]
    return draw_saved_shapes(frame, shapes_from_config(config, w, h))


setup_tab, run_tab = st.tabs(["1. Draw regions", "2. Run analysis"])

with setup_tab:
    st.subheader("Draw the exact areas yourself")
    st.info(
        "Draw each item separately. All counter logic is restricted to the counter polygon, "
        "and each gate crossing is accepted only inside that gate's polygon. The yellow line "
        "must be outside; the green line must be deeper inside."
    )
    setup_args = source_controls("setup")

    load_col, existing_col = st.columns(2)
    if load_col.button("Load first frame", type="primary"):
        try:
            frame = read_first_frame(setup_args)
            st.session_state.setup_frame = frame
            st.session_state.canvas_generation = st.session_state.get("canvas_generation", 0) + 1
            if CONFIG_PATH.is_file():
                cfg = RegionConfig.load(CONFIG_PATH)
                st.session_state.shapes = shapes_from_config(cfg, frame.shape[1], frame.shape[0])
            elif DRAFT_PATH.is_file():
                st.session_state.shapes = load_draft_shapes(frame.shape[1], frame.shape[0])
                st.info(f"Restored saved drawing progress from {DRAFT_PATH.name}.")
            else:
                st.session_state.shapes = {}
            st.success("First frame loaded.")
        except Exception as exc:
            st.error(str(exc))

    if existing_col.button(
        "Delete saved region configuration",
        disabled=not (CONFIG_PATH.is_file() or DRAFT_PATH.is_file()),
    ):
        CONFIG_PATH.unlink(missing_ok=True)
        DRAFT_PATH.unlink(missing_ok=True)
        st.session_state.shapes = {}
        st.session_state.canvas_generation = st.session_state.get("canvas_generation", 0) + 1
        st.success("Saved configuration deleted.")

    if "setup_frame" in st.session_state:
        original = st.session_state.setup_frame
        h, w = original.shape[:2]
        display_w = min(1100, w)
        display_h = max(300, int(round(h * display_w / w)))
        shapes = st.session_state.setdefault("shapes", {})

        # Keep the editor on an explicit current item and advance automatically
        # after each successful save. The selectbox remains available so any
        # item can be revisited and edited later.
        current_index = int(st.session_state.get("current_item_index", 0))
        current_index = max(0, min(current_index, len(ITEMS) - 1))
        selector_generation = int(st.session_state.get("selector_generation", 0))
        item = st.selectbox(
            "Item to draw",
            ITEMS,
            index=current_index,
            key=f"item_to_draw_{selector_generation}",
        )
        selected_index = ITEMS.index(item)
        if selected_index != current_index:
            st.session_state.current_item_index = selected_index
        is_line = item.endswith("line")

        st.markdown("### Region checklist")
        checklist_lines = []
        for idx, item_name in enumerate(ITEMS):
            mark = "✅" if item_name in shapes else "⬜"
            pointer = " **← current**" if item_name == item else ""
            checklist_lines.append(f"{mark} {item_name}{pointer}")
        st.markdown("  \n".join(checklist_lines))

        st.caption(
            "Click directly on the image to place points. "
            + ("A line needs exactly 2 points." if is_line else "A polygon needs at least 3 points.")
        )

        editor_key = f"editor_points_{item}"
        event_key = f"editor_event_{item}"
        if editor_key not in st.session_state:
            st.session_state[editor_key] = []
        editor_points = st.session_state[editor_key]

        # Existing saved regions are burned into the image. The currently
        # selected item is shown using the unsaved click points so there is no
        # dependency on drawable-canvas JSON callbacks.
        preview = draw_saved_shapes(original, {k: v for k, v in shapes.items() if k != item})
        preview = cv2.resize(preview, (display_w, display_h), interpolation=cv2.INTER_AREA)

        if editor_points:
            if item == "Counter polygon":
                editor_color = (255, 0, 255)
            elif "yellow" in item:
                editor_color = (0, 255, 255)
            elif "green" in item:
                editor_color = (0, 255, 0)
            else:
                editor_color = (255, 180, 0)

            for idx, (px, py) in enumerate(editor_points, start=1):
                cv2.circle(preview, (px, py), 6, editor_color, -1, cv2.LINE_AA)
                cv2.putText(preview, str(idx), (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, editor_color, 1, cv2.LINE_AA)
            if len(editor_points) >= 2:
                cv2.polylines(
                    preview,
                    [np.asarray(editor_points, dtype=np.int32)],
                    bool(not is_line and len(editor_points) >= 3),
                    editor_color,
                    3,
                    cv2.LINE_AA,
                )

        click_image = Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
        click = streamlit_image_coordinates(
            click_image,
            key=f"click_editor_{item}",
        )

        if click and "x" in click and "y" in click:
            event_id = click.get("time", f"{click['x']}:{click['y']}")
            if st.session_state.get(event_key) != event_id:
                st.session_state[event_key] = event_id
                point = (int(click["x"]), int(click["y"]))
                if not is_line or len(editor_points) < 2:
                    editor_points.append(point)
                    st.session_state[editor_key] = editor_points
                    st.rerun()

        st.write(f"Current points: **{len(editor_points)}**")
        b1, b2, b3, b4 = st.columns(4)
        if b1.button("Undo last point", disabled=not editor_points):
            editor_points.pop()
            st.session_state[editor_key] = editor_points
            st.session_state.pop(event_key, None)
            st.rerun()
        if b2.button("Clear current points", disabled=not editor_points):
            st.session_state[editor_key] = []
            st.session_state.pop(event_key, None)
            st.rerun()
        if b3.button(f"Save {item}", type="primary"):
            if (is_line and len(editor_points) != 2) or (not is_line and len(editor_points) < 3):
                st.error("Add exactly 2 points for a line, or at least 3 points for a polygon.")
            else:
                sx, sy = w / display_w, h / display_h
                full_points = [
                    (int(round(x * sx)), int(round(y * sy)))
                    for x, y in editor_points
                ]
                shapes[item] = full_points
                st.session_state.shapes = shapes
                save_draft_shapes(shapes, w, h)
                st.session_state[editor_key] = []
                st.session_state.pop(event_key, None)
                st.success(f"Saved {item}.")

                # Move to the next incomplete item. If all later items are
                # already complete, wrap around and find the first remaining
                # incomplete item. Once all 10 are complete, stay on the item
                # just saved so it can still be reviewed or edited.
                saved_index = ITEMS.index(item)
                remaining = [
                    idx for idx, name in enumerate(ITEMS)
                    if name not in shapes
                ]
                if remaining:
                    later = [idx for idx in remaining if idx > saved_index]
                    next_index = later[0] if later else remaining[0]
                    st.session_state.current_item_index = next_index
                else:
                    st.session_state.current_item_index = saved_index
                # A Streamlit widget's session-state value cannot be changed
                # after that widget has been instantiated in the same run.
                # Change the selector key for the next run instead.
                st.session_state.selector_generation = selector_generation + 1
                st.rerun()
        if b4.button(f"Clear saved {item}", disabled=item not in shapes):
            shapes.pop(item, None)
            st.session_state.shapes = shapes
            save_draft_shapes(shapes, w, h)
            st.session_state[editor_key] = []
            st.session_state.pop(event_key, None)
            st.session_state.current_item_index = ITEMS.index(item)
            st.session_state.selector_generation = selector_generation + 1
            st.rerun()

        completed = len([item_name for item_name in ITEMS if item_name in shapes])
        st.progress(completed / len(ITEMS), text=f"Saved {completed} of {len(ITEMS)} required items")
        st.caption("Saved: " + (", ".join(shapes.keys()) if shapes else "nothing yet"))
        st.caption(f"Autosave file: {DRAFT_PATH.name}")

        if st.button("Save complete region configuration", type="primary"):
            errors = validate_shapes(shapes)
            if errors:
                st.error("Please fix: " + " | ".join(errors))
            else:
                config_from_shapes(shapes, w, h).save(CONFIG_PATH)
                save_draft_shapes(shapes, w, h)
                st.success(
                    f"Saved {CONFIG_PATH.name}. Each individual shape is also stored in "
                    f"{DRAFT_PATH.name}. You can now run the analysis."
                )

with run_tab:
    if not CONFIG_PATH.is_file():
        st.warning("Draw and save the complete configuration in the first tab.")
    else:
        config = RegionConfig.load(CONFIG_PATH)
        st.success("User-drawn region configuration loaded.")
        run_args = source_controls("run")

        c1, c2, c3, c4 = st.columns(4)
        preview_every = c1.slider("Preview every N frames", 1, 30, 10)
        max_frames = int(c2.number_input("Maximum frames (0 = unlimited)", 0, 10_000_000, 0, 100))
        missing_grace = int(c3.number_input("Counter disappearance grace", 1, 300, 25))
        counter_confirm = int(c4.number_input("Counter confirmation frames", 1, 60, 5))
        st.caption(
            "Every frame is detected, tracked and counted. Preview every N frames controls only how often the browser image refreshes."
        )

        recorded_start: datetime | None = None
        if run_args[0] in ("Uploaded video", "YouTube URL"):
            dc1, dc2 = st.columns(2)
            start_date = dc1.date_input("Recording start date", value=date.today())
            start_time = dc2.time_input("Recording start time", value=dt_time(0, 0))
            recorded_start = datetime.combine(start_date, start_time)

        if st.button("Start analysis", type="primary"):
            processor = get_processor()
            cap = None
            temp_path = None
            try:
                with st.spinner("Loading YOLO and OSNet models..."):
                    ensure_models(processor)

                cap, temp_path, source_name, _, backend_name = open_source(*run_args)
                if not cap.isOpened():
                    raise RuntimeError("Could not open the selected source.")
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError("The source opened, but no frame could be read.")

                h, w = frame.shape[:2]
                gates = build_gates(config, w, h)
                if len(gates) != 3:
                    raise RuntimeError("All three gate polygons and both lines per gate are required.")

                counter_polygon = denormalize_points(config.counter_polygon, w, h)
                processor.reset()
                processor.configure_gates(gates)
                journey = JourneyManager(counter_polygon, missing_grace, counter_confirm)

                fps = float(cap.get(cv2.CAP_PROP_FPS))
                if not np.isfinite(fps) or fps <= 0.1:
                    fps = 25.0

                image_holder = st.empty()
                status_holder = st.empty()
                metric_columns = st.columns(4)
                metric_slots = [column.empty() for column in metric_columns]
                frame_no = 0
                consumed_event_count = 0

                while frame is not None:
                    if recorded_start is not None:
                        frame_time = recorded_start + timedelta(seconds=frame_no / fps)
                    else:
                        frame_time = datetime.now()
                    processor.current_timestamp = frame_time

                    annotated, tracks = processor.process_frame(frame)

                    # Assign/re-identify persistent IDs as soon as somebody is
                    # inside the counter polygon. Tracks elsewhere remain
                    # irrelevant to counter logic unless they enter a gate.
                    for track in tracks:
                        foot = processor.bbox_foot(track.bbox)
                        if point_in_polygon(foot, counter_polygon):
                            processor.ensure_visitor_identity(track, "counter")

                    journey.update_tracks(tracks, frame_no, frame_time)
                    new_events = processor.events[consumed_event_count:]
                    journey.consume_entries(new_events)
                    consumed_event_count = len(processor.events)

                    annotated = draw_runtime_regions(annotated, config)
                    frame_no += 1

                    if frame_no == 1 or frame_no % preview_every == 0:
                        image_holder.image(annotated, channels="BGR", width="stretch")
                        metric_slots[0].metric("Visited counter", journey.total_counter_visitors)
                        metric_slots[1].metric("Counter-only so far", journey.provisional_counter_only)
                        metric_slots[2].metric("Later entered", journey.later_entered)
                        metric_slots[3].metric("Unique gate entrants", processor.total_entries)
                        status_holder.caption(
                            f"{source_name} via {backend_name} | {frame_no:,} frames | {processor.fps:.1f} processing FPS"
                        )

                    if max_frames and frame_no >= max_frames:
                        break
                    ok, next_frame = cap.read()
                    frame = next_frame if ok else None

                journey.save(OUTPUT_DIR)
                processor.save_events_csv()
                records = journey.records_dataframe()
                entries = processor.events_dataframe()
                audit = journey.audit_dataframe()

                st.success(
                    "Analysis finished. 'Counter-only so far' is now final for the processed footage; "
                    "any person matched entering later was removed automatically."
                )
                st.subheader("Customer journeys")
                st.dataframe(records, width="stretch", hide_index=True)
                st.download_button(
                    "Download customer journeys CSV",
                    records.to_csv(index=False).encode("utf-8"),
                    "customer_journeys.csv",
                    "text/csv",
                )
                st.download_button(
                    "Download gate crossings CSV",
                    entries.to_csv(index=False).encode("utf-8"),
                    "gate_crossings.csv",
                    "text/csv",
                )
                st.download_button(
                    "Download audit CSV",
                    audit.to_csv(index=False).encode("utf-8"),
                    "journey_audit.csv",
                    "text/csv",
                )
            except Exception as exc:
                st.error(str(exc))
                st.exception(exc)
            finally:
                if cap is not None:
                    cap.release()
                if temp_path:
                    temp_path.unlink(missing_ok=True)
