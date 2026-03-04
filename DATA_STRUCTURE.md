# Data Structure

The `hdepic_data/` and `hdepic_examples/` folders are not tracked in git. This document describes their expected structure.

## hdepic_data/

Egocentric video recordings organized by participant.

```
hdepic_data/
└── P{XX}/                          # Participant folder (e.g. P01, P02, ...)
    ├── durations.txt               # Per-recording duration in seconds (one entry per line: <filename> <seconds>)
    ├── frames.txt                  # Per-recording frame count (one entry per line: <filename> <frames>)
    └── {PXX}-{YYYYMMDD}-{HHMMSS}/ # One entry per recording session, named by participant + timestamp
        ├── *.mp4                               # Raw video file
        ├── *_mp4_to_vrs_time_ns.csv            # Frame-level timestamp alignment between mp4 and VRS device time
        └── *_vrs_to_mp4_log.json               # Summary stats for the VRS↔mp4 sync (frame counts, duration, skips)
```

### File formats

**`durations.txt`**
```
P01-20240202-110250.mp4 396.266667
P01-20240202-161354.mp4 327.600000
...
```

**`frames.txt`**
```
P01-20240202-110250.mp4 11888
P01-20240202-161354.mp4 9828
...
```

**`*_mp4_to_vrs_time_ns.csv`**

Maps each mp4 frame to its corresponding VRS device timestamp:

| Column | Description |
|--------|-------------|
| `mp4_time_ns` | Frame time in the mp4 timeline (nanoseconds) |
| `relative_vrs_device_time_ns` | Offset from VRS recording start (nanoseconds) |
| `vrs_device_time_ns` | Absolute VRS device timestamp (nanoseconds) |

**`*_vrs_to_mp4_log.json`**

Sync summary for a single recording:

| Field | Description |
|-------|-------------|
| `num_mp4_frames` | Total frames in the mp4 |
| `down_sampling_factor_` | Frame downsampling applied during conversion |
| `num_skipped_frames` | Frames dropped during sync |
| `num_duplicated_frames_` | Frames duplicated during sync |
| `num_invalid_frames_` | Frames with invalid timestamps |
| `first_video_timestamp_ns` | Start timestamp of video (ns) |
| `end_video_timestamp_ns` | End timestamp of video (ns) |
| `video_duration_ns` | Video duration (ns) |
| `audio_duration_ns` | Audio duration (ns) |

---

## hdepic_examples/

Contains one example mp4 per participant for quick reference/testing without loading the full dataset.

```
hdepic_examples/
└── {PXX}-{YYYYMMDD}-{HHMMSS}.mp4   # One representative clip per participant
```
