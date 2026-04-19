---
license: cc-by-nc-sa-4.0
task_categories:
  - robotics
tags:
  - agibot
  - imitation-learning
  - embodied-ai
  - lerobot
  - real-world
  - dual-arm
pretty_name: AgiBot World 2026
size_categories:
  - "1K<n<10K"
language:
  - en
---

<div align="center">

# AgiBot World 2026

**Real-World Embodied Intelligence Dataset**

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![LeRobot](https://img.shields.io/badge/LeRobot-v2.1-blue)](https://github.com/huggingface/lerobot)
[![Homepage](https://img.shields.io/badge/Homepage-agibot--world.com-green)](https://agibot-world.com)

<br/>

<img src="./assets/teaser.gif" alt="AgiBot World 2026 dataset preview (animated)" style="width: 100%; max-width: 100%; height: auto; display: block; margin: 0 auto;" />

<br/>

</div>

## Overview

As robotics research advances into real-world scenarios, the demand for authentic, high-quality data has become increasingly urgent. Following AGIBOT WORLD's "ImageNet moment," we now release the **AGIBOT WORLD 2026** dataset. Built upon massive real-world scenes, it systematically spans pivotal research directions in embodied intelligence, designed to power the next generation of embodied agents.

The AGIBOT WORLD 2026 dataset is collected from **100% real-world environments**, covering commercial spaces, home, and other general-purpose scenarios. Collected on the **AGIBOT G2** robot platform through a free-form collection mode, the dataset provides developers with structured, accurately annotated, high-quality data. Digital twin technology is leveraged to construct a 1:1 scale scenario in the simulation environment for data collection, with the simulation data concurrently open-sourced in the **GenieSim** project.

We invite researchers worldwide to leverage AGIBOT WORLD 2026 to drive robotic intelligence from the lab into the real world, empowering every industry and tangibly boosting production and service efficiency.

---

## Get Started

### Download the Dataset

To download the full dataset, you can use the following code. If you encounter any issues, please refer to the official Hugging Face documentation.

```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install

# When prompted for a password, use an access token with write permissions.
# Generate one from your settings: https://huggingface.co/settings/tokens
git clone https://huggingface.co/datasets/agibot-world/AgiBotWorld2026

# If you want to clone without large files - just their pointers
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/agibot-world/AgiBotWorld2026
```

If you only want to download a specific task, such as `task_3777`, you can use the following code.

```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install

# Initialize an empty Git repository
git init AgiBotWorld2026
cd AgiBotWorld2026

# Set the remote repository
git remote add origin https://huggingface.co/datasets/agibot-world/AgiBotWorld2026

# Enable sparse-checkout
git sparse-checkout init

# Specify the folders and files
git sparse-checkout set 20260315/tar/task_3777 20260315/task_3777

# Pull the data
git pull origin main
```

> To facilitate the inspection of the dataset's internal structure and examples, we also provide a **sample dataset** (~7 GB). Please refer to `task3777/380098_380609.tar.gz`.

---

## Dataset Structure

> LeRobot version reference: **v2.1**

### Overview

A LeRobot dataset is organized into three main parts:

| Part | Description |
|------|-------------|
| `meta/` | Dataset-level metadata and schema definitions |
| `data/` | Episode data stored as Apache Parquet files |
| `videos/` | Per-camera episode videos stored as MP4 files |

### Directory Layout

```
dataset_root/
├── meta/
│   ├── episodes.jsonl
│   ├── info.json
│   ├── episodes_stats.jsonl
│   ├── annotations.json
│   └── tasks.jsonl
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
└── videos/
    └── chunk-000/
        ├── observation.images.top_head/
        │   ├── episode_000000.mp4
        │   └── ...
        ├── observation.images.hand_left/
        │   ├── episode_000000.mp4
        │   └── ...
        └── observation.images.hand_right/
            ├── episode_000000.mp4
            └── ...
```

### Core Storage Rules

- Each Parquet file stores **one complete episode**.
- Each video file stores **one camera stream** for one episode.
- The dataset schema is primarily defined in `meta/info.json`.
- File paths are generated from the templates in `info.json`:
  - `data_path` — path template for Parquet episode files
  - `video_path` — path template for MP4 video files

---

## Metadata Files

| File | Required | Purpose | Note |
|------|:--------:|---------|------|
| `episodes.jsonl` | Yes | Per-episode metadata (index, task list, frame count) | |
| `info.json` | Yes | Main dataset manifest (features, path templates, video metadata) | |
| `episodes_stats.jsonl` | Yes | Auto-generated per-episode statistics | Video channel statistics (min, max, etc.) are set to 0 due to internal preprocessing. This does not affect usage. |
| `tasks.jsonl` | Yes | Task description metadata | |
| `annotations.json` | No | Optional raw episode info, labels, review data, source paths | |

### `episodes.jsonl`

```json
{"episode_index": 0, "tasks": ["G2"], "length": 1242}
```

| Field | Description |
|-------|-------------|
| `episode_index` | Episode ID inside the dataset |
| `tasks` | Task labels or task names for the episode |
| `length` | Number of frames in the episode |

### `info.json`

`info.json` is the most important metadata file. It contains:

<details>
<summary><b>Click to expand full schema</b></summary>

- **Dataset summary**: `robot_type`, `total_episodes`, `total_frames`, `fps`, `splits`
- **File templates**: `data_path`, `video_path`
- **Schema definition**: `features`
- **Optional custom metadata**: `instruction_segments`, `key_frame`, `high_level_instruction`, `take_over`, `h5_path`, `camera_parameters`

</details>

---

## Features Schema

The `features` object defines how data is stored in Parquet files and how video streams are represented.

### Main Feature Groups

| Feature | Required | Description |
|---------|:--------:|-------------|
| `observation.state` | Yes | Robot state vector definition |
| `action` | Yes | Robot action vector definition |
| `observation.images.*` | Yes | Video stream metadata for each camera channel |
| `episode_index` | Yes | Episode index field |
| `frame_index` | Yes | Frame index within the episode |
| `index` | Yes | Global frame index in the dataset |
| `task_index` | Yes | Task index for the episode |
| `timestamp` | Yes | Relative timestamp starting from 0 |

### State and Action Vectors

For `observation.state` and `action`, the actual values are stored in Parquet files, while the schema is described by `field_descriptions`.

Each entry in `field_descriptions` typically includes:

- `description` — human-readable description of the field
- `dimensions` — number of dimensions in this sub-field
- `dims` — optional names for each dimension
- `indices` — index positions inside the flattened Parquet vector

<details>
<summary><b>Example keys</b></summary>

- `state/left_effector/position`
- `state/right_effector/position`
- `state/joint/position`
- `state/joint/velocity`
- `action/joint/position`
- `action/robot/velocity`

</details>

### Camera Feature Keys

Camera streams are represented as `observation.images.*` keys. Each one describes video type, frame rate, codec, pixel format, and frame shape.

<details>
<summary><b>Common camera keys</b></summary>

- `observation.images.top_head`
- `observation.images.hand_left`
- `observation.images.hand_right`
- `observation.images.head_depth`
- `observation.images.head_left_fisheye`
- `observation.images.head_right_fisheye`
- `observation.images.head_back_fisheye_color`
- `observation.images.head_stereo_left_color`
- `observation.images.head_stereo_right_color`

</details>

---

## Annotation Layers

The dataset follows the LeRobot v2.1 directory structure. `meta/info.json` adds **three annotation layers** beyond the standard LeRobot spec, stored under the `key_frame` and `instruction_segments` keys, indexed by episode index (as a string).

```
Episode ── high-level label (tasks.jsonl)
│
└── Task Frame segments ── subtask instruction + [start, end)
    │
    ├── 2D Bounding Box ── object label + bbox + camera
    │
    └── Instruction Segments ── skill + step instruction + [start, end)
```

### Layer 1 — Task Frame (Subtask Instructions)

`key_frame[ep_idx]["dual"]` entries with `"frame_type_name": "Task Frame"` define subtask-level segments. Each entry specifies a time interval `[start, end)` in frame indices and a natural language instruction in `frame_detail.comment`.

```json
{
  "track": "subtask",
  "frame_type_name": "Task Frame",
  "start": 35,
  "end": 2516,
  "comment": "",
  "frame_detail": {
    "comment": "Place the red-capped drinks and the white triple-pack yogurts from the shopping cart into the fifth shelf of the refrigerated cabinet.",
    "is_result_succeed": true
  }
}
```

A single long-horizon episode may contain multiple Task Frame entries — each covering a distinct subtask the robot must complete. This is the annotation used by `split_episode.py` to produce single-instruction episodes.

### Layer 2 — 2D Bounding Box (Object Annotations)

`key_frame[ep_idx]["dual"]` entries with `"frame_type_name": "2D Bounding Box"` label key objects that the robot interacts with. Each entry includes the object category, the arm track it belongs to, the frame interval during which the object is relevant, and the bounding box in normalized coordinates.

```json
{
  "track": "Right arm",
  "frame_type_name": "2D Bounding Box",
  "start": 493,
  "end": 743,
  "comment": "",
  "frame_detail": {
    "box": { "h": 0.239, "w": 0.083, "x": 0.490, "y": 0.574 },
    "camera": "head_color",
    "comment": "Yogurt, white",
    "type": "box"
  }
}
```

These annotations can be used to train object-conditioned policies, ground language instructions to visual regions, or study grasp selection.

### Layer 3 — Instruction Segments (Step-Level Labels)

`instruction_segments[ep_idx]` is a list of fine-grained step-level segments, each covering a primitive skill (e.g., "Pick", "Bend waist forward") with a natural language instruction and frame boundaries.

```json
{
  "track": "default",
  "skill": "Pick",
  "instruction": "The left arm picks up the red-capped drink from the shopping cart.",
  "instruction_augmentation": {},
  "start_frame_index": 284,
  "success_frame_index": 493,
  "end_frame_index": 493
}
```

This layer enables research into skill-level imitation learning, primitive discovery, and multi-granularity language conditioning.

---

## Optional Annotation Fields in `info.json`

<details>
<summary><b>instruction_segments</b></summary>

Per-episode step annotations. Each item may contain:

- `track` — annotation track name
- `instruction` — step description
- `start_frame_index` — start frame of the segment
- `end_frame_index` — end frame of the segment

</details>

<details>
<summary><b>key_frame</b></summary>

Per-episode key frame annotations. Two common categories are:

- **`single`** — single-frame annotations
- **`dual`** — range annotations

Typical fields: `track`, `frame_type_name`, `start`, `end`, `frame_detail.comment`, `frame_detail.error_cause`, `frame_detail.restorable`, `frame_detail.extra`

Common built-in frame type names: `Error Frame`, `Success Frame`, `Intervention Frame`, `Single Frame`, `2D Bounding Box`, `Task Frame`

</details>

---

## Episode Data and Video Data

| Aspect | Episode data | Video data |
|--------|--------------|------------|
| **Location** | `data/` | `videos/` |
| **Format** | Apache Parquet | MP4 |
| **Granularity** | One file per episode | One file per camera per episode |
| **Schema** | `meta/info.json` → `features` | `meta/info.json` → `video_path` |

### Practical Reading Order

1. Read `meta/info.json` to get the schema and path templates.
2. Read `meta/episodes.jsonl` to learn episode-level metadata.
3. Read the target `field_descriptions` entries in `features` and collect the required `indices`.
4. Load episode Parquet files from `data/` and extract required dimensions.
5. Load matching camera videos from `videos/`.

---

## Installation

The following link from the Hugging Face team provides instructions for installing LeRobot, which requires **Python 3.10+** and **PyTorch 2.2+**.

> [LeRobot Installation Guide](https://github.com/huggingface/lerobot/tree/v0.3.3?tab=readme-ov-file#installation)

---

## Two Usage Modes

### Mode A — Original Format (Hierarchical Research)

Use the dataset as-is. Each episode covers a complete long-horizon restocking task. Access the multi-level annotations from `meta/info.json` to train hierarchical policies, task-conditioned planners, or multi-granularity instruction-following models.

```python
import json

with open("meta/info.json") as f:
    info = json.load(f)

# Iterate subtask segments for episode 0
for entry in info["key_frame"]["0"]["dual"]:
    if entry["frame_type_name"] == "Task Frame":
        print(entry["start"], entry["end"], entry["frame_detail"]["comment"])

# Iterate step-level segments for episode 0
for step in info["instruction_segments"]["0"]:
    print(step["skill"], step["instruction"], step["start_frame_index"], step["end_frame_index"])
```

### Mode B — Split Format (Standard LeRobot, Plug-and-Play)

A companion script `split_episode.py` is provided to convert the dataset into standard single-episode single-instruction LeRobot format, compatible with existing training pipelines out of the box.

Run `split_episode.py` to split long-horizon episodes into single-instruction episodes compatible with standard LeRobot training scripts (e.g., `lerobot/scripts/train.py`). The extended annotation fields are stripped from `info.json` in the output.

**Tool:** [split_episodes_tool.zip](split_episodes_tool.zip)

**Prerequisites:** Python 3.8+ &nbsp;|&nbsp; FFmpeg in `PATH`

```bash
pip install pyarrow numpy
```

**Usage**

```bash
python split_episode.py <input_dataset_path> [--output_path <output>] [--num_workers <N>]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `input_path` | *(required)* | Root directory of the input dataset |
| `--output_path` | `<input_path>_split` | Output directory |
| `--num_workers` | `4` | Parallel threads for parquet + video splitting |

**Example**

```bash
python split_episode.py ./my_dataset --output_path ./my_dataset_split --num_workers 8
```

<details>
<summary><b>What happens (demo example)</b></summary>

| Original Episode | Task Frames | Resulting Episodes |
|:----------------:|:-----------:|:------------------:|
| episode_000000 | 1 segment | episode 0 |
| episode_000001 | 4 segments | episodes 1–4 |
| **Total: 2** | | **Total: 5** |

Each resulting episode has exactly one instruction:

```
episode_000001 → "Put the pink bottled yogurt from the third layer of the chilled cabinet into the shopping cart."
episode_000002 → "Put the pink bottled yogurt from the third shelf of the chilled cabinet into the shopping cart."
episode_000003 → "Put the red bottled yogurt from the third shelf of the refrigerated case into the shopping cart."
episode_000004 → "Put the blue bottled yogurt from the third layer of the chilled display case into the shopping cart."
```

</details>

The script:

- Copies the full dataset to the output path (original is never modified)
- Slices parquet files by frame range; resets `frame_index`, `index`, `timestamp`, `task_index`
- Extracts video segments via FFmpeg (preserves codec and pixel format)
- Rewrites `meta/info.json`, `episodes.jsonl`, `tasks.jsonl`, `episodes_stats.jsonl`

---

## License and Citation

All the data and code within this repo are licensed under [**CC BY-NC-SA 4.0**](https://creativecommons.org/licenses/by-nc-sa/4.0/). Please consider citing our project if it contributes to your research.

```bibtex
@misc{agibotworld2026,
    title        = {AgiBot World 2026},
    author       = {AgiBot World Team},
    howpublished = {\url{https://huggingface.co/datasets/agibot-world/AgiBotWorld2026}},
    year         = {2026}
}
```
