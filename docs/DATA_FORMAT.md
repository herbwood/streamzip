# Data format

A training/evaluation manifest is a nonempty JSON array. This is a **fabricated schema demonstration**, not experimental data:

```json
[
  {
    "id": "format-demo-001",
    "source_id": "format-demo-video",
    "video": "videos/example.mp4",
    "question": "What happens after the person opens the door?",
    "choices": [
      "The person sits down.",
      "The person walks outside.",
      "The person picks up a book.",
      "The person turns on a lamp."
    ],
    "answer": 1,
    "timestamp": 93.4
  }
]
```

| Field | Meaning |
|---|---|
| `id` | Unique nonempty string or integer QA ID. Used for deterministic training order. |
| `source_id` | Optional stable video ID. Defaults to the video string. Use the same ID for questions sharing a source. |
| `video` | Local video path; relative paths resolve against the manifest directory. |
| `question` | Original nonempty question. |
| `choices` | Exactly four nonempty strings, ordered A/B/C/D. |
| `answer` | Zero-based GT index 0–3, required for training/evaluation. |
| `timestamp` | Positive finite causal endpoint in seconds, within the actual video duration. |

Only the question and choices enter the model prompt. Answer labels never enter vision selection or LM input. They are used in the objective and after-forward scoring. Source IDs support equal-source metric aggregation; the caller is responsible for correct grouping and train/evaluation separation.

Inference uses the same question/video/timestamp/choices contract without a label. The CLI accepts those fields directly. `examples/inference_input.example.json` illustrates the equivalent record.

The decoder uses actual presentation timestamps, not assumed constant-rate frame ordinals. Each eight-second interval has four uniform temporal-bin targets; the last real frame at or before each target is selected. A final shorter interval is allowed. Frames after the query endpoint are never used.

Duplicate QA IDs, malformed choices, invalid labels and invalid endpoints are rejected. Video/model existence is checked by actual execution, not by `--validate-only`, which validates schema/configuration only. No examples are silently dropped.

Bring your own legally obtained videos and annotations. No video, dataset, download token or dataset-specific adapter ships in this repository.
