# Implementation contracts

The public package extracts the final method's tensor/optimizer logic and replaces research-directory lookup with configurable manifests and a causal context builder. It contains no alternative public method.

## Memory and compression

Each eight-second causal interval samples four source frames at bin midpoints, selecting actual display PTS at or before the target. Frames are letterboxed to 128×128 using the original preprocessing. Native vision produces a temporal grid `[2,8,8]`, followed by 32 merged visual positions and three aligned DeepStack feature streams.

The fixed Recency score is `(chunk_id + 1) × token_cost`. Feasible costs are {0,8,16,32}, constrained by recoverable tokens and current-chunk positivity. The original exact dynamic program maximizes the fixed score at the largest attainable spend below the cap. No learned allocator, utility head, retriever, or token-selection alternative is provided.

The coreset computes FP32 normalized feature cosine distances separately within each temporal group, starts from the point closest to its centroid, uses farthest-point updates with stable source-index ties, and interleaves groups round-robin. Integer priorities survive subsequent reductions. Selected feature storage is compact and owns no discarded full-feature backing buffer.

Irreversibility holds across arrivals within each independently replayed view. Chunk-level priority reductions are nested. Different budget replays need not form globally nested index sets, and FULL is the explicitly requested uncompressed reference.

## Coordinates and streams

Original token IDs are never renumbered. A retained index selects final features, all three DeepStack streams, temporal/spatial coordinates, timestamps and prompt positions together. Sparse prompt construction retains all original text/delimiter tokens, gathers original three-axis M-RoPE positions, and replaces only retained visual placeholders with their actual feature vectors.

The original timestamp text and dense visual-grid geometry are rebuilt from saved causal sampling metadata. Dummy zero pixels at that stage produce prompt IDs/positions only; they are never used as the language model's visual features and are never run through vision. Frozen visual features from the actual source frames are the only visual embeddings supplied to the LM.

## Budget token

The four FP32 vectors map to FULL/50%/25%/12.5% in that order. The inserted token occurs immediately before the original question/options sequence, after all visual positions. A single existing text vocabulary ID stands in for its learned embedding. Subsequent text M-RoPE positions increment by one on all three axes; visual positions are untouched.

## Post-training and checkpoints

The native LoRA walk freezes the original parameters and adapts top LM blocks 28–35. It preserves separate local Q/K/V adapters and the native grouped-QKV packing fix, attention output and one fused-FC1 plus FC2 adapter per block. Rank 8, alpha 16 and dropout zero are fixed.

Native BF16 LoRA / FP32 budget optimizer mappings, gradient finalization, clipping, and compact eight-layer checkpoint sharding are retained. The native overlay includes optimizer state and is loaded together with the unchanged base. No simplified HF adapter compatibility is claimed.

## Equivalence and release boundary

Default tests exercise exact cap rounding, deterministic selection, nested subsets, irreversible compact storage, DeepStack alignment, sparse M-RoPE/budget-token shifts, answer-free prompts, FP32 loss arithmetic, full-vocabulary NLL and source PTS decoding.

Release preparation additionally compares extracted functions structurally with the verified originals and replays saved original-bank descriptors at four budgets. This checks scientific tensor logic without executing the 8B model. New config, source-only context construction, conversion and launch wrappers require opt-in GPU integration qualification before being described as model-tested public entrypoints.

No model, dataset, bank, prediction dump, server configuration or comparison implementation is part of the export.
