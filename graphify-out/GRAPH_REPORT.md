# Graph Report - .  (2026-07-21)

## Corpus Check
- Corpus is ~16,493 words - fits in a single context window. You may not need a graph.

## Summary
- 203 nodes · 250 edges · 15 communities (10 shown, 5 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 16 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Pose Estimator ONNX Runtime
- YOLO Pose Detection
- Graphify Pipeline & Tooling
- Camera Capture Interface
- YOLO ONNX Session Setup
- Keypoint Data Model
- MediaPipe Python Benchmark
- CSV Data Logger
- Camera Implementation
- CMake Build System
- Watch Folder Feature
- FalkorDB Export
- MCP Server Export
- Neo4j Export

## God Nodes (most connected - your core abstractions)
1. `YoloPose` - 26 edges
2. `PoseEstimator` - 24 edges
3. `graphify Full Pipeline Skill` - 17 edges
4. `Camera` - 13 edges
5. `YoloPoseDetection` - 10 edges
6. `Keypoint` - 8 edges
7. `main()` - 8 edges
8. `tesis_pose C++ Executable Target` - 8 edges
9. `run()` - 7 edges
10. `FpsMeter` - 7 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `openVideo`  [INFERRED]
  src/main.cpp → include/camera.hpp
- `main()` --calls--> `read`  [INFERRED]
  src/main.cpp → include/camera.hpp
- `main()` --calls--> `release`  [INFERRED]
  src/main.cpp → include/camera.hpp
- `Camera::openVideo()` --calls--> `frameCount`  [INFERRED]
  src/camera.cpp → include/camera.hpp
- `main()` --calls--> `update`  [INFERRED]
  src/main.cpp → include/fps_meter.hpp

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **graphify Full Build Pipeline Steps** — _claude_skills_graphify_skill_ast_extraction, _claude_skills_graphify_skill_semantic_extraction, _claude_skills_graphify_skill_graph_build_cluster, _claude_skills_graphify_skill_community_labeling, _claude_skills_graphify_skill_html_viz [EXTRACTED 1.00]
- **tesis_pose C++ Source Files** — cmakeliststxt_src_main, cmakeliststxt_src_camera, cmakeliststxt_src_fps_meter, cmakeliststxt_src_csv_logger, cmakeliststxt_src_pose_estimator, cmakeliststxt_src_yolo_pose [EXTRACTED 1.00]
- **graphify Optional Export Formats** — _claude_skills_graphify_references_exports_wiki, _claude_skills_graphify_references_exports_neo4j, _claude_skills_graphify_references_exports_falkordb, _claude_skills_graphify_references_exports_mcp_server [EXTRACTED 1.00]

## Communities (15 total, 5 thin omitted)

### Community 0 - "Pose Estimator ONNX Runtime"
Cohesion: 0.08
Nodes (24): Env, int64_t, Session, SessionOptions, string, vector, PoseEstimator, decodeOutput (+16 more)

### Community 1 - "YOLO Pose Detection"
Cohesion: 0.11
Nodes (27): Rect, vector, letterbox, nonMaximumSuppression, YoloPoseDetection, box, keypoints, score (+19 more)

### Community 2 - "Graphify Pipeline & Tooling"
Cohesion: 0.09
Nodes (25): graphify Skill Entry Point, graphify add URL Ingest Flow, Wiki Export (--wiki flag), Confidence Score Rubric (EXTRACTED/INFERRED/AMBIGUOUS), Node ID Format Rules (stem_entity), Extraction Subagent Prompt Template, GitHub Repo Clone and Cross-Repo Merge, Native CLAUDE.md graphify Integration (+17 more)

### Community 3 - "Camera Capture Interface"
Cohesion: 0.11
Nodes (22): Camera, capture_, fps, frameCount, height, isOpened, open, openVideo (+14 more)

### Community 4 - "YOLO ONNX Session Setup"
Cohesion: 0.09
Nodes (23): Env, int64_t, Session, SessionOptions, string, YoloPose, confThreshold_, decodeOutput (+15 more)

### Community 5 - "Keypoint Data Model"
Cohesion: 0.16
Nodes (20): Keypoint, score, x, y, Env, int64_t, Mat, Session (+12 more)

### Community 6 - "MediaPipe Python Benchmark"
Cohesion: 0.21
Nodes (11): draw_overlay(), draw_pose(), FpsCounter, parse_source(), percentile(), Calcula FPS en tiempo real usando una ventana deslizante de timestamps.     Evit, Percentil p ∈ [0, 100] sobre una lista de floats., Dibuja conexiones del esqueleto y puntos de landmark sobre el frame.     Operaci (+3 more)

### Community 7 - "CSV Data Logger"
Cohesion: 0.17
Nodes (10): CsvLogger, currentTimestamp, file_, isOpen, log, ofstream, string, CsvLogger::CsvLogger() (+2 more)

### Community 9 - "CMake Build System"
Cohesion: 0.22
Nodes (9): ONNX Runtime Dependency, OpenCV Dependency (core, imgproc, highgui, videoio), src/camera.cpp, src/csv_logger.cpp, src/fps_meter.cpp, src/main.cpp, src/pose_estimator.cpp, src/yolo_pose.cpp (+1 more)

## Knowledge Gaps
- **76 isolated node(s):** `open`, `isOpened`, `fps`, `capture_`, `isOpen` (+71 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `YoloPose` connect `YOLO ONNX Session Setup` to `Pose Estimator ONNX Runtime`, `YOLO Pose Detection`?**
  _High betweenness centrality (0.155) - this node is a cross-community bridge._
- **Why does `Keypoint` connect `Keypoint Data Model` to `Pose Estimator ONNX Runtime`, `YOLO Pose Detection`?**
  _High betweenness centrality (0.129) - this node is a cross-community bridge._
- **What connects `open`, `isOpened`, `fps` to the rest of the system?**
  _76 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Pose Estimator ONNX Runtime` be split into smaller, more focused modules?**
  _Cohesion score 0.082010582010582 - nodes in this community are weakly interconnected._
- **Should `YOLO Pose Detection` be split into smaller, more focused modules?**
  _Cohesion score 0.1111111111111111 - nodes in this community are weakly interconnected._
- **Should `Graphify Pipeline & Tooling` be split into smaller, more focused modules?**
  _Cohesion score 0.09333333333333334 - nodes in this community are weakly interconnected._
- **Should `Camera Capture Interface` be split into smaller, more focused modules?**
  _Cohesion score 0.10869565217391304 - nodes in this community are weakly interconnected._