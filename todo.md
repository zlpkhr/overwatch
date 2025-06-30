- [x] Create Django models for storing frames and metadata
- [x] Implement frame extraction from RTSP stream at 1 FPS
- [x] Add basic frame storage using Django's file storage
- [x] Set up Celery for asynchronous tasks
- [x] Create task for frame embedding generation
- [x] Implement frame embedding generation
- [x] Store embeddings alongside frames in database
- [x] Add simple search endpoint using embedding similarity
- [x] Replace in-memory search with ChromaDB for scalable similarity search
- [x] Implement batched frame processing for efficient embedding generation and storage
- [x] Design and implement a persistent embedding store and indexing process using ChromaDB (e.g., save embeddings and ChromaDB index to disk/database)
- [x] Integrate LLM-based query enhancement: multi-query expansion and re-ranking logic for improved search results


## June 10th

- [x] Integrate HLS live playback in web interface (HLS.js player with reconnect logic)
- [x] Implement FFmpeg-based `restream_hls` command (single-tenant, uses settings.RTSP_URL)
- [x] Create dedicated `stream` app with templates, views, and URLs to avoid clutter in core project
- [x] Display matched frames on timeline and searchable result list
- [x] Backend `search_timestamps` endpoint returning search-aligned frame timestamps
- [x] Timeline thumbnails link to player /live or frame-based playback
- [x] Implement `frame_player` archive viewer: play/pause toggle, next/prev, reset to initial frame, dynamic timestamp label
- [x] Extend `frame_sequence` API with forward/backward cursors (`after`, `before`, `inc`) to support smooth navigation
- [x] YOLOv8 + EasyOCR integration for object detection and text extraction
- [x] `DetectedObject` model with bounding boxes, OCR text, crop embeddings
- [x] Celery task `detect_objects` wired into ingest pipeline
- [x] Crop embeddings indexed in Chroma and merged into frame ranking
- [x] `/search/frame/<id>/detections/` API for detection metadata
- [x] Unified search UI with large timeline cards (320 px) and bounding-box overlays
- [x] Tooltip hover displaying full frame info and detection summary
- [x] Loading spinner while search executes
- [x] Frame-player overlays with labelled boxes and "Go to Latest" control


## June 30th – Sprint 5 (Alerting)

- [x] Design and implement rule-based alerting engine (metadata + CLIP embedding conditions)
- [x] Add post-save signal hooks so alerts trigger on every new `DetectedObject` and whole-frame embedding
- [x] Create `alerts` app with models (`AlertRule`, `Alert`) and evaluation engine
- [x] Alert management UI – create / edit / delete rules with similarity slider and description field
- [x] Alerts dashboard listing recent alerts with thumbnails and localised timestamps
- [x] Global navbar integration across Search, Live Stream, Frame-player, Alerts
- [x] Browser notifications for new (unacknowledged) alerts on any page; auto-ack after delivery
- [x] `/api/alerts/*` endpoints: unacked, recent, acknowledge
- [x] Added detailed logging in `alerts.engine` for easier debugging


## June 30th – Sprint 6 (Alerting v2)

- [x] Implement image-based alert triggers (CLIP similarity vs. reference images)
- [x] Allow uploading reference images during rule creation and via dedicated management page
- [x] Store reference image embeddings automatically, reuse during evaluation
- [x] Update alert evaluation engine to prioritise reference-image similarity over description embedding
- [x] Extend Alerts API payloads with frame IDs & deep-link to frame player; make dashboard rows clickable
- [x] Fixed async signal bug in `_notify_clients` and file-pointer bug in reference image save logic
