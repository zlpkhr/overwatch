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
