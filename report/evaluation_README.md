# CLIP-based Image-Text Retrieval Evaluation

This script extracts the core search functionality from your CCTV surveillance system and evaluates it on the Flickr30k dataset.

## Overview

The evaluation implements the same CLIP-based embedding and similarity search pipeline used in your surveillance system:

1. **CLIP Embeddings**: Uses OpenAI's CLIP ViT-B/32 model to generate embeddings for both images and text queries
2. **Vector Database**: Uses Chroma for efficient similarity search (same as your main system)
3. **Evaluation Metrics**: Implements standard information retrieval metrics (Recall@K and MRR)

## Setup

1. Install dependencies:
```bash
pip install -r evaluation_requirements.txt
```

2. Download Flickr30k dataset:
   - Images: Place all images in a directory (e.g., `./flickr30k_images/`)
   - Captions: Download the captions file (`results.csv` or similar format)

## Usage

Run the evaluation script:

```bash
python evaluation_core.py <images_dir> <captions_file> [max_images]
```

Example:
```bash
python evaluation_core.py ./flickr30k_images ./results.csv 1000
```

### Parameters

- `images_dir`: Directory containing Flickr30k images
- `captions_file`: Path to captions file (format: `image_name|caption`)
- `max_images`: Maximum number of images to use (default: 1000)

## Expected Caption File Format

The captions file should have one caption per line in the format:
```
image1.jpg|A man riding a bicycle down the street
image1.jpg|Person on bike traveling on road
image2.jpg|A dog running in the park
...
```

## Output

The script will output:
- **Recall@K scores** (K=1,5,10): Measures how often relevant images appear in the top K results
- **Mean Reciprocal Rank (MRR)**: Measures the average rank of the first relevant image
- **JSON results file**: Detailed results saved to `evaluation_results_<N>images.json`

## Implementation Notes

### Core Components Extracted

1. **CLIPSearchEngine**: 
   - Loads CLIP ViT-B/32 model
   - Generates normalized embeddings for images and text
   - Manages Chroma vector database for similarity search

2. **Flickr30kEvaluator**:
   - Loads and indexes Flickr30k dataset
   - Implements standard IR evaluation metrics
   - Handles ground truth mapping between images and captions

### Key Differences from Main System

- **Simplified**: Removed Django dependencies, web interface, and streaming components
- **Batch Processing**: Optimized for offline evaluation rather than real-time processing
- **No LLM Query Expansion**: Uses original queries only (can be added if needed)
- **No Reranking**: Uses CLIP similarity only (Jina reranker removed for core evaluation)

## Performance Considerations

- **Device**: Automatically uses CUDA if available, otherwise CPU
- **Memory**: Processes images in batches to avoid memory issues
- **Storage**: Embeddings are persisted to disk via Chroma database

## For Your Report

This evaluation demonstrates:
1. **Core Technical Implementation**: The CLIP embedding pipeline you built
2. **Quantitative Results**: Standard IR metrics for comparison with state-of-the-art
3. **Practical Application**: How your surveillance-focused system performs on a general image-text retrieval task

You can use these results to:
- Compare your implementation against published baselines
- Discuss the trade-offs between general image-text retrieval and surveillance-specific tasks
- Validate the effectiveness of your core technical approach 