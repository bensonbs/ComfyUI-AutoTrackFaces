# ComfyUI-AutoTrackFaces
Use RetinaFace to detect and automatically track faces in video sequences with smooth camera movement effects

![Auto Track Faces Workflow](images/demo.gif)

## Custom Node

### Auto Track Faces (Video)


Automatically track faces in video sequences with smooth camera movement effects. This node processes a batch of video frames and creates a cinematic tracking effect that follows the largest face in the scene.

* `images`: Input video frames as a batch of images with shape (batch, height, width, channels)

* `smoothness`: Controls camera movement smoothness (default: 0.85, range: 0.0-0.98). 0=no smoothing, 0.85=recommended, 0.95=ultra smooth (eliminates jitter)

* `responsiveness`: Controls reaction speed to face movement (default: 0.3, range: 0.1-1.0). 0.1=very stable, 0.3=recommended, 1.0=fast response (may jitter)

* `detect_interval`: Face detection frequency for performance optimization (default: 1, range: 1-30). 1=every frame (most accurate), 5=5x speed boost

* `aspect_ratio`: Output aspect ratio for the tracked video (default: 16:9)

* `output_width`: Output video width, 0=auto calculate based on aspect ratio (default: 0, range: 0-8192)

* `output_height`: Output video height, 0=auto calculate based on aspect ratio (default: 0, range: 0-8192)

* `scale_factor`: How much padding to add around detected faces (default: 2.0, range: 1.0-10.0)

* `shift_factor`: Vertical positioning of face in frame (default: 0.45, range: 0-1). 0=top edge, 0.5=center, 1.0=bottom edge

Returns:
* `tracked_video`: The processed video with smooth face tracking
* `tracking_info`: Detailed tracking report including statistics and frame-by-frame information

**Algorithm Features:**
- Intelligent keyframe selection for optimal performance
- Exponential Moving Average (EMA) smoothing with adaptive speed limits
- Subpixel smoothing to eliminate tiny camera jitter
- Automatic resolution calculation based on input and aspect ratio
- Smooth interpolation between detection frames
- Comprehensive tracking statistics and reporting

**Performance Tips:**
- Higher `detect_interval` values provide significant speed boosts (e.g., 5x faster with interval=5)
- The algorithm automatically balances detection accuracy with performance
- Use reasonable `scale_factor` values (1.5-3.0) for best results
- For videos with fast-moving subjects, increase `responsiveness` slightly

---
Forked and modified from [liusida/ComfyUI-AutoCropFaces](https://github.com/liusida/ComfyUI-AutoCropFaces)
