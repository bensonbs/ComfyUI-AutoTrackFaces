import torch
import comfy.utils
from .Pytorch_Retinaface.pytorch_retinaface import Pytorch_RetinaFace
from comfy.model_management import get_torch_device

class AutoCropFaces:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "number_of_faces": ("INT", {
                    "default": 5, 
                    "min": 1,
                    "max": 100,
                    "step": 1,
                }),
                "scale_factor": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.5,
                    "max": 10,
                    "step": 0.5,
                    "display": "slider"
                }),
                "shift_factor": ("FLOAT", {
                    "default": 0.45,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "display": "slider"
                }),
                "start_index": ("INT", {
                    "default": 0,
                    "step": 1,
                    "display": "number"
                }),
                "max_faces_per_image": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                }),
                # "aspect_ratio": ("FLOAT", {
                #     "default": 1, 
                #     "min": 0.2,
                #     "max": 5,
                #     "step": 0.1,
                # }),
                "aspect_ratio": (["9:16", "2:3", "3:4", "4:5", "1:1", "5:4", "4:3", "3:2", "16:9"], {
                    "default": "1:1",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "CROP_DATA")
    RETURN_NAMES = ("face",)

    FUNCTION = "auto_crop_faces"

    CATEGORY = "Faces"

    def aspect_ratio_string_to_float(self, str_aspect_ratio="1:1"):
        a, b = map(float, str_aspect_ratio.split(':'))
        return a / b

    def auto_crop_faces_in_image (self, image, max_number_of_faces, scale_factor, shift_factor, aspect_ratio, method='lanczos'): 
        image_255 = image * 255
        rf = Pytorch_RetinaFace(top_k=50, keep_top_k=max_number_of_faces, device=get_torch_device())
        dets = rf.detect_faces(image_255)
        cropped_faces, bbox_info = rf.center_and_crop_rescale(image, dets, scale_factor=scale_factor, shift_factor=shift_factor, aspect_ratio=aspect_ratio)

        # Add a batch dimension to each cropped face
        cropped_faces_with_batch = [face.unsqueeze(0) for face in cropped_faces]
        return cropped_faces_with_batch, bbox_info

    def auto_crop_faces(self, image, number_of_faces, start_index, max_faces_per_image, scale_factor, shift_factor, aspect_ratio, method='lanczos'):
        """ 
        "image" - Input can be one image or a batch of images with shape (batch, width, height, channel count)
        "number_of_faces" - This is passed into PyTorch_RetinaFace which allows you to define a maximum number of faces to look for.
        "start_index" - The starting index of which face you select out of the set of detected faces.
        "scale_factor" - How much crop factor or padding do you want around each detected face.
        "shift_factor" - Pan up or down relative to the face, 0.5 should be right in the center.
        "aspect_ratio" - When we crop, you can have it crop down at a particular aspect ratio.
        "method" - Scaling pixel sampling interpolation method.
        """
        
        # Turn aspect ratio to float value
        aspect_ratio = self.aspect_ratio_string_to_float(aspect_ratio)

        selected_faces, detected_cropped_faces = [], []
        selected_crop_data, detected_crop_data = [], []
        original_images = []

        # Loop through the input batches. Even if there is only one input image, it's still considered a batch.
        for i in range(image.shape[0]):

            original_images.append(image[i].unsqueeze(0)) # Temporarily the image, but insure it still has the batch dimension.
            # Detect the faces in the image, this will return multiple images and crop data for it.
            cropped_images, infos = self.auto_crop_faces_in_image(
                image[i],
                max_faces_per_image,
                scale_factor,
                shift_factor,
                aspect_ratio,
                method)

            detected_cropped_faces.extend(cropped_images)
            detected_crop_data.extend(infos)

        # If we haven't detected anything, just return the original images, and default crop data.
        if not detected_cropped_faces or len(detected_cropped_faces) == 0:
            selected_crop_data = [(0, 0, img.shape[3], img.shape[2]) for img in original_images]
            return (image, selected_crop_data)

         # Circular index calculation
        start_index = start_index % len(detected_cropped_faces)

        if number_of_faces >= len(detected_cropped_faces):
            selected_faces = detected_cropped_faces[start_index:] + detected_cropped_faces[:start_index]
            selected_crop_data = detected_crop_data[start_index:] + detected_crop_data[:start_index]
        else:
            end_index = (start_index + number_of_faces) % len(detected_cropped_faces)
            if start_index < end_index:
                selected_faces = detected_cropped_faces[start_index:end_index]
                selected_crop_data = detected_crop_data[start_index:end_index]
            else:
                selected_faces = detected_cropped_faces[start_index:] + detected_cropped_faces[:end_index]
                selected_crop_data = detected_crop_data[start_index:] + detected_crop_data[:end_index]

        # If we haven't selected anything, then return original images.
        if len(selected_faces) == 0: 
            # selected_crop_data = [(0, 0, img.shape[3], img.shape[2]) for img in original_images]
            return (image, None)

        # If there is only one detected face in batch of images, just return that one.
        elif len(selected_faces) <= 1:
            out = selected_faces[0]
            crop_data = selected_crop_data[0] # to be compatible with WAS
            return (out, crop_data)

        # Determine the index of the face with the maximum width
        max_width_index = max(range(len(selected_faces)), key=lambda i: selected_faces[i].shape[1])

        # Determine the maximum width
        max_width = selected_faces[max_width_index].shape[1]
        max_height = selected_faces[max_width_index].shape[2]
        shape = (max_height, max_width)

        out = None
        # All images need to have the same width/height to fit into the tensor such that we can output as image batches.
        for face_image in selected_faces:
            if shape != face_image.shape[1:3]: # Determine whether cropped face image size matches largest cropped face image. 
                face_image = comfy.utils.common_upscale( # This method expects (batch, channel, height, width)
                    face_image.movedim(-1, 1), # Move channel dimension to width dimension
                    max_height, # Height
                    max_width, # Width
                    method, # Pixel sampling method.
                    "" # Only "center" is implemented right now, and we don't want to use that.
                ).movedim(1, -1)
            # Append the fitted image into the tensor.
            if out is None:
                out = face_image
            else:
                out = torch.cat((out, face_image), dim=0)

        #TODO: WAS doesn't not support multiple faces, so this won't work with WAS.
        return (out, selected_crop_data)

class AutoTrackFaces:
    """
    Automatically track faces in video with smooth camera movement effects
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "smoothness": ("FLOAT", {
                    "default": 0.85,
                    "min": 0.0,
                    "max": 0.98,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Smoothness intensity: 0=no smoothing, 0.85=recommended, 0.95=ultra smooth (eliminates jitter)"
                }),
                "responsiveness": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Response sensitivity: 0.1=very stable, 0.3=recommended, 1.0=fast response (may jitter)"
                }),
                "detect_interval": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 30,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Detect every N frames: 1=every frame (most accurate), 5=5x speed boost"
                }),
                "aspect_ratio": (["16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16"], {
                    "default": "16:9",
                }),
            },
            "optional": {
                "output_width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 8192,
                    "step": 8,
                    "tooltip": "Output width, 0=auto calculate"
                }),
                "output_height": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 8192,
                    "step": 8,
                    "tooltip": "Output height, 0=auto calculate"
                }),
                "scale_factor": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "slider",
                }),
                "shift_factor": ("FLOAT", {
                    "default": 0.45,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "display": "slider",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("tracked_video", "tracking_info")
    FUNCTION = "track_faces"
    CATEGORY = "Faces/Video"

    def aspect_ratio_string_to_float(self, str_aspect_ratio="1:1"):
        a, b = map(float, str_aspect_ratio.split(':'))
        return a / b
    
    def calculate_output_resolution(self, input_shape, aspect_ratio_str, output_width=0, output_height=0):
        """
        Intelligently calculate output resolution

        Strategy:
        1. If user specifies both width and height, use them directly
        2. If only one is specified, calculate the other based on aspect ratio
        3. If neither is specified, intelligently calculate based on input resolution and aspect ratio

        Algorithm:
        - Maintain quality not lower than input resolution
        - Align to multiples of 8 (video encoding requirement)
        - Adjust according to aspect ratio
        """
        input_height, input_width = input_shape[0], input_shape[1]
        aspect_ratio = self.aspect_ratio_string_to_float(aspect_ratio_str)
        
        # Case 1: User specified both width and height
        if output_width > 0 and output_height > 0:
            return output_width, output_height

        # Case 2: Only width specified, calculate height
        if output_width > 0 and output_height == 0:
            calculated_height = int(output_width / aspect_ratio)
            # Align to multiples of 8
            calculated_height = (calculated_height + 7) // 8 * 8
            return output_width, calculated_height

        # Case 3: Only height specified, calculate width
        if output_height > 0 and output_width == 0:
            calculated_width = int(output_height * aspect_ratio)
            # Align to multiples of 8
            calculated_width = (calculated_width + 7) // 8 * 8
            return calculated_width, output_height

        # Case 4: Neither specified, intelligent calculation
        # Strategy: Use input resolution as baseline, adjust according to aspect ratio

        # Calculate input aspect ratio
        input_aspect = input_width / input_height
        
        if aspect_ratio >= input_aspect:
            # Output is wider, base on width
            final_width = input_width
            final_height = int(input_width / aspect_ratio)
        else:
            # Output is taller, base on height
            final_height = input_height
            final_width = int(final_height * aspect_ratio)

        # Ensure not smaller than common SD resolution
        min_dimension = 512
        if final_width < min_dimension or final_height < min_dimension:
            if aspect_ratio >= 1.0:
                final_width = max(final_width, min_dimension)
                final_height = int(final_width / aspect_ratio)
            else:
                final_height = max(final_height, min_dimension)
                final_width = int(final_height * aspect_ratio)

        # Align to multiples of 8
        final_width = (final_width + 7) // 8 * 8
        final_height = (final_height + 7) // 8 * 8

        return final_width, final_height

    def smooth_position(self, new_pos, prev_pos, smoothing_factor, size_smoothing_factor, max_movement_per_frame):
        """
        Intelligent smoothing algorithm: combines EMA, adaptive speed limits, and subpixel smoothing

        Algorithm features:
        1. Uses Exponential Moving Average (EMA) for basic smoothing
        2. Size changes are smoothed more than position changes to prevent sudden fat/thin effects
        3. Multi-level adaptive speed limits:
           - Extreme changes (>200%): fast response (1.5x)
           - Medium changes (100-200%): normal speed (1.0x)
           - Small changes (10-100%): reduced speed (0.6x)
           - Tiny changes (<10%): greatly reduced (0.3x), eliminates noise
        4. Subpixel smoothing: changes less than 1 pixel are additionally attenuated by 50%, eliminates tiny jitter
        5. Allows faster response for large changes, ultra-smooth for small changes

        Parameters:
        - smoothing_factor: position smoothing coefficient (0-1)
        - size_smoothing_factor: size smoothing coefficient (0-1)
        - max_movement_per_frame: maximum movement speed limit
        """
        if prev_pos is None:
            return new_pos, new_pos
        
        smoothed = {}
        for key in new_pos.keys():
            # 1. Select smoothing coefficient
            if key in ['center_x', 'center_y']:
                current_smoothing = smoothing_factor
                max_change = max_movement_per_frame
            else:
                # Size uses higher smoothing coefficient
                current_smoothing = size_smoothing_factor
                # Size change limits are stricter to prevent sudden fat/thin
                max_change = max_movement_per_frame * 0.8

            # 2. Calculate raw change amount
            raw_change = new_pos[key] - prev_pos[key]

            # 3. Apply exponential moving average
            ema_value = prev_pos[key] * current_smoothing + new_pos[key] * (1 - current_smoothing)
            ema_change = ema_value - prev_pos[key]

            # 4. Adaptive speed limit (optimized version: eliminates tiny jitter)
            # Allow faster response for large changes; smoother for small changes
            change_ratio = abs(raw_change) / max(max_change, 1)

            if change_ratio > 2.0:
                # Extreme changes: increase response speed
                adaptive_factor = 1.5
            elif change_ratio > 1.0:
                # Medium changes: normal speed
                adaptive_factor = 1.0
            elif change_ratio > 0.1:
                # Small changes: reduce speed, increase smoothing
                adaptive_factor = 0.6
            else:
                # Tiny changes (possibly noise): greatly reduce speed, eliminate jitter
                adaptive_factor = 0.3

            effective_max_change = max_change * adaptive_factor

            # 5. Subpixel smoothing: further attenuate tiny changes
            if abs(ema_change) < 1.0:
                # Changes less than 1 pixel, additional smoothing
                ema_change *= 0.5

            # 6. Apply speed limit
            if abs(ema_change) > effective_max_change:
                final_change = effective_max_change if ema_change > 0 else -effective_max_change
            else:
                final_change = ema_change

            smoothed[key] = prev_pos[key] + final_change
        
        return smoothed, smoothed

    def get_target_face(self, dets, tracking_mode, target_face_index, image_shape):
        """
        Select the face to track based on tracking mode
        """
        if len(dets) == 0:
            return None
        
        if tracking_mode == "largest_face":
            # Select the largest face
            largest_idx = 0
            largest_area = 0
            for i, bbox in enumerate(dets):
                x1, y1, x2, y2 = bbox[:4]
                area = (x2 - x1) * (y2 - y1)
                if area > largest_area:
                    largest_area = area
                    largest_idx = i
            return dets[largest_idx]

        elif tracking_mode == "specific_face":
            # Select the face at specified index
            if target_face_index < len(dets):
                return dets[target_face_index]
            else:
                return dets[0]  # Return first if index out of range

        elif tracking_mode == "center_face":
            # Select the face closest to image center
            img_center_x = image_shape[1] / 2
            img_center_y = image_shape[0] / 2
            closest_idx = 0
            min_distance = float('inf')

            for i, bbox in enumerate(dets):
                x1, y1, x2, y2 = bbox[:4]
                face_center_x = (x1 + x2) / 2
                face_center_y = (y1 + y2) / 2
                distance = ((face_center_x - img_center_x) ** 2 +
                           (face_center_y - img_center_y) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    closest_idx = i
            return dets[closest_idx]
        
        return dets[0]

    def calculate_crop_region(self, bbox, scale_factor, shift_factor, aspect_ratio, image_shape):
        """
        Calculate crop region (ensuring it doesn't exceed image boundaries)

        Algorithm:
        1. Calculate ideal crop region based on face position and parameters
        2. Check if it exceeds image boundaries
        3. If exceeded, adjust center point position to ensure crop box is completely within image
        4. Maintain crop box aspect ratio unchanged to avoid distortion
        """
        import math
        
        img_height, img_width = image_shape[0], image_shape[1]

        x1, y1, x2, y2 = map(int, bbox[:4])
        face_width = x2 - x1
        face_height = y2 - y1

        default_area = face_width * face_height
        default_area *= scale_factor
        default_side = math.sqrt(default_area)

        # Calculate new width and height based on aspect ratio
        new_width = int(default_side * math.sqrt(aspect_ratio))
        new_height = int(default_side / math.sqrt(aspect_ratio))

        # Ensure crop box is not larger than image dimensions
        if new_width > img_width or new_height > img_height:
            # Shrink crop box to fit image
            scale_down = min(img_width / new_width, img_height / new_height) * 0.95
            new_width = int(new_width * scale_down)
            new_height = int(new_height * scale_down)

        # Calculate ideal face center coordinates
        center_x = x1 + face_width // 2
        center_y = y1 + face_height // 2 + int(new_height * (0.5 - shift_factor))

        # === Boundary checking and adjustment ===
        # Calculate crop box boundaries
        half_width = new_width // 2
        half_height = new_height // 2

        # X-axis boundary adjustment
        if center_x - half_width < 0:
            # Left boundary exceeded, move right
            center_x = half_width
        elif center_x + half_width > img_width:
            # Right boundary exceeded, move left
            center_x = img_width - half_width

        # Y-axis boundary adjustment
        if center_y - half_height < 0:
            # Top boundary exceeded, move down
            center_y = half_height
        elif center_y + half_height > img_height:
            # Bottom boundary exceeded, move up
            center_y = img_height - half_height
        
        return {
            'center_x': center_x,
            'center_y': center_y,
            'width': new_width,
            'height': new_height
        }

    def crop_and_resize(self, image, crop_region, output_width, output_height):
        """
        Crop and resize based on crop region

        Algorithm improvements:
        1. Use precise crop boundaries (already ensured not to exceed bounds in calculate_crop_region)
        2. Cropped region maintains correct aspect ratio
        3. Prevent distortion caused by boundary restrictions
        """
        center_x = int(crop_region['center_x'])
        center_y = int(crop_region['center_y'])
        width = int(crop_region['width'])
        height = int(crop_region['height'])

        # Calculate crop boundaries (calculate_crop_region has already ensured no bounds exceeded, calculate directly here)
        half_width = width // 2
        half_height = height // 2

        crop_x1 = center_x - half_width
        crop_x2 = center_x + half_width
        crop_y1 = center_y - half_height
        crop_y2 = center_y + half_height

        # Double-check boundary safety (belt and suspenders)
        crop_x1 = max(0, crop_x1)
        crop_y1 = max(0, crop_y1)
        crop_x2 = min(image.shape[1], crop_x2)
        crop_y2 = min(image.shape[0], crop_y2)

        # Crop image
        cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]

        # Verify crop dimensions (for debugging)
        actual_width = crop_x2 - crop_x1
        actual_height = crop_y2 - crop_y1

        # If crop dimensions don't match expected (theoretically shouldn't happen), handle it
        if actual_width != width or actual_height != height:
            # This indicates boundary issues still exist, use padding to maintain aspect ratio
            pad_left = (width - actual_width) // 2
            pad_right = width - actual_width - pad_left
            pad_top = (height - actual_height) // 2
            pad_bottom = height - actual_height - pad_top

            # Use edge pixel padding (replicate edge pixels to avoid black borders)
            cropped = torch.nn.functional.pad(
                cropped.permute(2, 0, 1),  # (C, H, W)
                (pad_left, pad_right, pad_top, pad_bottom),
                mode='replicate'
            ).permute(1, 2, 0)  # Restore (H, W, C)

        # Add batch dimension and resize
        cropped = cropped.unsqueeze(0)
        resized = comfy.utils.common_upscale(
            cropped.movedim(-1, 1),
            output_width,
            output_height,
            'lanczos',
            ""
        ).movedim(1, -1)
        
        return resized

    def ease_in_out_cubic(self, t):
        """
        Easing function: cubic ease in out
        Makes interpolation transitions more natural, avoids mechanical feel of linear interpolation
        """
        if t < 0.5:
            return 4 * t * t * t
        else:
            return 1 - pow(-2 * t + 2, 3) / 2
    
    def interpolate_crop_region(self, region1, region2, alpha):
        """
        Intelligent interpolation: uses easing function for non-linear interpolation

        Algorithm features:
        1. Uses cubic easing function for more natural transitions
        2. Position uses eased interpolation for smoother movement
        3. Size uses linear interpolation for stability

        Parameters:
        - alpha: 0-1, interpolation coefficient
        """
        if region1 is None or region2 is None:
            return region2 if region2 is not None else region1

        # Position uses eased interpolation
        eased_alpha = self.ease_in_out_cubic(alpha)

        return {
            # Position uses eased interpolation for smoother, more natural movement
            'center_x': region1['center_x'] * (1 - eased_alpha) + region2['center_x'] * eased_alpha,
            'center_y': region1['center_y'] * (1 - eased_alpha) + region2['center_y'] * eased_alpha,
            # Size uses linear interpolation for stability
            'width': region1['width'] * (1 - alpha) + region2['width'] * alpha,
            'height': region1['height'] * (1 - alpha) + region2['height'] * alpha,
        }

    def track_faces(self, images, smoothness, responsiveness, detect_interval,
                    aspect_ratio, output_width=0, output_height=0, scale_factor=2.0,
                    shift_factor=0.45):
        """
        Main face tracking function

        Parameter description:
        - smoothness: Smoothness intensity (0-0.95), controls camera movement smoothness
        - responsiveness: Response sensitivity (0.1-1.0), controls reaction speed to face movement
        - detect_interval: Detect every N frames for speed boost
        - aspect_ratio: Output aspect ratio
        - output_width/height: Output resolution, 0 means auto calculate

        Algorithm design:
        1. smoothness controls both position and size smoothing, size smoothing is higher to prevent sudden fat/thin effects
        2. responsiveness controls maximum movement distance per frame, higher sensitivity allows faster movement
        3. Detection intervals maintain smooth transitions through intelligent interpolation
        4. Output resolution automatically calculated intelligently based on input and aspect ratio
        """
        # === Auto calculate output resolution ===
        input_shape = images[0].shape  # (height, width, channels)
        output_width, output_height = self.calculate_output_resolution(
            input_shape, aspect_ratio, output_width, output_height
        )

        # === Parameter mapping algorithm ===

        # 1. Smoothing coefficient: uses Exponential Moving Average (EMA)
        smoothing_factor = smoothness  # position smoothing
        # Size smoothing coefficient is higher to prevent sudden fat/thin and jitter
        # Uses non-linear mapping to make size more stable at high smoothness values
        size_smoothing_factor = min(0.98, smoothness + (1 - smoothness) * 0.5)

        # 2. Response speed: convert sensitivity to maximum movement pixels
        # Base speed range: 20-200 pixels/frame
        # responsiveness=0.1 -> 20px (very slow)
        # responsiveness=0.5 -> 110px (moderate)
        # responsiveness=1.0 -> 200px (very fast)
        base_speed = 20
        max_speed = 200
        max_movement_per_frame = base_speed + (max_speed - base_speed) * responsiveness

        # 3. Detection interval
        detect_every_n_frames = detect_interval

        # 4. Fixed parameters
        target_face_index = 0
        tracking_mode = "largest_face"
        fallback_to_center = True
        
        aspect_ratio_float = self.aspect_ratio_string_to_float(aspect_ratio)

        # Initialize RetinaFace detector
        rf = Pytorch_RetinaFace(
            top_k=50,
            keep_top_k=10,
            device=get_torch_device()
        )

        output_frames = []
        prev_crop_region = None
        last_detected_crop_region = None  # last detected crop region
        next_detected_crop_region = None  # next detected crop region
        tracking_info_list = []

        num_frames = images.shape[0]
        pbar = comfy.utils.ProgressBar(num_frames)

        # === Intelligent keyframe selection ===
        # Strategy: ensure beginning, end, and evenly spaced intermediate frames are detected
        detect_frames = set()

        # 1. Always detect first and last frames
        detect_frames.add(0)
        if num_frames > 1:
            detect_frames.add(num_frames - 1)

        # 2. Add evenly spaced intermediate frames
        for i in range(0, num_frames, detect_every_n_frames):
            detect_frames.add(i)

        # 3. If interval is large, add more keyframes in the middle to ensure quality
        if detect_every_n_frames > 5 and num_frames > 10:
            # Also detect at midpoints of each interval
            for i in range(detect_every_n_frames // 2, num_frames, detect_every_n_frames):
                detect_frames.add(i)

        detect_frames = sorted(list(detect_frames))  # sort

        detected_regions = {}  # store detected regions {frame_index: crop_region}

        # First pass: detect keyframes
        for detect_idx in detect_frames:
            image = images[detect_idx]
            image_255 = image * 255

            # Detect faces
            dets = rf.detect_faces(image_255)
            target_face = self.get_target_face(dets, tracking_mode, target_face_index, image.shape)

            if target_face is not None:
                crop_region = self.calculate_crop_region(
                    target_face,
                    scale_factor,
                    shift_factor,
                    aspect_ratio_float,
                    image.shape
                )
                detected_regions[detect_idx] = crop_region
            elif fallback_to_center:
                # Use image center or previous position
                if len(detected_regions) > 0:
                    last_key = max(detected_regions.keys())
                    detected_regions[detect_idx] = detected_regions[last_key]
                else:
                    detected_regions[detect_idx] = {
                        'center_x': image.shape[1] // 2,
                        'center_y': image.shape[0] // 2,
                        'width': min(image.shape[1], image.shape[0]),
                        'height': min(image.shape[1], image.shape[0])
                    }
        
        # Second pass: process all frames (including interpolation)
        for i in range(num_frames):
            pbar.update(1)
            image = images[i]

            # Determine if this is a detection frame
            if i in detected_regions:
                # Use detected region directly
                crop_region = detected_regions[i]

                # Apply smoothing
                if smoothing_factor > 0 and prev_crop_region is not None:
                    crop_region, _ = self.smooth_position(
                        crop_region,
                        prev_crop_region,
                        smoothing_factor,
                        size_smoothing_factor,
                        max_movement_per_frame
                    )

                tracking_info_list.append(
                    f"Frame {i}: Detected at ({int(crop_region['center_x'])}, {int(crop_region['center_y'])})"
                )
            else:
                # Interpolate between two detection frames
                prev_detect_frame = None
                next_detect_frame = None

                for detect_idx in detect_frames:
                    if detect_idx < i:
                        prev_detect_frame = detect_idx
                    elif detect_idx > i:
                        next_detect_frame = detect_idx
                        break
                
                if prev_detect_frame is not None and next_detect_frame is not None:
                    # Calculate interpolation coefficient
                    alpha = (i - prev_detect_frame) / (next_detect_frame - prev_detect_frame)
                    crop_region = self.interpolate_crop_region(
                        detected_regions[prev_detect_frame],
                        detected_regions[next_detect_frame],
                        alpha
                    )

                    # Apply smoothing to interpolation results too (prevent sudden fat/thin and jitter)
                    if smoothing_factor > 0 and prev_crop_region is not None:
                        crop_region, _ = self.smooth_position(
                            crop_region,
                            prev_crop_region,
                            smoothing_factor * 0.9,  # Interpolated frames use stronger smoothing to eliminate jitter
                            size_smoothing_factor * 0.95,  # Size smoothing remains very high
                            max_movement_per_frame * 0.7  # Interpolated frames have stricter speed limits
                        )

                    tracking_info_list.append(
                        f"Frame {i}: Interpolated ({int(crop_region['center_x'])}, {int(crop_region['center_y'])})"
                    )
                elif prev_detect_frame is not None:
                    # Only previous detection frame available, use it
                    crop_region = detected_regions[prev_detect_frame]

                    # Apply smooth transition even when using previous frame
                    if smoothing_factor > 0 and prev_crop_region is not None:
                        crop_region, _ = self.smooth_position(
                            crop_region,
                            prev_crop_region,
                            smoothing_factor * 0.9,
                            size_smoothing_factor * 0.95,
                            max_movement_per_frame
                        )

                    tracking_info_list.append(
                        f"Frame {i}: Using previous detected frame"
                    )
                else:
                    # Use default position
                    crop_region = {
                        'center_x': image.shape[1] // 2,
                        'center_y': image.shape[0] // 2,
                        'width': min(image.shape[1], image.shape[0]),
                        'height': min(image.shape[1], image.shape[0])
                    }
                    tracking_info_list.append(f"Frame {i}: Using default center")
            
            prev_crop_region = crop_region

            # Crop and resize
            cropped_frame = self.crop_and_resize(image, crop_region, output_width, output_height)
            output_frames.append(cropped_frame)

        # Combine all frames
        output_video = torch.cat(output_frames, dim=0)

        # === Generate tracking report ===
        detection_count = len(detected_regions)
        interpolated_count = num_frames - detection_count

        tracking_summary = "=" * 60 + "\n"
        tracking_summary += "  Face Tracking Report\n"
        tracking_summary += "=" * 60 + "\n\n"

        # Statistics
        tracking_summary += "📊 Statistics:\n"
        tracking_summary += f"  · Total frames: {num_frames}\n"
        tracking_summary += f"  · Detection frames: {detection_count} ({detection_count*100//num_frames}%)\n"
        tracking_summary += f"  · Interpolated frames: {interpolated_count} ({interpolated_count*100//num_frames}%)\n"
        tracking_summary += f"  · Detection interval: every {detect_every_n_frames} frames\n"
        tracking_summary += f"  · Speed boost: ~{detect_every_n_frames}x\n\n"

        # Parameter settings
        tracking_summary += "⚙️ Parameter Settings:\n"
        tracking_summary += f"  · Smoothness intensity: {smoothness:.2f}\n"
        tracking_summary += f"  · Response sensitivity: {responsiveness:.2f}\n"
        tracking_summary += f"  · Max movement speed: {max_movement_per_frame:.1f} px/frame\n"
        tracking_summary += f"  · Input size: {input_shape[1]}x{input_shape[0]}\n"
        tracking_summary += f"  · Output size: {output_width}x{output_height} (auto calculated)\n"
        tracking_summary += f"  · Aspect ratio: {aspect_ratio}\n\n"

        # Detailed log (show first 20 frames only)
        tracking_summary += "📝 Detailed Log (first 20 frames):\n"
        tracking_summary += "-" * 60 + "\n"
        for i, info in enumerate(tracking_info_list[:20]):
            tracking_summary += f"  {info}\n"

        if len(tracking_info_list) > 20:
            tracking_summary += f"\n  ... (omitted {len(tracking_info_list) - 20} frames)\n"

        tracking_summary += "\n" + "=" * 60 + "\n"
        tracking_summary += "✅ Tracking completed!\n"
        tracking_summary += "=" * 60
        
        return (output_video, tracking_summary)


NODE_CLASS_MAPPINGS = {
    "AutoCropFaces": AutoCropFaces,
    "AutoTrackFaces": AutoTrackFaces
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoCropFaces": "Auto Crop Faces",
    "AutoTrackFaces": "Auto Track Faces (Video)"
}
