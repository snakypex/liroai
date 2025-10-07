import math
import torch
import torch.nn.functional as F

class ResolutionFromImagePreset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (["480p", "540p", "720p"],),
            },
            "optional": {
                "round_to": ("INT", {"default": 8, "min": 1, "max": 512, "step": 1}),
                "strict_preset": ("BOOL", {"default": False}),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "calc"
    CATEGORY = "utils/aspect"

    @staticmethod
    def _round_to_multiple(x: int, m: int) -> int:
        if m <= 1:
            return int(x)
        return max(m, int(round(x / m) * m))

    def calc(self, image, preset, round_to=8, strict_preset=False):
        if image is None:
            raise ValueError("image is required")
        if len(image.shape) != 4:
            raise ValueError(f"Unexpected image shape: {tuple(image.shape)}; expected [B,H,W,C]")

        if strict_preset:
            if preset == "480p":
                return (854, 480)
            elif preset == "540p":
                return (960, 540)
            else:
                return (1280, 720)

        _, src_h, src_w, _ = image.shape
        tgt_h = 480 if preset == "480p" else 540 if preset == "540p" else 720

        scale = tgt_h / float(src_h)
        tgt_w_f = src_w * scale

        tgt_w = self._round_to_multiple(int(round(tgt_w_f)), round_to)
        tgt_h = self._round_to_multiple(int(round(tgt_h)), round_to)

        return (tgt_w, tgt_h)


class ResizeToPresetKeepAR:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (["480p", "540p", "720p"],),
            },
            "optional": {
                "strategy": (["video_mode", "long_side", "height", "width"], {"default": "video_mode"}),
                "round_to": ("INT", {"default": 8, "min": 1, "max": 512, "step": 1}),
                "no_upscale": ("BOOL", {"default": False}),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "resize"
    CATEGORY = "image/resize"

    @staticmethod
    def _round_to_multiple(x: float, m: int) -> int:
        if m <= 1:
            return max(1, int(round(x)))
        return max(m, int(round(x / m) * m))

    def resize(self, image, preset, strategy="video_mode", round_to=8, no_upscale=False):
        if image is None:
            raise ValueError("image is required")
        if len(image.shape) != 4:
            raise ValueError(f"Unexpected image shape: {tuple(image.shape)}; expected [B,H,W,C]")

        B, src_h, src_w, C = image.shape
        tgt = 480 if preset == "480p" else 540 if preset == "540p" else 720

        if strategy == "height":
            scale = tgt / float(src_h)
            new_h = tgt
            new_w = int(round(src_w * scale))
        elif strategy == "width":
            scale = tgt / float(src_w)
            new_w = tgt
            new_h = int(round(src_h * scale))
        elif strategy == "long_side":
            long_side = max(src_h, src_w)
            scale = tgt / float(long_side)
            new_w = int(round(src_w * scale))
            new_h = int(round(src_h * scale))
        else:  # video_mode
            if src_h > src_w:
                scale = tgt / float(src_w)
                new_w = tgt
                new_h = int(round(src_h * scale))
            elif src_w > src_h:
                scale = tgt / float(src_h)
                new_h = tgt
                new_w = int(round(src_w * scale))
            else:
                scale = tgt / float(src_w)
                new_w = tgt
                new_h = int(round(src_h * scale))

        if no_upscale:
            scale_limit = min(1.0, scale)
            if scale_limit != scale:
                new_w = int(round(src_w * scale_limit))
                new_h = int(round(src_h * scale_limit))

        new_w = self._round_to_multiple(new_w, round_to)
        new_h = self._round_to_multiple(new_h, round_to)

        new_w = max(1, new_w)
        new_h = max(1, new_h)

        return (new_w, new_h)


NODE_CLASS_MAPPINGS = {
    "ResolutionFromImagePreset": ResolutionFromImagePreset,
    "ResizeToPresetKeepAR": ResizeToPresetKeepAR,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ResolutionFromImagePreset": "Resolution From Image (480p/540p/720p)",
    "ResizeToPresetKeepAR": "Resize To 480p/540p/720p (Keep AR)",
}
