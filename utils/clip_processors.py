from transformers import CLIPImageProcessor


def get_clip_image_processor(model_name="openai/clip-vit-base-patch32"):
    """
    CLIP image processor (handles resize, crop, normalization)
    """
    return CLIPImageProcessor.from_pretrained(model_name)
