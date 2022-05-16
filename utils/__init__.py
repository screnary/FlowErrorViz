from .flow_wrapping import (Warper2d, load_img, h5_reader, img2tensor, 
                            tensor2img, load_img_and_resize)
from .viz import overlay_results
from .flow_viz import flow_to_color



def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
def is_image_file(filename: str):
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


__all__ = ['Warper2d', 'load_img', 'h5_reader', 'img2tensor', 'load_img_and_resize',
           'tensor2img', 'overlay_results', 'flow_to_color']
