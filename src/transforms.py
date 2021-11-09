import torch
from imagenet_c import corrupt, corruption_dict
import random
import numpy as np
import augly.image as imaugs
import augly.utils
import os


class Augly(torch.nn.Module):
    aug_names = (
        "RandomBlur",
        "RandomEmojiOverlay",
        "RandomNoise",
        "RandomPixelization",
        "RandomRotation",
        "VFlip",
        "HFlip",
        "ColorJitter",
        "OverlayImage",
        "Crop",
        "EncodingQuality",
        "Grayscale",
        "OverlayStripes",
        # "OverlayOntoScreenshot",
        # "OverlayText",
        "Pad",
        "PerspectiveTransform",
        "Sharpen",
        "ShufflePixels"
    )

    def __init__(self):
        super().__init__()
        print("Augly initialized")

    def __len__(self):
        return len(self.aug_names)

    def forward(self, img, rand_img=None, aug_name_or_id=None, combine=None):
        crop_x1 = random.uniform(0.0, 0.25)
        crop_y1 = random.uniform(0.0, 0.25)
        crop_x2 = crop_x1 + random.uniform(0.5, 0.75)
        crop_y2 = crop_y1 + random.uniform(0.5, 0.75)
        self.augs = {
            "RandomBlur": imaugs.RandomBlur(),
            "RandomEmojiOverlay": imaugs.RandomEmojiOverlay(emoji_size=random.uniform(0.15, 0.45),
                                                            x_pos=random.uniform(0.1, 0.9),
                                                            y_pos=random.uniform(0.1, 0.9)),
            "RandomNoise": imaugs.RandomNoise(var=random.uniform(0.01, 0.05)),
            "RandomPixelization": imaugs.RandomPixelization(max_ratio=0.8),
            "RandomRotation": imaugs.RandomRotation(),
            "VFlip": imaugs.VFlip(),
            "HFlip": imaugs.HFlip(),
            "ColorJitter": imaugs.ColorJitter(brightness_factor=random.uniform(0.2, 2.0),
                                              contrast_factor=random.uniform(0.2, 2.0),
                                              saturation_factor=random.uniform(0.2, 2.0)),
            "OverlayImage": imaugs.OverlayImage(
                overlay=rand_img,
                overlay_size=random.uniform(0.1, 1.0),
                x_pos=random.uniform(0.1, 0.9),
                y_pos=random.uniform(0.1, 0.9),
                opacity=min(1.0, random.uniform(0.5, 1.2))
            ),
            "Crop": imaugs.Compose([imaugs.Crop(crop_x1, crop_y1, crop_x2, crop_y2),
                                    imaugs.OneOf([imaugs.PadSquare(color=tuple(np.random.choice(range(256), size=3))),
                                                  imaugs.Resize(width=224, height=224)])]),
            "EncodingQuality": imaugs.EncodingQuality(random.randint(0, 20)),
            "Grayscale": imaugs.Grayscale(),
            "OverlayStripes": imaugs.OverlayStripes(
                line_width=random.uniform(0.01, 0.5),
                line_color=tuple(np.random.choice(range(256), size=3)),
                line_angle=random.randint(0, 180),
                line_density=random.uniform(0.1, 0.9),
                line_type=np.random.choice(('dotted', 'dashed', 'solid')),
                line_opacity=min(1.0, random.uniform(0.5, 1.2))
            ),
            "OverlayOntoScreenshot": imaugs.Compose([imaugs.OverlayOntoScreenshot(
                os.path.join(augly.utils.SCREENSHOT_TEMPLATES_DIR,
                             np.random.choice(('web.png', 'mobile.png')))),
                imaugs.OneOf([imaugs.PadSquare(color=tuple(np.random.choice(range(256), size=3))),
                              imaugs.Resize(width=224, height=224)])]),

            "OverlayText": imaugs.OverlayText(text=list(np.random.randint(0, 100, size=(random.randint(2, 20),))),
                                              font_file=np.random.choice((augly.utils.FONT_PATH,
                                                                          augly.utils.MEME_DEFAULT_FONT)),
                                              font_size=random.uniform(0.1, 0.8),
                                              opacity=min(1.0, random.uniform(0.5, 1.2)),
                                              color=tuple(np.random.choice(range(256), size=3)),
                                              x_pos=random.uniform(0.01, 0.4),
                                              y_pos=random.uniform(0.01, 0.75)),
            "Pad": imaugs.Compose([imaugs.Pad(w_factor=random.uniform(0, 0.25),
                                              h_factor=random.uniform(0, 0.25),
                                              color=tuple(np.random.choice(range(256), size=3))),
                                   imaugs.Resize(width=224, height=224)]),
            "PerspectiveTransform": imaugs.PerspectiveTransform(
                sigma=random.randint(10, 25),
                dx=random.uniform(0.0, 1.0),
                dy=random.uniform(0.0, 1.0),
                seed=random.randint(0, 1000)
            ),
            "Sharpen": imaugs.Sharpen(factor=random.uniform(1.0, 20.0)),
            "ShufflePixels": imaugs.ShufflePixels(factor=random.uniform(0.1, 0.4),
                                                  seed=random.randint(0, 1000))
        }

        if aug_name_or_id is not None:
            if type(aug_name_or_id) is int:
                aug = self.augs[self.aug_names[aug_name_or_id]]
            else:
                aug = self.augs[aug_name_or_id]
        else:
            if combine is None:
                combine = random.randint(1, 5)
            choices = np.random.choice(self.aug_names, size=combine)
            aug = augly.image.Compose([self.augs[k] for k in choices])

        return aug(img).convert('RGB').resize((224, 224))


class ImagenetC(torch.nn.Module):
    methods = [x for x in list(corruption_dict.keys()) if x not in ['elastic_transform', 'glass_blur']]

    def __len__(self):
        return len(self.methods)

    def __init__(self, max_sev=2, method_id=None):
        np.random.seed(42)
        random.seed(42)
        super().__init__()
        self.max_sev = max_sev
        self.random_choice = method_id is None
        self.method_id = method_id

    def forward(self, x):
        if self.random_choice:
            return self.random_forward(x)
        else:
            return self.method_forward(x)

    def random_forward(self, x):
        method = random.choice(self.methods)
        severity = random.randint(0, self.max_sev)

        x = np.array(x)
        if random.random() > 0.5:
            x = corrupt(x, severity, method)
        return x

    def method_forward(self, x):
        method = self.methods[self.method_id]
        severity = random.randint(1, self.max_sev)

        x = np.array(x)
        x = corrupt(x, severity, method)
        return x


