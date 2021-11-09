from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from PIL import ImageChops
import numpy as np
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import torch


def text_on_img(img, text, size=24, pos=[0, 0], col=(0, 0, 0), center=False):
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("timr45w.ttf", size)

    w, h = draw.textsize(text, font=font)
    if center:
        pos[1] -= h//2
    draw.text(pos, text, col, font=font)


def rect_on_img(img, fill, outline, pos, width=1):
    draw = ImageDraw.Draw(img)
    draw.rectangle(pos, fill=fill, outline=outline, width=width)


def convert_annotation(org_size, pho_size, m_size, ann):
    """convert mturk annotation to bounding box matching original size
    org_size, pho_size, m_size are size of original, photoshop images in PSBattles and joint image shown to turker.
        Each is a tuple of (x,y) format corresponding to width and height.
    ann is mturk bounding box annotation, a tuple of format (x, y, w, h) correspond to (left, top, width, height)

    Output: (ann_out, which) where ann_out is annotation wrt natural size, which is 1 if original image is annotated,
        otherwise 0
    """
    ## forward pass
    wo, ho = org_size
    wp, hp = pho_size
    # phase 1: resize photoshop
    r = ho / hp
    wp1, hp1 = int(r * wp), ho
    # phase 2: concat
    w2, h2 = wo + wp1, ho
    # phase 3: resize to fixed width of 800
    g = 800 / w2
    w3, h3 = 800, int(h2 * g)
    # phase 4: pad text
    w4, h4 = w3, h3 + 20

    ## backward pass
    wa, ha = m_size  # mturk image size, should directly correspond to w4,h4
    # back phase 4
    a4 = (ann[0] * w4 / wa, ann[1] * h4 / ha, ann[2] * w4 / wa, ann[3] * h4 / ha)
    # back phase 3
    top = min(a4[1], h3)
    a3 = (a4[0], top, a4[2], min(a4[3] + top, h3) - top)
    # back phase 2
    a2 = (a3[0] * w2 / w3, a3[1] * h2 / h3, a3[2] * w2 / w3, a3[3] * h2 / h3)
    # back phase 1
    if a2[0] > wo:  # annotation on photoshop
        a1 = (a2[0] - wo, a2[1], a2[2], a2[3])
        which = 0
        ann_out = (a1[0] / wp1, a1[1] / hp1, a1[2] / wp1, a1[3] / hp1)
    else:  # annotation on original
        # print('Rare case: annotation on original')
        # if not a2[0] + a2[2] <= wo:
        #     print('Bounding box covers both images')
        ann_out = a2
        which = 1
    return ann_out, which


def concat_h(im1, im2, mode=Image.BICUBIC):
    r = im1.height / im2.height
    im2 = im2.resize((int(r * im2.width), im1.height), mode)
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def concat_v(im1, im2, mode=Image.BICUBIC):
    r = im1.width / im2.width
    im2 = im2.resize((im1.width, int(r * im2.height)), mode)
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def unnormalise(y):
    # assuming x and y are Batch x 3 x H x W
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = y.new(*y.size())
    x[0, :, :] = y[0, :, :] * std[0] + mean[0]
    x[1, :, :] = y[1, :, :] * std[1] + mean[1]
    x[2, :, :] = y[2, :, :] * std[2] + mean[2]

    x = x - x.min()
    x = x / x.max()
    return x


def mask_processing(x, use_t=True):
    if use_t:
        if x > 90:
            return 140
        elif x < 80:
            return 0
        else:
            return 255
    return x


def grid_to_heatmap(grid, cmap='jet', size=1024):
    # TODO: pad grid with zeros to remove side stickiness ?

    mask = TF.to_pil_image(grid.view(7, 7))
    mask = mask.resize((size, size), Image.BICUBIC)
    mask = Image.eval(mask, mask_processing)

    # Heatmap
    colormap = plt.get_cmap(cmap)
    heatmap = np.array(colormap(mask))
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = Image.fromarray(heatmap)

    return heatmap, mask


def grayscale_to_heatmap(img, cmap='jet'):
    colormap = plt.get_cmap(cmap)
    heatmap = np.array(colormap(img))
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = Image.fromarray(heatmap)

    return heatmap


def make_pred_bar(preds):
    img = Image.new('RGB', (1024, 1024), color='white')
    annotations = Image.new('RGB', (1024, 50), color='white')
    cols = [(89, 117, 164), (204, 137, 99), (95, 158, 110)]
    cols_prob = [(185, 197, 219), (239, 206, 187), (188, 227, 197)]
    cls = ('Original', 'Manip.', 'Distinct')
    w = 200
    font_size = 120

    for y in np.arange(0, 1.01, 0.25):
        f = 1
        if y == 1:
            f *= -1
        rect_on_img(img, fill=(204, 204, 204), outline=None,
                    pos=[int(1024*y), 0, int(1024*y) + 5*f, 1024])

    offset = 20  # (1024 - (2*341 + w))//2
    for i, prob in enumerate(preds):
        xpos = i*341 + offset
        rect_on_img(img, fill=cols[i], outline=None, pos=[0, xpos, int(1024 * prob), xpos + w])
        text_on_img(img, cls[i], pos=[10, xpos + w], size=font_size, col=(40, 40, 40))

        if prob > 0.78:
            text_on_img(img, f'{prob * 100:.00f}%', pos=[max(10, int(1024 * prob) - 215), xpos + w//2 - 10], size=font_size, col=cols_prob[i], center=True)
        else:
            text_on_img(img, f'{prob * 100:.00f}%', pos=[int(1024 * prob) + 10, xpos + w // 2 - 10], size=font_size,
                        col=(140, 140, 140), center=True)

    # img = img.rotate(-90)
    # for i in range(3):
    #     xpos = i * 341 + offset
    #
    # img.show()
    return img


make_pred_bar(np.array([0.19, 0.2, 0.22]))


def summary_image(img, target, prediction):
    prediction -= prediction.min()
    prediction = prediction / prediction.max()
    size = 1024
    # Heatmap of prediction
    img1 = unnormalise(img[:, 224:, :224])
    img1 = TF.to_pil_image(img1).resize((size, size))
    heatmap, mask = grid_to_heatmap(prediction.view(7, 7).detach().numpy())
    img1.paste(heatmap, (0, 0), mask)

    # Heatmap of target
    img2 = unnormalise(img[:, :224, :224])
    img2 = TF.to_pil_image(img2).resize((size, size))
    heatmap, mask = grid_to_heatmap(target.view(7, 7).detach().numpy())
    img2.paste(heatmap, (0, 0), mask)

    target = TF.to_pil_image(target.view(7, 7))
    prediction = TF.to_pil_image(prediction.view(7, 7))

    col2 = concat_v(img1, img2)
    col1 = concat_v(prediction.resize((size, size), Image.NEAREST), target.resize((size, size), Image.NEAREST))
    full = concat_h(col1, col2)
    return full


def short_summary_image(img, target, prediction):
    prediction -= prediction.min()
    prediction = prediction / prediction.max()

    size = 1024

    # Photoshopped image
    img1 = unnormalise(img[3:, :, :])
    img1 = TF.to_pil_image(img1).resize((size, size))

    # Heatmap of target
    heatmap, mask = grid_to_heatmap(target, cmap='Wistia')
    img1.paste(heatmap, (0, 0), mask)

    # Heatmap of prediction
    heatmap, mask = grid_to_heatmap(prediction, cmap='winter')
    img1.paste(heatmap, (0, 0), mask)

    return img1


def short_summary_image_three(img, target, prediction, pho_clean):
    prediction -= prediction.min()
    prediction = prediction / prediction.max()

    size = 1024

    pho_clean = TF.to_pil_image(unnormalise(pho_clean)).resize((size, size))

    # Photoshopped image
    pho_noisy = unnormalise(img[3:, :, :])
    pho_noisy = TF.to_pil_image(pho_noisy).resize((size, size))

    org = unnormalise(img[:3, :, :])
    org = TF.to_pil_image(org).resize((size, size))

    # Heatmap of prediction
    heatmap, mask = grid_to_heatmap(prediction, cmap='winter', size=size)
    pho_clean.paste(heatmap, (0, 0), mask)

    # Heatmap of target
    heatmap, mask = grid_to_heatmap(target, cmap='Wistia')
    pho_clean.paste(heatmap, (0, 0), mask)



    full = concat_h(org, pho_noisy)
    full = concat_h(full, pho_clean)

    return full


def stn_summary_image(img, target, fake, prediction):
    prediction -= prediction.min()
    prediction = prediction / prediction.max()

    size = 1024

    # Photoshopped image
    img1 = unnormalise(img[:3, :, :])
    img1 = TF.to_pil_image(img1).resize((size, size))

    img2 = unnormalise(img[3:, :, :])
    img2 = TF.to_pil_image(img2).resize((size, size))

    img3 = unnormalise(fake)
    img3 = TF.to_pil_image(img3).resize((size, size))

    # Heatmap of target
    heatmap, mask = grid_to_heatmap(target, cmap='Wistia')
    img3.paste(heatmap, (0, 0), mask)

    # Heatmap of prediction
    heatmap, mask = grid_to_heatmap(prediction, cmap='winter')
    img3.paste(heatmap, (0, 0), mask)

    full = concat_h(img1, img2)
    full = concat_h(full, img3)

    return full


def stn_only_summary_image(img, fake):
    size = 1024

    # Photoshopped image
    img1 = unnormalise(img[:3, :, :])
    img1 = TF.to_pil_image(img1).resize((size, size))

    img2 = unnormalise(img[3:, :, :])
    img2 = TF.to_pil_image(img2).resize((size, size))

    img3 = unnormalise(fake)
    img3 = TF.to_pil_image(img3).resize((size, size))

    full = concat_h(img1, img2)
    full = concat_h(full, img3)

    return full


def dewarper_summary_image(dewarped_mask, img, warped, dewarped):
    size = 512

    img0 = unnormalise(img)
    img0 = TF.to_pil_image(img0).resize((size, size))

    img1 = unnormalise(warped)
    img1 = TF.to_pil_image(img1).resize((size, size))

    img2 = unnormalise(dewarped)
    img2 = TF.to_pil_image(img2).resize((size, size))

    img3 = TF.to_pil_image(torch.abs(unnormalise(img * dewarped_mask[0]) - unnormalise(dewarped * dewarped_mask[0]))).resize((size, size)).convert('L')
    img3 = grayscale_to_heatmap(img3)

    text_on_img(img0, 'Original')
    text_on_img(img1, 'Warped')
    text_on_img(img2, 'Dewarped')
    text_on_img(img3, 'Difference', col=(255, 255, 255))

    top = concat_h(img0, img1)
    bot = concat_h(img2, img3)

    full = concat_v(top, bot)

    return full





def grid_to_binary(grid):
    mask = TF.to_pil_image(grid.view(7, 7))
    mask = mask.resize((1024, 1024), Image.BICUBIC)
    mask = Image.eval(mask, lambda x: 255 if x > 80 else 0)

    return mask


def heatmap_iou(target, prediction):
    prediction -= prediction.min()
    prediction = prediction / prediction.max()

    binary_mask_target = grid_to_binary(target)
    binary_mask_pred = grid_to_binary(prediction)

    intersection = np.count_nonzero(np.logical_and(binary_mask_target, binary_mask_pred))
    union = np.count_nonzero(np.logical_or(binary_mask_target, binary_mask_pred))

    if union == 0:
        return 0
    return intersection / union


def heatmap_emptiness(prediction):
    binary_mask_pred = grid_to_binary(prediction)

    return 1 - np.count_nonzero(binary_mask_pred) / np.array(binary_mask_pred).size
