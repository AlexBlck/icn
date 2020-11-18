from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


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


def concat_h(im1, im2):
    r = im1.height / im2.height
    im2 = im2.resize((int(r * im2.width), im1.height), Image.BICUBIC)
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def concat_v(im1, im2):
    r = im1.width / im2.width
    im2 = im2.resize((im1.width, int(r * im2.height)), Image.BICUBIC)
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def unnormalise(y):
    # assuming x and y are Batch x 3 x H x W and ,
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = y.new(*y.size())
    x[0, :, :] = y[0, :, :] * std[0] + mean[0]
    x[1, :, :] = y[1, :, :] * std[1] + mean[1]
    x[2, :, :] = y[2, :, :] * std[2] + mean[2]

    return x


def mask_processing(x):
    if x > 90:
        return 140
    elif x < 80:
        return 0
    else:
        return 255


def grid_to_heatmap(grid, cmap='jet'):
    # TODO: pad grid with zeros to remove side stickiness ?

    mask = TF.to_pil_image(grid.view(7, 7))
    mask = mask.resize((1024, 1024), Image.BICUBIC)
    mask = Image.eval(mask, mask_processing)

    # Heatmap
    colormap = plt.get_cmap(cmap)
    heatmap = np.array(colormap(mask))
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = Image.fromarray(heatmap)

    return heatmap, mask


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
    heatmap, mask = grid_to_heatmap(target, cmap='winter')
    img1.paste(heatmap, (0, 0), mask)

    # Heatmap of prediction
    heatmap, mask = grid_to_heatmap(prediction, cmap='Wistia')
    img1.paste(heatmap, (0, 0), mask)

    return img1


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

    return intersection / union
