from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import cv2
import matplotlib.pyplot as plt

def draw_bbox(img, bbox, hh, ww, col=(232, 32, 221), width=4):
    ww /= 2
    hh -= 20
    x, y, w, h = bbox['left'], bbox['top'], bbox['width'], bbox['height']
    x = int((x-400) / ww * img.shape[1])
    w = int(w / ww * img.shape[1])
    y = int(y / hh * img.shape[0])
    h = int(h / hh * img.shape[0])
    cv2.rectangle(img, (x, y), (x + w, y + h), col, width)


def imshow(img):
    while True:
        cv2.imshow('img', img)
        if cv2.waitKey(33) == ord('q'):
            break
    cv2.destroyAllWindows()


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
    im2 = im2.resize((int(r * im2.width), im1.height), Image.NEAREST)
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def concat_v(im1, im2):
    r = im1.width / im2.width
    im2 = im2.resize((im1.width, int(r * im2.height)), Image.NEAREST)
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


def grid_to_heatmap(grid):
    # TODO: pad grid with zeros to remove side stickiness
    # mask = cv2.resize(grid, (1024, 1024), interpolation=cv2.INTER_CUBIC)[int(1024 / 9):int(8 * 1024 / 9),
    #        int(1024 / 9):int(8 * 1024 / 9)]
    mask = cv2.resize(grid, (1024, 1024), interpolation=cv2.INTER_CUBIC)

    # Heatmap
    colormap = plt.get_cmap('jet')
    heatmap = colormap(mask)
    #heatmap /= np.max(heatmap)

    # Convert to PIL
    mask = np.clip((mask * 255), 0, 255).astype(np.uint8)
    mask[mask > 200] = 200
    mask = Image.fromarray(mask)

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
    img2 = unnormalise(img[:, 224:, :224])
    img2 = TF.to_pil_image(img2).resize((size, size))
    heatmap, mask = grid_to_heatmap(target.view(7, 7).detach().numpy())
    img2.paste(heatmap, (0, 0), mask)

    target = TF.to_pil_image(target.view(7, 7))
    prediction = TF.to_pil_image(prediction.view(7, 7))

    col2 = concat_v(img1, img2)
    col1 = concat_v(prediction.resize((size, size), Image.NEAREST), target.resize((size, size), Image.NEAREST))
    full = concat_h(col1, col2)
    return full
