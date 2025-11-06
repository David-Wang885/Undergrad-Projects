import io
import requests
import PIL
from PIL import Image

import torch
import torch.nn.functional as F

import torchvision.transforms as T
import torchvision.transforms.functional as TF

from dall_e import map_pixels, unmap_pixels, load_model
# from IPython.display import display, display_markdown

target_image_size = 256


def open_image(file):
    return PIL.Image.open(file)


def download_image(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return PIL.Image.open(io.BytesIO(resp.content))


def preprocess(img):
    s = min(img.size)

    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')

    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return map_pixels(img)


if __name__ == '__main__':
    dev = torch.device('cpu')

    enc = load_model("https://cdn.openai.com/dall-e/encoder.pkl", dev)
    dec = load_model("https://cdn.openai.com/dall-e/decoder.pkl", dev)

    x = preprocess(download_image('https://assets.bwbx.io/images/users/iqjWHBFdfxIU/iKIWgaiJUtss/v2/1000x-1.jpg'))
    original_image = T.ToPILImage(mode='RGB')(x[0])
    # original_image.show()
    # display_markdown('Original image:')
    # display(original_image)

    z_logits = enc(x)
    z = torch.argmax(z_logits, axis=1)
    z = F.one_hot(z, num_classes=enc.vocab_size).permute(0, 3, 1, 2).float()

    x_stats = dec(z).float()
    x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
    x_rec = T.ToPILImage(mode='RGB')(x_rec[0])
    # x_rec.show()
    # display_markdown('Reconstructed image:')
    # display(x_rec)

