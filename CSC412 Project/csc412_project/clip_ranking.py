import torch
import clip
import numpy as np


class CLIP:
    """
    A text-label ranking system using CLIP developed by Open.ai.
    For paper please check: https://arxiv.org/pdf/2103.00020.pdf
    For official codebase please check: https://github.com/openai/CLIP
    """
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def get_score(self, img, label):
        """
        Return the score of the image corresponding to label according to the
        pretrained CLIP model. 
        """
        image = self.preprocess(img).unsqueeze(0).to(self.device)
        text = clip.tokenize([label]).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)

            logits_per_image, logits_per_text = self.model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        return logits_per_image.cpu().numpy()[0, 0]
