
<div align='center'><h1> pytorch-anime-style-transfer </h1></div>

An educational project to implement image-to-image translation for anime style transfer using GAN models.

# Main goal

The goal of this project is to implement and understand how style transfer works [1], [2]. After that, aplied it to:

**Anime sketches colorization:** 

![Anime sketches colorization](/assets/pix2pix.gif)
- Paired dataset: [Anime Sketch Colorization (Pair)](https://www.kaggle.com/datasets/ktaebum/anime-sketch-colorization-pair)
- Result [`notebooks/pix2pix_colorization.ipynb`](./notebooks/pix2pix_colorization.ipynb)

**Human faces to anime faces style transfer:**
![cycle gan train loop](/assets/cycle_gan_train_loop.gif) 

- unpaired dataset (a combination of two datasets): 
    - Human faces: [Faces Dataset (Small)](https://www.kaggle.com/datasets/tommykamaz/faces-dataset-small)
    - Anime faces: [High-Resolution Anime Face Dataset (512x512)](https://www.kaggle.com/datasets/subinium/highresolution-anime-face-dataset-512x512)

- There are some examples of style transfer and code for the train [`notebooks/cycle_gan.ipynb`](./notebooks/cycle_gan.ipynb)

# How to use

1. **Clone the repository**

   ```bash
   git clone https://github.com/ThViviani/pytorch-anime-style-transfer.git
   cd pytorch-anime-style-transfer


2. (Optional) Create and activate a virtual environment:
    ```bash 
    python -m venv venv
    source venv/bin/activate  # for Linux/macOS
    venv\Scripts\activate     # for Windows


2. **Installation**

    ```bash
    pip install -r requirements.txt

3. **Human face to anime face transfer**

   Run inference on a photo:

   ```bash
   python inference.py path/to/your/photo.jpg path/to/result/
- path/to/your/photo.jpg — required, path to the input image
- path/to/result/ — optional, path to the output directory (default: ./)

## Checkpoints

You can download the pretrained checkpoints from the 
- [Pix2pix checkpoint](https://huggingface.co/ThViviani/pix2pix_for_colorization_anime_sketches/tree/main)
- [cycle gan checkpoint](https://huggingface.co/ThViviani/cycle_gan_for_anime2human_style_transfer)

How to do it see to get_last_checkpoint method in the [here](./inference.py)


## References

[1] Isola P., Zhu J.-Y., Zhou T., Efros A.A. Image-to-Image Translation with Conditional Adversarial Networks. URL: https://doi.org/10.48550/arXiv.1611.07004

[2] Zhu J.-Y., Park T., Isola P., Efros A.A. Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. URL: https://doi.org/10.48550/arXiv.1703.10593

---

*Last updated: August 8, 2025*
