# 🖼️ pic2words — Image Captioning with CNN + RNN

A deep learning pipeline that generates natural language captions for images using an **Inception V3 encoder** and an **LSTM-based decoder**, trained on the **Flickr8k** dataset.

---

## 📌 Overview

This project implements the classic **encoder-decoder** architecture for image captioning:

- **Encoder**: A pretrained `Inception V3` CNN extracts visual features from the input image.
- **Decoder**: An LSTM-based RNN generates a caption word-by-word, conditioned on the encoded features.

---

## 🗂️ Project Structure

```
pic2words/
├── test.py        # Caption generation for new images
├── model.py          # EncoderCNN, DecoderRNN, CNNtoRNN
├── train.py          # Training loop
├── utils.py          # Helper functions (save/load checkpoints)
├── get_loader.py      # Custom PyTorch Dataset + vocabulary builder
├── flickr8k/         # Dataset directory (see below)
│   ├── Images/
│   └── captions.txt
└── README.md
```

---

## 📦 Dataset — Flickr8k

**Flickr8k** is a benchmark dataset for image captioning, containing:

| Property | Details |
|---|---|
| Images | 8,091 photographs sourced from Flickr |
| Captions | 5 human-annotated captions per image (40,455 total) |
| Split | 6,000 train / 1,000 validation / 1,000 test |
| Image types | Everyday activities, animals, outdoor scenes |

Each image is paired with five independently written captions, providing diverse natural language descriptions of the same scene.

---

## ⬇️ Downloading the Flickr8k Dataset

### Option 1 — Kaggle (Recommended)

The easiest way to get the dataset is via Kaggle.

**Step 1**: Install the Kaggle CLI (if not already installed)
```bash
pip install kaggle
```

**Step 2**: Set up your Kaggle API key
- Go to [https://www.kaggle.com/settings](https://www.kaggle.com/settings)
- Scroll to the **API** section and click **Create New Token**
- This downloads a `kaggle.json` file — place it at:
  - **Linux/macOS**: `~/.kaggle/kaggle.json`
  - **Windows**: `C:\Users\<YourUsername>\.kaggle\kaggle.json`

**Step 3**: Download the dataset
```bash
kaggle datasets download -d adityajn105/flickr8k
unzip flickr8k.zip -d flickr8k/
```

**Step 4**: Verify the folder structure
```
flickr8k/
├── Images/          # 8,091 .jpg images
└── captions.txt     # Image filenames + 5 captions each
```

---

### Option 2 — Direct Download (University of Illinois)

You can request access to the original dataset from the authors:

1. Visit: [https://forms.illinois.edu/sec/1713398](https://forms.illinois.edu/sec/1713398)
2. Fill out the form to receive a download link via email.
3. Extract the downloaded archive into the `flickr8k/` directory.

---

### Option 3 — Manual Kaggle Download (No CLI)

1. Go to [https://www.kaggle.com/datasets/adityajn105/flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)
2. Click **Download** (requires a free Kaggle account)
3. Unzip and place contents into `flickr8k/`

---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/your-username/pic2words.git
cd pic2words

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install torch torchvision pillow pandas nltk tqdm

# flickr8k folder
Place the folder in the same directory as the train.py
```

---

## 🚀 Training

```bash
python train.py
```

Key hyperparameters (configurable in `train.py`):

| Parameter | Default |
|---|---------|
| `embed_size` | 256     |
| `hidden_size` | 256     |
| `num_layers` | 1       |
| `learning_rate` | 3e-4    |
| `num_epochs` | 60      |

---

## 🔍 Inference

```bash
python inference.py --image path/to/your/image.jpg
```

---

## 🧠 Model Architecture

```
Image (3×299×299)
      │
      ▼
 Inception V3                ← Pretrained on ImageNet, fc layer replaced
      │
      ▼
 Linear (embed_size)         ← Projects CNN features into embedding space
      │
      ▼
 LSTM Decoder                ← Generates caption token by token
      │
      ▼
 Caption: "a dog runs on the beach"
```

---

## 📋 Caption Format (`captions.txt`)

```
image,caption
1000268201_693b08cb0e.jpg,A child in a pink dress is climbing up a set of stairs in an entry way .
1000268201_693b08cb0e.jpg,A girl going into a wooden building .
...
```

---

## 🙏 Acknowledgements

- Dataset: [Flickr8k — Hodosh et al., 2013](https://forms.illinois.edu/sec/1713398)
- Architecture inspired by: [Show and Tell — Vinyals et al., 2015](https://arxiv.org/abs/1411.4555)
- Encoder: [Inception V3 — Szegedy et al.](https://arxiv.org/abs/1512.00567)

---

## 📄 License

MIT License