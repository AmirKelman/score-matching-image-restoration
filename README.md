
# Score Matching Models and Generative Image Restoration

This project implements a denoising generative model using score-based methods to restore corrupted images. We use a U-Net architecture combined with score matching loss to learn the gradient of the data distribution, enabling restoration of noisy, downsampled, or partially-masked images.

##  Key Features

- **Score Matching Model**: Learns gradients of log-density using noise-injected samples.
- **Generative Restoration**: Capable of restoring:
  - Noisy images
  - Downscaled images (factor 2 or 4)
  - Images with missing quarters or halves (inpainting)
- **Noise Estimation** using:
  - PCA-based estimation
  - Block-wise standard deviation
  - Median Absolute Deviation (MAD)

##  Structure

- `ex4.py`: Main training and restoration script
- `src/`:
  - `blocks.py`: U-Net implementation
  - `score_matching.py`: ScoreMatchingModel and configuration
- `examples/`: Folder to save sample results (e.g., `ex_mnist.png`)
- `ex4.pdf`: Project documentation (Amir Kelman, Omer Ben Haim)

##  Usage

### 1. Train or Load Model
```bash
python ex4.py --iterations 2000 --device cuda --load-trained 0
