## A Comparative Study of WGAN StyleGAN Unrolled GAN and Diffusion GAN for Mode Collapse in GAN
GANs are artificial intelligence algorithms used in unsupervised machine learning, particularly for image processing tasks. A discriminator and a generator, often Convolutional Neural Networks (CNNs), are the two neural networks involved. Mode collapse, a significant issue in GAN training, occurs when the generator focuses on a restricted range of samples, leading to a loss of diversity and low-quality output. To address mode collapse, various approaches aim to broaden the generator's scope by preventing it from optimizing a fixed discriminator. Four notable solutions are the Wasserstein Generative Adversarial Network (WGAN), the Unrolled GAN, the StyleGAN3, and the Diffusion GAN. WGAN introduces the Wasserstein distance as a stable objective function, enhancing training stability and mitigating mode collapse. Unrolled GAN involves several steps of discriminator optimization unrolling before adjusting the generator's parameters, which results in more reliable training dynamics and higher-quality samples. StyleGAN3 (Style-Based Generative Adversarial Network), developed by NVIDIA, generates high-quality, realistic images by controlling different levels of style through adaptive instance normalization (AdaIN). It mitigates mode collapse using AdaIN and a progressive growing architecture, enabling diverse and high-quality image generation. Diffusion GAN enhances sample diversity and reduces mode collapse by introducing a denoising-based training mechanism, improving the overall quality and stability of generated images. This paper conducts a comparative analysis of WGAN, Unrolled GAN, StyleGAN3, and Diffusion GAN to determine their effectiveness and appropriateness under specific circumstances, contributing valuable insights into mitigating mode collapse in GANs, particularly in the context of image processing tasks.

---

## 📊 Datasets Used

We evaluate our models on both *synthetic* and *real-world datasets* to assess performance across different data complexities.


### 1. 2D Ring Dataset

The 2D Ring dataset is a synthetic dataset generated using make_circles. It consists of points arranged in a circular structure and is commonly used to evaluate a model’s ability to capture continuous distributions.

```bash
from sklearn.datasets import make_circles
import torch
from torch.utils.data import DataLoader, TensorDataset

X_ring, _ = make_circles(n_samples=10000, noise=0.05, factor=0.5)

ring_loader = DataLoader(
    TensorDataset(torch.tensor(X_ring, dtype=torch.float32)),
    batch_size=batch_size,
    shuffle=True
)
```
---

### 2. 2D Grid Dataset (25 Gaussians)

The 2D Grid dataset is a synthetic dataset composed of 25 Gaussian distributions arranged in a 5×5 grid. It is widely used to evaluate *mode coverage* and detect Mode Collapse in generative models.

```bash
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

grid_points = []
for i in range(-2, 3):
    for j in range(-2, 3):
        points = np.random.randn(400, 2) * 0.05 + np.array([i*2, j*2])
        grid_points.append(points)

X_grid = np.vstack(grid_points)

grid_loader = DataLoader(
    TensorDataset(torch.tensor(X_grid, dtype=torch.float32)),
    batch_size=batch_size,
    shuffle=True
)
```

---

### 3. MNIST Dataset

The MNIST dataset is a benchmark dataset of handwritten digits, containing *60,000 training* and *10,000 test images*. It is widely used for evaluating generative models such as Generative Adversarial Network.

```bash
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

transform_mnist = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

mnist_loader = DataLoader(
    torchvision.datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transform_mnist
    ),
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)
```

---
### 4. MRI Dataset 

#### 🔗 Kaggle Source

* [https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c?select=Astrocitoma+T1*)


---

### 5. CelebA Dataset

#### 🔗 Official Source

* [https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

👉 From Chinese University of Hong Kong


---

