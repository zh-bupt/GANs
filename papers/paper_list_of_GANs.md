# Papers of Generative Adversarial Nets

### 1. [Generative Adversarial Nets](./1406.2661.pdf)

> 10 Jun. 2014

#### 1.1 Abstract

We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G. The training procedure for G is to maximize the probability of D making a mistake. This framework corresponds to a minimax two-player game. In the space of arbitrary functions G and D, a unique solution exists, with G recovering the training data distribution and D equal to 1/2 everywhere. In the case where G and D are defined by multilayer perceptrons, the entire system can be trained with backpropagation. There is no need for any Markov chains or unrolled approximate inference networks during either training or generation of samples. Experiments demonstrate the potential of the framework through qualitative and quantitative evaluation of the generated samples.

#### 1.2 Dataset

* *MNIST*
* *Toronto Face Database*
* *CIFAR-10*

#### 1.3 Code

[official code](http://www.github.com/goodfeli/adversarial)

#### 1.4 Why I chose

> 生成对抗网络的第一篇paper，讲解了生成对抗模型的结构、原理。

### 2.[Conditional Generative Adversarial Nets](./1411.1784.pdf)

> 6 Nov. 2014

#### 2.1 Abstract

Generative Adversarial Nets were recently introduced as a novel way to train generative models. In this work we introduce the conditional version of generative adversarial nets, which can be constructed by simply feeding the data, y, we wish to condition on to both the generator and discriminator. We show that this model can generate MNIST digits conditioned on class labels. We also illustrate how this model could be used to learn a multi-modal model, and provide preliminary examples of an application to image tagging in which we demonstrate how this approach can generate descriptive tags which are not part of training labels.

#### 2.2 Dataset

* *MNIST digit dataset*

* *MIR Flickr 25,000 dataset*
* *VFCC100M2dataset*

#### 2.3 Code

[Reference](https://github.com/zhangqianhui/Conditional-Gans)

#### 2.4 Why I chose

> 在unconditioned的生成模型中不能控制生成的数据，而conditioned模型提供了一种能够控制生成模型输出的方法。

### 3.[Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks](./1506.05751.pdf)

> 18 Jun. 2015

#### 3.1 Abstract

In this paper we introduce a generative parametric model capable of producing high quality samples of natural images. Our approach uses a cascade of convolutional networks within a Laplacian pyramid framework to generate images in a coarse-to-fine fashion. At each level of the pyramid, a separate generative convnet model is trained using the Generative Adversarial Nets (GAN) approach [10]. Samples drawn from our model are of significantly higher quality than alternate approaches. In a quantitative assessment by human evaluators, our CIFAR10 samples were mistaken for real images around 40% of the time, compared to 10% for samples drawn from a GAN baseline model. We also show samples from models trained on the higher resolution images of the LSUN scene dataset.

#### 3.2 Dataset

#### 3.3 Code

[official code](https://github.com/facebook/eyescream)

#### 3.4 Why I chose

> 为生成模型产生更加精细的图像提供了一种思路。

### 4.[ Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](./1511.06434.pdf)

> 19 Nov. 2015

#### 4.1 Abstract

In recent years, supervised learning with convolutional networks (CNNs) has seen huge adoption in computer vision applications. Comparatively, unsupervised learning with CNNs has received less attention. In this work we hope to help bridge the gap between the success of CNNs for supervised learning and unsupervised learning. We introduce a class of CNNs called deep convolutional generative adversarial networks (DCGANs), that have certain architectural constraints, and demonstrate that they are a strong candidate for unsupervised learning. Training on various image datasets, we show convincing evidence that our deep convolutional adversarial pair learns a hierarchy of representations from object parts to scenes in both the generator and discriminator. Additionally, we use the learned features for novel tasks - demonstrating their applicability as general image representations.

#### 4.2 Dataset

* *Large-scale Scene Understanding (LSUN)*
* *Imagenet 1k*
* *Faces dataset*

#### 4.3 Code

[official code](https://github.com/Newmu/dcgan_code)

#### 4.4 Why I chose

> 使用CNNs来扩展GANs，并且取得了很好的效果；
>
> 将GANs用于Unsupervised Learning；
>
> GANs可视化。

### 5. [Improved Techniques for Training GANs](./1606.03498.pdf)

>  10 Jun 2016

#### 5.1 Abstract

We present a variety of new architectural features and training procedures that we apply to the generative adversarial networks (GANs) framework. We focus on two applications of GANs: semi-supervised learning, and the generation of images that humans find visually realistic. Unlike most work on generative models, our primary goal is not to train a model that assigns high likelihood to test data, nor do we require the model to be able to learn well without using any labels. Using our new techniques, we achieve state-of-the-art results in semi-supervised classification on MNIST, CIFAR-10 and SVHN. The generated images are of high quality as confirmed by a visual Turing test: our model generates MNIST samples thathumans cannot distinguish from real data, and CIFAR-10 samples that yield a human error rate of 21.3%. We also present ImageNet samples with unprecedented resolution and show that our methods enable the model to learn recognizable features of ImageNet classes

#### 5.2 Dataset

#### 5.3 Code

#### 5.4 Why I chose

> GAN的提出者Ian Goodfellow提出了一些训练GAN网络的方法，希望能够解决GAN网络训练中的一些问题。

### 6. [Generative Image Modeling using Style and Structure Adversarial Networks](./1603.05631.pdf)

> 7 Mar 2016

#### 6.1 Abstract

Current generative frameworks use end-to-end learning and generate images by sampling from uniform noise distribution. However, these approaches ignore the most basic principle of image formation: images are product of: (a) Structure: the underlying 3D model; (b) Style: the texture mapped onto structure. In this paper, we factorize the image generation process and propose Style and Structure Generative Adversarial Network (S2-GAN). Our S2-GAN has two components: the Structure-GAN generates a surface normal map; the Style-GAN takes the surface normal map as input and generates the 2D image. Apart from a real vs. generated loss function, we use an additional loss with computed surface normals from generated images. The two GANs are first trained independently, and then merged together via joint learning. We show our S2-GAN model is interpretable, generates more realistic images and can be used to learn unsupervised RGBD representations.

#### 6.2 Dataset

* *NYUv2 RGBD dataset*

#### 6.3 Code

No official code

#### 6.4 Why I chose

> 通过分别训练两个GAN分别产生Structure和Style，然后合并来解决GAN网络不好训练以及不能产生高质量的高分辨率图像的问题。

### 7. [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](./1609.04802.pdf)

> 15 Sep 2016

#### 7.1 Abstract

Despite the breakthroughs in accuracy and speed of single image super-resolution using faster and deeper convolutional neural networks, one central problem remains largely unsolved: how do we recover the finer texture details when we super-resolve at large upscaling factors? The behavior of optimization-based super-resolution methods is principally driven by the choice of the objective function. Recent work has largely focused on minimizing the mean squared reconstruction error. The resulting estimates have high peak signal-to-noise ratios, but they are often lacking high-frequency details and are perceptually unsatisfying in the sense that they fail to match the fidelity expected at the higher resolution. In this paper, we present SRGAN, a generative adversarial network (GAN) for image super-resolution (SR). To our knowledge, it is the first framework capable of inferring photo-realistic natural images for 4× upscaling factors. To achieve this, we propose a perceptual loss function which consists of an adversarial loss and a content loss. The adversarial loss pushes our solution to the natural image manifold using a discriminator network that is trained to differentiate between the super-resolved images and original photo-realistic images. In addition, we use a content loss motivated by perceptual similarity instead of similarity in pixel space. Our deep residual network is able to recover photo-realistic textures from heavily downsampled images on public benchmarks. An extensive mean-opinion-score (MOS) test shows hugely significant gains in perceptual quality using SRGAN. The MOS scores obtained with SRGAN are closer to those of the original high-resolution images than to those obtained with any state-of-the-art method.

#### 7.2 Dataset

- *Set5*
- *Set14*
- *BSD100*

#### 7.3 Code

[reference](https://github.com/tensorlayer/SRGAN)

#### 7.4 Why I chose

> 如何用GAN来解决超分辨率问题，并且取得了比较好的效果。

### 8. [Least Squares Generative Adversarial Networks](./1611.04076.pdf)

> 13 Nov. 2016

#### 8.1 Abstract

Unsupervised learning with generative adversarial networks (GANs) has proven hugely successful. Regular GANs hypothesize the discriminator as a classifier with the sigmoid cross entropy loss function. However,
we found that this loss function may lead to the vanishing gradients problem during the learning process. To overcome such a problem, we propose in this paper the Least Squares Generative Adversarial Networks (LS-GANs) which adopt the least squares loss function for the discriminator.We show that minimizing the objective function of LSGAN yields minimizing the Pearson χ2 divergence. There are two benefits of LSGANs
over regular GANs. First, LSGANs are able to generate higher quality images than regular GANs. Second, LSGANs perform more stable during the learning process. We evaluate LSGANs on five scene datasets and the experimental results show that the images generated by LSGANs are of better quality than the ones generated by regular GANs. We also conduct two comparison experiments between LSGANs and regular GANs to illustrate the stability of LSGANs.

#### 8.2 Dataset

* *LSUN*
* *HWDB1.0*

#### 8.3 Code

no offcial code

#### 8.4 Why I chose

> 将目标函数将改成了一个平方误差函数，想通过不通的使用不同的损失函数来解决原来GAN的缺陷。

### 9. [Image-to-Image Translation with Conditional Adversarial Networks](./1611.07004.pdf)

> 21 Nov 2016

#### 9.1 Abstract

We investigate conditional adversarial networks as a general-purpose solution to image-to-image translation problems. These networks not only learn the mapping from input image to output image, but also learn a loss function to train this mapping. This makes it possible to apply the same generic approach to problems that traditionally would require very different loss formulations. We demonstrate that this approach is effective at synthesizing photos from label maps, reconstructing objects from edge maps, and colorizing images, among other tasks. Indeed, since the release of the pix2pix software associated with this paper, a large number of internet users (many of them artists) have posted their own experiments with our system, further demonstrating its wide applicability and ease of adoption without the need for parameter tweaking. As a community, we no longer hand-engineer our mapping functions, and this work suggests we can achieve reasonable results without hand-engineering our loss functions either.

#### 9.2 Dataset

* *Cityscape dataset*
* *CMP Facades*
* *data scraped from Google Maps*

#### 9.3 Code

[official code](https://github.com/phillipi/pix2pix)

#### 9.4 Why I chose

> * 设计了一种普遍通用的方法来处理各种pix-to-pix问题，包括图像上色，分割->真实图像，真实图像->分割，线稿->图像，航拍->地图等；
> * U-Net
> * PachGAN

### 10. [StackGAN: Text to Photo-realistic Image Synthesiswith Stacked Generative Adversarial Networks](./1612.03242.pdf)

> 10 Dec. 2016

#### 10.1 Abstract

Synthesizing high-quality images from text descriptions is a challenging problem in computer vision and has many practical applications. Samples generated by existing textto-image approaches can roughly reflect the meaning of the given descriptions, but they fail to contain necessary details and vivid object parts. In this paper, we propose Stacked Generative Adversarial Networks (StackGAN) to generate 256×256 photo-realistic images conditioned on text descriptions. We decompose the hard problem into more manageable subproblems through a sketch-refinement process. The Stage-I GAN sketches the primitive shape and colors of the object based on the given text description, yielding Stage-I low-resolution images. The Stage-II GAN takes Stage-I results and text descriptions as inputs, and generates high-resolution images with photo-realistic details. It is able to rectify defects in Stage-I results and add compelling details with the refinement process. To improve the diversity of the synthesized images and stabilize the training of the conditional-GAN, we introduce a novel Conditioning Augmentation technique that encourages smoothness in the latent conditioning manifold. Extensive experiments and comparisons with state-of-the-arts on benchmark datasets demonstrate that the proposed method achieves significant improvements on generating photo-realistic images conditioned on text descriptions.

#### 10.2 Dataset

* *CUB*
* *Oxford-102*
* *MS COCO*

#### 10.3 Code

[official code](https://github.com/hanzhanggit/StackGAN)

#### 10.4 Why I chose

> 将产生高分辨率的图像的任务分解为两个阶段：先产生形状和基本的颜色信息，然后再针对第一阶段产生的图像增加细节和错误修正，最终产生比较真实的高分辨率图像。

### 11. [Wasserstein GAN](./1701.07875.pdf)

> 26 Jan. 2017

#### 11.1 Abstract

The contributions of this paper are: 

* In Section 2, we provide a comprehensive theoretical analysis of how the Earth Mover (EM) distance behaves in comparison to popular probability distances and divergences used in the context of learning distributions. 

* In Section 3, we define a form of GAN called Wasserstein-GAN that mini- mizes a reasonable and efficient approximation of the EM distance, and we theoretically show that the corresponding optimization problem is sound. 

* In Section 4, we empirically show that WGANs cure the main training prob- lems of GANs. In particular, training WGANs does not require maintaining a careful balance in training of the discriminator and the generator, and does not require a careful design of the network architecture either. The mode dropping phenomenon that is typical in GANs is also drastically reduced. One of the most compelling practical benefits of WGANs is the ability to continuously estimate the EM distance by training the discriminator to op- timality. Plotting these learning curves is not only useful for debugging and hyperparameter searches, but also correlate remarkably well with the observed sample quality. 

#### 11.2 Dataset

* *LSUN-Bedrooms*

#### 11.3 Code

[official code](https://github.com/martinarjovsky/WassersteinGAN)

#### 11.4 Why I chose

> 从理论上分析了原始GAN的问题所在，并给出了改进的算法实现流程。

### 12. [Loss-Sensitive Generative Adversarial Networks on Lipschitz Densities](./1701.06264.pdf)

> 23 Jan. 2017

#### 12.1 Abstract

In this paper, we present the Lipschitz regularization theory and algorithms for a novel Loss-Sensitive Generative Adversarial Network (LS-GAN). Specifically, it trains a loss function to distinguish between real and fake samples by designated margins, while learning a generator alternately to produce realistic samples by minimizing their losses. The LS-GAN further regularizes its loss function with a Lipschitz regularity condition on the density of real data, yielding a regularized model that can better generalize to produce new data from a reasonable number of training examples than the classic GAN. We will further present a Generalized LS-GAN (GLS-GAN) and show it contains a large family of regularized GAN models, including both LS-GAN and Wasserstein GAN, as its special cases. Compared with the other GAN models, we will conduct experiments to show both LS-GAN and GLS-GAN exhibit competitive ability in generating new images in terms of the Minimum Reconstruction Error (MRE) assessed on a separate test set. We further extend the LS-GAN to a conditional form for supervised and semi-supervised learning problems, and demonstrate its outstanding performance on image classification tasks.

#### 12.2 Dataset

- *CelebA*
- *MNIST*
- *CIFAR-10*
- *SVHN*

#### 12.3 Code

[official code](https://github.com/maple-research-lab)

#### 12.4 Why I chose

> 提出了解决GAN难以训练的一种解决方法，并给出了定量的分析证明，并和之前的WGAN进行了比较。

### 13. [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](./1703.10593.pdf)

> 30 Mar 2017

#### 13.1 Abstract

Image-to-image translation is a class of vision and graphics problems where the goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs. However, for many tasks, paired training data will not be available. We present an approach for learning to translate an image from a source domain X to a target domain Y in the absence of paired examples. Our goal is to learn a mapping G : X → Y such that the distribution of images from G(X) is indistinguishable from the distribution Y using an adversarial loss. Because this mapping is highly under-constrained, we couple it with an inverse mapping F : Y → X and introduce a cycle consistency loss to enforce F(G(X)) ≈ X (and vice versa). Qualitative results are presented on several tasks where paired training data does not exist, including collection style transfer, object transfiguration, season transfer, photo enhancement, etc. Quantitative comparisons against several prior methods demonstrate the superiority of our approach.

#### 13.2 Dataset

* *Cityscapes*
* *images scraped from Google Maps*
* *CMP Facade Database*
* *Zappos50K dataset*
* *Flickr*
* *art images downloaded from Wikiart.org*
* *DSLR dataset*

#### 13.3 Code

[official code](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

#### 13.4 Why I chose

> CycleGAN可以在没有成对训练数据的情况下，实现图像风格的转换。

### 14. [Improved Training of Wasserstein GANs](./1704.00028.pdf)

> 31 Mar. 2017

#### 14.1 Abstract

Generative Adversarial Networks (GANs) are powerful generative models, but suffer from training instability. The recently proposed Wasserstein GAN (WGAN) makes progress toward stable training of GANs, but sometimes can still generate only poor samples or fail to converge. We find that these problems are often due to the use of weight clipping in WGAN to enforce a Lipschitz constraint on the critic, which can lead to undesired behavior. We propose an alternative to clipping weights: penalize the norm of gradient of the critic with respect to its input. Our proposed method performs better than standard WGAN and enables stable training of a wide variety of GAN architectures with almost no hyperparameter tuning, including 101-layer ResNets and language models with continuous generators. We also achieve high quality generations on CIFAR-10 and LSUN bedrooms.

#### 14.2 Dataset

* *Swiss Roll dataset*
* *LSUN bedroom*

#### 14.3 Code

[official code](https://github.com/igul222/improved_wgan_training)

#### 14.4 Why I chose

> 解释了在WGAN中为了保证Lipshitz约束的Weight Clipping的问题，提出了新的解决方法——Gradient penalty。

### 15. [MMGAN: Manifold-Matching Generative Adversarial Network](./1707.08273.pdf)

> 26 Jul 2017

#### 15.1 Abstract

It is well-known that GANs are difficult to train, and several different techniques have been proposed in order to stabilize their training. In this paper, we propose a novel training method called manifold-matching, and a new GAN model called manifold-matching GAN (MMGAN). MMGAN finds two manifolds representing the vector representations of real and fake images. If these two manifolds match, it means that real and fake images are statistically identical. To assist the manifold- matching task, we also use i) kernel tricks to find better manifold structures, ii) moving-averaged manifolds across mini-batches, and iii) a regularizer based on correlation matrix to suppress mode collapse. 

We conduct in-depth experiments with three image datasets and compare with several state-of-the-art GAN models. 32.4% of images generated by the proposed MMGAN are recognized as fake images during our user study (16% enhancement compared to other state-of-the-art model). MMGAN achieved an unsuper- vised inception score of 7.8 for CIFAR-10. 

#### 15.2 Dataset

* *MNIST*
* *CelebA*
*  *CIFAR-10*

#### 15.3 Code

No official code

#### 15.4 Why I chose

> 通过让生成数据和真实数据的Mainfold相匹配来使生成数据的分布接近真实数据的分布，从而来稳定训练。

### 16. [High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs](./1711.11585.pdf)

> 30 Nov 2017

#### 16.1 Abstract

We present a new method for synthesizing high- resolution photo-realistic images from semantic label maps using conditional generative adversarial networks (condi- tional GANs). Conditional GANs have enabled a variety of applications, but the results are often limited to low- resolution and still far from realistic. In this work, we gen- erate 2048 × 1024 visually appealing results with a novel adversarial loss, as well as new multi-scale generator and discriminator architectures. Furthermore, we extend our framework to interactive visual manipulation with two ad- ditional features. First, we incorporate object instance seg- mentation information, which enables object manipulations such as removing/adding objects and changing the object category. Second, we propose a method to generate di- verse results given the same input, allowing users to edit the object appearance interactively. Human opinion stud- ies demonstrate that our method significantly outperforms existing methods, advancing both the quality and the reso- lution of deep image synthesis and editing. 

#### 16.2 Dataset

* *Cityscapes dataset*
* *NYU dataset*
* *ADE20K dataset*

#### 16.3 Code

[offcial code](https://github.com/NVIDIA/pix2pixHD)

#### 16.4 Why I chose

> 利用条件 cGAN 生成 2048 x 1024 分辨率的图像，利用Multi-scale discriminators来解决对高分辨率图像的判别问题，使用了改进的损失函数，利用instance map作为condition。

### 17. [Spectral Normalization for Generative Adversarial Networks](./1802.05957.pdf)

> 16 Feb. 2018

#### 17.1 Abstact

One of the challenges in the study of generative adversarial networks is the instability of its training. In this paper, we propose a novel weight normalization technique called spectral normalization to stabilize the training of the discriminator. Our new normalization technique is computationally light and easy to incorporate into existing implementations. We tested the efficacy of spectral normalization on CIFAR10, STL-10, and ILSVRC2012 dataset, and we experimentally confirmed that spectrally normalized GANs (SN-GANs) is capable of generating images of better or equal quality relative to the previous training stabilization techniques. The code with Chainer (Tokui et al., 2015), generated images and pretrained models are available at https://github.com/pfnet-research/sngan_projection.

#### 17.2 Dataset

* *CIFAR10*
* *STL-10*
* *ILSVRC2012*

#### 17.3 Code

[official code](https://github.com/pfnet-research/sngan_projection)

#### 17.4 Why I chose

> 这篇文章提出了一种简单有效的标准化方法来限制GAN中分辨器D的优化过程，从而达到整个模型能学习到更好的生成器G的结果。

### 18. [Progressive Growing of GANs for Improved Quality, Stability, and Variation](./1710.10196.pdf)

> 26 Feb. 2018

#### 18.1 Abstract

We describe a new training methodology for generative adversarial networks. The key idea is to grow both the generator and discriminator progressively: starting from a low resolution, we add new layers that model increasingly fine details as training progresses. This both speeds the training up and greatly stabilizes it, allowing us to produce images of unprecedented quality, e.g., CELEBA images at 1024^2^. We also propose a simple way to increase the variation in generated images, and achieve a record inception score of 8.80 in unsupervised CIFAR10. Additionally, we describe several implementation details that are important for discouraging unhealthy competition between the generator and discriminator. Finally, we suggest a new metric for evaluating GAN results, both in terms of image quality and variation. As an additional contribution, we construct a higher-quality version of the CELEBA dataset.

#### 18.2 Dataset

- *CelebA dataset*
- *CIFAR10*

#### 18.3 [Code](https://github.com/tkarras/progressive_growing_of_gans)

#### 18.4 Why I chose

> 增大训练的方式，能够产生高清图像

### 19. [WESPE: Weakly Supervised Photo Enhancer for Digital Cameras](./1709.01118.pdf)[[arXiv]](https://arxiv.org/pdf/1704.02470.pdf)

> 3 Mar. 2018

#### 19.1 Abstract

Low-end and compact mobile cameras demonstrate limited photo quality mainly due to space, hardware and budget constraints. In this work, we propose a deep learning solution that translates photos taken by cameras with limited capabilities into DSLR-quality photos automatically. We tackle this problem by introducing a weakly supervised photo enhancer (WESPE) – a novel image-to-image Generative Adversarial Network-based architecture. The proposed model is trained by under weak supervision: unlike previous works, there is no need for strong supervision in the form of a large annotated dataset of aligned original/enhanced photo pairs. The sole requirement is two distinct datasets: one from the source camera, and one composed of arbitrary high-quality images that can be generally crawled from the Internet – the visual content they exhibit may be unrelated. Hence, our solution is repeatable for any camera: collecting the data and training can be achieved in a couple of hours. In this work, we emphasize on extensive evaluation of obtained results. Besides standard objective metrics and subjective user study, we train a virtual rater in the form of a separate CNN that mimics human raters on Flickr data and use this network to get reference scores for both original and enhanced photos. Our experiments on the DPED, KITTI and Cityscapes datasets as well as pictures from several generations of smartphones demonstrate that WESPE produces comparable or improved qualitative results with state-of-the-art strongly supervised methods.

#### 19.2 Dataset

- *DPED*
- *KITTI*
- *Cityscapes datasets*

#### 19.3 Code

[official code](https://github.com/aiff22/DPED)

#### 19.4 Why I chose

> 相对于之前的图像质量提升方法，本方法无需和低端质量图像相关的高质量监督数据。训练时只需要提供需要提高到的图像质量参考。 

### 20. [Are GANs Created Equal? A Large-Scale Study](./1711.10337.pdf)

> 3 Mar. 2018

#### 20.1 Abstract

Generative adversarial networks (GAN) are a powerful subclass of generative models. Despite a very rich research activity leading to numerous interesting GAN algorithms, it is still very hard to assess which algorithm(s) perform better than others. We conduct a neutral, multi-faceted largescale empirical study on state-of-the art models and evaluation measures. We find that most models can reach similar scores with enough hyperparameter optimization and random restarts. This suggests that improvements can arise from a higher computational budget and tuning more than fundamental algorithmic changes. To overcome some limitations of the current metrics, we also propose several data sets on which precision and recall can be computed. Our experimental results suggest that future GAN research should be based on more systematic and objective evaluation procedures. Finally, we did not find evidence that any of the tested algorithms consistently outperforms the original one.

#### 20.2 Dataset

#### 20.3 Code

#### 20.4 Why I chose

> 自2014年GAN网络提出以来，GAN网络发展迅速，产生了很多变体来解决原版GAN的问题。Google Brain团队将这些变体和原版GAN做了对比。他们得出的结论是，没有实证证据能证明这些GAN变体在所有数据集上明显优于原版。

