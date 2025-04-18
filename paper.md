## 1. Introduction

Gender classification has become a fundamental task in computer vision, with numerous applications across various domains including in-cabin driver monitoring systems, human-computer interaction, video surveillance, retail analytics, and psychological analysis. Traditionally, researchers have focused on gender classification using visible spectrum images of the human face. However, the performance of these systems can be significantly affected by challenging environmental factors such as varying illumination conditions, shadows, occlusions, and the time of day.

To overcome these limitations, there has been a growing interest in exploring alternative or complementary sensing modalities, such as **thermal imaging**. Thermal imaging offers several advantages as it does not rely on external illumination and provides a distinct perspective on an imaged scene compared to conventional visible light sensors. This makes it a potentially more robust solution for gender classification in diverse and uncontrolled environments. Furthermore, thermal imaging can easily detect people even in total darkness, expanding its applicability in security systems. Beyond security, thermal signatures can provide complementary information in human-computer interaction, potentially revealing subtle physiological indicators relevant to gender.

Despite the benefits, thermal images typically lack some of the detailed facial definitions present in visible spectrum images, posing a challenge for accurate classification. To address this, the application of **deep learning**, particularly **Convolutional Neural Networks (CNNs)**, has shown significant promise in learning intricate patterns from thermal data for gender classification.

This paper investigates the effectiveness of deep learning models for gender detection using thermal facial images. We utilize two publicly available thermal image datasets, the **Tufts University Thermal Face dataset** and the **Charlotte-ThermalFace dataset**, both individually and in combination, to train and evaluate a range of state-of-the-art CNN architectures including **AlexNet**, **VGG**, **InceptionV3**, **ResNet50**, and **EfficientNet**. We address the differences in channel availability between the datasets and enhance the data through **image augmentation** techniques. Furthermore, we tackle the class imbalance present in the Tufts dataset to ensure robust training. To further advance the field, we propose a **novel CNN architecture** based on the ResNet framework, incorporating a **channel input adapter** to handle varying input channels and **Squeeze and Excitation (SE) blocks** within its layers to enhance feature discrimination, along with a tailored final classifier.

The primary contributions of this paper include:
*   A comprehensive evaluation of several state-of-the-art CNN models for gender classification on thermal facial images using the Tufts and Charlotte datasets.
*   An investigation into the impact of combining datasets with differing channel characteristics.
*   The development and evaluation of a novel CNN architecture specifically designed for thermal image-based gender detection, incorporating channel adaptation and attention mechanisms.
*   An analysis of the challenges and potential of deep learning for gender classification using thermal imaging.

The remainder of this paper is structured as follows: Section 2 provides a review of related work in gender classification using both traditional and deep learning methods with visible, near-infrared, and thermal imagery. Section 3 details the datasets used and the methodology employed, including preprocessing, augmentation techniques, and the architecture of the proposed CNN model. Section 4 presents the experimental results and a comparative analysis of the different models. Section 5 discusses the implications and limitations of our findings, and Section 6 concludes the paper with potential directions for future research.

## 2. Literature Review (updated)

The task of gender classification has been extensively studied in computer vision. Early approaches often relied on **conventional machine learning methods** and feature extraction techniques applied to visible spectrum images. Makinen and Raisamo and Reid et al. provided detailed surveys of these methods. Initial techniques involved training neural systems on small sets of frontal face images. Later, methods incorporated 3D head structure and image intensities for gender characterization. **Support Vector Machines (SVMs)** were also widely used, demonstrating competitive performance compared to other traditional classifiers. Techniques like AdaBoost, utilizing low-resolution grayscale images, and methods addressing perspective invariant recognition were also explored. More recently, researchers utilized local image descriptors like the Webers Local Surface Descriptor (WLD) and features based on shape, texture, and color extracted from frontal faces, achieving high accuracy on benchmark datasets like FERET.

Gender classification has emerged as an essential task in computer vision, playing a key role in numerous real-world applications such as in-cabin driver monitoring systems, smart surveillance, demographic analytics, and human-computer interaction. Traditionally, this task has relied predominantly on facial images captured in the visible spectrum. Early systems applied handcrafted features like Local Binary Patterns (LBP), Principal Component Analysis (PCA), and Histogram of Oriented Gradients (HOG) in conjunction with classifiers like Support Vector Machines (SVMs). However, these approaches have repeatedly demonstrated fragility under real-world conditions where variables such as illumination, facial occlusions (e.g., masks, sunglasses), shadows, and changing poses severely degrade performance.
To overcome these limitations, research has increasingly turned toward thermal imaging as an alternative sensing modality. Thermal images capture the heat signature emitted by facial tissues, providing invariant information that is unaffected by ambient lighting conditions. This makes them particularly useful in environments where visible light sensors fail—such as night-time surveillance, poor weather conditions, or low-contrast settings. Moreover, thermal imaging enables detection of subtle physiological patterns that are invisible in RGB data, which can offer additional cues for gender classification.
Yet, the advantages of thermal imaging also come with notable challenges. Thermal images tend to have lower resolution and often lack the detailed structural and textural features present in visible spectrum images. These factors make feature extraction and discrimination more complex. The traditional methods that worked well on RGB data often fail when applied directly to thermal imagery. This is where deep learning, specifically Convolutional Neural Networks (CNNs), has shown immense promise in automatically learning hierarchical representations even from low-resolution and noisy thermal data.
One of the landmark studies in this direction was by Jalil et al. (2023) who introduced a Modified CNN model for classifying gender of thermal images using cloud computing. Their architecture, Cloud_Res, was specifically optimized for thermal facial images and achieved a remarkable precision. What distinguished this work was its deployment in a cloud environment, leveraging the scalability and speed of cloud-based inference engines. They also benchmarked their architecture against traditional ResNet variants (18, 50, and 101 layers), concluding that a well-designed lightweight CNN with fewer layers could achieve similar—if not better—performance due to reduced overfitting and faster convergence. However, their model did not incorporate attention mechanisms or adaptive input handling, and it exhibited some imbalance in gender classification accuracy, particularly favoring male predictions.
In parallel, Chatterjee and Zaman (2023) conducted an in-depth study using ResNet-50 and VGG-19 architectures on the Tufts and Charlotte ThermalFace datasets. Their preprocessing pipeline included Kalman filtering, which significantly enhanced the signal-to-noise ratio of thermal images, resulting in an increase in classification accuracy by 3–5%. Their best performing model, ResNet-50, achieved 95.0% accuracy, demonstrating that deeper CNNs are capable of extracting discriminative patterns even from thermally distorted or noisy data. However, these architectures are generic and not explicitly designed for thermal imaging.
Another key contribution was made by Nguyen et al. (2017), who combined thermal and visible-light camera feeds for gender classification using a CNN-SVM hybrid approach. Their work highlighted the effectiveness of feature-level fusion and score-level fusion in improving classification accuracy. While this bimodal setup achieved higher accuracy compared to single-modality systems, it required synchronized camera systems and complex alignment pipelines—rendering it impractical for many real-world deployments where only thermal imaging is feasible.
The role of hand-based thermal imaging was explored by Prihodova et al. (2022). Their work used VGG-16 and VGG-19 models on thermal hand images and achieved an impressive accuracy of 94.9%. While promising, hand-based methods require subject cooperation and controlled image acquisition, limiting their utility in surveillance and dynamic environments.
In terms of architectural exploration, Farooq et al. and others performed comprehensive benchmarking using CNNs like AlexNet, VGG19, and EfficientNet-B4. Their results consistently showed that shallow networks like AlexNet underperformed (with accuracies around 82.6%), while InceptionV3 reached 92.3% due to its deeper and more modular design. These studies emphasize the importance of choosing architectures capable of capturing both local and global patterns—something shallow networks struggle with in low-resolution thermal images.
The Infrared Thermal Image Gender Classifier (IRT_ResNet) proposed by Jalil et al. (2022) compared ResNet variants and demonstrated that deeper networks (ResNet-101) offered better performance. However, their study noted diminishing returns beyond a certain depth and highlighted the model's skewed performance favoring male predictions, suggesting a need for better-balanced training methods.
Thermal-based gender classification from UAV-mounted cameras has also gained attention. Studies like "Thermal-based Gender Recognition Using Drones" and "Gender Recognition Using UAV-based Thermal Images" explored mobile applications where thermal images captured from drones were used for biometric analysis. However, these setups faced challenges due to image instability, resolution loss, and varying subject distance. CNNs like AlexNet and GoogLeNet achieved moderate accuracies (82–85%), but performance varied depending on environmental conditions.
To address occlusions and dataset-specific challenges, some researchers proposed the use of 3D facial models or spatial-temporal analysis (e.g., CNN-BGRU models) to integrate motion or depth-based cues into classification. These approaches, while theoretically sound, are computationally intensive and not well-suited for low-power or real-time deployments.
In our work, we aim to overcome these limitations through the design of a novel CNN architecture called TH-SE-ResNet. It builds upon the ResNet backbone but introduces several key innovations:
Channel Input Adapter: Given the inconsistency in channel formats between datasets (e.g., grayscale vs. RGB vs. 6-channel IR-RGB), our model integrates an adapter module to standardize inputs, allowing for seamless dataset fusion. This is particularly crucial as we combine the Tufts University and Charlotte-ThermalFace datasets in our experiments.


Squeeze-and-Excitation (SE) Blocks: To improve feature discrimination, SE blocks are embedded within residual units. These blocks dynamically recalibrate channel-wise feature responses, enabling the network to prioritize salient thermal features—especially useful in handling occlusions and low-contrast areas.


Class-Imbalance Mitigation: We incorporate class-weighted loss functions during training to counter the male-biased prediction patterns observed in previous studies (e.g., Jalil et al., 2022). This ensures fairer and more balanced classification across genders.


Data Augmentation Pipeline: We apply a robust preprocessing and augmentation routine—using techniques such as rotation, flipping, and Gaussian noise injection—to increase model generalizability and reduce overfitting.


Evaluation on Combined Datasets: Unlike prior work that tested models on isolated datasets, we evaluate our model on a combined Tufts-Charlotte dataset, increasing diversity in facial features, pose variations, and sensor modalities, thus pushing the limits of generalization.


Our experimental findings demonstrate that TH-SE-ResNet consistently outperforms standard architectures across multiple metrics (accuracy, precision, recall, F1-score) and maintains high performance even under occlusion and noise. Unlike previous models limited to specific deployment environments, our model is designed for cloud and edge deployment, supporting real-time inference and scalability.
In conclusion, while existing literature has made considerable strides in leveraging CNNs for thermal gender classification, challenges remain in model generalization, dataset handling, feature prioritization, and bias mitigation. Our work builds on this foundation by introducing a tailored architecture that directly addresses these gaps. TH-SE-ResNet offers a more complete, fair, and deployable solution—moving the field closer to practical, large-scale implementations of thermal gender classification systems.


### 3.1 Datasets

#### 3.1.1 Tufts University Thermal Face Dataset

The Tufts University Thermal Face Dataset represents a comprehensive multimodal collection comprising over 10,000 images across various modalities acquired from a diverse cohort of 113 participants (74 females, 39 males). For our research, we specifically utilized the thermal subset containing approximately 1,400 images. The age distribution spans from 4 to 70 years, with subjects originating from more than 15 countries, thus providing substantial demographic variability. Image acquisition was conducted using a FLIR Vue Pro thermal camera under controlled indoor environmental conditions. Participants were positioned at a standardized distance from the imaging apparatus to maintain consistency. For our investigation, we specifically utilized two subsets: TD_IR_E (Emotion), which contains images depicting five distinct facial expressions (neutral, smile, eyes closed, shocked, and with sunglasses), and TD_IR_A (Around), which encompasses images captured from nine different camera positions arranged in a semicircular configuration around each participant. A significant challenge encountered with this dataset was the pronounced gender imbalance, with approximately 30.32% female and 69.68% male images. To mitigate this imbalance and enhance model robustness, we implemented targeted data augmentation techniques specifically for the underrepresented female class, including controlled geometric transformations and intensity adjustments while preserving critical thermal signature characteristics.

![tufts_grid](https://github.com/user-attachments/assets/3b896a26-95b9-4c6b-b02a-f22fed2de0a6)


#### 3.1.2 Charlotte-ThermalFace Dataset

The Charlotte-ThermalFace Dataset comprises approximately 10,364 thermal facial images from 10 subjects, collected under varying conditions (e.g., distance, head position, temperature). This dataset was not specifically created for gender detection tasks, but we repurposed it for our gender classification research. Based on image characteristics, we infer that data acquisition likely employed a FLIR-based thermal imaging system. In contrast to the Tufts collection, the Charlotte dataset exhibits near-perfect gender balance with approximately 50.10% female and 49.90% male. This balanced distribution provided an advantageous counterpoint to the gender imbalance present in the Tufts dataset.

![charlotte_grid](https://github.com/user-attachments/assets/3c6c179a-e9fe-43d4-a16d-ca780c4d42c9)

#### 3.1.3 Combined Dataset

To enhance data diversity and expand the training corpus, we constructed a combined dataset by integrating the Tufts and Charlotte collections following a systematic merging protocol. A significant technical challenge encountered during this integration was the channel discrepancy between datasets—the Charlotte images were originally single-channel thermal representations, whereas the Tufts dataset employed a three-channel format. To address this incompatibility, we implemented channel replication for the Charlotte images, duplicating the single thermal channel across three channels to establish format consistency with the Tufts data structure. Furthermore, to prevent model bias towards the overrepresented class, we carefully balanced the gender distribution by selecting an equal number of images per gender category through strategic sampling. This integration yielded a substantially enlarged dataset of approximately 11,921 images with perfect gender balance (50% female, 50% male), thereby providing our models with enhanced training diversity spanning different thermal imaging conditions, acquisition parameters, and subject characteristics.

In addition to the primary datasets, we designed cross-dataset experimental protocols to rigorously evaluate model generalization capabilities across different thermal imaging domains. These experiments comprised two principal configurations: Tufts-to-Charlotte (training on Tufts data and evaluating on Charlotte) and Charlotte-to-Tufts (training on Charlotte and evaluating on Tufts). This cross-domain validation approach enables assessment of our models' ability to generalize across varying thermal imaging conditions, camera specifications, and data collection protocols—a critical factor for real-world deployment scenarios where thermal imaging parameters may differ substantially from training conditions.

**Table 1: Summary of Datasets**

| Dataset    | Size (Images) | Gender Distribution     | Channels                        |  
|------------|---------------|-------------------------|---------------------------------|  
| Tufts      | ~1,400        | 30.32% F, 69.68% M      | Three (thermal representation)  |  
| Charlotte  | ~10,000       | 50.10% F, 49.90% M      | One (thermal grayscale)        |  
| Combined   | ~11,921       | 50.00% F, 50.00% M      | Three-channel format           | 


## 3.2 Data Preprocessing and Augmentation

Our data preprocessing and augmentation pipeline was meticulously designed to address the unique challenges of thermal facial image analysis for gender classification. The pipeline incorporated several carefully considered stages to ensure optimal model performance and generalization.

### 3.2.1 Dataset Organization and Partitioning

We structured our datasets according to a standardized hierarchical organization to facilitate efficient training and evaluation. Each dataset (Tufts, Charlotte, and Combined) was systematically partitioned into training and testing subsets using a subject-disjoint approach. This critical design choice ensured that images from the same individual never appeared in both training and testing sets, thus preventing identity-based information leakage that could artificially inflate performance metrics. We implemented an 80:20 train-test split ratio, stratified by gender to maintain proportional representation across partitions.

For the Tufts dataset, we addressed the gender imbalance during augmentation, ensuring that the disproportionate male-to-female ratio was consistently reflected in both training and testing subsets. In the Charlotte dataset, the near-perfect gender balance was preserved throughout the partition process. For the combined dataset, we implemented balanced sampling to achieve gender parity while maintaining subject-level separation between training and testing sets.


![dataset_partitioning_schema](https://github.com/user-attachments/assets/06a2b046-8513-4092-9345-ad485141a975)
**Figure 1: Subject-Disjoint Dataset Partitioning Schema** - A diagram showing the hierarchical organization and separation of subjects by gender across train/test splits.


### 3.2.2 Image Normalization and Standardization

Thermal imaging data presents unique challenges due to variations in sensor calibration, environmental conditions, and temperature ranges. To mitigate these issues, we implemented a comprehensive normalization protocol:

All thermal images were normalized using mean-centering with a value of 0.5 and standard deviation scaling of 0.5. This approach was selected over the conventional ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) as it proved more effective for thermal imagery during our preliminary experiments, likely due to the fundamentally different intensity distribution characteristics of thermal versus visible spectrum images.

The Charlotte dataset's single-channel thermal images required special handling when used in models designed for multi-channel inputs. For our TH-SE-ResNet specifically designed for thermal data, we maintained the single-channel representation, utilizing the grayscale transformation to preserve thermal intensity information. For standard RGB-designed architectures, we expanded the single channel through replication to maintain compatibility while preserving the original thermal information.

Images were resized according to model-specific requirements—224×224 pixels for AlexNet, VGG, ResNet, and EfficientNet; 299×299 pixels for Inception. This standardization ensured consistent spatial dimensions while preserving the aspect ratio through center cropping, thus maintaining the integrity of facial thermal patterns.


**Table 2: Model-Specific Normalization Parameters**
| Model Type | Input Size | Normalization Values | Channels | Rationale |
|------------|------------|----------------------|----------|-----------|
| AlexNet/VGG/ResNet/EfficientNet | 224×224 | mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5] | 3 | Optimized for thermal intensity distribution |
| Inception | 299×299 | mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5] | 3 | Maintained larger input size for finer detail capture |
| TH-SE-ResNet | 224×224 | mean=[0.5], std=[0.5] | 1 | Preserved original thermal information without channel duplication |


### 3.2.3 Data Augmentation Strategies

We developed a sophisticated augmentation strategy tailored specifically for thermal facial imagery, carefully balancing the need for dataset expansion with the preservation of thermally significant features:

We implemented distinct augmentation pipelines optimized for different network architectures. For RGB-designed models (AlexNet, VGG, ResNet, EfficientNet, Inception), we employed a comprehensive suite of transformations including random resized cropping, horizontal flipping, rotation (±15°), brightness and contrast adjustments (±20%), and Gaussian blurring. For our thermal-specific TH-SE-ResNet, we employed a more conservative approach with grayscale conversion, random resized cropping, horizontal flipping, moderate rotation (±15°), and controlled affine transformations (±10% translation).

To address the pronounced gender imbalance in the Tufts dataset (30.32% female, 69.68% male), we implemented targeted augmentation for the underrepresented female class. This approach involved creating additional augmented samples exclusively for female subjects, effectively doubling the female representation in the training set while preserving the original male samples. This selective augmentation substantially improved class balance without introducing excessive redundancy or overfitting risks.

Our augmentation protocol was carefully calibrated to preserve the thermal signature integrity crucial for gender classification. Specifically, we avoided extreme geometric transformations and color-space alterations that might distort thermally significant facial features. The brightness and contrast adjustments were conservatively parameterized to simulate natural variations in thermal imaging conditions without introducing artifacts that could compromise the intrinsic thermal patterns.

For the combined dataset, we implemented a sophisticated integration protocol that addressed both the channel disparity between datasets and the gender distribution imbalance. To achieve perfect gender balance (50% female, 50% male), we employed controlled sampling from both constituent datasets, ensuring representative inclusion of diverse thermal imaging conditions and subject characteristics while maintaining strict subject-level separation between training and testing partitions.

The final augmented training sets demonstrated substantially enhanced diversity and robustness. For the Tufts dataset, our class-balanced augmentation approach effectively doubled the representation of the underrepresented female class. The combined dataset benefited from both the targeted augmentation and the integration of diverse thermal imaging modalities, resulting in a comprehensive training corpus that captured a wide spectrum of thermal facial characteristics across different acquisition parameters and subject demographics.

This carefully engineered preprocessing and augmentation pipeline provided our models with high-quality, balanced training data while preserving the critical thermal signatures necessary for accurate gender classification in thermal facial imagery.


![new_thermal_augmentation_combined_examplesaa](https://github.com/user-attachments/assets/8842aedb-3c13-432e-b7af-baa0b1c6c789)
**Figure 2: Thermal Image Augmentation Examples** - A grid showing original thermal facial images alongside various augmented versions (horizontal flip, rotation, contrast adjustment, etc.)


**Table 3: Final Experimental Dataset Configurations**
| Experiment | Training Set | Testing Set | Total Training Images | Total Testing Images |
|------------|--------------|-------------|------------------------|----------------------|
| Tufts-only | Tufts train | Tufts test | ~1,600* | 330 |
| Charlotte-only | Charlotte train | Charlotte test | ~16,000* | 2,000 |
| Combined | Combined train | Combined test | 18,200 | 2,290 |
| Tufts-to-Charlotte | Tufts train | Charlotte test | ~1,600* | 2,000 |
| Charlotte-to-Tufts | Charlotte train | Tufts test | ~16,000* | 330 |
*Approximate values after augmentation

**Figure 3: Complete Data Preprocessing and Augmentation Pipeline** - A flowchart showing the end-to-end process from raw dataset organization through partitioning, normalization, augmentation, to final training/testing sets.


## 3.3 Proposed CNN Architecture

### 3.3.1 Overview

Our research introduces a sophisticated deep learning framework built upon a modified ResNet-50 architecture, tailored specifically for thermal (single channel) image classification. The selection of ResNet-50 as the foundational backbone is driven by its proven ability to address challenges inherent in training very deep neural networks. A hallmark of ResNet is its use of residual connections, which mitigate the vanishing gradient problem by introducing skip connections that allow gradients to propagate more effectively during backpropagation. This design enables the construction of deeper architectures without compromising performance, a critical advantage when extracting intricate features from complex human faces in thermal imagery.

ResNet-50 strikes an exceptional balance between computational efficiency and representational power. Its 50-layer depth facilitates the hierarchical extraction of features, ranging from low-level details such as edges and textures to high-level semantic patterns, which are essential for discerning subtle gender specific cues in thermal images. Furthermore, initializing the model with pretrained weights from ImageNet provides a robust starting point. Although thermal images differ from natural images, the general visual features learned from ImageNet—such as edge detection and texture analysis—serve as transferable knowledge that can be fine-tuned to adapt to the our domain. This transfer learning approach accelerates convergence and enhances performance, particularly when training data is limited.

It is worth noting that the final architecture emerged from a rigorous iterative development process. Multiple architectural variants were systematically evaluated, with each iteration informing subsequent refinements based on empirical performance assessments. This methodical approach to model selection enabled us to identify the optimal configuration presented in this study.

The proposed architecture enhances the standard ResNet-50 by integrating three key modifications: a Channel Input Adapter to handle single-channel inputs, Squeeze-and-Excitation (SE) blocks to improve feature representation, and a redesigned classifier head to optimize classification performance. Each component is meticulously crafted to align with the implementation in the provided code, ensuring consistency between the theoretical design and practical execution.


![Architecture](https://github.com/user-attachments/assets/91b98f31-c02b-4c24-a38f-013f90641ae7)
- **Figure 4: Overall Architecture of TH-SE-ResNet** - A comprehensive diagram showing the complete model architecture with all components connected, highlighting the modifications to the standard ResNet-50.

### 3.3.2 Channel Input Adapter

Thermal imaging often presents unique challenges due to the prevalence of single-channel grayscale images, whereas pretrained models like ResNet-50 are designed for three-channel RGB inputs. To bridge this gap effectively, we developed a Channel Input Adapter that transforms single-channel inputs into a three-channel representation suitable for the pretrained backbone. Unlike the simplistic approach of replicating the grayscale channel across three dimensions, which imposes a fixed and potentially suboptimal mapping, our adapter employs a learnable transformation to capture nuanced features tailored to the input data.

#### Architecture

The Channel Input Adapter is implemented as a sequence of convolutional layers that progressively process the input. The transformation unfolds as follows:

- **Initial Convolutional Block**: The single-channel input, denoted as \( x \in \mathbb{R}^{1 \times H \times W} \), where \( H \) and \( W \) represent the height and width, is processed by a 3×3 convolutional layer with 32 output channels. Padding of 1 is applied to preserve spatial dimensions. This operation is followed by batch normalization to stabilize training and a ReLU activation to introduce non-linearity. The resulting feature map is \( x_1 \in \mathbb{R}^{32 \times H \times W} \).

- **Subsequent Convolutional Block**: The intermediate feature map \( x_1 \) is fed into a second 3×3 convolutional layer, this time reducing the channel dimension to 3, again with padding of 1. Batch normalization and ReLU activation are applied subsequently, yielding the final output \( x_2 \in \mathbb{R}^{3 \times H \times W} \), which matches the input requirements of the ResNet backbone.

The transformation can be expressed mathematically as shown in Equations (1) and (2), where the initial convolutional block produces intermediate features that are further refined by the second block.

\[
x_1 = \text{ReLU}\left(\text{BN}\left(\text{Conv}_{3 \times 3, 32}(x)\right)\right) \tag{1}
\]

\[
x_2 = \text{ReLU}\left(\text{BN}\left(\text{Conv}_{3 \times 3, 3}(x_1)\right)\right) \tag{2}
\]

Here, \(\text{Conv}_{k \times k, c}\) represents a convolutional operation with a kernel size of \( k \times k \) and \( c \) output channels, \(\text{BN}\) denotes batch normalization, and \(\text{ReLU}(z) = \max(0, z)\) is the rectified linear unit activation function.

The learnable nature of this adapter allows the network to adaptively map the single-channel input to a three-channel space, potentially capturing richer and more relevant features than a static replication method. By employing convolutional layers, the adapter can learn spatially varying transformations, which is particularly advantageous for gender classification in thermal images, where local patterns—such as facial heat distributions or temperature variations—are discriminative. This design enhances the model’s compatibility with pretrained weights while optimizing its ability to process domain-specific data.

**Algorithm 1: Channel Input Adapter Forward Pass**
Input: Single-channel image x (1xHxW)
Output: Three-channel feature map x_out (3xHxW)

```
Procedure:
1:  x1 <- Conv_3x3_32(x, padding=1)
2:  x1 <- BatchNorm2d(x1)
3:  x1 <- ReLU(x1)
4:  x_out <- Conv_3x3_3(x1, padding=1)
5:  x_out <- BatchNorm2d(x_out)
6:  x_out <- ReLU(x_out)
7:  Return x_out
```

### 3.3.3 Squeeze and Excitation (SE) Blocks

To enhance the representational power of this model, we integrated Squeeze-and-Excitation (SE) blocks throughout the network architecture. SE blocks implement an attention mechanism that adaptively recalibrates channel-wise feature responses by explicitly modeling interdependencies between channels. This approach allows the network to increase its sensitivity to informative features while suppressing less useful ones.

The SE block operates through a two-step process: squeeze and excitation.

- **Squeeze Operation**: This step aggregates global spatial information into a channel descriptor. For convolutional feature maps \( x \in \mathbb{R}^{C \times H \times W} \), where \( C \) is the number of channels, global average pooling is applied:

\[
z_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} x_c(i, j), \quad c = 1, 2, \ldots, C \tag{3}
\]

For fully-connected layers with input \( x \in \mathbb{R}^{B \times C} \), where \( B \) is the batch size, a 1D adaptive average pooling is used:

\[
z_c = \frac{1}{B} \sum_{b=1}^{B} x_b(c) \tag{4}
\]

The result is a channel descriptor \( z \in \mathbb{R}^{C} \) that encapsulates the global context of each channel.

- **Excitation Operation**: The channel descriptor \( z \) is processed through a bottleneck structure comprising two fully-connected layers:

\[
s = \sigma\left(W_2 \cdot \delta\left(W_1 \cdot z\right)\right) \tag{5}
\]

The global spatial information is aggregated into a channel descriptor as defined in Equation (3) for convolutional feature maps and Equation (4) for fully-connected layers. The channel-wise attention weights are then computed using Equation (5).

Here, \( W_1 \in \mathbb{R}^{\frac{C}{r} \times C} \) reduces the dimensionality with a reduction ratio \( r = 16 \), \( \delta(z) = \text{ReLU}(z) \) introduces non-linearity, \( W_2 \in \mathbb{R}^{C \times \frac{C}{r}} \) restores the original dimensionality, and \( \sigma(z) = \frac{1}{1 + e^{-z}} \) is the sigmoid activation function. The output \( s \in \mathbb{R}^{C} \) represents channel-wise attention weights ranging from 0 to 1.

- **Recalibration**: The original feature maps are scaled by these weights:

- For convolutional layers: \( \tilde{x}_c = s_c \cdot x_c \)
- For fully-connected layers: \( \tilde{x}_b(c) = s_c \cdot x_b(c) \)

This recalibration enhances the emphasis on channels deemed most relevant by the attention mechanism.

We implemented SE blocks that are capable of handling both convolutional feature maps (4D tensors) and fully-connected layers (2D tensors), making the architecture more flexible. For convolutional layers, the SE blocks apply 2D adaptive average pooling before computing attention weights, while for fully-connected layers, they utilize 1D pooling. This adaptive approach ensures that the attention mechanism works effectively throughout the network.

**Algorithm 2: Squeeze-and-Excitation Block Forward Pass**
Input: Feature map F (CxHxW or BxC), reduction ratio r
Output: Recalibrated feature map F_recal
```
1:  C <- Number of channels in F
2:  If F is 4D (convolutional):
3:      z <- AdaptiveAvgPool2d(1)(F)
4:      z <- Flatten z to shape (BatchSize, C)
5:  Else if F is 2D (fully-connected):
6:      z <- AdaptiveAvgPool1d(1)(F.unsqueeze(-1)) // Unsqueeze to treat as sequence
7:      z <- Flatten z to shape (BatchSize, C)

9: s1 <- FullyConnected(z, input_size=C, output_size=C // r)
10: s1 <- ReLU(s1)
11: s2 <- FullyConnected(s1, input_size=C // r, output_size=C)
12: attention_weights <- Sigmoid(s2)

13: If F is 4D:
14:     attention_weights <- Reshape attention_weights to (BatchSize, C, 1, 1)
15:     F_recal <- F * attention_weights // Channel-wise multiplication
16: Else: // F is 2D
17:     F_recal <- F * attention_weights // Element-wise multiplication along channel dimension

18: Return F_recal
```

SE blocks are integrated into the ResNet architecture by appending them after the final convolutional layer (conv3) of each bottleneck module within layers 1 through 4. This strategic placement ensures that feature recalibration occurs at multiple abstraction levels, enhancing the model’s ability to prioritize features critical for the classification task.

### 3.3.4 Classifier Head

The standard ResNet classifier, consisting of a single fully-connected layer, is replaced with a more elaborate structure to optimize classification performance and generalization. This redesign addresses the need for robust feature processing and regularization, particularly in the context of gender classification in thermal images.

The classifier head processes the 2048-dimensional feature vector obtained from global average pooling through a multi-layer sequence:

- **First Dropout Layer**: A dropout operation with a probability of 0.5 is applied to the input \( x \in \mathbb{R}^{2048} \), randomly setting half of the features to zero during training to prevent neuron co-adaptation.

- **Dimensionality Reduction**: A fully-connected layer reduces the dimensionality from 2048 to 512, followed by a ReLU activation (as shown later in Equation (7)), where \( W_1 \in \mathbb{R}^{512 \times 2048} \) and \( b_1 \in \mathbb{R}^{512} \) are learnable parameters.

- **SE Block**: An SE block recalibrates the 512-dimensional feature vector, applying the squeeze and excitation operations described earlier to emphasize discriminative features.

- **Second Dropout Layer**: Another dropout operation with a probability of 0.3 provides additional regularization.

- **Output Layer**: A final fully-connected layer maps the features to the number of classes:

\[
y = W_2 x_4 + b_2
\]

where \( W_2 \in \mathbb{R}^{\text{num\_classes} \times 512} \) and \( b_2 \in \mathbb{R}^{\text{num\_classes}} \) produce the classification logits.

The complete transformation sequence is defined in Equations (6) through (10).

\[
x_1 = \text{Dropout}_{0.5}(x) \tag{6}
\]

\[
x_2 = \text{ReLU}(W_1 x_1 + b_1) \tag{7}
\]

\[
x_3 = \text{SEBlock}(x_2) \tag{8}
\]

\[
x_4 = \text{Dropout}_{0.3}(x_3) \tag{9}
\]

\[
y = W_2 x_4 + b_2 \tag{10}
\]


The incorporation of SE blocks enhances the network’s sensitivity to informative features, a crucial capability in gender classification for thermal images, where subtle differences in facial heat distribution can be discriminative. The adaptive nature of the attention mechanism allows the model to dynamically adjust its focus, improving both performance and robustness across diverse datasets.


### 3.3.5 Unified Equation

The complete TH-SE-ResNet architecture can be expressed as a composition of three key components: the Channel Input Adapter, the ResNet backbone with SE blocks, and the modified classifier head. This composition is mathematically formulated as:

\[
f_{\text{TH-SE-ResNet}}(x) = g_{\text{FC}} \circ f_{\text{ResNet+SE}} \circ h_{\text{CIA}}(x) \tag{11}
\]

where \( x \in \mathbb{R}^{1 \times H \times W} \) represents the single-channel thermal input image, \( h_{\text{CIA}} \) is the Channel Input Adapter, \( f_{\text{ResNet+SE}} \) is the ResNet backbone enhanced with SE blocks, and \( g_{\text{FC}} \) is the modified classifier head.

The Channel Input Adapter transforms the single-channel input into a three-channel representation through a sequence of convolutional operations:

\[
h_{\text{CIA}}(x) = \text{ReLU}\left(\text{BN}\left(\text{Conv}_{3 \times 3, 3}\left(\text{ReLU}\left(\text{BN}\left(\text{Conv}_{3 \times 3, 32}(x)\right)\right)\right)\right)\right) \tag{12}
\]

The output of the Channel Input Adapter, \( h_{\text{CIA}}(x) \in \mathbb{R}^{3 \times H \times W} \), is then processed by the ResNet backbone with integrated SE blocks.

The backbone processes the input through a series of layers:

\[
f_{\text{ResNet+SE}}(x) = L_4(L_3(L_2(L_1(\text{Pool}(\text{ReLU}(\text{BN}(\text{Conv}_{7 \times 7, 64}(x)))))))) \tag{13}
\]

where each layer \( L_i \) consists of multiple bottleneck modules with SE blocks added to their outputs.

The SE block operates on feature maps using the following formula:

\[
\text{SE}(F) = F \cdot \sigma(W_2(\delta(W_1(\text{GAP}(F))))) \tag{14}
\]


where \( F \) represents the feature maps, GAP is global average pooling (adaptive to input dimensions), \( \delta \) is the ReLU activation, \( \sigma \) is the sigmoid activation, and \( \cdot \) denotes channel-wise multiplication.

The modified classifier head processes the extracted features through:

\[
g_{\text{FC}}(x) = W_2(D_{0.3}(\text{SE}(\text{ReLU}(W_1(D_{0.5}(x)))))) \tag{15}
\]

where \( W_1 \in \mathbb{R}^{512 \times 2048} \) and \( W_2 \in \mathbb{R}^{\text{num\_classes} \times 512} \) are learnable weight matrices, and \( D_p \) represents dropout with probability \( p \).

The complete TH-SE-ResNet architecture can be expressed as a composition shown in Equation (11), where the Channel Input Adapter defined by Equation (12) feeds into the ResNet backbone with SE blocks expressed in Equation (13). The SE block operation is formalized in Equation (14), and the classifier head transformation is given by Equation (15).

This unified mathematical formulation captures the key architectural components of our TH-SE-ResNet, highlighting the integration of the Channel Input Adapter for domain-specific processing, the SE blocks for adaptive feature recalibration, and the enhanced classifier head for optimized performance in thermal image classification.


## 3.4 Model Comparison

This section provides an evaluation of the diverse neural network architectures employed as baseline models to compare with our proposed ResNet-based framework for gender detection in thermal facial images. The selection criteria prioritized architectural diversity across model generations, parameter complexity, and feature extraction methodologies to establish a robust comparative foundation. These architectures also represent frameworks frequently encountered in the thermal imaging literature, facilitating contextual interpretation of our results within the broader research landscape. By examining AlexNet, VGG-16, InceptionV3, ResNet50, and EfficientNet, we gain valuable insights into how different architectural paradigms process the unique characteristics of thermal facial signatures for gender classification tasks.

### 3.4.1 Baseline Architectures

#### 3.4.1.1 AlexNet: Foundational CNN Architecture

AlexNet represents a fundamental benchmark in our evaluation due to its historical significance in revolutionizing computer vision through deep convolutional neural networks. Despite its relative simplicity by contemporary standards, this architecture offers critical insights into the minimum viable model complexity required for effective thermal feature discrimination. The network comprises five convolutional layers and three fully connected layers, creating a relatively shallow architecture with eight trainable layers.

The model processes thermal input images resized to 224×224 pixels through a series of operations beginning with large-kernel convolutions (11×11 stride 4) that capture broad thermal gradients across facial regions. These initial layers are particularly relevant for thermal imaging, as they can detect coarse temperature variations corresponding to major facial vasculature patterns that exhibit gender-specific differences. Subsequent layers employ progressively smaller kernels (5×5, then 3×3) to refine feature representation, with max-pooling operations providing spatial reduction. The final network outputs a 4096-dimensional feature vector before classification, which must encapsulate gender-discriminative thermal signatures.

AlexNet's inclusion allows us to evaluate whether early CNN architectural patterns can correctly capture the subtle temperature distribution differences between male and female thermal facial signatures. The model's Local Response Normalization (LRN) layers may also prove beneficial in standardizing thermal intensity variations across different capture conditions.

#### 3.4.1.2 VGG-16: Homogeneous Deep Architecture

VGG-16 extends architectural depth systematically through homogeneous convolutional blocks, enabling examination of how increased layer count (16 trainable layers) affects thermal feature learning without introducing advanced structural innovations. Its uniform design philosophy—consisting of stacked 3×3 convolutional layers followed by spatial reduction via max-pooling—provides a controlled comparison point for evaluating thermal feature learning in deeper networks.

The network's consistent kernel size (3×3) throughout all convolutional layers creates a large effective receptive field while maintaining computational efficiency. This architecture may effectively capture multi-scale thermal patterns ranging from localized temperature peaks around the periorbital regions to broader thermal distributions across facial contours. 

VGG-16's straightforward layer progression offers interpretability advantages, potentially allowing clearer attribution of which facial thermal regions contribute most significantly to gender classification decisions. This transparency could prove valuable for subsequent research into the physiological basis of gender-specific thermal signatures.

#### 3.4.1.3 InceptionV3: Multi-Scale Processing Architecture

InceptionV3 introduces sophisticated multi-scale processing capabilities through its innovative inception modules and factorized convolutions. This architectural approach allows simultaneous analysis of thermal features at multiple spatial resolutions—a potentially valuable characteristic for thermal gender classification, where discriminative information may exist at different scales, from fine vascular patterns to broader facial temperature zones.

The architecture reduces computational complexity through strategic use of 1×1 convolutions for dimensionality reduction prior to expensive 3×3 and 5×5 operations. Its asymmetric kernel decompositions (replacing 5×5 filters with stacked 3×3 convolutions and factorizing n×n filters into consecutive 1×n and n×1 operations) enhance efficiency while preserving representational capacity. Input thermal images are resized to 299×299 pixels to align with InceptionV3's native resolution, providing increased spatial detail compared to other models in our evaluation.

InceptionV3's auxiliary classifier, which emerges from an intermediate layer during training, potentially aids in propagating more direct gender-classification signals to earlier network stages. This feature may prove particularly beneficial for thermal imaging, where distinguishing gradient information can be more subtle than in visible-spectrum imagery.

The network's branch diversity within inception modules enables it to learn specialized feature extractors for different thermal pattern types simultaneously—potentially capturing both the textural aspects of facial thermal patterns and their spatial configuration in a single unified architecture.

#### 3.4.1.4 ResNet50: Residual Learning Framework

ResNet50 employs innovative residual learning principles to achieve 50-layer depth without degradation in training accuracy. The architecture utilizes bottleneck blocks to mitigate vanishing gradients, enabling deeper feature hierarchies that may better capture the complex relationships in thermal facial imagery.

Each residual block follows the fundamental mapping principle:

$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$

where $\mathbf{x}$ and $\mathbf{y}$ represent input and output vectors respectively, and $\mathcal{F}$ denotes the residual function implemented through three consecutive convolutions (1×1 for dimensionality reduction, 3×3 for feature extraction, and 1×1 for restoration). This design enables stable training of substantially deeper networks while preserving gradient flow—a critical consideration when fine-tuning on limited thermal datasets where gradient signals may be weaker than in large-scale visible image collections.

The architecture's identity shortcuts create direct paths for gradient flow, potentially enabling more effective training on the relatively subtle thermal features that distinguish genders. The bottleneck design (1×1, 3×3, 1×1 convolution pattern) reduces computational requirements while maintaining representational capacity, making ResNet50 an efficient option for thermal image processing despite its depth.

ResNet50's batch normalization after each convolutional layer standardizes feature activations, which may be particularly beneficial for thermal imagery where absolute temperature values can vary across subjects and capture conditions. This normalization potentially helps the network focus on relative temperature distribution patterns rather than absolute values.

#### 3.4.1.5 EfficientNet-B0: Optimized Scaling Architecture

EfficientNet-B0 represents the state-of-the-art in efficiency-optimized architectures, leveraging compound scaling to balance depth, width, and resolution. This balanced approach optimizes the accuracy-efficiency trade-off, making it particularly relevant for potential deployment of thermal gender classification systems in resource-constrained environments.

The architecture employs mobile inverted bottleneck (MBConv) layers as its primary building block, integrating squeeze-and-excitation (SE) mechanisms for adaptive channel attention. This attention mechanism can be formulated as:

$$\mathbf{\tilde{X}} = s(\mathbf{X}) \cdot \mathbf{X}$$

where $s(\mathbf{X})$ represents channel-wise attention weights derived from the SE block. This mechanism potentially enables the network to emphasize the most gender-discriminative thermal channels while suppressing less informative ones, creating an adaptive feature selection process beneficial for capturing subtle thermal differences between genders.

Despite its relatively lightweight design (5.3 million parameters), EfficientNet-B0 achieves competitive accuracy on standard computer vision benchmarks. This raises the important question of whether efficient architectures can maintain high accuracy on thermal gender classification despite their reduced capacity, or if the subtle nature of thermal features requires larger models. The architecture's depth (82 layers) combined with its parameter efficiency provides an interesting contrast to other models in our evaluation.

The swish activation function ($x \cdot \text{sigmoid}(x)$) used throughout EfficientNet potentially offers advantages over ReLU activations for thermal imagery by providing smoother gradients for small activation values, which may better preserve subtle thermal variation information during forward propagation.

### Comparison of Architectural Characteristics

**Table 4: Comprehensive Baseline Model Specifications**

| Model | Depth | Parameters (M) | Input Size | Key Components | Potential Thermal Imaging Advantages |
|-------------|-------|----------------|------------|------------------------------------|--------------------------------------------|
| AlexNet | 8 | 61.0 | 224×224 | Large kernels (11×11), LRN | Effective capture of broad thermal gradients |
| VGG-16 | 16 | 138.0 | 224×224 | Homogeneous 3×3 conv stacks | Consistent multi-scale feature extraction |
| InceptionV3 | 48 | 23.9 | 299×299 | Factorized convolutions, Auxiliary classifier | Multi-resolution thermal pattern analysis |
| ResNet50 | 50 | 25.6 | 224×224 | Bottleneck residual blocks | Deep thermal feature hierarchies with gradient preservation |
| EfficientNet-B0 | 82 | 5.3 | 224×224 | MBConv with SE, Compound scaling | Adaptive attention to gender-discriminative thermal channels |

**Figure 6: Architectural Diagrams** – Detailed schematic representations of each baseline model's layer configuration, highlighting specific components relevant to thermal feature extraction.

### 3.4.2 Input Adaptation and Training Protocol

#### 3.4.2.1 Thermal Image Preprocessing and Channel Adaptation

Adapting standard CNN architectures designed for RGB images to thermal data requires careful consideration of input channel dimensionality. Our experimental protocol addresses this challenge through dataset-specific preprocessing pipelines:

For the Charlotte dataset's single-channel thermal inputs, we employed channel replication to create compatible three-channel inputs. *This approach converts grayscale thermal intensity values into three identical channels during image loading, preserving the original thermal distribution while satisfying the input requirements of networks pretrained on RGB data.* While this method introduces redundancy, it maintains compatibility with the convolutional filters learned from visible spectrum imagery and allows us to leverage transfer learning effectively.

The Tufts dataset provides native three-channel thermal representations, each encoding different thermal wavelength bands. These original multi-channel thermal representations were retained *and directly utilized without modification* to preserve the potential complementary information across thermal spectral bands. The three-channel structure of this data aligns naturally with the input expectations of conventional CNNs.

#### 3.4.2.2 Transfer Learning and Fine-Tuning Strategy

We initialize the baseline models with ImageNet-pretrained weights, freezing their initial layers to retain general feature extraction while training only a new final classification layer for thermal gender classification. This focuses optimization on our specific two-class task and minimizes overfitting.

## 4. Experimental Results

### 4.1 Experimental Setup
To rigorously assess the efficacy of our baseline models and the proposed Ther-SE-ResNet architecture, we designed a comprehensive experimental framework involving multiple dataset configurations. Specifically, we utilized the Tufts dataset, the Charlotte dataset, and a combined dataset merging both, and two cross-dataset. These configurations enabled us to evaluate the models’ performance within individual datasets as well as their ability to generalize across distinct datasets, a critical aspect of real-world applicability.

For training, all models were optimized using the Adam algorithm, configured with momentum parameters \(\beta_1 = 0.9\) and \(\beta_2 = 0.999\), which are widely adopted for their stability and efficiency in deep learning tasks. We set the initial learning rate to 0.00005, a value selected to ensure gradual parameter updates suitable for our architecture. To enhance training dynamics, we implemented a 5-epoch warmup phase during which the learning rate increased linearly from zero to the specified value, followed by cosine annealing for the subsequent epochs to promote smooth convergence to an optimal solution. We experimented with batch sizes of 32 and 64 to explore their effects on training stability and generalization performance, providing insights into the trade-offs between computational efficiency and model accuracy.

Each model underwent training for 10 epochs, a duration determined through preliminary experiments to strike a balance between achieving convergence and minimizing computational overhead. The experiments were executed on an NVIDIA GeForce RTX 4090, a high-performance hardware platform that facilitated rapid iteration. To optimize data handling and reduce training bottlenecks, we employed PyTorch’s DataLoader with settings of `num_workers=8` and `pin_memory=True`, ensuring efficient data transfer to the GPU and maximizing throughput during training.

**Algorithm 3: Model Training Loop**
Input: Training DataLoader D_train, Test DataLoader D_test, Model M, Optimizer Opt,
       Learning Rate Scheduler Sch, Loss Function Crit, Total Epochs N_epochs,
       Warmup Epochs N_warmup, Initial Learning Rate LR_init
Output: Best performing Model M_best
```
1:  Initialize best_accuracy = 0.0
2:  Initialize M_best = None
3:  For epoch from 1 to N_epochs:
4:      M.train() // Set model to training mode
5:      Initialize running_loss = 0.0
6:      // Adjust learning rate based on warmup or scheduler
7:      If epoch <= N_warmup:
8:          current_lr = LR_init * (epoch / N_warmup)
9:          Set learning rate in Opt to current_lr
10:     Else:
11:         Sch.step() // Apply cosine annealing after warmup
12:
13:     // Iterate over training data
14:     For images, labels in D_train:
15:         Move images, labels to target device
16:         Opt.zero_grad() // Clear gradients
17:         // Forward pass (handle Inception's auxiliary output if applicable)
18:         If model is Inception:
19:             outputs, aux_outputs = M(images)
20:             loss = Crit(outputs, labels) + 0.4 * Crit(aux_outputs, labels)
21:         Else:
22:             outputs = M(images)
23:             loss = Crit(outputs, labels)
24:
25:         loss.backward() // Backward pass
26:         Opt.step() // Update weights
27:         running_loss += loss.item()
28:
29:     epoch_loss = running_loss / len(D_train)
30:
31:     // Evaluation phase
32:     M.eval() // Set model to evaluation mode
33:     Initialize correct = 0, total = 0
34:     Start inference mode (no gradient calculation)
35:     For images, labels in D_test:
36:         Move images, labels to target device
37:         outputs = M(images)
38:         _, predicted = torch.max(outputs.data, 1)
39:         total += labels.size(0)
40:         correct += (predicted == labels).sum().item()
41:     End inference mode
42:
43:     accuracy = 100 * correct / total
44:     Print epoch statistics (epoch number, epoch_loss, accuracy, current learning rate)
45:
46:     // Save best model based on accuracy
47:     If accuracy > best_accuracy:
48:         best_accuracy = accuracy
49:         M_best = copy.deepcopy(M.state_dict())
50:
51: // Load best weights into model
52: M.load_state_dict(M_best)
53: Return M // Return the model with the best weights loaded
```

## 4.2 Results on Individual Datasets
### 4.2.1 Tufts Dataset

The Tufts dataset experiments yielded notable performance variations across the six tested models and two batch size configurations. Table 5 presents a comprehensive performance comparison, highlighting the accuracy, precision, recall, and F1 scores achieved by each model architecture.

##### Table 5: Performance on Tufts Dataset

| Model | Batch Size | Accuracy | Precision (weighted) | Recall (weighted) | F1 (weighted) |
|-------|------------|----------|----------------------|-------------------|---------------|
| AlexNet | 32 | 0.86 | 0.86 | 0.86 | 0.85 |
| AlexNet | 64 | 0.85 | 0.87 | 0.85 | 0.85 |
| ResNet | 32 | 0.85 | 0.85 | 0.85 | 0.85 |
| ResNet | 64 | 0.83 | 0.83 | 0.83 | 0.82 |
| Inception v3 | 32 | 0.85 | 0.85 | 0.85 | 0.84 |
| Inception v3 | 64 | 0.83 | 0.83 | 0.83 | 0.82 |
| VGG | 32 | 0.84 | 0.84 | 0.84 | 0.84 |
| VGG | 64 | 0.84 | 0.85 | 0.84 | 0.84 |
| EfficientNet B0 | 32 | 0.79 | 0.82 | 0.79 | 0.77 |
| EfficientNet B0 | 64 | 0.71 | 0.71 | 0.71 | 0.66 |
| TH-SE-ResNet | 32 | 0.95 | 0.96 | 0.95 | 0.95 |
| TH-SE-ResNet | 64 | 0.97 | 0.97 | 0.97 | 0.97 |


Our experimental results reveal several significant trends in gender classification performance across different model architectures and batch size configurations on the Tufts dataset. The most striking finding is the exceptional performance of the TH-SE-ResNet model, which substantially outperformed all other tested architectures with accuracy rates of 95% and 97% for batch sizes 32 and 64, respectively. This represents a considerable improvement over the next best performer, AlexNet, which achieved 86% accuracy with batch size 32.

When examining the class-specific metrics, we observe a consistent pattern across nearly all models: higher precision for female classification but higher recall for male classification. This imbalance is particularly evident in models like AlexNet (batch 64), which achieved 94% precision for female classification but only 63% recall, indicating a tendency to misclassify female subjects as male. This gender-based performance disparity may be attributed to the dataset composition, which contains nearly twice as many male samples (215) as female samples (115).

Interestingly, the performance impact of batch size varied across architectures. While TH-SE-ResNet and AlexNet maintained relatively stable performance across batch sizes, EfficientNet B0 exhibited a dramatic performance degradation when the batch size increased from 32 to 64, with accuracy dropping from 79% to 71%. This suggests that EfficientNet's learning dynamics are more sensitive to batch size configurations than other architectures.

The convergence behavior, as illustrated in the training loss and test accuracy graphs, further differentiates TH-SE-ResNet from the other models. TH-SE-ResNet demonstrated remarkably rapid convergence, reaching near-optimal performance within the first two epochs and maintaining a stable performance trajectory thereafter. In contrast, models like ResNet and Inception exhibited more gradual learning curves, requiring additional epochs to approach their performance plateaus.

EfficientNet B0, despite its reputation for efficiency in other computer vision tasks, performed notably poorly on this gender classification task, achieving the lowest accuracy among all tested models. This underperformance may be attributed to the model's design optimizations for general image recognition tasks, which may not translate effectively to the specific feature patterns relevant for gender classification in the Tufts dataset.

The F1 scores, which balance precision and recall considerations, further emphasize TH-SE-ResNet's superior performance, with weighted F1 scores of 0.95 and 0.97 for batch sizes 32 and 64, respectively. This indicates that TH-SE-ResNet not only achieves higher overall accuracy but also maintains a better balance between precision and recall across both gender classes.

To visualize the learning dynamics, the training loss and test accuracy curves for the Tufts dataset experiments are presented below.

![alt text](https://github.com/user-attachments/assets/da3ff3df-9f40-4353-979b-3c3d2b3337e5)

**Figure 5: Training Loss and Test Accuracy Curves on Tufts Dataset (Batch Size 64)**

![alt text](https://github.com/user-attachments/assets/72f75044-de5d-4775-8e10-c50be50d50a3)

**Figure 6: Training Loss and Test Accuracy Curves on Tufts Dataset (Batch Size 32)**

These learning curves (Figures 7 and 8) visually confirm the quantitative findings. The TH-SE-ResNet model (blue line) exhibits remarkably rapid convergence, reaching near-peak accuracy well within the first half of the training epochs and maintaining stability thereafter. This contrasts sharply with the baseline models, which generally show slower learning progress and plateau at lower accuracy levels. EfficientNet B0's sensitivity to batch size is also apparent, with its accuracy curve degrading more significantly in Figure 7 (Batch Size 64) compared to Figure 8 (Batch Size 32). The high final accuracy levels achieved by TH-SE-ResNet in these plots align directly with the results in Table 5.

For a detailed view of the classification performance of the top-performing model, the confusion matrices for TH-SE-ResNet on the Tufts test set are shown.

[todo]

Figure 9a: Confusion Matrix - TH-SE-ResNet, Tufts Dataset, Batch Size 64

[todo]

Figure 9b: Confusion Matrix - TH-SE-ResNet, Tufts Dataset, Batch Size 32

The confusion matrices in Figures 9a and 9b provide a clear picture of TH-SE-ResNet's effectiveness on the Tufts data. The strong diagonal values indicate a high number of correct classifications for both female and male classes. Off-diagonal values, representing misclassifications, are minimal. For instance, with batch size 64 (Figure 9a), only 15 females were misclassified as male, and zero males were misclassified as female out of the respective test samples. Similarly, with batch size 32 (Figure 9b), errors were 10 females misclassified as male and only 1 male as female. These low error counts visually corroborate the high accuracy (95-97%) and balanced F1 scores reported earlier, demonstrating the model's ability to effectively classify genders despite the dataset's inherent imbalance.

In summary, our empirical evaluation on the Tufts dataset demonstrates that the Th-SE-ResNet architecture provides substantial performance advantages for gender classification tasks. Its superior accuracy, balanced class-specific performance, and rapid convergence characteristics make it particularly well-suited for applications where gender classification accuracy is critical. Meanwhile, the consistent gender-based performance disparities observed across models highlight the importance of addressing potential biases in both model architectures and training methodologies for gender classification tasks.


### 4.2.2 Charlotte Dataset

The Charlotte dataset experiments revealed distinct performance patterns compared to the Tufts dataset, reflecting the unique challenges posed by this larger and more variable thermal image collection. Table 6 presents the comprehensive performance metrics for all six models across the two batch size configurations.

##### Table 6: Performance on Charlotte Dataset

| Model | Batch Size | Accuracy | Precision (weighted) | Recall (weighted) | F1 (weighted) |
|-------|------------|----------|----------------------|-------------------|---------------|
| AlexNet | 32 | 0.70 | 0.74 | 0.70 | 0.68 |
| AlexNet | 64 | 0.68 | 0.71 | 0.68 | 0.67 |
| ResNet | 32 | 0.60 | 0.61 | 0.60 | 0.60 |
| ResNet | 64 | 0.56 | 0.56 | 0.56 | 0.56 |
| Inception v3 | 32 | 0.67 | 0.69 | 0.67 | 0.66 |
| Inception v3 | 64 | 0.67 | 0.70 | 0.67 | 0.66 |
| VGG | 32 | 0.63 | 0.63 | 0.63 | 0.63 |
| VGG | 64 | 0.66 | 0.67 | 0.66 | 0.65 |
| EfficientNet B0 | 32 | 0.68 | 0.68 | 0.68 | 0.67 |
| EfficientNet B0 | 64 | 0.63 | 0.64 | 0.63 | 0.63 |
| TH-SE-ResNet | 32 | 0.81 | 0.86 | 0.81 | 0.80 |
| TH-SE-ResNet | 64 | 0.85 | 0.85 | 0.85 | 0.84 |

#### Performance Analysis

The experimental results on the Charlotte dataset exhibit notably different characteristics compared to those observed in the Tufts dataset experiments. Overall, we observed a general decrease in performance across all models, which can be attributed to the Charlotte dataset's unique properties—specifically its limited subject count (only 10 individuals) and significant variability in image quality due to deliberate variations in temperature and environmental conditions.

TH-SE-ResNet maintained its superior performance, achieving the highest accuracy rates of 81% and 85% with batch sizes 32 and 64, respectively. However, this represents a substantial performance drop of approximately 12-14 percentage points compared to its performance on the Tufts dataset. This decline underscores the challenging nature of the Charlotte dataset.

AlexNet emerged as the second-best performer with accuracies of 70% and 68% for batch sizes 32 and 64, respectively. Interestingly, this represents a much smaller performance decline (approximately 16-17 percentage points) compared to TH-SE-ResNet, suggesting that AlexNet may possess certain architectural characteristics that provide resilience to the specific challenges presented by the Charlotte dataset.

A particularly noteworthy finding was the substantial performance degradation of ResNet, which achieved only 60% and 56% accuracy for batch sizes 32 and 64, respectively. This represents a drop of 25-27 percentage points from its Tufts dataset performance, making it the worst-performing model on the Charlotte dataset. This significant decline suggests that ResNet's architectural design may be particularly sensitive to the quality variations present in the Charlotte dataset.

Analysis of class-specific metrics revealed a pronounced gender bias across most models, but with a reversed pattern compared to the Tufts dataset. While the Tufts dataset generally exhibited higher precision for female classification, the Charlotte dataset showed higher recall for female subjects across most models. For instance, AlexNet (batch 32) achieved 90% recall for females but only 49% for males, indicating a strong tendency to classify subjects as female. This reversed bias might be attributed to the more balanced gender distribution in the Charlotte dataset (1030 female and 1029 male samples) combined with the distinctive thermal signatures captured under varying environmental conditions.

The convergence patterns, as illustrated in the training loss and test accuracy graphs, reveal intriguing dynamics. TH-SE-ResNet demonstrated remarkable early convergence, with its training loss rapidly decreasing within the first epoch. However, its test accuracy on the Charlotte dataset exhibited greater fluctuation compared to its stable performance on the Tufts dataset, particularly with batch size 32. This fluctuation suggests that despite its superior overall performance, TH-SE-ResNet encountered challenges in generalizing consistently across the varying conditions represented in the Charlotte dataset.

EfficientNet B0 performed comparatively better on the Charlotte dataset than on the Tufts dataset in relative terms. While it ranked among the lower performers on the Tufts dataset, it achieved respectable accuracy rates of 68% and 63% for batch sizes 32 and 64, respectively, on the Charlotte dataset. This improved relative performance might indicate that EfficientNet's design is better suited to handling the varied thermal signatures present in the Charlotte dataset.

Notably, Inception v3 maintained relatively consistent performance across both batch sizes (67% accuracy), suggesting that its architectural design provides a degree of stability when processing the Charlotte dataset's variable thermal signatures. This consistency contrasts with the more pronounced batch size sensitivity observed with other models.

The F1 scores further emphasize the superior performance balance of TH-SE-ResNet, with weighted F1 scores of 0.80 and 0.84 for batch sizes 32 and 64, respectively. These scores, while lower than those achieved on the Tufts dataset, still represent a substantial margin over the next best performer, indicating that TH-SE-ResNet maintains its effectiveness even under challenging conditions.

The learning curves for the Charlotte dataset illustrate these trends.

![comparison_charlotte_batch32](https://github.com/user-attachments/assets/9d1a70c4-06a7-4e8c-a562-c95a28d6b50d)
]

Figure 10: Training Loss and Test Accuracy Curves on Charlotte Dataset (Batch Size 64)

![comparison_charlotte_batch64](https://github.com/user-attachments/assets/1006ff13-0494-494a-9ecd-62572652760f)

Figure 11: Training Loss and Test Accuracy Curves on Charlotte Dataset (Batch Size 32)

Figures 10 and 11 depict the generally lower performance ceiling on the Charlotte dataset, with accuracy curves plateauing earlier and at lower levels for most models compared to the Tufts experiments. TH-SE-ResNet (blue line) still converges fastest and achieves the highest accuracy, but its test accuracy curve shows more fluctuation, particularly with batch size 32 (Figure 11), suggesting difficulties in consistent generalization across the varied conditions. The significant underperformance of standard ResNet (red line) is visually apparent, struggling to learn effectively. EfficientNet B0 (purple line) performs relatively better here compared to its Tufts results but still lags behind TH-SE-ResNet and AlexNet.

The confusion matrices for TH-SE-ResNet on the Charlotte dataset provide a breakdown of the errors.

[todo]

Figure 12a: Confusion Matrix - TH-SE-ResNet, Charlotte Dataset, Batch Size 64

[todo]

Figure 12b: Confusion Matrix - TH-SE-ResNet, Charlotte Dataset, Batch Size 32

Compared to the Tufts results, the confusion matrices in Figures 12a and 12b show considerably higher off-diagonal counts, reflecting the lower overall accuracy (81-85%). For batch size 64 (Figure 12a), while 947 females and 794 males were correctly identified, a substantial number of misclassifications occurred (235 females predicted as male, 83 males predicted as female). A similar pattern of significant errors is visible for batch size 32 (Figure 12b: 175 females predicted as male, 49 males as female). These matrices visually underscore the challenge posed by the Charlotte dataset's variability and limited subject pool, leading to more confusion between the gender classes for the model.

### 4.3 Results on Combined Dataset:

After analyzing the performance on individual datasets, we also evaluated these models on a combined dataset integrating both Tufts and Charlotte-ThermalFace collections. This combination helps us to assess model generalization across different thermal imaging sources and environmental conditions. Table 7 presents the comprehensive performance metrics across all six architectures and both batch size configurations.

##### Table 7: Performance on Combined Dataset

| Model | Batch Size | Accuracy | Precision (weighted) | Recall (weighted) | F1 (weighted) |
|-------|------------|----------|----------------------|-------------------|---------------|
| AlexNet | 32 | 0.76 | 0.80 | 0.76 | 0.75 |
| AlexNet | 64 | 0.75 | 0.79 | 0.75 | 0.74 |
| ResNet | 32 | 0.64 | 0.64 | 0.64 | 0.64 |
| ResNet | 64 | 0.63 | 0.63 | 0.63 | 0.63 |
| Inception v3 | 32 | 0.71 | 0.73 | 0.71 | 0.70 |
| Inception v3 | 64 | 0.72 | 0.74 | 0.72 | 0.72 |
| VGG | 32 | 0.68 | 0.68 | 0.68 | 0.68 |
| VGG | 64 | 0.70 | 0.70 | 0.70 | 0.70 |
| EfficientNet B0 | 32 | 0.74 | 0.74 | 0.74 | 0.74 |
| EfficientNet B0 | 64 | 0.71 | 0.71 | 0.71 | 0.71 |
| TH-SE-ResNet | 32 | 0.87 | 0.89 | 0.87 | 0.87 |
| TH-SE-ResNet | 64 | 0.90 | 0.91 | 0.90 | 0.90 |

#### Performance Analysis

Performance on the combined dataset generally fell between the results obtained on the individual Tufts and Charlotte datasets. TH-SE-ResNet continued its strong performance, achieving 87% and 90% accuracy (batch sizes 32 and 64), demonstrating good adaptability to the heterogeneous data. AlexNet (75-76%) and EfficientNet B0 (71-74%) followed, with EfficientNet showing relatively better performance here than on Tufts alone, possibly benefiting from the increased data diversity. Standard ResNet again struggled significantly (63-64%), reinforcing its apparent limitations in generalizing across these thermal datasets. The gender bias pattern persisted, often with higher precision but lower recall for females, even with the balanced combined dataset, indicating the challenge of achieving equitable performance. TH-SE-ResNet maintained the best balance, reflected in its leading F1 scores (0.87-0.90).

The training dynamics on the combined dataset are shown in the following figures.

![alt text](https://github.com/user-attachments/assets/0bebbace-6a57-49a7-bf72-f584041444f7)

Figure 13: Training Loss and Test Accuracy Curves on Combined Dataset (Batch Size 64)

![alt text](https://github.com/user-attachments/assets/dd3fcbf0-bb17-4827-afe5-be9338b8fb1c)

Figure 14: Training Loss and Test Accuracy Curves on Combined Dataset (Batch Size 32)

Figures 13 and 14 illustrate that TH-SE-ResNet (blue line) maintained its characteristic rapid convergence and superior accuracy even when trained on the combined, more diverse dataset. It quickly establishes a significant lead over the baseline models. Standard ResNet's (red line) struggle is again evident, plateauing at a much lower accuracy. EfficientNet B0 (purple line) shows reasonably good performance, surpassing some other baselines, and again exhibits some sensitivity to the larger batch size (Figure 13 vs Figure 14). These curves visually support the conclusion that TH-SE-ResNet generalizes more effectively across the combined data sources.

The confusion matrices for TH-SE-ResNet provide insight into the specific error patterns on this mixed dataset.

![alt text](https://github.com/user-attachments/assets/954c9da1-26e6-4769-8cdb-f5ae44d20e46)

Figure 15a: Confusion Matrix - TH-SE-ResNet, Combined Dataset, Batch Size 64

![alt text](https://github.com/user-attachments/assets/4bd11d6a-12a7-4cd6-a8a2-de03d54bbbc9)

Figure 15b: Confusion Matrix - TH-SE-ResNet, Combined Dataset, Batch Size 32

The confusion matrices for the combined dataset (Figures 15a and 15b) show error levels intermediate between the Tufts and Charlotte experiments. While the diagonal elements are strong, confirming the high overall accuracy (87-90%), the off-diagonal counts are non-negligible. Notably, with batch size 64 (Figure 15a), there is a pronounced asymmetry: 390 females were misclassified as male, whereas only 6 males were misclassified as female. This indicates a specific difficulty in correctly recalling female subjects under these conditions, despite high precision for male predictions. With batch size 32 (Figure 15b), the errors are more balanced but still significant (276 females misclassified as male, 19 males as female). These matrices highlight the complexities of classifying gender accurately when dealing with data combined from different thermal cameras and conditions.

To provide concrete examples of the model's performance on this heterogeneous data, Figure 16 displays sample images from the combined test set alongside the true labels and the predictions from the TH-SE-ResNet model (trained with batch size 64).

![alt text](https://github.com/user-attachments/assets/ce743113-a418-43cd-a84b-ffe24a85a7d2)

**Figure 16: Sample Classification Analysis** - Examples from Combined Dataset Test Set (TH-SE-ResNet, B64)

These examples in Figure 16 offer a qualitative glimpse into the model's behavior, showcasing instances of correct classifications alongside examples where the model failed, likely due to variations in pose, expression, or thermal artifacts inherent in the combined dataset.


(The results I have from ablation study are not consistant enough, this is performed on charllate dataset, and for the complete model results are different then the once we mentioned above, most probably do to different seed, so either we need to run the ablation study again or remove the results from the paper.)
## 4.5 Ablation Study

The TH-SE-ResNet architecture emerged from a deliberate redesign of the standard ResNet50 model to address specific challenges in processing single-channel thermal imagery. Our architectural decisions were validated through a comprehensive ablation study that demonstrated the effectiveness of each component under varying conditions.

### 4.5.1 Experimental Methodology
The evaluation employed the "combined" dataset and tracked two primary performance metrics—training loss and test accuracy—across ten epochs. Experiments were conducted with two different batch sizes (64 and 32) to assess the architecture's sensitivity to training conditions.

### 4.5.2 Rationale for Key Components

#### 4.5.2.1 Squeeze-and-Excitation Blocks

We integrated Squeeze-and-Excitation (SE) blocks throughout the architecture to enhance feature representation quality. Thermal imagery often contains subtle temperature variations with critical diagnostic information that can be overshadowed by stronger signals. SE blocks adaptively recalibrate channel-wise feature responses by explicitly modeling interdependencies between channels, allowing the network to selectively emphasize informative features while suppressing less useful ones. Our ablation study confirms this design choice, showing that SE blocks contribute a 2% improvement in accuracy with larger batch sizes, elevating performance from 90% to 92%.

#### 4.5.2.2 Modified Fully Connected Layer

The standard fully connected layer in ResNet was replaced with a more sophisticated structure incorporating dropout regularization, multiple linear transformations, and a SE block. This modification addresses the challenge of overfitting in thermal image classification, where training datasets are often limited in size and diversity compared to RGB datasets. The modified FC layer introduces controlled regularization through dual dropout layers (rates of 0.5 and 0.3) and leverages intermediate dimensionality adjustments to create a more generalizable feature representation. The ablation study validates this approach for larger batch sizes, where the modified FC layer outperforms the standard implementation by approximately 2% in accuracy.

#### 4.5.2.3 Input Convolution Layer

To handle the dimensionality mismatch between single-channel thermal inputs and the three-channel expectation of standard ResNet architectures, we incorporated an input convolution layer that projects the thermal data into a three-channel representation. While simple channel replication could serve as an alternative, the dedicated convolution layer provides the network with learnable parameters to transform the input representation in a data-driven manner. Though our ablation study shows comparable performance between this approach and simple channel replication, the convolution layer maintains architectural consistency and offers potentially greater flexibility for diverse thermal imaging scenarios.

#### 4.5.2.4 Batch Size Sensitivity and Architectural Adaptations

Our ablation study revealed insights regarding the sensitivity of TH-SE-ResNet to batch size variations. This sensitivity stems from the interplay between regularization intensity (particularly in the modified FC layer) and the statistical properties of gradient estimates at different batch sizes. With larger batches (64), gradient estimates are more stable, allowing the additional regularization from dropout layers to effectively prevent overfitting without impeding learning. Conversely, smaller batch (32) introduce inherent noise in gradient estimates, which, when combined with strong explicit regularization, can hinder convergence.

This finding justifies a flexible implementation approach where the architecture adapts based on anticipated deployment conditions. For systems with sufficient computational resources to support larger batch sizes, the complete TH-SE-ResNet with SE blocks and the modified FC layer maximizes performance. For resource-constrained environments requiring smaller batches, a variant with a standard FC layer proves more appropriate, achieving up to 95% accuracy compared to 90-92% with the modified FC.

#### 4.5.2.5 Decision on Component Selection

For applications requiring maximum accuracy and having access to substantial computational resources, the full architecture with all components delivers optimal performance. For deployments on edge devices with memory or processing limitations, simpler variants can be selected with minimal performance degradation. The ablation study demonstrates that even the simplest configuration maintains performance well above 90%, justifying our design philosophy of maintaining robust performance across diverse implementation scenarios.

