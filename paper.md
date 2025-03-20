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

## 2. Literature Review

The task of gender classification has been extensively studied in computer vision. Early approaches often relied on **conventional machine learning methods** and feature extraction techniques applied to visible spectrum images. Makinen and Raisamo and Reid et al. provided detailed surveys of these methods. Initial techniques involved training neural systems on small sets of frontal face images. Later, methods incorporated 3D head structure and image intensities for gender characterization. **Support Vector Machines (SVMs)** were also widely used, demonstrating competitive performance compared to other traditional classifiers. Techniques like AdaBoost, utilizing low-resolution grayscale images, and methods addressing perspective invariant recognition were also explored. More recently, researchers utilized local image descriptors like the Webers Local Surface Descriptor (WLD) and features based on shape, texture, and color extracted from frontal faces, achieving high accuracy on benchmark datasets like FERET.

Recognizing the limitations of visible spectrum-based methods, researchers began to explore the potential of deducing gender information from other modalities, including **thermal and Near-Infrared (NIR) spectra**. Chen and Ross are noted as early proponents of human face-based gender classification systems using both thermal and NIR data, employing conventional feature extraction methods and classifiers like SVM, LDA, Adaboost, random forest, Gaussian mixture models, and multi-layer perceptrons. Their findings suggested that SVM with histogram-based gender classification yielded better performance on NIR and thermal spectra. Nguyen and Park proposed a gender classification system using joint visible and thermal spectrum data of the human body, utilizing feature extractors like Histogram of Oriented Gradients (HoG) and Multi-Local Binary Patterns (MLBP). Their results indicated improved accuracy by combining data from both modalities. Similarly, Abouelenien et al. explored multimodal datasets including audiovisual, thermal, and physiological recordings for automatic gender classification, again relying on conventional machine learning algorithms.

The advent of **deep learning** and the success of **CNNs** in various computer vision tasks, particularly where high accuracy and robustness are required, led to their application in gender classification. Canziani et al. listed numerous pretrained models suitable for various applications. Dwivedi and Singh provided a comprehensive review of deep learning methodologies for robust gender classification using visible spectrum datasets. Ozbulak et al. investigated fine-tuning and SVM classification using CNN features for age and gender classification on visible datasets, demonstrating that transferred models can outperform task-specific models. Manyala et al. explored CNN-based methods for gender classification using NIR periocular images, achieving promising results. Baek et al. used combined visible and NIR data with two CNN architectures for robust gender classification from full human body images in surveillance environments.

In the domain of **thermal image-based gender classification**, Farooq et al. conducted a comprehensive performance estimation of state-of-the-art CNNs, including AlexNet, VGG-19, MobileNet-V2, Inception-V3, ResNet-50, ResNet-101, DenseNet-121, DenseNet-201, and EfficientNet-B4, using the **Tufts thermal faces dataset** and the **Carl thermal faces dataset**. They also proposed a new CNN architecture, **GENNet**, specifically for this task. Li et al. focused on detecting age and gender from thermal images for personal thermal comfort prediction, utilizing a newly established dataset of thermal and visible-light images. They evaluated the performance of ResNet-50, ResNet-101, EfficientNet, and Inception v3, finding ResNet-50 to achieve a high gender accuracy on their thermal dataset. Chatterjee and Zaman proposed a deep-learning approach for general thermal image classification, utilizing pretrained ResNet-50 and VGGNet-19 and exploring the impact of Kalman filtering for denoising on the **Tufts** and **Charlotte-ThermalFace datasets**. Keerthi et al. investigated gender classification optimization with thermal images using InceptionV3 and AlexNet, also utilizing the "tufts" dataset.

These studies highlight the growing interest and potential of using deep learning techniques for gender classification based on thermal imagery. Our work builds upon this foundation by providing a comparative analysis of several prominent CNN architectures on the **Tufts** and **Charlotte** datasets, addressing the challenges of varying input channels, class imbalance, and further proposing and evaluating a novel architecture tailored for this specific task with the incorporation of channel adaptation and Squeeze-and-Excitation mechanisms. This research aims to contribute to the advancement of robust and accurate gender detection systems using thermal imaging in diverse real-world applications.

## 3. Methodology

This study focuses on gender classification using thermal facial images from two publicly available datasets: the **Tufts University Thermal Face Dataset** and the **Charlotte-ThermalFace Dataset**. Additionally, a combined dataset and cross-dataset experiments were conducted to enhance data diversity and assess model generalizability. The methodology encompasses data preprocessing, augmentation, a variety of deep learning models (including a novel `HybridResNet` architecture), experimental setup, training procedures, and comprehensive evaluation metrics.

### 3.1 Datasets

#### 3.1.1 Tufts University Thermal Face Dataset

The **Tufts Face Database** is a multimodal dataset featuring over 10,000 images from 113 participants (74 females, 39 males), aged 4 to 70 years, representing over 15 countries. The thermal imagery, captured using a **FLIR Vue Pro** camera (operating in the 7.5–13.5 μm long-wave infrared spectrum), was collected in a controlled indoor environment (~22°C ambient temperature) with participants seated at a fixed distance. Two subsets were utilized:

- **TD_IR_E (Emotion)**: Images of five expressions (neutral, smile, eyes closed, shocked, sunglasses).
- **TD_IR_A (Around)**: Images from nine camera positions in a semicircle around the participant.

The dataset exhibits a gender imbalance:
- **Training Set**: 389 female, 838 male images (30.32% female).
- **Test Set**: 115 female, 215 male images (34.85% female).

To address this, targeted augmentation was applied to the female class (see Section 3.2.3).

*(Placeholder for Figure: Diagram of FLIR Vue Pro setup and TD_IR_A camera positions)*

#### 3.1.2 Charlotte-ThermalFace Dataset

The **Charlotte-ThermalFace Dataset** comprises approximately 10,364 thermal facial images from 10 subjects, collected under varying conditions (e.g., distance, head position, temperature). Assumed to be captured with a FLIR-based thermal camera, this dataset was curated for gender classification, achieving near-perfect balance:
- **Training Set**: 4,161 female, 4,144 male images (50.10% female).
- **Test Set**: 1,030 female, 1,029 male images (50.02% female).

#### 3.1.3 Combined Dataset Construction

A **combined dataset** was created by merging the Tufts and Charlotte datasets:
1. Images were organized into a unified directory (`gender_data/combined`).
2. Single-channel Charlotte images were replicated to three channels for compatibility with Tufts data.
3. Gender classes were balanced by selecting an equal number of images per class.

This yielded approximately 11,921 images, enhancing data volume and diversity.

#### 3.1.4 Cross-Dataset Experiments

To evaluate model robustness across domains, cross-dataset experiments were conducted:
- **Tufts-to-Charlotte**: Trained on Tufts, tested on Charlotte.
- **Charlotte-to-Tufts**: Trained on Charlotte, tested on Tufts.

These setups assess generalization to unseen thermal imaging conditions.

#### 3.1.5 Dataset Summary

**Table 1**: Dataset Characteristics

| Dataset       | Training Samples | Test Samples | Female (%) | Male (%) | Notes                     |
|---------------|------------------|--------------|------------|----------|---------------------------|
| Tufts         | 1,227            | 330          | 30.32      | 69.68    | Imbalanced                |
| Charlotte     | 8,305            | 2,059        | 50.10      | 49.90    | Balanced                  |
| Combined      | ~11,921          | -            | 50.00      | 50.00    | Merged and balanced       |

*(Placeholder for Figure: Representative samples from Tufts and Charlotte datasets)*

### 3.2 Preprocessing

#### 3.2.1 Standard Preprocessing Pipeline

All images underwent a standardized preprocessing pipeline:
1. **Resizing**: To 256×256 pixels (or 342×342 for InceptionV3) to standardize dimensions.
2. **Center Cropping**: To model-specific sizes:
   - 224×224 (AlexNet, VGG, ResNet, EfficientNet, HybridResNet).
   - 299×299 (InceptionV3).
3. **Tensor Conversion**: Converted to PyTorch tensors for GPU processing.
4. **Normalization**:
   - **Standard Models** (AlexNet, VGG, ResNet, EfficientNet, Inception): Mean=[0.5, 0.5, 0.5], Std=[0.5, 0.5, 0.5].
   - **HybridResNet**: Single-channel input, Mean=[0.5], Std=[0.5].

#### 3.2.2 Data Augmentation Techniques

Augmentation enhanced training data variability:
- **Random Resized Crop**: Scale=(0.08, 1.0), Ratio=(0.75, 1.33).
- **Random Horizontal Flip**: p=0.5.
- **Random Rotation**: ±15° (simulates head tilt).
- **Color Jitter**: Brightness=0.2, Contrast=0.2 (mimics thermal intensity variations).
- **Gaussian Blur**: Kernel=3, Sigma=(0.1, 2.0).
- **Grayscale Conversion**: For HybridResNet variants.

Parameters were empirically tuned for thermal image characteristics.

#### 3.2.3 Class Imbalance Mitigation

For the Tufts dataset, gender imbalance was mitigated by:
1. Creating an augmented female subset using the above techniques.
2. Concatenating it with the original training data via `ConcatDataset`.

This increased female representation to approximately 48%, reducing bias without duplicating original samples.

### 3.3 Deep Learning Models

We evaluated five established CNNs and proposed a novel architecture:

#### 3.3.1 Overview of Evaluated Models

1. **AlexNet**:
   - **Architecture**: 5 convolutional layers, 3 fully connected layers; large kernels (11×11, 5×5); ReLU activations; dropout (p=0.5).
   - **Key Feature**: Pioneered deep CNNs, winning ImageNet 2012.
   - **Suitability**: Baseline for thermal image classification.
   *(Space for a diagram of AlexNet architecture)*

2. **VGG-19**:
   - **Architecture**: 16 convolutional layers (3×3 kernels) + 3 fully connected layers; deep, uniform design.
   - **Key Feature**: Stacked small convolutions for hierarchical feature learning.
   - **Suitability**: Captures fine details but risks overfitting on small datasets.
   *(Space for a diagram of VGG-19 layer stack)*

3. **InceptionV3**:
   - **Architecture**: Inception modules with parallel convolutions (1×1, 3×3, 5×5); factorized convolutions.
   - **Key Feature**: Multi-scale feature extraction.
   - **Suitability**: Effective for thermal images with varying feature scales.
   *(Space for a diagram of an Inception module)*

4. **ResNet50**:
   - **Architecture**: 50 layers with residual connections: \( y = F(x, \{W_i\}) + x \).
   - **Key Feature**: Mitigates vanishing gradients, enabling deep training.
   - **Suitability**: Learns complex thermal patterns.
   *(Space for a diagram of a residual block)*

5. **EfficientNet-B0**:
   - **Architecture**: Compound scaling of depth, width, resolution (e.g., \(\phi = 1\)).
   - **Key Feature**: Balances efficiency and accuracy.
   - **Suitability**: Resource-efficient for thermal tasks.
   *(Space for a diagram of EfficientNet scaling)*

#### 3.3.2 Novel CNN Architecture (HybridResnet)

Our **HybridResnet**, built on ResNet50, includes:

1. **ResNet Base**:
   - Residual block: \( y = F(x, \{W_i\}) + x \), where \( F \) includes convolutions, batch normalization, and ReLU.

2. **Channel Input Adapter**:
   - **Function**: Converts single-channel thermal inputs to three channels via replication or a learned 1×1 convolution.
   - **Implementation**: Replication used for simplicity, leveraging ImageNet weights.
   *(Space for a diagram of the Channel Input Adapter)*

3. **Squeeze and Excitation (SE) Blocks**:
   - **Squeeze**: Global average pooling: \( z_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} x_c(i,j) \).
   - **Excitation**: \( s = \sigma(W_2 \delta(W_1 z)) \), where \(\delta\) is ReLU, \(\sigma\) is sigmoid.
   - **Rescaling**: \( \tilde{x}_c = s_c \cdot x_c \).
   - **Integration**: Added post-convolution in residual blocks.
   *(Space for a diagram of an SE block within a residual block)*

4. **Tailored Final Classifier**:
   - Replaced 1000-class layer with a 2-class layer (male/female); added a fully connected layer (512 units, dropout p=0.5); softmax output: \( p(\text{class}) = \frac{e^{z_{\text{class}}}}{\sum e^{z_i}} \).

5. **Training Strategy**: Initialized with ImageNet weights, fine-tuned end-to-end.


### 3.4 Experimental Setup

- **Dataset Types**: Tufts, Charlotte, Combined, Tufts-to-Charlotte, Charlotte-to-Tufts.
- **Training Parameters**:
  - **Optimizer**: Adam (\(\beta_1=0.9, \beta_2=0.999\)).
  - **Learning Rate**: 0.00005, with 5-epoch warmup (linear increase from 0) and cosine annealing thereafter.
  - **Batch Sizes**: 32, 64.
  - **Epochs**: 10.
- **Hardware**: NVIDIA RTX 3090 GPU, with `DataLoader` optimizations (`num_workers=8`, `pin_memory=True`)

### 3.5 Evaluation Metrics

Performance was evaluated using:
- **Accuracy**: \( \frac{\text{TP} + \text{TN}}{\text{Total}} \).
- **Precision**: \( \frac{\text{TP}}{\text{TP} + \text{FP}} \).
- **Recall**: \( \frac{\text{TP}}{\text{TP} + \text{FN}} \).
- **F1-score**: \( 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \).
- **Confusion Matrix**: Saved as heatmap.
- **Classification Report**: Detailed per-class metrics, saved to file.

*(Placeholder for Figure: Example confusion matrix)*

## 4. Experimental Results and Comparative Analysis

This section details the experimental results obtained by training and evaluating various deep learning models for gender classification using thermal facial images. We utilized the **Tufts University Thermal Face dataset** and the **Charlotte-ThermalFace dataset**, both individually and in combination, to assess the performance of several state-of-the-art Convolutional Neural Networks (CNNs), including **AlexNet**, **VGG**, **ResNet**, **InceptionV3**, and **EfficientNet**. Furthermore, we evaluated our proposed novel architecture, **HybridResNet**, and conducted ablation studies and hyperparameter tuning to understand the contributions of its components and optimize its performance.

### 4.1 Performance on Individual Datasets

To understand the inherent characteristics of each dataset and the suitability of different architectures for them, we first trained and evaluated a subset of key models separately on the **Tufts** and **Charlotte** datasets.

#### 4.1.1 Results on the Tufts University Thermal Face Dataset

[ADD: Table showing the performance metrics (accuracy, precision, recall, F1-score) of AlexNet, ResNet, EfficientNet, and HybridResNet trained and evaluated solely on the Tufts dataset. Specify the training hyperparameters used (e.g., batch size, learning rate, optimizer).]

Our experiments on the Tufts dataset revealed the following trends:

*   **[ADD ANALYSIS HERE: Compare the performance of the different models on the Tufts dataset. Highlight the best-performing model and note any significant differences in accuracy, precision, recall, and F1-score. Discuss potential reasons for these differences based on the dataset's characteristics, such as the smaller sample size and potential class imbalance mentioned in].** For example, [ADD SENTENCE ABOUT CLASS IMBALANCE IMPACT].

#### 4.1.2 Results on the Charlotte-ThermalFace Dataset

[ADD: Table showing the performance metrics (accuracy, precision, recall, F1-score) of AlexNet, ResNet, EfficientNet, and HybridResNet trained and evaluated solely on the Charlotte dataset. Ensure the training hyperparameters are consistent with those used for the Tufts experiments for fair comparison.]

The results obtained on the Charlotte dataset showed:

*   **[ADD ANALYSIS HERE: Compare the performance of the same set of models on the Charlotte dataset. Again, highlight the top performer and analyze the relative strengths and weaknesses of each architecture. Discuss if the trends observed on the Tufts dataset hold for Charlotte, or if there are notable discrepancies, potentially due to the larger size and balanced class distribution of the Charlotte dataset].** For instance, [ADD A SENTENCE COMPARING PERFORMANCE ACROSS DATASETS FOR A SPECIFIC MODEL].

#### 4.1.3 Comparative Analysis Across Individual Datasets

Comparing the performance of the models across the Tufts and Charlotte datasets provides valuable insights into the dataset-specific challenges and the generalization capabilities of the architectures.

*   **[ADD ANALYSIS HERE: Discuss the overall performance differences observed between the Tufts and Charlotte datasets for the models tested. For example, were the accuracies generally higher on one dataset compared to the other? Hypothesize reasons for these overall differences, considering factors like dataset size, image variability, and potential annotation quality. Refer to any information about data augmentation techniques used and how they might have influenced the results on each dataset.]**

*   **[ADD ANALYSIS HERE: Compare how each specific model (e.g., AlexNet, ResNet) performed relative to others *within* each dataset and *across* the two datasets. Did some models show more consistent performance across datasets? Did any model exhibit a significant performance disparity between Tufts and Charlotte, suggesting sensitivity to specific dataset characteristics?]**

### 4.2 Impact of the Channel Input Adapter in HybridResNet

Our proposed **HybridResNet** architecture incorporates a **channel input adapter** to handle the potential variations in input channels in thermal data. To assess the effectiveness of this component, we conducted an ablation study on the combined dataset.

*   **Experimental Setup:** We trained and evaluated the **HybridResNet architecture *without* the channel input adapter** on the combined dataset using the same training hyperparameters as the original HybridResNet experiments on the combined dataset.

*   **Results:**

    [ADD: Table showing the performance metrics (accuracy, precision, recall, F1-score) of HybridResNet with and without the channel input adapter on the combined dataset.]

*   **Analysis:**

    *   **[ADD ANALYSIS HERE: Compare the performance of HybridResNet with and without the channel input adapter. Quantify the difference in accuracy and other metrics. Discuss whether the presence of the adapter significantly improved performance, remained neutral, or even hindered it. Provide a potential explanation for the observed impact based on the role of the adapter in handling input channel variations.]** This analysis will directly address the contribution of the channel input adapter to the overall performance of the HybridResNet architecture.

### 4.3 Hyperparameter Tuning: Learning Rate

The learning rate is a critical hyperparameter that significantly influences the training process. To optimize the performance of our models, we experimented with different learning rate values on the combined dataset for a selection of promising architectures.

*   **Models Tested:** [Specify the models on which learning rate tuning was performed, likely the best-performing baselines from the individual dataset experiments and the HybridResNet.]

*   **Learning Rate Values Tested:** [List the specific learning rate values that were evaluated (e.g., 0.001, 0.0001, 0.00001).]

*   **Experimental Setup:** Each selected model was trained on the combined dataset for a fixed number of epochs (or until convergence) using each of the chosen learning rate values, while keeping other hyperparameters constant.

*   **Results:**

    [ADD: Table(s) showing the performance metrics (e.g., final test accuracy, best validation accuracy) for each model across the different learning rate values tested. we might want separate tables for each model if the number of learning rates is large.]

*   **Analysis:**

    *   **[ADD ANALYSIS HERE: For each model, analyze the impact of different learning rates on its performance. Discuss the trends observed (e.g., lower learning rates leading to slower convergence but potentially better final accuracy, or vice versa). Identify the learning rate that yielded the best performance for each model on the combined dataset. Discuss potential reasons why a particular learning rate might be optimal for a given architecture and dataset.]**

### 4.4 Cross-Dataset Generalization

To evaluate the ability of the models to generalize across different thermal datasets, we performed cross-dataset evaluation using the best-performing model identified in our earlier experiments.

*   **Best Performing Model:** [Identify the model that showed the most promising results on the individual or combined datasets.]

*   **Experimental Setup:**
    *   Trained the best model *solely* on the **Tufts dataset** and evaluated it on the **Charlotte dataset**.
    *   Trained the best model *solely* on the **Charlotte dataset** and evaluated it on the **Tufts dataset**.

*   **Results:**

    [ADD: Table showing the performance metrics (accuracy, precision, recall, F1-score) for both cross-dataset evaluation scenarios.]

*   **Analysis:**

    *   **[ADD ANALYSIS HERE: Analyze the performance of the best model when trained on one dataset and tested on the other. Quantify the performance drop compared to the in-dataset evaluation results. Discuss the degree of generalization observed. Hypothesize potential reasons for any performance degradation, considering differences in image characteristics, data distribution, and potential domain shift between the Tufts and Charlotte datasets.]**

### 4.5 Comparison with Existing State-of-the-Art

[ADD: This section requires we to revisit the literature on thermal gender classification (mentioned in Section 2 of this paper paper) and compare our best-performing model's results (on the combined dataset, if applicable) with the state-of-the-art results reported in other studies. Acknowledge that direct comparison might be challenging due to variations in datasets, evaluation protocols, and reported metrics. Focus on a qualitative comparison, highlighting whether our results are competitive with or surpass existing findings. If possible, mention the models and datasets used in the studies we are comparing against and note any significant differences in our approach (e.g., novel architecture, specific data preprocessing or augmentation techniques).]

### 4.6 Analysis of Model Complexity

[ADD: Here, we can briefly discuss the model complexity of the main architectures we experimented with. We can compare the number of parameters of AlexNet, VGG, ResNet, EfficientNet, and our HybridResNet. If we have information about the computational cost (e.g., FLOPs), we can include that as well. This analysis provides context for the performance achieved by each model, considering the trade-off between accuracy and computational resources.]

### 4.7 Further Investigations (Space for Future Work)

*   **In-depth Ablation Study of SE Blocks in HybridResNet:** [SPACE TO BE ELABORATED IN FUTURE WORK]
*   **Experimentation with Different Optimizers:** [SPACE TO BE ELABORATED IN FUTURE WORK]
*   **Increased Training Epochs and Early Stopping:** [SPACE TO BE ELABORATED IN FUTURE WORK]
*   **More Sophisticated Data Augmentation Techniques:** [SPACE TO BE ELABORATED IN FUTURE WORK, currently using [MENTION THE AUGMENTATIONS USED BASED ON SOURCE]].
*   **Visualization and Interpretation of Learned Features:** [SPACE TO BE ELABORATED IN FUTURE WORK]
*   **Investigating the Impact of Image Resolution:** [SPACE TO BE ELABORATED IN FUTURE WORK]
