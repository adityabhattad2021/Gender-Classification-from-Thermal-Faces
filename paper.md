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

This study utilized two publicly available thermal facial image datasets: the **Tufts University Thermal Face dataset** and the **Charlotte-ThermalFace dataset**. These datasets were selected for their established benchmark status in thermal image-based facial analysis and their complementary characteristics.

### 3.1.1 Tufts Face Database

The **Tufts Face Database** represents a **comprehensive, large-scale multimodal face dataset** featuring **six distinct imaging modalities**, including thermal imaging. This extensive collection encompasses:

*   **Over 10,000 images** acquired from **113 participants** (74 females and 39 males) [3, our prior turn]
*   Age distribution ranging from **4 to 70 years**
*   Demographic diversity representing **more than 15 countries**


The thermal imagery component is accessible through two specific modalities:
1. **TD_IR_E (E stands for emotion) The images were captured using a FLIR Vue Pro camera. Each participant was asked to pose with (1) a neutral expression, (2) a smile, (3) eyes closed, (4) exaggerated shocked expression, (5) sunglasses."**
2. **TD_IR_A (A stands for around):  The images were captured using a FLIR Vue Pro camera. Each participant was asked to look at a fixed view-point while the cameras were moved to 9 equidistant positions forming an approximate semi-circle around the participant."**

Analysis of the Tufts thermal subset revealed a significant **class imbalance**:
- **Training set**: 389 female and 838 male images (30.32% female, 69.68% male)
- **Test set**: 115 female and 215 male images (34.85% female, 65.15% male)

To address this gender disparity, we implemented **targeted data augmentation techniques for the female class** during the training phase.

### 3.1.2 Charlotte-ThermalFace Dataset

The Charlotte-ThermalFace dataset is a publicly available collection of approximately 10,000 thermal facial images, captured under varying environmental conditions, distances, and head positions. The dataset was originally collected by researchers at UNC Charlotte and includes detailed annotations such as facial landmarks, environmental properties, and subjective thermal sensations.

While the original dataset was not specifically designed for gender classification, we processed the data to create a curated version with a balanced gender distribution, ensuring its suitability for gender classification tasks:

- **Training set**: 4,161 female and 4,144 male images (50.10% female, 49.90% male)
- **Test set**: 1,030 female and 1,029 male images (50.02% female, 49.98% male)

### 3.1.3 Combined Dataset Construction

For experiments investigating the impact of increased data volume and diversity, we created a combined dataset by merging the Tufts and Charlotte datasets. The integration process followed these steps:

1. Creation of a new directory structure ('gender_data/combined')
2. Balanced combination of both datasets' training and testing splits
3. Selection of the minimum number of images from each gender category to ensure equal representation

This methodical approach enabled us to systematically investigate how increased data volume and cross-dataset diversity influence model performance while maintaining gender balance.

### 3.1.4 Dataset Comparison

Table 1 provides a comparative overview of the key characteristics of the datasets used in this study:

| Dataset | Subjects (Approx.) | Images (Approx.) | Illumination Dependence | Channels | Class Balance |
|:----------------------|:---------------------------|:---------------------------|:----------------------|:---------|:------------|
| Tufts University Thermal | 112 | 1,557 | Independent | 3 | Imbalanced |
| Charlotte-ThermalFace | 10 | 10,364 | Independent | 1 | Balanced |
| Combined | ~122 | 11,921 | Independent | 1 (processed to 3 or used as 1 for HybridResnet) | Balanced |

Figure 1 shows representative sample images from each dataset to illustrate their visual characteristics and differences:

![Representative samples from the Tufts and Charlotte-ThermalFace datasets showing variations in thermal facial imaging](https://github.com/user-attachments/assets/35c16d77-e720-413a-9ef0-0cd5664037b7

### 3.2 Data Preprocessing and Augmentation

Prior to training, the thermal images underwent several preprocessing steps. These steps aimed to standardize the input data and improve the learning process. The specific preprocessing steps applied might have included resizing and cropping the images to a consistent input size required by the CNN architectures.

To enhance the robustness and generalization ability of the models, **image augmentation** techniques were applied to the training data. These techniques introduce variations in the training samples, preventing overfitting and improving the models' performance on unseen data. Common augmentation techniques used in this research included:

*   **Resizing and Random Cropping**: To vary the scale and position of the face in the image.
*   **Random Horizontal Flip**: To introduce symmetry variations.
*   **Random Rotation**: To make the models invariant to slight rotations.
*   **Color Jittering**: Although thermal images are typically grayscale, this technique might have been adapted if the thermal data had multiple channels or was converted to an RGB-like format.
*   **Gaussian Blurring**: To simulate variations in image sharpness.

The specific augmentation pipeline used for each dataset and model configuration have been tailored to optimize performance. The test datasets were typically subjected only to resizing and center cropping to ensure a consistent evaluation.

### 3.3 Deep Learning Models

A range of state-of-the-art CNN architectures were evaluated for gender classification using thermal facial images. These included:

*   **AlexNet**: An early deep CNN that demonstrated the power of convolutional networks for image classification.
*   **VGG (e.g., VGG-19)**: Architectures known for their deep stacks of convolutional layers with small receptive fields.
*   **InceptionV3**: A network with a more complex architecture utilizing parallel convolutional layers with varying kernel sizes to capture multi-scale features.
*   **ResNet50**: A very deep residual network that utilizes skip connections to mitigate the vanishing gradient problem, enabling the training of much deeper networks.
*   **EfficientNet**: A family of models that efficiently scales network dimensions (depth, width, resolution) using a compound scaling method.

Additionally, a **novel CNN architecture** was proposed in this research. This architecture was based on the **ResNet framework** and incorporated several key modifications:

*   **Channel Input Adapter**: To handle the potential differences in the number of channels in the thermal image datasets (e.g., single-channel grayscale vs. multi-channel representations). This adapter converts single-channel thermal images to a 3-channel format (or other required input channel size) to be compatible with pre-trained models designed for RGB images.
    *(Space for a diagram illustrating the architecture of the Channel Input Adapter).*

*   **Squeeze and Excitation (SE) Blocks**: Integrated within the layers of the proposed ResNet-based architecture to enhance feature discrimination. SE blocks are attention mechanisms that allow the network to learn which features are most important for the task at hand by explicitly modeling the channel interdependencies.
    *(Space for a diagram illustrating the architecture of an SE Block).*

*   **Tailored Final Classifier**: The final classification layer of the proposed network was specifically designed for the gender classification task.

### 3.4 Experimental Setup

The experiments were conducted by training and evaluating each of the selected CNN architectures on the individual datasets (Tufts and Charlotte), as well as the combined dataset. For each experiment, the datasets were split into training and testing sets. The split ratios were chosen to have a sufficient amount of data for training while retaining a representative set for evaluation.

The models were trained using **Adam** as the optimizer, and  **Cross-Entropy Loss** as a loss function. The training was performed for a fixed number of **epochs**, and the learning rate was set to a specific value. Hyperparameter tuning (e.g., learning rate, batch size) might have been performed to optimize the performance of each model on each dataset.

Experiments were conducted with different **batch sizes** (e.g., 32, 64, 128) to observe their impact on training dynamics and model performance.

### 3.5 Training Procedures

The training procedure involved feeding batches of preprocessed and augmented thermal images to the CNN models. The models learned to map the input thermal data to the output gender label (male or female) by adjusting their internal weights based on the error between the predicted and true labels, as calculated by the loss function.

The models were trained for a predefined number of epochs, with performance on a held-out validation set (if used) being monitored to prevent overfitting and to select the best performing model.

### 3.6 Evaluation Metrics

The performance of the trained gender classification models was evaluated using several standard metrics:

*   **Accuracy**: The overall percentage of correctly classified samples.
*   **Precision**: The ability of the classifier to avoid labeling a negative sample as positive.
*   **Recall**: The ability of the classifier to correctly identify all positive samples.
*   **F1-score**: The harmonic mean of precision and recall, providing a balanced measure of the model's performance.
*   **Confusion Matrix**: A table summarizing the number of true positives, true negatives, false positives, and false negatives, providing insights into the types of errors made by the model.
    *(Space for an example confusion matrix with labels).*
*   **Classification Report**: A comprehensive report containing precision, recall, F1-score, and support for each class.

These metrics were computed on the test sets for each dataset and model configuration to provide a comprehensive evaluation of the effectiveness of the proposed methodology. The results were then compared across different models and datasets to draw conclusions about the suitability of deep learning for thermal image-based gender classification and the effectiveness of the proposed novel architecture.


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
