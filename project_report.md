
<br><br><br><br>

<div align="center">

# **DEEP LEARNING-BASED GENDER CLASSIFICATION USING THERMAL FACIAL IMAGES**


<br><br>

Internship Report submitted to <br>
Shri Ramdeobaba College of Engineering & Management, Nagpur <br>
in partial fulfillment of requirement for the award of <br>
degree of <br>

<br>

## **Bachelor of Technology**
**(16pt, Bold, Title Case)**

<br>

In <br>

<br>

## **COMPUTER SCIENCE AND ENGINEERING**

<br><br>

By <br>
<br>

### **Aditya Bhattad**
### **Etisha Shastri**

<br><br>

Guide <br>
**(12pt)** <br>
### **Dr. Pravin Sonsare**
### **Dr. Khushboo Khurana**
### **Dr. Preeti Voditel**


<br><br><br>

**Department of Computer Science and Engineering** <br>
Shri Ramdeobaba College of Engineering & Management, Nagpur 440 013 <br>
(An Autonomous Institute affiliated to Rashtrasant Tukdoji Maharaj Nagpur University Nagpur)

<br><br>

**May 2025** <br>

</div>

<br><br><br><br>

---
<!-- Page Break -->

<br>

This is to certify that the Internship Report on **“Deep Learning-Based Gender Classification Using Thermal Facial Images”** is a bonafide work of **Aditya Bhattad and Etisha Shastri** submitted to the Rashtrasant Tukdoji Maharaj Nagpur University, Nagpur in partial fulfillment of the award of a Bachelor of Technology, in Computer Science and Engineering has been carried out at **RCOEM** during the academic year **2024-25**.

<br><br>

Date: 12-05-2025 <br>
Place: Nagpur

<br><br><br>

|                                         |                                                       |
| :-------------------------------------- | :---------------------------------------------------- |
| Dr. Pravin Sonsare    | Dr. P. Voditel                                        |
| Project Guide                         | H.O.D                                                 |
| Department of Computer Science Engineering                    | Department of Computer Science Engineering            |

<br><br>


<br><br>

<div align="center">
Dr. M. B. Chandak <br>
Principal
</div>

---
<!-- Page Break -->


<div align="center">
**DECLARATION**
</div>

<br>

We, **Aditya Bhattad and Etisha Shastri**, students of Bachelor of Technology in Computer Science and Engineering at Shri Ramdeobaba College of Engineering & Management, Nagpur, hereby declare that the Internship Report titled **“Deep Learning-Based Gender Classification Using Thermal Facial Images”** is an authentic record of my own work carried out at **RCOEM**, under the guidance of **Dr. Pravin Sonsare**.

The matter embodied in this report has not been submitted by us for the award of any other degree or diploma. I further declare that this report is prepared in accordance with the Non-Disclosure Agreement (NDA) specified by the respective industry/organization, if applicable.

To the best of my knowledge and belief, the report is free from plagiarism and contains no material previously published or written by another person except where due reference is made in the text.

<br><br>

Date: [Date of Submission] <br>
Place: Nagpur

<br><br>

**Aditya Bhattad** <br>
Roll No.: 28 <br>
Branch: Computer Science and Engineering
**Etisha Shastri** <br>
Roll No.: [Roll Number] <br>
Branch: Computer Science and Engineering

---
<!-- Page Break -->

# ACKNOWLEDGEMENTS

We wish to express my sincere gratitude to my internal guide, **Dr. Pravin Sonsare**, Professor, Department of Computer Science and Engineering, Shri Ramdeobaba College of Engineering and Management, Nagpur, for providing invaluable guidance, encouragement, and support throughout the course of this internship project. His insightful suggestions and critical feedback were instrumental in shaping this work.

We are also immensely grateful to **Dr. Pravin Sonsare**, for granting me the opportunity to undertake this internship and for his/her constant supervision, valuable inputs, and for providing the necessary resources and environment to conduct this research.

We would like to thank **Dr. P. Voditel**, Head of the Department of Computer Science and Engineering, and **Dr. Parag Jawarkar**, Dean-CDPC, for their support and for fostering an environment conducive to research and learning.

Our heartfelt appreciation goes to **Dr. M. B. Chandak**, Principal, Shri Ramdeobaba College of Engineering and Management, for his leadership and for providing the platform for such enriching experiences.

We are thankful to the faculty members and staff of the Department of Computer Science and Engineering for their cooperation and assistance.


<br>
**Aditya Bhattad**
**Etisha Shastri**

---
<!-- Page Break -->

# ABSTRACT

Gender classification is a pivotal task in computer vision with wide-ranging applications. Traditional methods using visible spectrum images often falter under challenging environmental conditions. This research investigates the efficacy of thermal facial imaging as a robust alternative, leveraging deep learning, specifically Convolutional Neural Networks (CNNs), to overcome these limitations. The study conducts a comprehensive evaluation of established CNN architectures (AlexNet, VGG, InceptionV3, ResNet50, EfficientNet-B0) and proposes a novel architecture, TH-SE-ResNet, tailored for thermal data. TH-SE-ResNet integrates a Channel Input Adapter for handling disparate dataset channel formats, Squeeze-and-Excitation (SE) blocks for enhanced feature discrimination, and a custom classifier head. Experiments were performed on the Tufts University Thermal Face dataset, the Charlotte-ThermalFace dataset, and a combined dataset. Preprocessing, data augmentation, and class imbalance mitigation strategies were employed. TH-SE-ResNet demonstrated superior performance, achieving accuracies up to 97% on the Tufts dataset, 85% on the Charlotte dataset, and 90% on the combined dataset, outperforming baseline models and exhibiting better generalization and gender balance in predictions. The findings underscore the potential of specialized deep learning models for robust gender classification using thermal imagery, offering a promising solution for real-world deployment.

---
<!-- Page Break -->

# TABLE OF CONTENTS


i.  Technology/Tools Used
ii.  Project Domain
iii. Working Methodology
iv.  About Project (Project Flow Diagram)
v.  About Project Intern Role in the Industry
vi.  About Internship Work
vii. Applications

List of Figures
List of Tables
List of Symbols, Abbreviations or Nomenclature

**Chapter 1: Introduction**
1.1 Background
1.2 Motivation and Problem Statement
1.3 Objectives and Contributions
1.4 Report Organization

**Chapter 2: Literature Review**
2.1 Conventional Machine Learning Approaches
2.2 Deep Learning for Visible Spectrum Gender Classification
2.3 Gender Classification using Thermal Imaging
2.4 Advancements with CNNs in Thermal Imaging
2.5 Gaps in Existing Research and Our Approach

**Chapter 3: Methodology**
3.1 Datasets
    3.1.1 Tufts University Thermal Face Dataset
    3.1.2 Charlotte-ThermalFace Dataset
    3.1.3 Combined Dataset
3.2 Data Preprocessing and Augmentation
    3.2.1 Dataset Organization and Partitioning
    3.2.2 Image Normalization and Standardization
    3.2.3 Data Augmentation Strategies
3.3 Proposed CNN Architecture (TH-SE-ResNet)
    3.3.1 Overview
    3.3.2 Channel Input Adapter
    3.3.3 Squeeze and Excitation (SE) Blocks
    3.3.4 Classifier Head
    3.3.5 Unified Equation
3.4 Baseline Model Architectures and Comparison
    3.4.1 Baseline Architectures
        3.4.1.1 AlexNet: Foundational CNN Architecture
        3.4.1.2 VGG-16: Homogeneous Deep Architecture
        3.4.1.3 InceptionV3: Multi-Scale Processing Architecture
        3.4.1.4 ResNet50: Residual Learning Framework
        3.4.1.5 EfficientNet-B0: Optimized Scaling Architecture
    3.4.2 Input Adaptation and Training Protocol
        3.4.2.1 Thermal Image Preprocessing and Channel Adaptation
        3.4.2.2 Transfer Learning and Fine-Tuning Strategy

**Chapter 4: Experimental Setup and Results**
4.1 Experimental Setup
4.2 Results on Individual Datasets
    4.2.1 Tufts Dataset
    4.2.2 Charlotte Dataset
4.3 Results on Combined Dataset
4.4 Comparative Analysis and Key Observations

**Chapter 5: Discussion and Future Work**
5.1 Interpretation of Results
5.2 Comparison with Existing Literature
5.3 Limitations of the Study
5.4 Future Scope and Potential Enhancements

**Chapter 6: Conclusion**

**References**

---
<!-- Page Break -->

## i. Technology/Tools Used

The development and experimentation for this project involved a range of modern technologies and software tools, primarily centered around the Python ecosystem for deep learning:

*   **Programming Language:** Python (Version 3.8+)
*   **Deep Learning Framework:** PyTorch (Version 1.10+), utilized for building, training, and evaluating all Convolutional Neural Network (CNN) models.
*   **Core CNN Architectures:**
    *   AlexNet
    *   VGG-16
    *   InceptionV3
    *   ResNet50
    *   EfficientNet-B0
    *   Custom TH-SE-ResNet (developed as part of this work)
*   **Data Handling and Manipulation:**
    *   NumPy: For numerical operations and array manipulations.
    *   Pandas: For data organization and analysis (if tabular metadata was used).
    *   Torchvision: For dataset loading, image transformations, and accessing pre-trained models.
*   **Image Processing:** OpenCV (cv2) or PIL (Pillow): For basic image reading, writing, and manipulation tasks if not handled directly by Torchvision.
*   **Data Visualization:** Matplotlib and Seaborn: For plotting graphs of training/validation loss and accuracy, confusion matrices, and displaying sample images.
*   **Development Environment:**
    *   Jupyter Notebooks or IDEs like VS Code with Python extensions.
    *   Google Colaboratory (Colab) or local setup with GPU support.
*   **Hardware:** NVIDIA GeForce RTX 4090 GPU (as mentioned) for accelerated model training.
*   **Version Control:** Git and GitHub (as evidenced by image links) for code management and collaboration.
*   **Operating System:** Linux (commonly used for deep learning development) or Windows.

---
<!-- Page Break -->

## ii. Project Domain

This internship project falls under several interconnected domains within computer science and engineering:

*   **Computer Vision:** The core domain, as the project involves processing and analyzing visual information (thermal facial images) to make predictions.
*   **Deep Learning:** A subfield of machine learning that utilizes deep artificial neural networks, specifically Convolutional Neural Networks (CNNs), for feature extraction and classification.
*   **Artificial Intelligence (AI):** The broader field encompassing machine learning and deep learning, aiming to create systems that can perform tasks typically requiring human intelligence.
*   **Biometrics:** The project explores gender classification, which is a form of soft biometrics – identifying physiological or behavioral characteristics. Thermal imaging for identity or characteristic recognition is a growing area in biometrics.
*   **Thermal Imaging Analysis:** This specific niche focuses on extracting meaningful information from images captured in the thermal spectrum, distinct from visible light imagery.
*   **Image Processing:** Techniques for enhancing, normalizing, and augmenting images are fundamental to preparing data for deep learning models.
*   **Pattern Recognition:** The models learn to recognize patterns in thermal facial data that are indicative of gender.

---
<!-- Page Break -->

## iii. Working Methodology

The project followed a systematic research and development methodology, encompassing the following key phases:

1.  **Literature Review and Problem Definition:**
    *   Comprehensive review of existing research on gender classification using visible, near-infrared, and thermal imagery.
    *   Analysis of traditional machine learning methods and recent deep learning approaches.
    *   Identification of limitations in current methods, particularly performance degradation in challenging environments and the underexplored potential of thermal imaging with advanced CNNs.
    *   Defining the core problem: To investigate and improve deep learning-based gender classification using thermal facial images.

2.  **Dataset Acquisition and Preparation:**
    *   Identification and acquisition of publicly available thermal face datasets: Tufts University Thermal Face Dataset and Charlotte-ThermalFace Dataset.
    *   Understanding dataset characteristics: size, gender distribution, image properties, channel information.
    *   Creation of a Combined Dataset by merging Tufts and Charlotte datasets, addressing channel discrepancies (single-channel vs. three-channel).

3.  **Data Preprocessing and Augmentation:**
    *   **Organization:** Structuring datasets for training and testing, ensuring subject-disjoint splits (80:20 ratio).
    *   **Normalization:** Applying mean-centering and standard deviation scaling appropriate for thermal images.
    *   **Resizing:** Standardizing image dimensions based on model input requirements.
    *   **Channel Adaptation:** Handling single-channel (Charlotte) and three-channel (Tufts) data, including channel replication for baseline models and a learnable adapter for the proposed model.
    *   **Augmentation:** Implementing a robust augmentation pipeline (rotation, flipping, cropping, Gaussian blur, affine transformations) to increase data diversity, improve model generalization, and address class imbalance in the Tufts dataset.

4.  **Model Design and Selection:**
    *   **Baseline Models:** Selection of state-of-the-art CNN architectures (AlexNet, VGG-16, InceptionV3, ResNet50, EfficientNet-B0) for comparative evaluation.
    *   **Proposed Architecture (TH-SE-ResNet):**
        *   Iterative design based on the ResNet-50 backbone.
        *   Incorporation of a Channel Input Adapter for flexible single/multi-channel input.
        *   Integration of Squeeze-and-Excitation (SE) blocks to enhance feature discrimination.
        *   Development of a tailored classifier head with dropout for regularization.

5.  **Experimental Setup and Training:**
    *   **Environment:** Python with PyTorch framework, leveraging NVIDIA RTX 4090 GPU.
    *   **Training Protocol:**
        *   Optimizer: Adam.
        *   Learning Rate: 0.00005 with 5-epoch warmup and cosine annealing.
        *   Batch Sizes: 32 and 64.
        *   Epochs: 10.
        *   Loss Function: Cross-Entropy Loss.
        *   Transfer Learning: Utilizing ImageNet pre-trained weights for baseline models and the ResNet backbone of TH-SE-ResNet, followed by fine-tuning.
    *   **Evaluation Metrics:** Accuracy, Precision (weighted), Recall (weighted), F1-score (weighted), Confusion Matrices.

6.  **Results Analysis and Evaluation:**
    *   Systematic evaluation of all models on individual datasets (Tufts, Charlotte) and the Combined dataset.
    *   Analysis of performance metrics, training/test curves, and confusion matrices.
    *   Comparison of the proposed TH-SE-ResNet against baseline models.
    *   Assessment of model robustness, generalization, and handling of class imbalance.

7.  **Discussion, Conclusion, and Future Work:**
    *   Interpretation of results in the context of existing literature.
    *   Highlighting the contributions and advantages of the proposed TH-SE-ResNet.
    *   Identifying limitations of the study.
    *   Suggesting potential avenues for future research and improvements.

8.  **Reporting:**
    *   Documentation of the entire research process, methodology, experiments, and findings in the format of this internship report.

---
<!-- Page Break -->

## iv. About Project 
The project aims to develop and evaluate robust deep learning models for gender classification using thermal facial images. The overall workflow can be visualized as follows:


**Project Flow:**

1.  **Data Acquisition:**
    *   Collect thermal facial images from Tufts University Dataset and Charlotte-ThermalFace Dataset.

2.  **Dataset Preparation & Preprocessing:**
    *   **Combine Datasets (Optional):** Merge datasets, handling channel inconsistencies (e.g., single channel to three channels via replication or adapter).
    *   **Split Data:** Divide into Training (80%) and Testing (20%) sets (subject-disjoint).
    *   **Image Normalization:** Standardize pixel values (e.g., mean 0.5, std 0.5).
    *   **Image Resizing:** Resize images to fit model input requirements (e.g., 224x224 or 299x299).
    *   **Data Augmentation (Training Set Only):** Apply transformations like random rotation, horizontal flipping, resized cropping, Gaussian blur to increase dataset size and diversity, and to address class imbalance.

3.  **Model Selection / Design:**
    *   **Baseline Models:** AlexNet, VGG-16, InceptionV3, ResNet50, EfficientNet-B0 (using pre-trained ImageNet weights).
    *   **Proposed Model (TH-SE-ResNet):**
        *   ResNet-50 backbone (pre-trained).
        *   Channel Input Adapter (for single-channel thermal inputs).
        *   Integration of Squeeze-and-Excitation (SE) blocks.
        *   Custom Classifier Head.

4.  **Model Training:**
    *   **Optimizer:** Adam.
    *   **Learning Rate Strategy:** Initial LR, warmup, cosine annealing.
    *   **Loss Function:** Cross-Entropy Loss.
    *   **Batch Processing:** Iterate through training data in batches.
    *   **Forward Pass:** Input batch through the model to get predictions.
    *   **Loss Calculation:** Compare predictions with true labels.
    *   **Backward Pass (Backpropagation):** Calculate gradients.
    *   **Weight Update:** Adjust model weights using the optimizer.
    *   **Epochs:** Repeat for a defined number of epochs.

5.  **Model Evaluation (on Test Set):**
    *   Load the best model weights (based on validation performance during training).
    *   Perform inference on the unseen test set.
    *   Calculate Performance Metrics:
        *   Accuracy
        *   Precision (weighted)
        *   Recall (weighted)
        *   F1-Score (weighted)
        *   Confusion Matrix

6.  **Results Analysis & Comparison:**
    *   Compare performance of TH-SE-ResNet against baseline models across all datasets (Tufts, Charlotte, Combined).
    *   Analyze learning curves (training loss, test accuracy).
    *   Discuss findings, limitations, and contributions.

**Key Diagrammatic Reference:**

The architecture of the proposed TH-SE-ResNet, as depicted in **Figure 7** of this report, is central to understanding the novel contribution of this project. The overall training loop is also detailed in **Algorithm 3: Model Training Loop** in Chapter 4.

---
<!-- Page Break -->

## v. About Project Intern Role in the Industry/Research Lab

As a research intern contributing to the project **"Deep Learning-Based Gender Classification Using Thermal Facial Images"** at **[Name of Company/Research Lab]**, my role encompassed a variety of responsibilities critical to the research lifecycle. These responsibilities were designed to provide hands-on experience in cutting-edge computer vision and deep learning research:

1.  **Literature Survey and Gap Analysis:**
    *   Conducted an extensive review of scientific papers and articles related to gender classification, thermal imaging, and deep learning techniques.
    *   Identified current state-of-the-art methods, their limitations, and potential areas for improvement, which helped in formulating the research objectives.

2.  **Dataset Management and Preprocessing:**
    *   Sourced, downloaded, and organized the Tufts University Thermal Face Dataset and the Charlotte-ThermalFace Dataset.
    *   Developed and implemented Python scripts for data cleaning, normalization, resizing, and channel adaptation (handling single vs. three-channel images).
    *   Designed and applied data augmentation strategies to enhance dataset diversity and address class imbalance issues, particularly in the Tufts dataset.

3.  **Model Implementation and Development:**
    *   Implemented various baseline CNN architectures (AlexNet, VGG, ResNet50, InceptionV3, EfficientNet-B0) using the PyTorch framework.
    *   Played a key role in the design, development, and iterative refinement of the novel **TH-SE-ResNet** architecture. This involved:
        *   Implementing the Channel Input Adapter.
        *   Integrating Squeeze-and-Excitation (SE) blocks within the ResNet framework.
        *   Designing and coding the custom classifier head.

4.  **Experimentation and Training:**
    *   Set up the experimental environment, configured training parameters (learning rates, batch sizes, optimizers, loss functions).
    *   Conducted numerous training runs for all models on the individual and combined datasets using high-performance GPU resources (NVIDIA RTX 4090).
    *   Monitored training processes, logged performance metrics, and debugged issues as they arose.

5.  **Results Analysis and Evaluation:**
    *   Collected and processed experimental results, including accuracy, precision, recall, F1-scores, and confusion matrices.
    *   Generated visualizations (graphs of learning curves, confusion matrices) to aid in the interpretation of results.
    *   Performed comparative analysis of the proposed TH-SE-ResNet against baseline models.

6.  **Documentation and Reporting:**
    *   Maintained detailed records of experiments, code versions, and findings.
    *   Contributed significantly to the preparation of this research internship report, including drafting methodology, results, and discussion sections.

7.  **Collaboration and Learning:**
    *   Regularly interacted with my mentors (**[Name of Industry Mentor]** and **Dr. [Name of Internal Guide]**) to discuss progress, challenges, and future directions.
    *   Engaged in learning new concepts and techniques in deep learning and thermal image analysis as required by the project.

This multifaceted role provided a comprehensive experience in applied research, from conceptualization and literature review through to implementation, experimentation, and analysis of results.

---
<!-- Page Break -->

## vi. About Internship Work

The internship work focused on advancing the field of gender classification by leveraging the unique advantages of thermal facial imaging combined with state-of-the-art deep learning techniques. The core objective was to develop and rigorously evaluate a robust system capable of accurately determining gender from thermal face images, particularly in scenarios where traditional visible-light systems might fail.

The internship commenced with an in-depth study of existing literature to understand the challenges and opportunities in thermal image-based biometrics. This foundational work highlighted the potential of thermal imaging's invariance to illumination but also pointed to difficulties arising from lower image resolution and less distinct facial features compared to RGB images.

A significant portion of the internship was dedicated to data management. Two key public datasets, Tufts University Thermal Face and Charlotte-ThermalFace, were procured and meticulously prepared. This involved developing preprocessing pipelines for normalization, resizing, and critically, handling differences in image channel formats between the datasets. Data augmentation techniques were researched and implemented not only to expand the training data and improve model generalization but also to strategically address the class imbalance present in the Tufts dataset.

The main technical contribution was the design and implementation of a novel Convolutional Neural Network (CNN) architecture, named **TH-SE-ResNet**. This model was built upon a ResNet-50 backbone and enhanced with:
1.  A **Channel Input Adapter** to seamlessly process single-channel thermal images (like those in the Charlotte dataset) with a model structure typically expecting three-channel inputs.
2.  **Squeeze-and-Excitation (SE) blocks** integrated within the residual layers to enable the network to learn channel-wise feature interdependencies and focus on more discriminative thermal patterns.
3.  A **customized classifier head** with appropriate dropout layers for better regularization and final classification.

Alongside the development of TH-SE-ResNet, several established CNN architectures (AlexNet, VGG, InceptionV3, ResNet50, EfficientNet-B0) were implemented as baselines. All models were trained and evaluated systematically on the individual datasets and a combined version of both. The training utilized transfer learning from ImageNet pre-trained weights, Adam optimizer, and a learning rate schedule with warmup and cosine annealing. Experiments were conducted with different batch sizes to observe their impact.

The final phase of the internship involved a thorough analysis of the experimental results. Performance was benchmarked using metrics such as accuracy, precision, recall, F1-score, and confusion matrices. The TH-SE-ResNet consistently outperformed the baseline models across all dataset configurations, demonstrating its superior ability to learn gender-specific features from thermal images and generalize well.

Throughout the internship, rigorous experimentation, careful analysis, and detailed documentation were maintained, culminating in this comprehensive report which outlines the methodology, findings, and potential future directions of this research.

---
<!-- Page Break -->

## vii. Applications

The development of robust gender classification systems using thermal facial images, as explored in this project, has a wide array of potential real-world applications across various domains. The key advantage of thermal imaging—its independence from ambient illumination and ability to operate in complete darkness—opens up possibilities where visible-spectrum systems are ineffective or impractical:

1.  **In-Cabin Driver Monitoring Systems:**
    *   Enhanced safety features in autonomous and semi-autonomous vehicles by tailoring responses or alerts based on driver demographics (e.g., gender-specific fatigue patterns, although this requires further research).
    *   Personalization of vehicle settings (e.g., seat position, climate control, infotainment preferences) based on recognized driver gender.

2.  **Human-Computer Interaction (HCI):**
    *   Development of more intuitive and personalized user interfaces that adapt to the gender of the user.
    *   Enriching interactive experiences in gaming, virtual reality (VR), and augmented reality (AR) applications.
    *   Social robotics, where robots can interact more naturally and appropriately based on perceived gender cues.

3.  **Video Surveillance and Security:**
    *   Improved demographic analysis in public spaces for security monitoring and threat assessment, especially in low-light or nighttime conditions where visible cameras fail.
    *   Enhanced forensic analysis by providing gender information from thermal footage captured at crime scenes.
    *   Access control systems that can use gender as a soft biometric feature in multi-factor authentication.

4.  **Retail Analytics and Smart Environments:**
    *   Gathering anonymized demographic data (gender distribution of shoppers) to optimize store layouts, product placements, and marketing strategies, even in dimly lit areas or through certain types of glass.
    *   Smart building systems that can adjust environmental controls (lighting, temperature) based on the demographic makeup of occupants.

5.  **Psychological and Physiological Analysis:**
    *   Non-invasive research into gender-specific physiological responses (e.g., stress, emotional states) by analyzing thermal facial patterns, as thermal signatures can reflect subtle blood flow changes.
    *   Complementary information in medical diagnostics or monitoring where gender might be a relevant factor.

6.  **Forensics and Law Enforcement:**
    *   Assisting in the identification process of individuals from thermal imagery where visible imagery is unavailable or of poor quality.
    *   Narrowing down suspect pools based on gender cues in investigations.

7.  **Missing Persons and Disaster Relief:**
    *   In search and rescue operations, especially in dark or obscured environments, thermal cameras can detect human presence, and subsequent gender classification could aid in identification efforts.

The robustness of thermal imaging to varying light conditions and its potential for privacy preservation (as it doesn't capture detailed visible facial features like traditional cameras) make it an attractive modality for these and other emerging applications. The advancements in deep learning, as demonstrated by the TH-SE-ResNet model, are crucial for unlocking the full potential of thermal data for accurate and reliable gender classification.

---
<!-- Page Break -->

# LIST OF FIGURES

**Figure 1:** Sample Images from the Tufts University Thermal Face Dataset Grid
**Figure 2:** Sample Images from the Charlotte-ThermalFace Dataset Grid
**Figure 3:** Subject-Disjoint Dataset Partitioning Schema
**Figure 4:** Thermal Image Augmentation Examples
**Figure 5:** Detailed Schematic of the AlexNet Architecture (Conceptual)
**Figure 6:** Detailed Schematic of the VGG-16 Architecture (Conceptual)
**Figure 7:** Overall Architecture of the Proposed TH-SE-ResNet Model
**Figure 8:** Channel Input Adapter Architecture (Conceptual)
**Figure 9:** Squeeze-and-Excitation (SE) Block Architecture (Conceptual)
**Figure 10:** Classifier Head Architecture of TH-SE-ResNet (Conceptual)
**Figure 11:** Training Loss and Test Accuracy Curves on Tufts Dataset (Batch Size 64)
**Figure 12:** Training Loss and Test Accuracy Curves on Tufts Dataset (Batch Size 32)
**Figure 13a:** Confusion Matrix - TH-SE-ResNet, Tufts Dataset, Batch Size 64 (Placeholder)
**Figure 13b:** Confusion Matrix - TH-SE-ResNet, Tufts Dataset, Batch Size 32 (Placeholder)
**Figure 14:** Training Loss and Test Accuracy Curves on Charlotte Dataset (Batch Size 64)
**Figure 15:** Training Loss and Test Accuracy Curves on Charlotte Dataset (Batch Size 32)
**Figure 16a:** Confusion Matrix - TH-SE-ResNet, Charlotte Dataset, Batch Size 64 (Placeholder)
**Figure 16b:** Confusion Matrix - TH-SE-ResNet, Charlotte Dataset, Batch Size 32 (Placeholder)
**Figure 17:** Training Loss and Test Accuracy Curves on Combined Dataset (Batch Size 64)
**Figure 18:** Training Loss and Test Accuracy Curves on Combined Dataset (Batch Size 32)
**Figure 19a:** Confusion Matrix - TH-SE-ResNet, Combined Dataset, Batch Size 64 (Placeholder)
**Figure 19b:** Confusion Matrix - TH-SE-ResNet, Combined Dataset, Batch Size 32 (Placeholder)
**Figure 20:** Sample Classification Analysis from Combined Dataset Test Set (TH-SE-ResNet, B64)

---
<!-- Page Break -->

# LIST OF TABLES

**Table 1:** Summary of Datasets
**Table 2:** Model-Specific Normalization Parameters
**Table 3:** Final Experimental Dataset Configurations
**Table 4:** Comprehensive Baseline Model Specifications
**Table 5:** Performance on Tufts Dataset
**Table 6:** Performance on Charlotte Dataset
**Table 7:** Performance on Combined Dataset
**Table 8:** Comparative Summary of TH-SE-ResNet Performance against Literature

---
<!-- Page Break -->

# LIST OF SYMBOLS, ABBREVIATIONS OR NOMENCLATURE

*   **AI:** Artificial Intelligence
*   **BN:** Batch Normalization
*   **CNN:** Convolutional Neural Network
*   **EER:** Equal Error Rate
*   **F:** Female (in dataset distributions/results)
*   **FC:** Fully Connected (Layer)
*   **FERET:** Face Recognition Technology (Dataset)
*   **FLIR:** Forward-Looking Infrared
*   **GAP:** Global Average Pooling
*   **GPU:** Graphics Processing Unit
*   **HCI:** Human-Computer Interaction
*   **HOG:** Histogram of Oriented Gradients
*   **IR:** Infrared
*   **IRT:** Infrared Thermal
*   **LBP:** Local Binary Patterns
*   **LRN:** Local Response Normalization
*   **M:** Male (in dataset distributions/results)
*   **MBConv:** Mobile Inverted Bottleneck Convolution
*   **MSX:** Multi-Spectral Dynamic Imaging
*   **NDA:** Non-Disclosure Agreement
*   **PCA:** Principal Component Analysis
*   **RCOEM:** Shri Ramdeobaba College of Engineering & Management
*   **ReLU:** Rectified Linear Unit
*   **RGB:** Red, Green, Blue (color model)
*   **SE Block:** Squeeze-and-Excitation Block
*   **SVM:** Support Vector Machine
*   **TH-SE-ResNet:** Thermal Squeeze-and-Excitation Residual Network (Proposed Model)
*   **UAV:** Unmanned Aerial Vehicle
*   **WLD:** Weber's Local Descriptor

---
<!-- Page Break -->

# Chapter 1: Introduction

## 1.1 Background
Gender classification has become a fundamental task in computer vision, with numerous applications across various domains including in-cabin driver monitoring systems, human-computer interaction, video surveillance, retail analytics, and psychological analysis. Traditionally, researchers have focused on gender classification using visible spectrum images of the human face. However, the performance of these systems can be significantly affected by challenging environmental factors such as varying illumination conditions, shadows, occlusions, and the time of day.

## 1.2 Motivation and Problem Statement
To overcome these limitations, there has been a growing interest in exploring alternative or complementary sensing modalities, such as **thermal imaging**. Thermal imaging offers several advantages as it does not rely on external illumination and provides a distinct perspective on an imaged scene compared to conventional visible light sensors. This makes it a potentially more robust solution for gender classification in diverse and uncontrolled environments. Furthermore, thermal imaging can easily detect people even in total darkness, expanding its applicability in security systems. Beyond security, thermal signatures can provide complementary information in human-computer interaction, potentially revealing subtle physiological indicators relevant to gender.

Despite the benefits, thermal images typically lack some of the detailed facial definitions present in visible spectrum images, posing a challenge for accurate classification. This research addresses the problem of robust gender classification under such challenging conditions by exploring the capabilities of thermal imaging combined with advanced deep learning techniques. The core problem is to develop a model that can effectively learn discriminative features from thermal facial images, outperforming traditional methods and standard deep learning architectures not specifically tailored for thermal data.

## 1.3 Objectives and Contributions
This paper investigates the effectiveness of deep learning models for gender detection using thermal facial images. We utilize two publicly available thermal image datasets, the **Tufts University Thermal Face dataset** and the **Charlotte-ThermalFace dataset**, both individually and in combination, to train and evaluate a range of state-of-the-art CNN architectures including **AlexNet**, **VGG**, **InceptionV3**, **ResNet50**, and **EfficientNet**. We address the differences in channel availability between the datasets and enhance the data through **image augmentation** techniques. Furthermore, we tackle the class imbalance present in the Tufts dataset to ensure robust training. To further advance the field, we propose a **novel CNN architecture** based on the ResNet framework, incorporating a **channel input adapter** to handle varying input channels and **Squeeze and Excitation (SE) blocks** within its layers to enhance feature discrimination, along with a tailored final classifier.

The primary objectives of this internship project are:
*   To conduct a comprehensive evaluation of several state-of-the-art CNN models for gender classification on thermal facial images using the Tufts and Charlotte datasets.
*   To investigate the impact of combining datasets with differing channel characteristics on model performance and generalization.
*   To develop and evaluate a novel CNN architecture (TH-SE-ResNet) specifically designed for thermal image-based gender detection, incorporating channel adaptation and attention mechanisms.
*   To analyze the challenges, potential, and performance benchmarks of deep learning for gender classification using thermal imaging.

The primary contributions of this paper include:
*   A comprehensive evaluation of several state-of-the-art CNN models for gender classification on thermal facial images using the Tufts and Charlotte datasets.
*   An investigation into the impact of combining datasets with differing channel characteristics.
*   The development and evaluation of a novel CNN architecture specifically designed for thermal image-based gender detection, incorporating channel adaptation and attention mechanisms.
*   An analysis of the challenges and potential of deep learning for gender classification using thermal imaging.

## 1.4 Report Organization
The remainder of this report is structured as follows: Chapter 2 provides a review of related work in gender classification using both traditional and deep learning methods with visible, near-infrared, and thermal imagery. Chapter 3 details the datasets used and the methodology employed, including preprocessing, augmentation techniques, and the architecture of the proposed CNN model and baseline models. Chapter 4 presents the experimental setup, results, and a comparative analysis of the different models. Chapter 5 discusses the implications, limitations of our findings, and compares them with existing literature. Finally, Chapter 6 concludes the report, summarizing the key findings and suggesting potential directions for future research.

---
<!-- Page Break -->

# Chapter 2: Literature Review

The task of gender classification has been extensively studied in computer vision. This chapter reviews relevant literature, covering conventional machine learning approaches, the rise of deep learning for visible spectrum images, and the specific advancements and challenges associated with using thermal imaging for this task.

## 2.1 Conventional Machine Learning Approaches
Early approaches often relied on **conventional machine learning methods** and feature extraction techniques applied to visible spectrum images. Makinen and Raisamo and Reid et al. provided detailed surveys of these methods. Initial techniques involved training neural systems on small sets of frontal face images. Later, methods incorporated 3D head structure and image intensities for gender characterization. **Support Vector Machines (SVMs)** were also widely used, demonstrating competitive performance compared to other traditional classifiers. Techniques like AdaBoost, utilizing low-resolution grayscale images, and methods addressing perspective invariant recognition were also explored. More recently, researchers utilized local image descriptors like the Weber's Local Descriptor (WLD) and features based on shape, texture, and color extracted from frontal faces, achieving high accuracy on benchmark datasets like FERET. Early systems applied handcrafted features like Local Binary Patterns (LBP), Principal Component Analysis (PCA), and Histogram of Oriented Gradients (HOG) in conjunction with classifiers like Support Vector Machines (SVMs). However, these approaches have repeatedly demonstrated fragility under real-world conditions where variables such as illumination, facial occlusions (e.g., masks, sunglasses), shadows, and changing poses severely degrade performance.

## 2.2 Deep Learning for Visible Spectrum Gender Classification
With the advent of deep learning, Convolutional Neural Networks (CNNs) became the dominant approach for various computer vision tasks, including gender classification from visible spectrum images, often outperforming traditional methods by automatically learning hierarchical feature representations.

## 2.3 Gender Classification using Thermal Imaging
To overcome the limitations of visible spectrum systems, research has increasingly turned toward thermal imaging as an alternative sensing modality. Thermal images capture the heat signature emitted by facial tissues, providing invariant information that is unaffected by ambient lighting conditions. This makes them particularly useful in environments where visible light sensors fail—such as night-time surveillance, poor weather conditions, or low-contrast settings. Moreover, thermal imaging enables detection of subtle physiological patterns that are invisible in RGB data, which can offer additional cues for gender classification.

Yet, the advantages of thermal imaging also come with notable challenges. Thermal images tend to have lower resolution and often lack the detailed structural and textural features present in visible spectrum images. These factors make feature extraction and discrimination more complex. The traditional methods that worked well on RGB data often fail when applied directly to thermal imagery. This is where deep learning, specifically Convolutional Neural Networks (CNNs), has shown immense promise in automatically learning hierarchical representations even from low-resolution and noisy thermal data.

## 2.4 Advancements with CNNs in Thermal Imaging
One of the landmark studies in this direction was by Jalil et al. (2023) who introduced a Modified CNN model for classifying gender of thermal images using cloud computing. Their architecture, Cloud_Res, was specifically optimized for thermal facial images and achieved a remarkable precision. What distinguished this work was its deployment in a cloud environment, leveraging the scalability and speed of cloud-based inference engines. They also benchmarked their architecture against traditional ResNet variants (18, 50, and 101 layers), concluding that a well-designed lightweight CNN with fewer layers could achieve similar—if not better—performance due to reduced overfitting and faster convergence. However, their model did not incorporate attention mechanisms or adaptive input handling, and it exhibited some imbalance in gender classification accuracy, particularly favoring male predictions.

In parallel, Chatterjee and Zaman (2023) conducted an in-depth study using ResNet-50 and VGG-19 architectures on the Tufts and Charlotte ThermalFace datasets. Their preprocessing pipeline included Kalman filtering, which significantly enhanced the signal-to-noise ratio of thermal images, resulting in an increase in classification accuracy by 3–5%. Their best performing model, ResNet-50, achieved 95.0% accuracy, demonstrating that deeper CNNs are capable of extracting discriminative patterns even from thermally distorted or noisy data. However, these architectures are generic and not explicitly designed for thermal imaging.

Another key contribution was made by Nguyen et al. (2017), who combined thermal and visible-light camera feeds for gender classification using a CNN-SVM hybrid approach. Their work highlighted the effectiveness of feature-level fusion and score-level fusion in improving classification accuracy. While this bimodal setup achieved higher accuracy compared to single-modality systems, it required synchronized camera systems and complex alignment pipelines—rendering it impractical for many real-world deployments where only thermal imaging is feasible.

The role of hand-based thermal imaging was explored by Prihodova et al. (2022). Their work used VGG-16 and VGG-19 models on thermal hand images and achieved an impressive accuracy of 94.9%. While promising, hand-based methods require subject cooperation and controlled image acquisition, limiting their utility in surveillance and dynamic environments.

In terms of architectural exploration, Farooq et al. and others performed comprehensive benchmarking using CNNs like AlexNet, VGG19, and EfficientNet-B4. Their results consistently showed that shallow networks like AlexNet underperformed (with accuracies around 82.6%), while InceptionV3 reached 92.3% due to its deeper and more modular design. These studies emphasize the importance of choosing architectures capable of capturing both local and global patterns—something shallow networks struggle with in low-resolution thermal images.

The Infrared Thermal Image Gender Classifier (IRT_ResNet) proposed by Jalil et al. (2022) compared ResNet variants and demonstrated that deeper networks (ResNet-101) offered better performance. However, their study noted diminishing returns beyond a certain depth and highlighted the model's skewed performance favoring male predictions, suggesting a need for better-balanced training methods.

Thermal-based gender classification from UAV-mounted cameras has also gained attention. Studies like "Thermal-based Gender Recognition Using Drones" and "Gender Recognition Using UAV-based Thermal Images" explored mobile applications where thermal images captured from drones were used for biometric analysis. However, these setups faced challenges due to image instability, resolution loss, and varying subject distance. CNNs like AlexNet and GoogLeNet achieved moderate accuracies (82–85%), but performance varied depending on environmental conditions.

To address occlusions and dataset-specific challenges, some researchers proposed the use of 3D facial models or spatial-temporal analysis (e.g., CNN-BGRU models) to integrate motion or depth-based cues into classification. These approaches, while theoretically sound, are computationally intensive and not well-suited for low-power or real-time deployments.

## 2.5 Gaps in Existing Research and Our Approach
In our work, we aim to overcome these limitations through the design of a novel CNN architecture called TH-SE-ResNet. It builds upon the ResNet backbone but introduces several key innovations:
*   **Channel Input Adapter:** Given the inconsistency in channel formats between datasets (e.g., grayscale vs. RGB), our model integrates an adapter module to standardize inputs, allowing for seamless dataset fusion. This is particularly crucial as we combine the Tufts University and Charlotte-ThermalFace datasets in our experiments.
*   **Squeeze-and-Excitation (SE) Blocks:** To improve feature discrimination, SE blocks are embedded within residual units. These blocks dynamically recalibrate channel-wise feature responses, enabling the network to prioritize salient thermal features—especially useful in handling occlusions and low-contrast areas.
*   **Class-Imbalance Mitigation:** We incorporate class-weighted loss functions during training and targeted augmentation to counter the male-biased prediction patterns observed in previous studies (e.g., Jalil et al., 2022). This ensures fairer and more balanced classification across genders.
*   **Data Augmentation Pipeline:** We apply a robust preprocessing and augmentation routine—using techniques such as rotation, flipping, and Gaussian noise injection—to increase model generalizability and reduce overfitting.
*   **Evaluation on Combined Datasets:** Unlike prior work that tested models on isolated datasets, we evaluate our model on a combined Tufts-Charlotte dataset, increasing diversity in facial features, pose variations, and sensor modalities, thus pushing the limits of generalization.

Our experimental findings demonstrate that TH-SE-ResNet consistently outperforms standard architectures across multiple metrics (accuracy, precision, recall, F1-score) and maintains high performance even under occlusion and noise. Unlike previous models limited to specific deployment environments, our model is designed for robust performance across varied thermal data sources.

In conclusion, while existing literature has made considerable strides in leveraging CNNs for thermal gender classification, challenges remain in model generalization, dataset handling, feature prioritization, and bias mitigation. Our work builds on this foundation by introducing a tailored architecture that directly addresses these gaps. TH-SE-ResNet offers a more complete, fair, and potentially deployable solution—moving the field closer to practical, large-scale implementations of thermal gender classification systems.

---
<!-- Page Break -->

# Chapter 3: Methodology

This chapter details the datasets utilized, the comprehensive data preprocessing and augmentation pipeline, the architecture of our proposed TH-SE-ResNet model, and the baseline CNN architectures employed for comparative analysis.

## 3.1 Datasets

Our research utilizes two publicly available thermal facial image datasets, along with a combined version, to ensure a robust evaluation of the proposed and baseline models.

### 3.1.1 Tufts University Thermal Face Dataset

The Tufts University Thermal Face Dataset represents a comprehensive multimodal collection comprising over 10,000 images across various modalities acquired from a diverse cohort of 113 participants (74 females, 39 males). For our research, we specifically utilized the thermal subset containing approximately 1,400 images. The age distribution spans from 4 to 70 years, with subjects originating from more than 15 countries, thus providing substantial demographic variability. Image acquisition was conducted using a FLIR Vue Pro thermal camera under controlled indoor environmental conditions. Participants were positioned at a standardized distance from the imaging apparatus to maintain consistency. For our investigation, we specifically utilized two subsets: TD_IR_E (Emotion), which contains images depicting five distinct facial expressions (neutral, smile, eyes closed, shocked, and with sunglasses), and TD_IR_A (Around), which encompasses images captured from nine different camera positions arranged in a semicircular configuration around each participant. A significant challenge encountered with this dataset was the pronounced gender imbalance, with approximately 30.32% female and 69.68% male images. To mitigate this imbalance and enhance model robustness, we implemented targeted data augmentation techniques specifically for the underrepresented female class, including controlled geometric transformations and intensity adjustments while preserving critical thermal signature characteristics.

![tufts_grid](https://github.com/user-attachments/assets/3b896a26-95b9-4c6b-b02a-f22fed2de0a6)
**Figure 1:** Sample Images from the Tufts University Thermal Face Dataset Grid
*(Source: Original Paper)*

### 3.1.2 Charlotte-ThermalFace Dataset

The Charlotte-ThermalFace Dataset comprises approximately 10,364 thermal facial images from 10 subjects, collected under varying conditions (e.g., distance, head position, temperature). This dataset was not specifically created for gender detection tasks, but we repurposed it for our gender classification research. Based on image characteristics, we infer that data acquisition likely employed a FLIR-based thermal imaging system. In contrast to the Tufts collection, the Charlotte dataset exhibits near-perfect gender balance with approximately 50.10% female and 49.90% male. This balanced distribution provided an advantageous counterpoint to the gender imbalance present in the Tufts dataset.

![charlotte_grid](https://github.com/user-attachments/assets/3c6c179a-e9fe-43d4-a16d-ca780c4d42c9)
**Figure 2:** Sample Images from the Charlotte-ThermalFace Dataset Grid
*(Source: Original Paper)*

### 3.1.3 Combined Dataset

To enhance data diversity and expand the training corpus, we constructed a combined dataset by integrating the Tufts and Charlotte collections following a systematic merging protocol. A significant technical challenge encountered during this integration was the channel discrepancy between datasets—the Charlotte images were originally single-channel thermal representations, whereas the Tufts dataset employed a three-channel format. To address this incompatibility, we implemented channel replication for the Charlotte images, duplicating the single thermal channel across three channels to establish format consistency with the Tufts data structure when used with baseline models. For our proposed TH-SE-ResNet, the channel input adapter handled this. Furthermore, to prevent model bias towards the overrepresented class from the Tufts dataset when combining, we carefully balanced the gender distribution by selecting an equal number of images per gender category through strategic sampling or ensuring augmentation balanced the final combined set. This integration yielded a substantially enlarged dataset of approximately 11,921 images with perfect gender balance (50% female, 50% male), thereby providing our models with enhanced training diversity spanning different thermal imaging conditions, acquisition parameters, and subject characteristics.

**Table 1: Summary of Datasets**

| Dataset    | Size (Images) | Gender Distribution     | Channels                        |
|------------|---------------|-------------------------|---------------------------------|
| Tufts      | ~1,400        | 30.32% F, 69.68% M      | Three (thermal representation)  |
| Charlotte  | ~10,000       | 50.10% F, 49.90% M      | One (thermal grayscale)        |
| Combined   | ~11,921       | 50.00% F, 50.00% M      | Three-channel format (standardized) |

## 3.2 Data Preprocessing and Augmentation

Our data preprocessing and augmentation pipeline was meticulously designed to address the unique challenges of thermal facial image analysis for gender classification. The pipeline incorporated several carefully considered stages to ensure optimal model performance and generalization.

### 3.2.1 Dataset Organization and Partitioning

We structured our datasets according to a standardized hierarchical organization to facilitate efficient training and evaluation. Each dataset (Tufts, Charlotte, and Combined) was systematically partitioned into training and testing subsets using a subject-disjoint approach. This critical design choice ensured that images from the same individual never appeared in both training and testing sets, thus preventing identity-based information leakage that could artificially inflate performance metrics. We implemented an 80:20 train-test split ratio, stratified by gender to maintain proportional representation across partitions.

For the Tufts dataset, we addressed the gender imbalance during augmentation, ensuring that the disproportionate male-to-female ratio was managed effectively for balanced training impact. In the Charlotte dataset, the near-perfect gender balance was preserved throughout the partition process. For the combined dataset, we implemented balanced sampling and augmentation strategies to achieve gender parity while maintaining subject-level separation between training and testing sets.

![dataset_partitioning_schema](https://github.com/user-attachments/assets/06a2b046-8513-4092-9345-ad485141a975)
**Figure 3:** Subject-Disjoint Dataset Partitioning Schema - A diagram showing the hierarchical organization and separation of subjects by gender across train/test splits.
*(Source: Original Paper, renumbered)*

### 3.2.2 Image Normalization and Standardization

Thermal imaging data presents unique challenges due to variations in sensor calibration, environmental conditions, and temperature ranges. To mitigate these issues, we implemented a comprehensive normalization protocol:

All thermal images were normalized using mean-centering with a value of 0.5 and standard deviation scaling of 0.5. This approach was selected over the conventional ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) as it proved more effective for thermal imagery during our preliminary experiments, likely due to the fundamentally different intensity distribution characteristics of thermal versus visible spectrum images.

The Charlotte dataset's single-channel thermal images required special handling. For standard RGB-designed architectures (AlexNet, VGG, ResNet, EfficientNet, InceptionV3), we expanded the single channel through replication to create a three-channel input, maintaining compatibility while preserving the original thermal information. For our TH-SE-ResNet, which is specifically designed for thermal data, its Channel Input Adapter directly processed the single-channel representation from Charlotte (after grayscale conversion if needed for Tufts), or the three-channel representation from Tufts.

Images were resized according to model-specific requirements—224×224 pixels for AlexNet, VGG, ResNet, EfficientNet, and TH-SE-ResNet; 299×299 pixels for InceptionV3. This standardization ensured consistent spatial dimensions while preserving the aspect ratio through center cropping where appropriate, thus maintaining the integrity of facial thermal patterns.

**Table 2: Model-Specific Normalization Parameters**
| Model Type                         | Input Size | Normalization Values                       | Channels (Input to Backbone) | Rationale                                                       |
|------------------------------------|------------|--------------------------------------------|-----------------------------|-----------------------------------------------------------------|
| AlexNet/VGG/ResNet50/EfficientNet-B0 | 224×224    | mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]  | 3                           | Optimized for thermal intensity distribution; compatibility   |
| InceptionV3                        | 299×299    | mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]  | 3                           | Maintained larger input size; compatibility                   |
| TH-SE-ResNet                       | 224×224    | mean=[0.5(x1 or x3)], std=[0.5(x1 or x3)] | 3 (via adapter)             | Adapter handles 1 or 3 channel raw input to produce 3 for backbone |

### 3.2.3 Data Augmentation Strategies

We developed a sophisticated augmentation strategy tailored specifically for thermal facial imagery, balancing the need for dataset expansion with the preservation of thermally significant features:

For baseline models (AlexNet, VGG, ResNet, EfficientNet), we implemented an augmentation pipeline including random resized cropping (scale 0.8-1.0), horizontal flipping (probability 0.5), moderate rotation (±15°), and light Gaussian blurring (kernel size=3, sigma=0.1-2.0). The Inception model follows a similar strategy but uses architecture-specific dimensions (299px crop size, typically from a 342px resized image).

For our novel TH-SE-ResNet, when dealing with the Tufts dataset (3-channel), it undergoes similar augmentation. When handling single-channel input (e.g. from Charlotte or converted Tufts for specific tests of the adapter), it would be converted to grayscale if not already, and then a modified augmentation pipeline is applied. This includes horizontal flipping, reduced rotation (±10°), and carefully controlled affine transformations (degrees=5°, translation=±5%, scaling=±5%). We used black fill (value=0) for all geometric transformations to maintain thermal signature consistency, and a slightly reduced Gaussian blur (sigma=0.1-1.5) to preserve subtle thermal gradients. This conservative parameterization helps preserve the thermal signature integrity crucial for gender classification.

To address the pronounced gender imbalance in the Tufts dataset, we implemented targeted augmentation for the underrepresented female class. The system identifies female samples and applies additional augmentations exclusively to these instances, effectively increasing the female representation in the training set towards balance. This selective augmentation substantially improves class balance without introducing excessive redundancy or overfitting risks to the majority class.

Our implementation combines multiple dataset variations during training: (1) a base dataset with minimal preprocessing, (2) a fully augmented version of the entire dataset (Charlotte and Combined), and (3) an additional augmented subset containing only samples from the minority class (Tufts dataset) to achieve balance. This approach provides models with both the original thermal signatures and systematically expanded variations, with particular emphasis on improving representation of the underrepresented gender.

For all models, we maintained separate transformation pipelines for training and testing. Test-time preprocessing was kept minimal (resize, center crop, and normalization) to evaluate model performance on thermal signatures closer to their original form.

This carefully engineered preprocessing and augmentation pipeline provided our models with high-quality, balanced training data while preserving the critical thermal signatures necessary for accurate gender classification in thermal facial imagery.

![new_thermal_augmentation_combined_examplesaa](https://github.com/user-attachments/assets/8842aedb-3c13-432e-b7af-baa0b1c6c789)
**Figure 4:** Thermal Image Augmentation Examples - A grid showing original thermal facial images alongside various augmented versions (without normalization).
*(Source: Original Paper, renumbered)*

**Table 3: Final Experimental Dataset Configurations**
| Experiment     | Training Set                 | Testing Set                | Total Training Images (Approx. after Aug.) | Total Testing Images |
|----------------|------------------------------|----------------------------|--------------------------------------------|----------------------|
| Tufts-only     | Tufts train (balanced aug.)  | Tufts test                 | ~1,600*                                    | 330                  |
| Charlotte-only | Charlotte train (std. aug.)  | Charlotte test             | ~16,000*                                   | 2,000                |
| Combined       | Combined train (balanced aug.)| Combined test              | ~18,200*                                   | 2,290                |
*\*Approximate values after augmentation strategies aimed at dataset expansion and balancing.*

## 3.3 Proposed CNN Architecture (TH-SE-ResNet)

### 3.3.1 Overview

Our research introduces a novel deep learning framework, TH-SE-ResNet, built upon a modified ResNet-50 architecture, tailored specifically for thermal image classification, capable of handling both single-channel and three-channel inputs effectively. The selection of ResNet-50 as the foundational backbone is driven by its proven ability to address challenges inherent in training very deep neural networks. A hallmark of ResNet is its use of residual connections, which mitigate the vanishing gradient problem by introducing skip connections that allow gradients to propagate more effectively during backpropagation. This design enables the construction of deeper architectures without compromising performance, a critical advantage when extracting intricate features from complex human faces in thermal imagery.

ResNet-50 strikes an exceptional balance between computational efficiency and representational power. Its 50-layer depth facilitates the hierarchical extraction of features, ranging from low-level details such as edges and textures to high-level semantic patterns, which are essential for discerning subtle gender-specific cues in thermal images. Furthermore, initializing the model with pretrained weights from ImageNet provides a robust starting point. Although thermal images differ from natural images, the general visual features learned from ImageNet—such as edge detection and texture analysis—serve as transferable knowledge that can be fine-tuned to adapt to our domain. This transfer learning approach accelerates convergence and enhances performance, particularly when training data is limited.

It is worth noting that the final architecture emerged from a rigorous iterative development process. Multiple architectural variants were systematically evaluated, with each iteration informing subsequent refinements based on empirical performance assessments. This methodical approach to model selection enabled us to identify the optimal configuration presented in this study.

The proposed architecture enhances the standard ResNet-50 by integrating three key modifications: a Channel Input Adapter to handle varying input channels (single or three), Squeeze-and-Excitation (SE) blocks to improve feature representation, and a redesigned classifier head to optimize classification performance. Each component is meticulously crafted to align with the implementation, ensuring consistency between the theoretical design and practical execution.

![Architecture](https://github.com/user-attachments/assets/91b98f31-c02b-4c24-a38f-013f90641ae7)
**Figure 7:** Overall Architecture of the Proposed TH-SE-ResNet Model - A comprehensive diagram showing the complete model architecture with all components connected, highlighting the modifications to the standard ResNet-50.
*(Source: Original Paper, renumbered from Figure 4)*

### 3.3.2 Channel Input Adapter

Thermal imaging often presents unique challenges due to the prevalence of single-channel grayscale images (e.g., Charlotte dataset), whereas pretrained models like ResNet-50 are designed for three-channel RGB inputs. The Tufts dataset, conversely, provides three-channel thermal data. To bridge this gap effectively and allow seamless integration of diverse datasets, we developed a Channel Input Adapter. This adapter transforms inputs into a three-channel representation suitable for the pretrained ResNet backbone. Unlike simplistic channel replication, our adapter employs a learnable transformation.

**Architecture:**
The Channel Input Adapter is implemented as a small sequence of convolutional layers that preprocess the input before it enters the main ResNet backbone.
*   **Input Handling:** The adapter first checks the number of input channels.
    *   If the input $x$ is single-channel ($x \in \mathbb{R}^{1 \times H \times W}$), it is passed through two convolutional layers:
        1.  A 3×3 convolutional layer with 32 output channels, padding of 1, followed by Batch Normalization and ReLU activation. Let this output be $x_1 \in \mathbb{R}^{32 \times H \times W}$.
        2.  A second 3×3 convolutional layer reducing the channel dimension from 32 to 3, with padding of 1, followed by Batch Normalization and ReLU. The output is $x_2 \in \mathbb{R}^{3 \times H \times W}$.
    *   If the input $x$ is already three-channel ($x \in \mathbb{R}^{3 \times H \times W}$), it bypasses these initial convolutional layers and is passed through directly.

The transformation for single-channel input can be expressed mathematically:
$$ x_1 = \text{ReLU}\left(\text{BN}\left(\text{Conv}_{3 \times 3, 32}(x)\right)\right) \tag{1} $$
$$ x_{\text{out}} = \text{ReLU}\left(\text{BN}\left(\text{Conv}_{3 \times 3, 3}(x_1)\right)\right) \tag{2} $$
Where $x_{\text{out}}$ is the three-channel output fed to the ResNet backbone.

This learnable adapter allows the network to adaptively map single-channel thermal information to a three-channel space more effectively than static replication, potentially capturing richer features.

**Algorithm 1: Channel Input Adapter Forward Pass (Simplified)**
Input: Image x (1xHxW or 3xHxW)
Output: Three-channel feature map x_out (3xHxW)
```
Procedure:
1:  If num_channels(x) == 1:
2:      x1 <- Conv_3x3_32(x, padding=1)
3:      x1 <- BatchNorm2d(x1)
4:      x1 <- ReLU(x1)
5:      x_adapted <- Conv_3x3_3(x1, padding=1)
6:      x_adapted <- BatchNorm2d(x_adapted)
7:      x_adapted <- ReLU(x_adapted)
8:  Else if num_channels(x) == 3:
9:      x_adapted <- x // Or minimal processing if needed
10: Else:
11:     Error: Unsupported number of input channels
12: Return x_adapted
```

**Figure 8:** Channel Input Adapter Architecture (Conceptual for single-to-three channel conversion path)
*(Conceptual Diagram: 1-Ch Input -> Conv(1->32, 3x3)+BN+ReLU -> Conv(32->3, 3x3)+BN+ReLU -> 3-Ch Output)*

### 3.3.3 Squeeze and Excitation (SE) Blocks

To enhance the representational power of our model, we integrated Squeeze-and-Excitation (SE) blocks within the ResNet architecture. SE blocks are a form of channel attention mechanism that allows the network to adaptively recalibrate channel-wise feature responses.

The SE block operation consists of two main steps: Squeeze and Excitation.

1.  **Squeeze Operation (Global Information Embedding):**
    For a given feature map $U \in \mathbb{R}^{C \times H \times W}$, global average pooling is applied:
    $$ z_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} U_c(i, j), \quad c = 1, 2, \ldots, C \tag{3} $$
    The result is a channel descriptor $z \in \mathbb{R}^{C}$.
    For fully-connected layers with input $x \in \mathbb{R}^{B \times C}$, a 1D adaptive average pooling is used if the input were treated as a sequence of length C for each batch item, or more commonly, the vector $x_b$ for batch item $b$ is already the "squeezed" representation for that item. The paper describes this for both convolutional and FC layers.

2.  **Excitation Operation (Adaptive Recalibration):**
    The channel descriptor $z$ is processed through two fully-connected layers:
    $$ s = \sigma\left(W_2 \cdot \delta\left(W_1 \cdot z\right)\right) \tag{4} $$
    Here, $W_1 \in \mathbb{R}^{\frac{C}{r} \times C}$ reduces dimensionality (reduction ratio $r=16$), $\delta$ is ReLU, $W_2 \in \mathbb{R}^{C \times \frac{C}{r}}$ restores dimensionality, and $\sigma$ is Sigmoid. The output $s \in \mathbb{R}^{C}$ represents channel-wise attention weights.

3.  **Recalibration (Scaling):**
    The original feature maps $U$ are scaled by these weights:
    For convolutional layers: $\tilde{X}_c = s_c \cdot U_c$
    For fully-connected layers: $\tilde{x}_b(c) = s_c \cdot x_b(c)$

SE blocks are integrated into the ResNet architecture by appending them after the final convolutional layer (conv3) of each bottleneck module within layers 1 through 4.

**Algorithm 2: Squeeze-and-Excitation Block Forward Pass**
Input: Feature map F (CxHxW or BxC), reduction ratio r
Output: Recalibrated feature map F_recal
```
1:  C <- Number of channels in F
2:  If F is 4D (convolutional, BxCxHxW):
3:      z <- AdaptiveAvgPool2d((1,1))(F) // Output BxCx1x1
4:      z_flat <- Flatten z to shape (B, C)
5:  Else if F is 2D (fully-connected, BxC):
6:      // Assuming input F is already the "squeezed" representation for FC (BatchSize, Features)
7:      z_flat <- F 
8:      // Or if F is BxCxL, then apply AdaptiveAvgPool1d(1)
9:
10: s1 <- FullyConnected(z_flat, input_size=C, output_size=C // r)
11: s1 <- ReLU(s1)
12: s2 <- FullyConnected(s1, input_size=C // r, output_size=C)
13: attention_weights <- Sigmoid(s2) // Shape (B, C)
14:
15: If F is 4D:
16:     attention_weights_reshaped <- Reshape attention_weights to (B, C, 1, 1)
17:     F_recal <- F * attention_weights_reshaped
18: Else if F is 2D: 
19:     F_recal <- F * attention_weights // Element-wise if z_flat was F
20:
21: Return F_recal
```
*(Note: The paper implies 1D adaptive pooling for FC layers. This is for cases where the FC input might still have a sequence dimension. If the input to SE is already (Batch, Channels) from an FC layer, then this `z_flat` is the input itself.)*

**Figure 9:** Squeeze-and-Excitation (SE) Block Architecture (Conceptual)
*(Conceptual Diagram: Input Features -> Squeeze (Global Avg Pool) -> Excitation (FC-ReLU-FC-Sigmoid) -> Scale (Multiply with Input Features))*

### 3.3.4 Classifier Head

The standard ResNet classifier is replaced with a more elaborate structure. It processes the 2048-dimensional feature vector ($x_{gap}$) from global average pooling:

1.  **First Dropout Layer**: Dropout with $p=0.5$.
    $$ x_1 = \text{Dropout}_{0.5}(x_{gap}) \tag{5} $$
2.  **Dimensionality Reduction**: Fully-connected layer (2048 to 512) + ReLU.
    $$ x_2 = \text{ReLU}(W_1 x_1 + b_1) \tag{6} $$
3.  **SE Block**: Applied to the 512-dim vector.
    $$ x_3 = \text{SEBlock}(x_2) \tag{7} $$
4.  **Second Dropout Layer**: Dropout with $p=0.3$.
    $$ x_4 = \text{Dropout}_{0.3}(x_3) \tag{8} $$
5.  **Output Layer**: Fully-connected layer (512 to num_classes).
    $$ y_{logits} = W_2 x_4 + b_2 \tag{9} $$

**Figure 10:** Classifier Head Architecture of TH-SE-ResNet (Conceptual)
*(Conceptual Diagram: GAP_Output (2048D) -> Dropout(0.5) -> FC(512D)+ReLU -> SE_Block -> Dropout(0.3) -> FC(NumClasses) -> Logits)*

### 3.3.5 Unified Equation

The complete TH-SE-ResNet architecture operates as:
$$ f_{\text{TH-SE-ResNet}}(x_{raw}) = g_{\text{FCHead}} \left( \text{GAP} \left( f_{\text{ResNet+SE}} \left( h_{\text{CIA}}(x_{raw}) \right) \right) \right) \tag{10} $$

Where:
*   $x_{raw}$ is the single or three-channel thermal input image.
*   $h_{\text{CIA}}$ is the Channel Input Adapter (Eq. 1-2 for single channel).
*   $f_{\text{ResNet+SE}}$ is the ResNet backbone with SE blocks integrated into its residual modules. The SE block itself operates as: $\text{SE}(F) = F \odot \sigma(W_2(\delta(W_1(\text{GlobalAvgPool}(F)))))$, where $\odot$ is channel-wise multiplication.
*   $\text{GAP}$ is Global Average Pooling.
*   $g_{\text{FCHead}}$ is the modified classifier head (Eq. 5-9).

This unified formulation captures the sequential processing of input adaptation, feature extraction with attention, feature aggregation, and robust classification.

## 3.4 Baseline Model Architectures and Comparison

This section details the diverse neural network architectures used as baselines for comparison against our proposed TH-SE-ResNet.

### 3.4.1 Baseline Architectures

#### 3.4.1.1 AlexNet: Foundational CNN Architecture
AlexNet comprises five convolutional layers and three fully connected layers.
*   **Input Size:** 224×224 pixels.
*   **Key Features:** Large initial kernels (11x11), ReLU, LRN, Dropout.

**Figure 5:** Detailed Schematic of the AlexNet Architecture (Conceptual)
*(Conceptual Diagram: Series of Conv -> Pool -> Conv -> Pool -> Conv -> Conv -> Conv -> Pool -> FC -> FC -> FC layers)*

#### 3.4.1.2 VGG-16: Homogeneous Deep Architecture
VGG-16 uses stacked 3×3 convolutional layers and 2x2 max-pooling.
*   **Input Size:** 224×224 pixels.
*   **Key Features:** Uniform 3x3 convolutions, increasing depth.

**Figure 6:** Detailed Schematic of the VGG-16 Architecture (Conceptual)
*(Conceptual Diagram: Blocks of (Conv3x3 -> Conv3x3 -> Pool) repeated, followed by FC layers)*

#### 3.4.1.3 InceptionV3: Multi-Scale Processing Architecture
InceptionV3 uses "Inception modules" for parallel multi-scale convolutions.
*   **Input Size:** 299×299 pixels.
*   **Key Features:** Inception modules, factorized convolutions, auxiliary classifier.

#### 3.4.1.4 ResNet50: Residual Learning Framework
ResNet50 employs "residual learning" with skip connections.
*   **Input Size:** 224×224 pixels.
*   **Key Features:** Bottleneck residual blocks, Batch Normalization. This is the backbone for TH-SE-ResNet.

#### 3.4.1.5 EfficientNet-B0: Optimized Scaling Architecture
EfficientNet-B0 uses compound scaling and MBConv blocks with SE mechanisms.
*   **Input Size:** 224×224 pixels.
*   **Key Features:** MBConv, SE blocks, Swish activation, compound scaling.

**Table 4: Comprehensive Baseline Model Specifications**

| Model         | Depth (Layers) | Parameters (M) | Input Size | Key Components                                  | Potential Thermal Imaging Advantages                                  |
|---------------|----------------|----------------|------------|-------------------------------------------------|-----------------------------------------------------------------------|
| AlexNet       | 8              | 61.0           | 224×224    | Large kernels (11×11), ReLU, LRN, Dropout      | Effective capture of broad thermal gradients, LRN for intensity variation |
| VGG-16        | 16             | 138.0          | 224×224    | Homogeneous 3×3 conv stacks, ReLU, MaxPool     | Consistent multi-scale feature extraction, deep feature hierarchy     |
| InceptionV3   | 48 (conceptual)| 23.9           | 299×299    | Inception modules, Factorized conv, BN, Aux. Class. | Multi-resolution thermal pattern analysis, efficient deep learning    |
| ResNet50      | 50             | 25.6           | 224×224    | Bottleneck residual blocks, BN, Global Avg Pool | Deep thermal feature hierarchies with gradient preservation, stable training |
| EfficientNet-B0| ~82 (effective)| 5.3            | 224×224    | MBConv with SE blocks, Compound scaling, Swish  | Adaptive attention to thermal channels, high efficiency               |

### 3.4.2 Input Adaptation and Training Protocol

#### 3.4.2.1 Thermal Image Preprocessing and Channel Adaptation
*   **Single-Channel Data (e.g., Charlotte):** For baseline models, the single thermal channel was replicated to three identical channels.
*   **Three-Channel Data (e.g., Tufts):** Used directly with baseline models.
*   All images normalized with mean 0.5 and std 0.5.

#### 3.4.2.2 Transfer Learning and Fine-Tuning Strategy
*   All baseline models and the ResNet-50 backbone of TH-SE-ResNet were initialized with ImageNet pre-trained weights.
*   Classifier layers were replaced for binary gender classification.
*   Initial layers were frozen, and only the new final classification layer was trained initially, followed by potential fine-tuning of more layers.

---
<!-- Page Break -->

# Chapter 4: Experimental Setup and Results

This chapter details the experimental configuration used for training and evaluating the models, followed by a presentation and analysis of the results obtained on the Tufts, Charlotte, and Combined datasets.

## 4.1 Experimental Setup
To rigorously assess the efficacy of our baseline models and the proposed TH-SE-ResNet architecture, we designed a comprehensive experimental framework.
Models were trained using the **Adam algorithm** ($\beta_1 = 0.9, \beta_2 = 0.999$).
The initial **learning rate was 0.00005** (5e-5), with a **5-epoch linear warmup** followed by **cosine annealing**.
**Batch sizes of 32 and 64** were used. The loss function was **Cross-Entropy Loss**.
Training was conducted for **10 epochs** on an **NVIDIA GeForce RTX 4090 GPU**.
PyTorch’s `DataLoader` was used with `num_workers=8` and `pin_memory=True`.

Evaluation metrics: Accuracy, Precision (weighted), Recall (weighted), F1-score (weighted), Confusion Matrix.

**Algorithm 3: Model Training Loop**
Input: Training DataLoader `D_train`, Test DataLoader `D_test`, Model `M`, Optimizer `Opt`,
       Learning Rate Scheduler `Sch` (warmup + cosine annealing),
       Loss Function `Crit` (CrossEntropyLoss), Total Epochs `N_epochs` (10),
       Warmup Epochs `N_warmup` (5), Initial Learning Rate `LR_init` (0.00005)
Output: Best performing Model `M_best` (based on test accuracy)

```python
# Pseudocode representation
# Initialize best_accuracy = 0.0, M_best_state_dict = None
# For epoch from 1 to N_epochs:
#     M.train()
#     running_loss = 0.0
#     # Adjust learning rate (warmup or scheduler.step())
#     # ...
#     For images, labels in D_train:
#         # Move to device
#         Opt.zero_grad()
#         # Forward pass (handle Inception auxiliary output if M is InceptionV3)
#         # ...
#         outputs = M(images) # or outputs, aux_outputs for Inception
#         loss = Crit(outputs, labels) # or loss1 + 0.4 * loss2 for Inception
#         loss.backward()
#         Opt.step()
#         running_loss += loss.item() * images.size(0)
#     epoch_loss = running_loss / len(D_train.dataset)
#
#     # Evaluation phase
#     M.eval()
#     correct = 0, total = 0
#     # with torch.no_grad():
#     For images, labels in D_test:
#         # Move to device
#         outputs = M(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#     accuracy = 100 * correct / total
#     # Print stats
#
#     If accuracy > best_accuracy:
#         best_accuracy = accuracy
#         # M_best_state_dict = M.state_dict()
# # After loop, M.load_state_dict(M_best_state_dict) if saving best
# # Return M
```

## 4.2 Results on Individual Datasets
### 4.2.1 Tufts Dataset

**Table 5: Performance on Tufts Dataset**

| Model           | Batch Size | Accuracy | Precision (weighted) | Recall (weighted) | F1 (weighted) |
|-----------------|------------|----------|----------------------|-------------------|---------------|
| AlexNet         | 32         | 0.86     | 0.86                 | 0.86              | 0.85          |
| AlexNet         | 64         | 0.85     | 0.87                 | 0.85              | 0.85          |
| ResNet50 (Std.) | 32         | 0.85     | 0.85                 | 0.85              | 0.85          |
| ResNet50 (Std.) | 64         | 0.83     | 0.83                 | 0.83              | 0.82          |
| Inception v3    | 32         | 0.85     | 0.85                 | 0.85              | 0.84          |
| Inception v3    | 64         | 0.83     | 0.83                 | 0.83              | 0.82          |
| VGG-16          | 32         | 0.84     | 0.84                 | 0.84              | 0.84          |
| VGG-16          | 64         | 0.84     | 0.85                 | 0.84              | 0.84          |
| EfficientNet B0 | 32         | 0.79     | 0.82                 | 0.79              | 0.77          |
| EfficientNet B0 | 64         | 0.71     | 0.71                 | 0.71              | 0.66          |
| **TH-SE-ResNet**| **32**     | **0.95** | **0.96**             | **0.95**          | **0.95**      |
| **TH-SE-ResNet**| **64**     | **0.97** | **0.97**             | **0.97**          | **0.97**      |

**Performance Analysis (Tufts):**
TH-SE-ResNet significantly outperformed baselines, achieving 97% accuracy (BS 64). AlexNet (86%) and standard ResNet50 (85%) followed. EfficientNet-B0 lagged, especially at BS 64.

![tufts_bs64_curves](https://github.com/user-attachments/assets/da3ff3df-9f40-4353-979b-3c3d2b3337e5)
**Figure 11:** Training Loss and Test Accuracy Curves on Tufts Dataset (Batch Size 64)

![tufts_bs32_curves](https://github.com/user-attachments/assets/72f75044-de5d-4775-8e10-c50be50d50a3)
**Figure 12:** Training Loss and Test Accuracy Curves on Tufts Dataset (Batch Size 32)

Learning curves show TH-SE-ResNet's rapid convergence to higher accuracy.

**Figure 13a:** Confusion Matrix - TH-SE-ResNet, Tufts Dataset, Batch Size 64 (Placeholder - Insert Actual Image)
**Figure 13b:** Confusion Matrix - TH-SE-ResNet, Tufts Dataset, Batch Size 32 (Placeholder - Insert Actual Image)
*(Confusion matrices would show high true positives/negatives for TH-SE-ResNet).*

### 4.2.2 Charlotte Dataset

**Table 6: Performance on Charlotte Dataset**

| Model           | Batch Size | Accuracy | Precision (weighted) | Recall (weighted) | F1 (weighted) |
|-----------------|------------|----------|----------------------|-------------------|---------------|
| AlexNet         | 32         | 0.70     | 0.74                 | 0.70              | 0.68          |
| AlexNet         | 64         | 0.68     | 0.71                 | 0.68              | 0.67          |
| ResNet50 (Std.) | 32         | 0.60     | 0.61                 | 0.60              | 0.60          |
| ResNet50 (Std.) | 64         | 0.56     | 0.56                 | 0.56              | 0.56          |
| Inception v3    | 32         | 0.67     | 0.69                 | 0.67              | 0.66          |
| Inception v3    | 64         | 0.67     | 0.70                 | 0.67              | 0.66          |
| VGG-16          | 32         | 0.63     | 0.63                 | 0.63              | 0.63          |
| VGG-16          | 64         | 0.66     | 0.67                 | 0.66              | 0.65          |
| EfficientNet B0 | 32         | 0.68     | 0.68                 | 0.68              | 0.67          |
| EfficientNet B0 | 64         | 0.63     | 0.64                 | 0.63              | 0.63          |
| **TH-SE-ResNet**| **32**     | **0.81** | **0.86**             | **0.81**          | **0.80**      |
| **TH-SE-ResNet**| **64**     | **0.85** | **0.85**             | **0.85**          | **0.84**      |

**Performance Analysis (Charlotte):**
Overall accuracies are lower. TH-SE-ResNet maintained superiority (85%, BS 64). AlexNet (70%) and EfficientNet-B0 (68%) were next. Standard ResNet50 performed poorly (56%).

![comparison_charlotte_batch64_curves](https://github.com/user-attachments/assets/9d1a70c4-06a7-4e8c-a562-c95a28d6b50d)
**Figure 14:** Training Loss and Test Accuracy Curves on Charlotte Dataset (Batch Size 64)

![comparison_charlotte_batch32_curves](https://github.com/user-attachments/assets/1006ff13-0494-494a-9ecd-62572652760f)
**Figure 15:** Training Loss and Test Accuracy Curves on Charlotte Dataset (Batch Size 32)

Learning curves show lower performance ceilings. TH-SE-ResNet still converges fastest.

**Figure 16a:** Confusion Matrix - TH-SE-ResNet, Charlotte Dataset, Batch Size 64 (Placeholder - Insert Actual Image)
**Figure 16b:** Confusion Matrix - TH-SE-ResNet, Charlotte Dataset, Batch Size 32 (Placeholder - Insert Actual Image)
*(Confusion matrices would show more errors than on Tufts).*

## 4.3 Results on Combined Dataset

**Table 7: Performance on Combined Dataset**

| Model           | Batch Size | Accuracy | Precision (weighted) | Recall (weighted) | F1 (weighted) |
|-----------------|------------|----------|----------------------|-------------------|---------------|
| AlexNet         | 32         | 0.76     | 0.80                 | 0.76              | 0.75          |
| AlexNet         | 64         | 0.75     | 0.79                 | 0.75              | 0.74          |
| ResNet50 (Std.) | 32         | 0.64     | 0.64                 | 0.64              | 0.64          |
| ResNet50 (Std.) | 64         | 0.63     | 0.63                 | 0.63              | 0.63          |
| Inception v3    | 32         | 0.71     | 0.73                 | 0.71              | 0.70          |
| Inception v3    | 64         | 0.72     | 0.74                 | 0.72              | 0.72          |
| VGG-16          | 32         | 0.68     | 0.68                 | 0.68              | 0.68          |
| VGG-16          | 64         | 0.70     | 0.70                 | 0.70              | 0.70          |
| EfficientNet B0 | 32         | 0.74     | 0.74                 | 0.74              | 0.74          |
| EfficientNet B0 | 64         | 0.71     | 0.71                 | 0.71              | 0.71          |
| **TH-SE-ResNet**| **32**     | **0.87** | **0.89**             | **0.87**          | **0.87**      |
| **TH-SE-ResNet**| **64**     | **0.90** | **0.91**             | **0.90**          | **0.90**      |

**Performance Analysis (Combined):**
TH-SE-ResNet achieved 90% accuracy (BS 64), demonstrating good generalization. AlexNet (76%) and EfficientNet-B0 (74%) followed. Standard ResNet50 struggled.

![combined_bs64_curves](https://github.com/user-attachments/assets/0bebbace-6a57-49a7-bf72-f584041444f7)
**Figure 17:** Training Loss and Test Accuracy Curves on Combined Dataset (Batch Size 64)

![combined_bs32_curves](https://github.com/user-attachments/assets/dd3fcbf0-bb17-4827-afe5-be9338b8fb1c)
**Figure 18:** Training Loss and Test Accuracy Curves on Combined Dataset (Batch Size 32)

TH-SE-ResNet maintained rapid convergence and superior accuracy on combined data.

**Figure 19a:** Confusion Matrix - TH-SE-ResNet, Combined Dataset, Batch Size 64 (Placeholder - Insert Actual Image)
**Figure 19b:** Confusion Matrix - TH-SE-ResNet, Combined Dataset, Batch Size 32 (Placeholder - Insert Actual Image)

![sample_classification_analysis](https://github.com/user-attachments/assets/ce743113-a418-43cd-a84b-ffe24a85a7d2)
**Figure 20:** Sample Classification Analysis from Combined Dataset Test Set (TH-SE-ResNet, B64)
*(Source: Original Paper, renumbered from Figure 16)*

## 4.4 Comparative Analysis and Key Observations

*   **TH-SE-ResNet consistently outperformed all baselines** across all datasets and batch sizes due to its architectural enhancements (Channel Adapter, SE Blocks, custom classifier).
*   **Dataset-Specific Performance:** Highest on Tufts, lowest on Charlotte (challenging due to variability and few subjects), strong generalization on Combined.
*   **Baseline Performance:** AlexNet was robust for an older model. Standard ResNet50 needed specific adaptations for thermal data. EfficientNet-B0 was inconsistent.
*   **Batch Size Impact:** TH-SE-ResNet favored BS 64. Some models like EfficientNet-B0 were sensitive.
*   **Convergence:** TH-SE-ResNet converged faster and to higher accuracy.
*   **Gender Bias:** High weighted F1-scores for TH-SE-ResNet suggest balanced classification.

---
<!-- Page Break -->

# Chapter 5: Discussion and Future Work

## 5.1 Interpretation of Results
The proposed **TH-SE-ResNet** consistently outperformed baselines, with accuracies up to 97% (Tufts), 85% (Charlotte), and 90% (Combined). This success is attributed to its tailored design:
*   **Channel Input Adapter:** Effective for heterogeneous data.
*   **Squeeze-and-Excitation (SE) Blocks:** Enhanced feature discrimination by focusing on relevant thermal patterns.
*   **Customized Classifier Head:** Improved regularization and decision-making.

Performance variations across datasets highlight their unique characteristics. Tufts (controlled) yielded higher accuracies. Charlotte (varied conditions, few subjects) was more challenging. Strong performance on the Combined dataset indicates good generalization, crucial for real-world use.

Standard ResNet50's struggles on Charlotte/Combined emphasize the need for domain-specific adaptations for thermal imagery. AlexNet's relative robustness might be due to its large initial filters. EfficientNet-B0's inconsistency suggests its design may not be universally optimal for specialized thermal tasks.

## 5.2 Comparison with Existing Literature
Our TH-SE-ResNet results are competitive or superior to recent studies:
*   Exceeds Chatterjee & Zaman (2023)'s 95.0% on Tufts (ResNet-50 + Kalman) with architectural improvements.
*   Provides robust benchmarks against Jalil et al. (2023)'s Cloud_Res, with a focus on balanced gender performance, an issue in Jalil et al. (2022)'s earlier IRT_ResNet.
*   Outperforms AlexNet (82.6%) and InceptionV3 (92.3%) benchmarks by Gurram et al. (2024) on Tufts.
*   Compares favorably with Prihodova et al. (2022)'s 94.9% on thermal *hand* images.
*   Provides a strong static dataset baseline relevant to UAV thermal studies (e.g., Prihodova & Jech, 2024).

Key differentiators: novel TH-SE-ResNet architecture, rigorous evaluation on combined datasets, and focus on balanced gender classification. The Channel Input Adapter is a practical contribution.

**Table 8: Comparative Summary of TH-SE-ResNet Performance against Literature**

| S.N. | Name of Architecture           | Dataset Used & Conditions (as described/implied)                     | Accuracy / Key Result (with Comparative Note)                                                                                                                               |
| :--- | :---------------------------------------------- | :------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.   | Modified CNN (Cloud_Res) (Jalil et al., 2023)   | Two thermal DBs (Tufts + other); 3366 images; Cloud environment      | 99% (Overall Accuracy; *achieved on specific dataset setup; TH-SE-ResNet tested across multiple datasets with transparent methodology*)                                                                     |
| 2.   | ResNet-50 (Best Model) (Chatterjee & Zaman, 2023)| Tufts and Charlotte; Used Kalman filtering preprocessing             | 95.0% Accuracy (*TH-SE-ResNet exceeded this [up to 97% on Tufts] on same datasets via architectural enhancements over standard ResNet-50*)                                                                     |
| 3.   | Custom CNN + SVM (Fusion) (Nguyen et al., 2017) | Visible-Light **and** Thermal **Human Body** Images (DBGender-DB1)     | 11.439% EER (best result via feature-level fusion+PCA); *Lower EER is better (~88.6% accuracy); Requires **multimodal input (Vis+Th)** & uses **body images**, not solely facial thermal.* |
| 4.   | VGG-19 (Best Model) (Prihodova et al., 2022)           | **Fused Multispectral (MSX) Hand Images (Palm)**; 46 subjects        | **94.9% Accuracy** (*High accuracy, but uses **fused Visible+Thermal Hand Images (MSX)**, a different modality & input type than TH-SE-ResNet's thermal-only facial focus*)              |
| 5.   | AlexNet (Gurram et al., 2024)                   | Thermal Images (Tufts with augmentation, 2031 images)                           | **82.6% Accuracy** (*Our AlexNet baseline achieved 85-86% on Tufts; TH-SE-ResNet (97%) significantly outperforms both*)                                         |
| 6.   | InceptionV3 (Gurram et al., 2024)               | Thermal Images (Tufts with augmentation, 2031 images)                           | **92.3% Accuracy** (*Our InceptionV3 baseline achieved ~85%; TH-SE-ResNet (97%) achieves higher results with specific thermal adaptations*)                                           |
| 7.   | IRT_ResNet-101 (Best Model) (Jilal & Reda, 2022) | Two thermal DBs (Tufts + other); 3366 images                          | **99% (Overall Accuracy - Combined datasets)**; *Very high result, though details on dataset combination and gender balance crucial; TH-SE-ResNet (90% on our Combined) shows strong, balanced performance on publicly verifiable datasets.* |
| 8.   | VGG-19 (Best Model) (Prihodova & Jech, 2024)    | Thermal **Facial** Images (Train: FATFD, Test: **Drone** OCD); Outdoor | **85.5% Accuracy** (*Moderate accuracy reflecting challenging outdoor **drone-based capture**; TH-SE-ResNet achieved higher accuracy on static data, providing a strong baseline*)       |
| 9.   | **TH-SE-ResNet (This Report)**                   | **Tufts / Charlotte / Combined** (Thermal Facial); Std preprocessing | **Up to 97% (Tufts), 85% (Charlotte), 90% (Combined)**; *Demonstrates SOTA or highly competitive accuracy & superior generalization on **thermal-only facial data** via tailored architecture (SE Blocks, Channel Adapter) across multiple public datasets.* |

## 5.3 Limitations of the Study
1.  **Dataset Scale and Diversity:** Public datasets used are limited in scale and demographic range. Charlotte has few unique subjects.
2.  **Real-World Conditions:** Datasets are from controlled/semi-controlled environments. Performance in highly uncontrolled scenarios (extreme weather, occlusions beyond augmentation, motion blur) is untested.
3.  **Computational Resources:** TH-SE-ResNet, while built on ResNet50, would need optimization for edge deployment.
4.  **Interpretability:** Deep learning models like TH-SE-ResNet are largely "black boxes."
5.  **Scope of "Gender":** Treats gender as binary (male/female), not addressing the broader spectrum of gender identity.

## 5.4 Future Scope and Potential Enhancements
1.  **Larger and More Diverse Datasets:** Crucial for robust, generalizable models.
2.  **Advanced Augmentation and Synthetic Data:** Tailored thermal augmentation or GANs for synthetic data generation.
3.  **Architectural Innovations:**
    *   **Vision Transformers (ViTs):** Explore for capturing long-range thermal patterns.
    *   **Lightweight Architectures:** Adapt TH-SE-ResNet principles to MobileNet/ShuffleNet for edge devices.
    *   **Multi-Task Learning:** Combine gender classification with age/emotion recognition.
4.  **Multimodal Fusion:** Explore robust fusion with NIR, depth, or anonymized visible data.
5.  **Real-World Deployment and Evaluation:** Test in actual application environments.
6.  **Ethical Considerations and Bias Mitigation:** Address potential demographic biases.
7.  **Interpretability and Explainability:** Use Grad-CAM/LIME to understand decision processes and salient thermal features.
8.  **Cross-Spectral Adaptation:** Research domain adaptation for other IR spectrum parts or visible light.

---
<!-- Page Break -->

# Chapter 6: Conclusion

Gender classification is a fundamental task in computer vision, crucial for applications ranging from driver monitoring and video surveillance to human-computer interaction and retail analytics. While traditional systems rely on visible spectrum facial images, their performance deteriorates under variable lighting, shadows, occlusions, and other real-world challenges. This research addressed these limitations by leveraging thermal facial imaging, which captures heat-based physiological patterns and remains robust in low-light and visually complex environments.

Although thermal imaging offers clear benefits, its use in gender classification poses challenges due to lower resolution and a lack of detailed facial features compared to RGB images. To address this, this internship project explored the use of deep learning, particularly Convolutional Neural Networks (CNNs), for learning meaningful features directly from thermal images.

We conducted an extensive evaluation of state-of-the-art CNN architectures—including AlexNet, VGG-16, InceptionV3, standard ResNet50, and EfficientNet-B0—on two publicly available thermal datasets: Tufts University Thermal Face and Charlotte-ThermalFace. Additionally, a combined dataset was created to enhance generalizability and test cross-domain performance. Since the datasets differ in channel configuration and class distribution, we introduced preprocessing pipelines, sophisticated data augmentation techniques, and class-balancing strategies to ensure fair and effective training.

To further push the performance boundary, we proposed a novel architecture, **TH-SE-ResNet**, a modified version of ResNet-50 designed specifically for thermal image-based gender classification. Key enhancements in TH-SE-ResNet included:
*   A **Channel Input Adapter** to harmonize inputs from datasets with varying channel formats (single or three-channel).
*   **Squeeze-and-Excitation (SE) blocks** integrated within residual layers to improve channel-wise attention, enabling the model to focus on discriminative thermal features.
*   A **redesigned classifier head** with dropout layers, optimized for binary gender classification and improved regularization.
*   Effective use of **transfer learning** by initializing with ImageNet-pretrained weights, accelerating training convergence and improving performance.

Our proposed TH-SE-ResNet model was derived through a rigorous empirical process and achieved state-of-the-art or highly competitive results across both individual and combined datasets. On the Tufts dataset, it achieved up to 97% accuracy; on the more challenging Charlotte dataset, 85% accuracy; and on the heterogeneous Combined dataset, 90% accuracy. In all scenarios, TH-SE-ResNet outperformed the standard CNN baselines in key metrics (accuracy, precision, recall, F1-score) and demonstrated robust performance and generalization. Notably, it also appeared to maintain a better gender balance in predictions compared to potential biases observed in some previous literature.

In comparison, standard ResNet-50 models showed limitations in generalizability across diverse thermal datasets, while other baselines like AlexNet and EfficientNet-B0, though having their own merits, underperformed TH-SE-ResNet due to architectural limitations in optimally capturing salient features in thermal data for this specific task.

Overall, this internship project makes several key contributions:
1.  A thorough benchmarking of contemporary CNN models for thermal facial gender classification.
2.  The development and successful implementation of effective preprocessing, augmentation, and class balancing strategies tailored for thermal image datasets.
3.  The introduction of a novel CNN architecture, TH-SE-ResNet, which demonstrates superior performance and generalization for thermal image-based gender detection due to its specialized components.
4.  Empirical evidence showcasing the model's robustness when tested on combined datasets from different sources.
5.  Valuable insights into the challenges, potential, and current state of deep learning for this application.

The proposed TH-SE-ResNet represents a significant step towards a scalable and deployable solution for real-world applications requiring robust gender classification from thermal imagery, offering enhanced accuracy, fairness, and efficiency. Future work can build upon this foundation to explore even larger datasets, further architectural refinements, and deployment in real-world systems.

---
<!-- Page Break -->

# REFERENCES

*(Full bibliographic details should be used in the final Word document. This is a list based on mentions in the provided text.)*

1.  Chatterjee, S., & Zaman, F. (2023). *[Full title and publication details for Chatterjee & Zaman's 2023 paper on ResNet-50 and VGG-19 with Kalman filtering on Tufts and Charlotte datasets].*
2.  Farooq, M., et al. *[Full title and publication details for Farooq et al.'s benchmarking paper including AlexNet, VGG19, EfficientNet-B4].*
3.  Gurram, P., Goud, S., Saba, T., Rehman, A., & AlNahari, A. (2024). ThermalSQNet-Gender: Thermal Face Based Gender Classification using SqueezeNet Architecture. *IEEE Access*, *12*, 25897-25908.
4.  Jalil, M. A., Hasan, T., Redoy, ROR., Islam, R., & Reda, M.N. (2023). Cloud_Res: Gender Classification of Thermal Images using Modified CNN in Cloud Computing Environment. *2023 International Conference on Product Development and Smart Innovention (ICPDSI)*.
5.  Jalil, M. A., & Reda, M. N. (2022). IRT_ResNet: Gender Classification Based on Infrared Thermal (IRT) Images Using Deep Residual Networks. *2022 IEEE International Conference on Consumer Electronics-Asia (ICCE-Asia)*, 1-5.
6.  Makinen, E., & Raisamo, R. (2008). Evaluation of gender classification methods with publicly available face databases. *University of Tampere, Dept. of Computer Sciences*.
7.  Nguyen, D. T., Pham, T. D., Bae, K. B., & Kim, S. H. (2017). Gender classification based on fusion of visible and thermal_infrared images using CNN and SVM. *Journal of the Korea Institute of Information and Communication Engineering*, *21*(1), 149-156.
8.  Prihodova, S., & Jech, M. (2024). Gender Recognition in Outdoor Environment Based on Thermal Face Images Captured by Drone. *Applied Sciences*, *14*(5), 1865.
9.  Prihodova, S., Jech, M., & Kriz, P. (2022). Human Gender Recognition Based on Thermal Hand Images Using Convolutional Neural Networks. *Sensors*, *22*(19), 7241.
10. Reid, D., Samangooei, S., Chen, C., Nixon, M., & Ross, A. (2011). Soft biometrics: A survey. In *Handbook of remote biometrics* (pp. 213-240). Springer.
11. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR)*, 770-778.
12. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. *Advances in neural information processing systems (NIPS)*, *25*.
13. Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. *arXiv preprint arXiv:1409.1556*. (Published at ICLR 2015).
14. Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the inception architecture for computer vision. *Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR)*, 2818-2826.
15. Tan, M., & Le, Q. (2019). Efficientnet: Rethinking model scaling for convolutional neural networks. *International conference on machine learning (ICML)*, 6105-6114. PMLR.

---
