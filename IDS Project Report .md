# **Intrusion Detection System (IDS) Final Project**

 

 

**Students:**

Abdulaziz Alessa \- 202303806

Turki Alsalama \- 202303811

Ziyad Abdulqader – 202303781

Mutlaq Allahaydan \- 202303804

Omran Salam \- 202303796

 

 

 

 

 

**Instructor: Shahid Alam**

**Date: 12/10/2025**

**Introduction:**

This project focuses on building an Intrusion Detection System (IDS) using machine learning techniques to detect different types of network attacks. The dataset used (`data.csv`) contains various traffic features and labels representing normal activity and several attack types such as Blackhole, TCP-SYN, PortScan, Diversion, and Overflow.

Multiple machine learning classifiers were trained and evaluated, including Neural Networks, Random Forest, AdaBoost, Gradient Boosting, SVM, and Naive Bayes. Their performance was measured using metrics such as True Positive Rate (TPR), False Positive Rate (FPR), Accuracy, Precision, and F1-score.

In addition, a Voting Ensemble Classifier was implemented as a bonus requirement to combine several models and improve the overall detection rate. The goal of the project is to compare the classifiers and identify the most effective model for intrusion detection.

### **Python Libraries Used**

The Intrusion Detection System (IDS) was implemented using several key Python libraries:

* **NumPy** – Used for numerical operations and handling arrays required for computing confusion matrix values (TP, FP, FN, TN) and other statistical metrics.

* **Pandas** – Used to load, preprocess, and manage the dataset (`data.csv`). It provided tools for handling missing values and separating features from the class labels.

* **Matplotlib** – Used to generate visualizations such as ROC curves that help assess classifier performance.

* **Scikit-Learn (sklearn)** – The main machine learning framework used in this project.  
   It provided:

  * **Model Selection Tools:** `train_test_split` for splitting the dataset into training and testing sets.

  * **Evaluation Metrics:** `classification_report`, `confusion_matrix`, `accuracy_score`, `roc_curve`, `auc` for measuring classifier performance.

**TASK 1:**

### **1\. What is a True Positive Rate (TPR)?**

The **True Positive Rate (TPR),** also known as *Recall* or *Sensitivity*, measures how many of the actual positive cases (e.g., attacks) are correctly detected by the classifier.

**Where:**

* **TP (True Positives):** number of attack instances correctly classified as attacks.

* **FN (False Negatives):** number of attack instances incorrectly classified as normal (missed attacks).

**A higher TPR means the IDS is better at detecting attacks.**

### **2\. What is a False Positive Rate (FPR)?**

The **False Positive Rate (FPR)** measures how many normal instances are incorrectly classified as attacks.

**Where:**

* **FP (False Positives):** number of normal instances incorrectly classified as attacks (false alarms).

* **TN (True Negatives):** number of normal instances correctly classified as normal.

**A lower FPR means fewer false alarms in the IDS.**

### **3\. What TPR and FPR are produced by the MLPClassifier?**

For the MLPClassifier (Neural Networks), the True Positive Rate (TPR) and False Positive Rate (FPR) were calculated from the evaluation results as follows:

* With 20% test size, the MLPClassifier achieved an  
   Average TPR of approximately 0.1948 and an  
   Average FPR of approximately 0.1557.

* With 50% test size, the MLPClassifier achieved an  
   Average TPR of approximately 0.1822 and an  
   Average FPR of approximately 0.1578.

These results indicate that the MLPClassifier performs poorly in detecting attack classes, with a low TPR (around 18–19%) and a relatively high FPR (around 15–16%). This makes it unsuitable as a standalone Intrusion Detection System in this dataset

### **4\. Is it possible to improve these values? Give two approaches.**

