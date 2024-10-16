# **EMAIL SPAM DETECTION USING MACHINE LARNING**

# ![Spam Detection](https://thumbs.dreamstime.com/b/red-spam-detected-icon-phishing-scam-hacking-concept-cyber-security-concept-alert-message-red-spam-detected-icon-phishing-scam-242696284.jpg)

# **Project Overview**
This project focuses on detecting spam emails using machine learning techniques, particularly **Logistic Regression** for classification. The process involves **Natural Language Processing (NLP)**, feature engineering, and various data preprocessing steps to classify emails as spam or non-spam. We used a public dataset from Kaggle and visualized the model’s performance using a confusion matrix heatmap.

---

# **Dataset**
The dataset used for this project is taken from Kaggle:

- **Source**: [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset/code)  
  The dataset contains a collection of SMS messages labeled as "ham" (non-spam) or "spam."

---

# **Key Steps**

### 1. Data Preprocessing
The data preprocessing phase involves:
- Loading the data using **pandas**.
- Cleaning the text data with regular expressions (`re`).
- Tokenizing and stemming the text using **nltk** (Natural Language Toolkit).

### 2. Natural Language Processing (NLP)
NLP techniques are applied to convert textual data into meaningful numerical representations:
- Tokenization
- Removing stop words
- Stemming
- Vectorizing the text data using **TF-IDF**.

### 3. Feature Engineering
- Features like the presence of specific keywords, the length of the message, and other email-specific metadata are extracted.

### 4. Model Building: **Decision Tree, Random Forest, Multinomial Naïve Bayes**
The classification task is handled using three models to determine the one that performs best on text data.

### 5. Model Evaluation
- **Confusion Matrix**: We visualize the confusion matrix using **Seaborn** to understand the performance of the model.
- **Classification Report**: Precision, Recall, F1-score, and Accuracy are calculated to measure the model’s effectiveness.

### 6. Visualization
- A heatmap of the confusion matrix is plotted to visualize the model’s performance.

---

# **Required Libraries**
The following Python packages are required:
- `pandas`
- `re`
- `nltk`
- `sklearn`
- `seaborn`
- `matplotlib`
- `tqdm`
- `time`

---

# **Results**
The **Multinomial Naïve Bayes** model performs the best based on its high precision, recall, and F1-score. It efficiently identifies both positive and negative cases while minimizing false positives and false negatives, making it the best model for this dataset.

---

# **Future Improvements**

- **Deep Learning**: Exploring neural networks for more complex feature extraction and classification.
- **Unsupervised Learning**: Implementing unsupervised learning techniques like clustering for anomaly detection in spam classification.

---

# **Key Concepts for Spam Detection**

- **Email Filtering**: Categorizing emails as spam or non-spam using machine learning.
- **Natural Language Processing (NLP)**: Extracting meaningful features from email content.
- **Text Classification**: Using supervised learning to label emails based on features like keywords, tone, and grammar.
- **Feature Engineering**: Selecting relevant features from emails to improve classification.
- **Supervised Learning**: Training models with labeled data for text classification tasks.
- **Unsupervised Learning**: Finding hidden patterns in the data without labels.
- **Deep Learning & Neural Networks**: Exploring advanced models for spam detection tasks.

---

# **Running the Code**

To run the project, follow these steps:
1. Clone the repository.
2. Install the required libraries.
3. Execute the Jupyter Notebook or Python script to preprocess the data and train the models.
4. Visualize the results.

---

# **Conclusion**

This project successfully demonstrates how machine learning, combined with NLP techniques, can be used for email spam detection. The model provides high accuracy in classifying emails, and further improvements can be made using deep learning techniques.

---

