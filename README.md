# Fake News Detection

This project aims to detect fake news using a machine learning pipeline with Logistic Regression as the classifier and TF-IDF for text vectorization. The dataset contains news articles labeled as real or fake.

## Dataset

The dataset used in this project is assumed to be in CSV format with the following columns:
- `title`: The title of the news article.
- `text`: The full text of the news article.
- `subject`: The category of the news.
- `date`: The date the news article was published.
- `class`: The label indicating whether the news is real (1) or fake (0).

## Requirements

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

Install the required packages using:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
Project Structure
Fake_News_Detection.py: Main script to train and evaluate the fake news detection model.
News.csv: The dataset file (not included in the repository; must be added separately).
fake_news_detector.pkl: The trained Logistic Regression model (generated after running the script).
tfidf_vectorizer.pkl: The TF-IDF vectorizer (generated after running the script).
Usage
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your_username/fake-news-detection.git
Navigate to the project directory:

bash
Copy
Edit
cd fake-news-detection
Place the News.csv file in the project directory.

Run the script to train and evaluate the model:

bash
Copy
Edit
python Fake_News_Detection.py
The script will output the accuracy, classification report, and display graphs of the data distribution and confusion matrix.

The trained model and vectorizer will be saved as fake_news_detector.pkl and tfidf_vectorizer.pkl.

Results
The project evaluates the performance of the Logistic Regression model using accuracy, precision, recall, and F1-score. Additionally, it visualizes the confusion matrix to provide insights into the model's predictions.

Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any improvements or new features.
