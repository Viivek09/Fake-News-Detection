import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the dataset
file_path = 'News.csv'  # Update this path as needed
df = pd.read_csv(file_path)

# Step 2: Preprocess the data
df.dropna(inplace=True)

# Step 3: Visualize the data distribution
sns.countplot(x='class', data=df)
plt.title('Distribution of Real vs Fake News')
plt.xlabel('Class (0: Fake, 1: Real)')
plt.ylabel('Count')
plt.show()

# Step 4: Split the data into training and testing sets
X = df['text']
y = df['class']  # Update target variable name
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Convert text data to numerical data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Step 6: Train the model using Logistic Regression
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Step 7: Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 9: Plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Optional: Save the model and vectorizer
import joblib
joblib.dump(model, 'fake_news_detector.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
