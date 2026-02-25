# Movie Night 🎬 - Movie Recommendation System

An intelligent content-based movie recommendation system built with machine learning and natural language processing. This project combines data science, predictive modeling, and an interactive Streamlit web application to help users discover movies they'll love.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Project Overview

This comprehensive movie recommendation system was developed as part of MAT124 coursework and includes:
- **Data Collection** from TMDb API (20,000+ movies)
- **Exploratory Data Analysis** on movie trends and ratings
- **Machine Learning Models** for IMDb rating prediction
- **Content-Based Recommendation Engine** using semantic embeddings
- **Interactive Web Application** built with Streamlit

## ✨ Key Features

- 🎯 **Smart Recommendations** - Uses SentenceTransformer embeddings and cosine similarity
- 🎨 **Netflix-Style UI** - Modern, responsive design with custom CSS
- 📊 **Multi-Feature Analysis** - Considers genres, descriptions, directors, cast, and ratings
- 🔍 **Detailed Movie Info** - Displays posters, ratings, cast, and synopses
- ⚡ **Fast Performance** - Cached embeddings for instant recommendations
- 📱 **Responsive Design** - Works on desktop and mobile devices

## 🎯 Project Achievements

### IMDb Rating Prediction
Developed and compared 9 machine learning models:

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| **XGBoost** ⭐ | 0.0660 | 0.0505 | **0.491** |
| Gradient Boosting | 0.0666 | 0.0510 | 0.481 |
| Random Forest | 0.0679 | 0.0520 | 0.461 |
| Decision Tree | 0.0709 | 0.0550 | 0.412 |
| Ridge Regression | 0.0740 | 0.0582 | 0.360 |
| Neural Network | 0.0753 | 0.0592 | 0.336 |
| Elastic Net | 0.0756 | 0.0597 | 0.332 |
| Linear Regression | 0.0772 | 0.0610 | 0.303 |
| Lasso Regression | 0.0793 | 0.0631 | 0.265 |

**Best Model: XGBoost** achieved 49.1% variance explanation in IMDb ratings.

### Recommendation System
- Weighted similarity calculation combining:
  - **Genres** (40% weight)
  - **Description** (30% weight)
  - **Director** (20% weight)
  - **Cast** (10% weight)
  - **IMDb Rating** (10% weight)

## 🗂️ Project Structure

```
movie-recommendation-system/
├── app.py                          # Main Streamlit application
├── Milestone1.ipynb               # Data collection from TMDb API
├── Milestone2.ipynb               # Data cleaning and preprocessing
├── Milestone3.ipynb               # Exploratory Data Analysis (EDA)
├── Milestone4.ipynb               # ML models for rating prediction
├── Milestone_5_and_6.ipynb        # Recommendation system development
├── top20k_movies_project2.csv     # Complete movie dataset (20k movies)
├── top11k_movies_cleaned2.csv     # Cleaned dataset subset
├── project_report.docx            # Comprehensive project documentation
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## 🛠️ Technologies Used

### Core Technologies
- **Python 3.12** - Primary programming language
- **Streamlit** - Web application framework
- **Pandas & NumPy** - Data manipulation and analysis

### Machine Learning & NLP
- **Scikit-learn** - ML algorithms and preprocessing
- **Sentence-Transformers** - Semantic embeddings (all-MiniLM-L6-v2)
- **XGBoost** - Gradient boosting for rating prediction
- **PyTorch** - Deep learning backend

### Data & APIs
- **TMDb API** - Movie data collection
- **Matplotlib & Seaborn** - Data visualization

## 📊 Dataset Information

### Data Sources
- **Primary Source:** TMDb (The Movie Database) API
- **Movies Collected:** 20,000+
- **Time Range:** 1935 - 2024
- **Average Release Year:** ~1998

### Features Collected
- Title, Genres, Release Year
- IMDb Rating & Vote Count
- Movie Description/Synopsis
- Director & Cast Information
- Poster URLs

### Data Statistics
- **IMDb Ratings:** 2.1 - 9.95 (Average: 6.77)
- **Vote Counts:** 80 - 36,630 (Median: 284)
- **Genres:** Multi-label encoded (Action, Drama, Comedy, etc.)
- **Release Years:** Concentrated 1985-2020

## 🚀 Installation & Usage

### Prerequisites
```bash
Python 3.11 or higher
pip package manager
```

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
```txt
streamlit>=1.30.0
pandas>=2.0.0
numpy<2.0.0
scikit-learn>=1.3.0
sentence-transformers>=2.0.0
torch>=2.0.0
matplotlib>=3.5.0
seaborn>=0.12.0
xgboost>=2.0.0
```

### Step 3: Run the Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 💻 How to Use

1. **Select a Movie** - Choose from 20,000+ movies in the sidebar dropdown
2. **View Details** - See the movie's poster, rating, genres, cast, and description
3. **Get Recommendations** - Browse 9 similar movie suggestions
4. **Explore** - Click on recommendation cards to see full details

## 📈 Methodology

### 1. Data Collection (Milestone 1)
- Scraped 20,000+ movies from TMDb API
- Collected metadata: titles, genres, ratings, descriptions
- Gathered director and cast information from film credits

### 2. Data Preprocessing (Milestone 2)
- **Missing Values:** 
  - Numerical: Filled with median
  - Categorical: Filled with 'Unknown'
- **Normalization:** MinMaxScaler for ratings, votes, years
- **Encoding:** MultiLabelBinarizer for genres
- **Deduplication:** Removed duplicates using Title_Release_Year

### 3. Exploratory Data Analysis (Milestone 3)
- Analyzed rating distributions and trends
- Examined genre popularity over time
- Identified correlations between features
- Visualized vote count distributions

### 4. Predictive Modeling (Milestone 4)
- Trained 9 different models for IMDb rating prediction
- Performed hyperparameter tuning
- Evaluated using RMSE, MAE, and R² metrics
- Selected XGBoost as the best performer

### 5. Recommendation Engine (Milestones 5 & 6)
- Generated embeddings using SentenceTransformer
- Computed cosine similarity across multiple features
- Implemented weighted similarity scoring
- Built interactive Streamlit interface

## 🎨 UI Features

- **Custom CSS Styling** - Netflix-inspired dark theme
- **Responsive Design** - Mobile and desktop compatible
- **Smooth Animations** - Fade-in effects and hover states
- **Collapsible Cards** - Expandable recommendation details
- **Dynamic Content** - Real-time poster loading
- **Professional Layout** - Clean, modern interface

## 📊 Model Performance Insights

### XGBoost Success Factors
- **Feature Engineering:** Combined multiple movie attributes
- **Tree-Based Learning:** Captured non-linear relationships
- **Regularization:** Prevented overfitting on training data
- **Gradient Boosting:** Iteratively improved predictions

### Recommendation Quality
- **Semantic Understanding:** NLP embeddings capture meaning beyond keywords
- **Multi-Factor Scoring:** Balances genres, plot, and creative team
- **Personalization:** Adapts to user's movie selection
- **Diversity:** Provides varied yet relevant suggestions



## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



## 👤 Author

**Aitugan Shagyr**


## 📧 Contact

For questions or feedback, please reach out through GitHub issues or email.

---


⭐ If you found this project helpful, please consider giving it a star!
