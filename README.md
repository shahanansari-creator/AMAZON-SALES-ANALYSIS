# AMAZON-SALES-ANALYSIS

# 🛒 Amazon Sales Dataset — EDA, Visualization & Recommendation System

Exploratory data analysis, data visualization, and a multi-method product recommendation system built on a 1,465-row Amazon India product dataset (ratings, prices, discounts, and customer reviews).

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?logo=pandas&logoColor=white)
![Scikit--learn](https://img.shields.io/badge/scikit--learn-TF--IDF%20%7C%20Cosine%20Sim-F7931E?logo=scikitlearn&logoColor=white)
![Colab](https://img.shields.io/badge/Google%20Colab-Notebook-F9AB00?logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

This project analyzes 1,465 Amazon product listings — spanning 211 sub-categories — to answer three questions:

1. **What does the catalogue actually look like?** (pricing, discounting, ratings, category mix)
2. **How can that be communicated visually?**
3. **Given a product a shopper is looking at, what else should they see?**

The result is a single, self-contained Colab notebook that cleans the raw data, explores and visualizes it, and implements **three complementary recommendation approaches**.

---

## 📊 Dataset

| | |
|---|---|
| **Source** | Amazon Sales Dataset (Amazon.in product listings) |
| **Rows** | 1,465 (1,462 after cleaning) |
| **Unique products** | 1,351 |
| **Categories** | 211 sub-categories / 9 top-level categories |
| **Format** | Single CSV, `amazon_sales_dataset.csv` |

**Columns:**

| Column | Description |
|---|---|
| `product_id`, `product_name` | Unique ID and title of the product |
| `category` | Pipe-separated hierarchy, e.g. `Electronics\|...\|USBCables` |
| `discounted_price`, `actual_price` | Selling price and list price (₹) |
| `discount_percentage` | % off list price |
| `rating`, `rating_count` | Aggregate star rating (1–5) and number of raters |
| `about_product` | Bullet-point product description |
| `user_id`, `user_name`, `review_id` | Comma-separated reviewer identifiers (multiple per product) |
| `review_title`, `review_content` | Short and long review text |
| `img_link`, `product_link` | Product image and Amazon listing URL |

> ⚠️ **Note:** despite the "price" columns, this is a *listings* dataset, not a transactions/sales-volume dataset. There is no unit-sold data. `rating_count` is used throughout as a proxy for popularity.

---

## ✨ Features

### 🧹 Data Cleaning
- Strips `₹`, commas, and `%` from price/discount/count fields and casts them to numeric types
- Fixes a known malformed row (`rating == "|"`)
- Splits the pipe-separated `category` string into `main_category` / `sub_category`

### 🔍 Exploratory Data Analysis
- Rating, price, and discount distributions & summary statistics
- Category composition and average rating by category
- Correlation analysis between price, discount, rating, and rating_count
- Top-rated and most-reviewed product rankings

### 📈 Visualizations
Built with Matplotlib & Seaborn:
- Rating distribution histogram
- Top-10 categories by product count
- Discounted price distribution (log scale)
- Discount % vs. rating scatter plot
- Average rating by category
- Price vs. rating bubble chart (sized by review volume)
- Correlation heatmap
- Word cloud of customer review text

### 🤖 Recommendation System
Three complementary recommenders, each suited to a different scenario:

| Method | How it works | Best for |
|---|---|---|
| **Popularity-based** | IMDB-style weighted (Bayesian) rating that balances star rating against review volume | "Trending / best overall" lists, per category |
| **Content-based** | TF-IDF over product name + category + description, ranked by cosine similarity | "More like this" — works even for products with zero reviews |
| **Item-based Collaborative Filtering (demo)** | Explodes embedded reviewer IDs into a user–item matrix; computes item-item similarity from co-review overlap | Illustrates CF mechanics; a proxy for true CF since the dataset has no per-user star rating |

```python
# Popularity-based
popular_products(category='Electronics', top_n=10)

# Content-based
recommend_similar('headphone', top_n=10)

# Collaborative filtering (co-review overlap)
recommend_cf(product_id='B07JW9H4J1', top_n=10)
```

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Environment:** Google Colab
- **Data handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, WordCloud
- **Machine learning:** Scikit-learn (`TfidfVectorizer`, `cosine_similarity`)
- **Sparse matrices:** SciPy (`csr_matrix`)

---

## 🚀 Getting Started

### Option A — Run in Google Colab (recommended)
1. Open [Google Colab](https://colab.research.google.com).
2. `File → Upload notebook` and select `Amazon_Sales_EDA_Recommendation.ipynb`.
3. Run the first cell to install dependencies.
4. Run the **Load Data** cell — it will prompt you to upload `amazon_sales_dataset.csv`.
5. Run the remaining cells top to bottom.

### Option B — Run locally

```bash
git clone https://github.com/shahanansari-creator/amazon-sales-eda-recommendation.git
cd amazon-sales-eda-recommendation

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook Amazon_Sales_EDA_Recommendation.ipynb
```

**`requirements.txt`**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
wordcloud
```

---

## 📁 Project Structure

```
.
├── Amazon_Sales_EDA_Recommendation.ipynb   # Main notebook: EDA + visualization + recommenders
├── amazon_sales_dataset.csv                # Dataset (add your own copy — not committed)
├── Amazon_Sales_Project_Report.docx        # Written project report
├── requirements.txt
└── README.md
```

---

## 📈 Key Findings

- Average rating across the catalogue is **4.10 / 5**, tightly clustered between 3.8–4.5 — typical of a curated marketplace where poorly-rated items are delisted or under-reviewed.
- **Electronics** is the largest category (526 products), followed by Computers & Accessories.
- Discount and rating are only weakly correlated (**r ≈ -0.16**) — heavy discounting is not a strong signal of lower quality.
- Price and rating are also weakly correlated (**r ≈ 0.12**) — higher price does not reliably mean higher rating.
- The content-based recommender's TF-IDF vectors meaningfully capture product similarity purely from title/category/description text, with no ratings required.

---

## ⚠️ Limitations

- No individual per-user ratings exist, so collaborative filtering here approximates similarity via shared reviewers rather than true rating-based CF.
- No transaction or units-sold data — "sales" analysis is really "listing and rating" analysis.
- Category distribution is imbalanced (Electronics dominates), so category-level averages should be read alongside sample size.

## 🔭 Future Work

- Source true per-user ratings to enable matrix-factorization-based collaborative filtering (e.g. SVD)
- Build a hybrid recommender blending content similarity + popularity score + category filters
- Add sentiment analysis on `review_content` to weight recommendations by review sentiment
- Wrap the recommender in a simple web app (Streamlit/Flask) for interactive use

---

## 👤 Author

**Mohd Shahan Ansari**
- GitHub: [@shahanansari-creator](https://github.com/shahanansari-creator)
- LinkedIn: [mohd-shahan-ansari](https://www.linkedin.com/in/mohd-shahan-ansari-100479259/)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
