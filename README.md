# 🛍️ E-commerce Product Recommender with LLM Explanations

This project is a complete e-commerce product recommendation system that provides personalized suggestions using a collaborative filtering model and generates human-like explanations for why an item is recommended using a local Large Language Model (LLM).

## Features
- **Personalized Recommendations:** Uses the Alternating Least Squares (ALS) algorithm to generate recommendations for known users.
- **Popularity Fallback:** Provides recommendations for new (cold-start) users based on the most popular items.
- **AI-Powered Explanations:** Integrates with a local LLM via LM Studio to explain *why* a product is a good fit for a user's purchase history.
- **RESTful API:** Built with FastAPI to serve recommendations and explanations.
- **Simple Frontend:** A clean, no-framework user interface built with HTML, CSS, and vanilla JavaScript.

## Tech Stack
- **Backend:** Python, FastAPI
- **ML/Recommender:** Pandas, `implicit`, Scipy
- **LLM Integration:** LM Studio (local server)
- **Frontend:** HTML, CSS, JavaScript

## Setup and Installation
Follow these steps to run the project locally.

**1. Clone the Repository**
```bash
git clone [https://github.com/Simarpreet7876/E-commerce-Recommender-System.git](https://github.com/Simarpreet7876/E-commerce-Recommender-System.git)
cd E-commerce-Recommender-System
