# 🗺️ Tour Place Recommender System
### Applying Collaborative Filtering · Content-Based Filtering · Q-Learning (Reinforcement Learning)

![Bangladesh](https://img.shields.io/badge/Country-Bangladesh%20🇧🇩-006A4E?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![ML](https://img.shields.io/badge/ML-Hybrid%20Recommender-F42A41?style=for-the-badge)

---

## 👤 Student Information

| Field        | Details                       |
|--------------|-------------------------------|
| **Name**     | Khalid Mahmud Joy             |
| **ID**       | 2022-3-60-159                 |
| **Project**  | Tour Place Recommender System |

---

## ⁉️ Overview

This project develops a **Tour Place Recommender System** for Bangladesh using a hybrid approach that combines three machine learning techniques:

- 👥 **Collaborative Filtering** — recommends destinations based on ratings from similar users
- 🧠 **Content-Based Filtering** — recommends destinations based on the user's own profile (travel style, preferred type, budget)
- 🪖 **Reinforcement Learning (Q-Learning)** — an agent that learns from user feedback and improves recommendations over time

The system is presented through an **interactive HTML dashboard** with charts, user profiles, and side-by-side comparison of all three methods.

---

## 📈 Dataset

The dataset contains information about Bangladesh tour destinations, user profiles, and user ratings. It comprises three CSV files:

### 1. `Users.csv`
| Column         | Description                              |
|----------------|------------------------------------------|
| User-ID        | Unique identifier for each user          |
| Division       | User's home division (e.g., Dhaka)       |
| Age            | User's age                               |
| Travel-Style   | Travel preference (Luxury, Nature, etc.) |
| Preferred-Type | Preferred destination type (Beach, Hill) |
| Budget-Level   | Spending level from 1 (low) to 5 (high)  |

### 2. `Destinations.csv`
| Column       | Description                                     |
|--------------|-------------------------------------------------|
| Dest-ID      | Unique ID for each destination (e.g., BD001)    |
| Name         | Name of the destination                         |
| Division     | Division where the destination is located       |
| Type         | Type of destination (Beach, Forest, Hill, etc.) |
| Tags         | Keywords describing the destination             |
| Budget-Level | Cost level of visiting (1–5)                    |
| Best-Season  | Best time to visit (Winter, Summer, All)        |
| Avg-Rating   | Average rating given by all users (out of 10)   |
| Description  | Short description of the destination            |

### 3. `Ratings.csv`
| Column  | Description                           |
|---------|---------------------------------------|
| User-ID | Which user gave the rating            |
| Dest-ID | Which destination was rated           |
| Rating  | Score from 1 (lowest) to 10 (highest) |

---

## 🛠️ Implementation

### 👥 1. Collaborative Filtering

Collaborative Filtering works on the assumption that users who agreed in the past will continue to agree in the future. This system uses **User-Based Collaborative Filtering**.

**How it works:**
1. A User × Destination sparse rating matrix is built using `scipy.sparse.coo_matrix`
2. **Cosine similarity** is computed between all users using `sklearn.metrics.pairwise.cosine_similarity`
3. The **top 5 most similar users** are identified for the target user
4. Destinations rated **7 or above** by those similar users are collected as candidates
5. Already-rated destinations are excluded
6. Each candidate is scored — 50% weight on average rating from similar users, 50% weight on similarity score
7. Top 10 destinations are returned sorted by score

**Advantage:** Discovers destinations the user never considered, based on community taste and shared preferences.

---

### 🧠 2. Content-Based Filtering

Content-Based Filtering recommends destinations that closely match the user's own profile and stated preferences.

**How it works:**
1. The target user's profile is extracted (Preferred-Type, Budget-Level, Travel-Style)
2. Every unrated destination is scored using a weighted formula:
   - Preferred-Type matches destination Type → **+0.4**
   - Budget-Level is within ±1 of destination Budget → **+0.3**
   - Travel-Style keyword found in destination Tags → **+0.3**
3. Final score formula:
```
   Final Score = (0.5 × content_score) + (0.3 × avg_rating/10) + 0.2
```
4. Top 10 destinations returned sorted by final score

**Advantage:** Works even for new users with no ratings — only requires their profile information.

---

### 🪖 3. Reinforcement Learning (Q-Learning)

Reinforcement Learning treats recommendation as a **Markov Decision Process (MDP)** where the agent learns which destinations make users happy through trial and error.

**Key Components:**

| Component   | Definition                                       |
|-------------|--------------------------------------------------|
| Agent       | The recommender system                           |
| Environment | Users and destination pool                       |
| State       | Current user being served                        |
| Action      | Recommending a specific destination              |
| Reward      | Based on the user's rating of the recommendation |

**Reward Shaping:**

| Rating   | Reward           |
|----------|------------------|
| ≥ 8      | +1.0             |
| 6 – 7    | +0.3             |
| 4 – 5    | −0.2             |
| < 4      | −1.0             |
| Repeated | −0.5 extra penalty |

**Q-Learning Update Formula:**
```
Q(state, action) = Q(state, action) + α × (reward + γ × max Q(next) − Q(state, action))
```

**Training Parameters:**

| Parameter         | Value |
|-------------------|-------|
| Episodes          | 1000  |
| Learning Rate α   | 0.1   |
| Discount Factor γ | 0.9   |
| Epsilon ε         | 0.1   |

**Epsilon-Greedy Policy:**
- 10% of the time → **Explore** (random destination)
- 90% of the time → **Exploit** (best known Q-value)

**How recommendations are made:** After training, the Q-table stores the expected reward for every (user, destination) pair. To recommend, the system looks up the user's row in the Q-table and returns the top 10 destinations with the highest Q-values.

---

### 🔀 4. Hybrid Approach

All three methods are combined to overcome each other's weaknesses.

| Problem                         | Solution                               |
|---------------------------------|----------------------------------------|
| New user with no ratings        | Content-Based uses only profile        |
| Cold start for CF               | CB fills the gap immediately           |
| CB always recommends same type  | CF brings diversity from similar users |
| CF and CB are static            | RL learns and improves over time       |
| Sparse ratings data             | RL adapts to sparse, delayed rewards   |

---

## 📂 Files Included

```
tour-place-recommender/
│
├── Users.csv                    ← User profile dataset
├── Destinations.csv             ← Bangladesh destinations dataset
├── Ratings.csv                  ← User ratings dataset
│
├── export_data.py               ← Python script: trains RL + exports JSON
│
├── Knowledge-Based.ipynb        ← Notebook: Collaborative + Content-Based
├── Reinforcement-Learning.ipynb ← Notebook: Q-Learning agent
│
├── users.json                   ← Exported user data (auto-generated)
├── destinations.json            ← Exported destination data (auto-generated)
├── ratings.json                 ← Exported ratings data (auto-generated)
├── recommendations.json         ← CF + CB results (auto-generated)
├── rl_recommendations.json      ← RL results (auto-generated)
├── rl_training.json             ← RL training stats (auto-generated)
│
└── index.html                   ← Interactive dashboard
```

---

## 🌐 Dashboard Features

### 📊 Section 1 — Dataset Overview
Six stat cards: Total Users, Total Destinations, Total Ratings, Average Rating, RL Training Episodes, RL Final Average Reward.

### 📈 Section 2 — Data Visualizations
- **Bar Chart** — Destinations by Type
- **Doughnut Chart** — Destinations by Division
- **Bar Chart** — Ratings Distribution (1–10)
- **Polar Area Chart** — Destinations by Budget Level
- **Line Chart** — Q-Learning Training Progress over 1000 episodes

### 🤖 Section 3 — Recommender Engine
- Select any User-ID → view full profile + top 5 similar users
- **Tab 1 — Collaborative Filtering (Green):** Top 10 destinations from similar users
- **Tab 2 — Content-Based (Red):** Top 10 destinations matched to user profile
- **Tab 3 — Q-Learning RL (Purple):** Top 10 by Q-value + RL stats + CF vs CB vs RL comparison chart

### 🔍 Section 4 — Destination Explorer
Browse and filter all Bangladesh destinations by Type, Division, Season, and Budget Level.

---

## ⚙️ Requirements

```
Python 3.x
pandas
numpy
scipy
scikit-learn
```

Install all dependencies:
```bash
pip install pandas numpy scipy scikit-learn
```

---

## 🕹️ Usage

### Step 1 — Run the Export Script
```bash
python export_data.py
```

### Step 2 — Start Local Web Server
```bash
python -m http.server 8000
```

### Step 3 — Open Dashboard in Browser
```
http://localhost:8000/index.html
```

### Step 4 — Use the Dashboard
1. Explore dataset stats and charts
2. Go to **Recommender Engine**
3. Select a User-ID and click **Get Recommendations**
4. Browse the 3 tabs — CF, Content-Based, and RL results
5. Use **Destination Explorer** to browse and filter all spots

---

## 🚧 Obstacles & Solutions

### 1. Sparse Data Problem
**Problem:** The User × Destination matrix would require enormous memory.
**Solution:** Used `scipy.sparse.coo_matrix` which only stores non-zero values, reducing memory drastically and cutting runtime from 30+ seconds to milliseconds.

### 2. Cold Start Problem
**Problem:** New users with no ratings cannot be served by Collaborative Filtering.
**Solution:** Content-Based Filtering serves new users using only their profile without needing any ratings.

### 3. Non-Numeric Destination IDs
**Problem:** Dest-IDs like "BD001" cannot be used in matrix operations or Q-table indexing.
**Solution:** All Dest-IDs are encoded to integers (BD001→0, BD002→1, etc.) before building matrices, then decoded back for display.

### 4. RL in HTML Dashboard
**Problem:** Q-Learning requires Python to train — it cannot run inside a browser.
**Solution:** The agent is fully trained in `export_data.py`, results saved to `rl_recommendations.json` and `rl_training.json`, and the dashboard loads these pre-computed results.

### 5. Jupyter Notebook Interaction Limit
**Problem:** The RL model requires live user feedback but Jupyter does not support active GUI interaction.
**Solution:** An interactive `input()` session is implemented in the notebook allowing the user to rate recommendations one by one, updating the Q-table in real time.

---

## 🔮 Future Improvements

- Add a web backend (Flask or FastAPI) to run RL training live with real user feedback
- Include more destination features like photos, GPS coordinates, and visitor reviews
- Implement Deep Q-Network (DQN) for more accurate reinforcement learning
- Add matrix factorization (SVD) as an additional baseline comparison method
- Include seasonal and contextual filters (recommend based on current month or weather)

---

## 📚 References

- Kaggle Book Recommendation Dataset: www.kaggle.com/datasets/arashnic/book-recommendation-dataset
- Scikit-learn Cosine Similarity: scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
- SciPy Sparse Matrix: docs.scipy.org/doc/scipy/reference/sparse.html
- Sutton & Barto, Reinforcement Learning: An Introduction

---

*Tour Place Recommender System — Khalid Mahmud Joy — 2022-3-60-159* 🇧🇩
