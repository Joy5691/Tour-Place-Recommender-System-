# 🗺️ Tour Place Recommender System
### Applying Collaborative Filtering · Content-Based Filtering · Q-Learning (Reinforcement Learning)

![Bangladesh](https://img.shields.io/badge/Country-Bangladesh%20🇧🇩-006A4E?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![ML](https://img.shields.io/badge/ML-Hybrid%20Recommender-F42A41?style=for-the-badge)

---

## 👤 Student Information

| Field        | Details                  |
|--------------|--------------------------|
| **Name**     | Khalid Mahmud Joy        |
| **ID**       | 2022-3-60-159            |
| **Project**  | Tour Place Recommender System |

---

## ⁉️ Overview

This project develops a **Tour Place Recommender System** for Bangladesh
using a hybrid approach that combines three machine learning techniques:

- 👥 **Collaborative Filtering** — recommends destinations based on
  ratings from similar users
- 🧠 **Content-Based Filtering** — recommends destinations based on
  the user's own profile (travel style, preferred type, budget)
- 🪖 **Reinforcement Learning (Q-Learning)** — an agent that learns
  from user feedback and improves recommendations over time

The system is presented through an **interactive HTML dashboard** with
charts, user profiles, and side-by-side comparison of all three methods.

---

## 📈 Dataset

The dataset contains information about Bangladesh tour destinations,
user profiles, and user ratings. It comprises three CSV files:

### 1. `Users.csv`
Contains user profile information.

| Column           | Description                              |
|------------------|------------------------------------------|
| User-ID          | Unique identifier for each user          |
| Division         | User's home division (e.g., Dhaka)       |
| Age              | User's age                               |
| Travel-Style     | Travel preference (Luxury, Nature, etc.) |
| Preferred-Type   | Preferred destination type (Beach, Hill) |
| Budget-Level     | Spending level from 1 (low) to 5 (high)  |

### 2. `Destinations.csv`
Contains information about each Bangladesh tour destination.

| Column       | Description                                    |
|--------------|------------------------------------------------|
| Dest-ID      | Unique ID for each destination (e.g., BD001)   |
| Name         | Name of the destination                        |
| Division     | Division where the destination is located      |
| Type         | Type of destination (Beach, Forest, Hill, etc.)|
| Tags         | Keywords describing the destination            |
| Budget-Level | Cost level of visiting (1–5)                   |
| Best-Season  | Best time to visit (Winter, Summer, All)       |
| Avg-Rating   | Average rating given by all users (out of 10)  |
| Description  | Short description of the destination           |

### 3. `Ratings.csv`
Contains user ratings for destinations.

| Column   | Description                        |
|----------|------------------------------------|
| User-ID  | Which user gave the rating         |
| Dest-ID  | Which destination was rated        |
| Rating   | Score from 1 (lowest) to 10 (highest)|

---

## 🛠️ Implementation

The project involves three recommendation approaches combined
into a **Hybrid System**.

---

### 👥 1. Collaborative Filtering

Collaborative Filtering works on the assumption that users who
agreed in the past will continue to agree in the future.
This system uses **User-Based Collaborative Filtering**.

**How it works:**
1. A User × Destination sparse rating matrix is built using
   `scipy.sparse.coo_matrix`
2. **Cosine similarity** is computed between all users using
   `sklearn.metrics.pairwise.cosine_similarity`
3. The **top 5 most similar users** are identified for the target user
4. Destinations rated **7 or above** by those similar users are
   collected as candidates
5. Already-rated destinations are excluded
6. Each candidate is scored:
   - 50% weight on average rating from similar users
   - 50% weight on similarity score
7. Top 10 destinations are returned sorted by score

**Advantage:** Discovers destinations the user never considered,
based on community taste and shared preferences.

---

### 🧠 2. Content-Based Filtering

Content-Based Filtering recommends destinations that closely
match the user's own profile and stated preferences.

**How it works:**
1. The target user's profile is extracted (Preferred-Type,
   Budget-Level, Travel-Style)
2. Every unrated destination is scored using a weighted formula:
   - Preferred-Type matches destination Type → **+0.4**
   - Budget-Level is within ±1 of destination Budget → **+0.3**
   - Travel-Style keyword found in destination Tags → **+0.3**
3. Final score formula:
