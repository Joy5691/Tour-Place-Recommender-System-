# export_data.py
# Exports all JSON files including Q-Learning recommendations
# Student: Khalid Mahmud Joy | ID: 2022-3-60-159

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
import json
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
users_df        = pd.read_csv('Users.csv')
destinations_df = pd.read_csv('Destinations.csv')
ratings_df      = pd.read_csv('Ratings.csv')

print("✅ Data loaded!")
print(f"   Users: {len(users_df)} | Destinations: {len(destinations_df)} | Ratings: {len(ratings_df)}")

# ─────────────────────────────────────────────
# 2. PREPROCESS
# ─────────────────────────────────────────────
users_df        = users_df.dropna()
destinations_df = destinations_df.dropna()
ratings_df      = ratings_df.dropna()

users_df['Age']                 = users_df['Age'].astype(int)
users_df['Budget-Level']        = users_df['Budget-Level'].astype(int)
destinations_df['Budget-Level'] = destinations_df['Budget-Level'].astype(int)
destinations_df['Avg-Rating']   = destinations_df['Avg-Rating'].astype(float)
ratings_df['Rating']            = ratings_df['Rating'].astype(int)

for col in ['Division','Travel-Style','Preferred-Type']:
    users_df[col] = users_df[col].astype(str)
for col in ['Name','Division','Type','Tags','Best-Season','Description']:
    destinations_df[col] = destinations_df[col].astype(str)

# ─────────────────────────────────────────────
# 3. ENCODE IDs
# ─────────────────────────────────────────────
dest_ids       = destinations_df['Dest-ID'].tolist()
user_ids       = users_df['User-ID'].tolist()

dest_id_to_idx = {d: i for i, d in enumerate(dest_ids)}
user_id_to_idx = {u: i for i, u in enumerate(user_ids)}
idx_to_dest_id = {i: d for d, i in dest_id_to_idx.items()}

ratings_df['user_idx'] = ratings_df['User-ID'].map(user_id_to_idx)
ratings_df['dest_idx'] = ratings_df['Dest-ID'].map(dest_id_to_idx)
ratings_df = ratings_df.dropna(subset=['user_idx','dest_idx'])
ratings_df['user_idx'] = ratings_df['user_idx'].astype(int)
ratings_df['dest_idx'] = ratings_df['dest_idx'].astype(int)

print("✅ Encoding complete!")

# ─────────────────────────────────────────────
# 4. SPARSE MATRIX + COSINE SIMILARITY
# ─────────────────────────────────────────────
num_users = len(user_ids)
num_dests = len(dest_ids)

sparse_matrix = coo_matrix(
    (ratings_df['Rating'].values,
     (ratings_df['user_idx'].values, ratings_df['dest_idx'].values)),
    shape=(num_users, num_dests)
).tocsr()

user_similarity = cosine_similarity(sparse_matrix)
print("✅ Cosine similarity matrix built!")

# ─────────────────────────────────────────────
# 5. CONTENT-BASED SCORING
# ─────────────────────────────────────────────
def content_score(user_row, dest_row):
    score = 0.0
    if str(user_row['Preferred-Type']).lower() == str(dest_row['Type']).lower():
        score += 0.4
    if abs(int(user_row['Budget-Level']) - int(dest_row['Budget-Level'])) <= 1:
        score += 0.3
    if str(user_row['Travel-Style']).lower() in str(dest_row['Tags']).lower():
        score += 0.3
    return round(score, 4)

# ─────────────────────────────────────────────
# 6. Q-LEARNING AGENT
# ─────────────────────────────────────────────
class QLearningAgent:
    def __init__(self, state_space_size, action_space_size,
                 learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.q_table          = np.zeros((state_space_size, action_space_size))
        self.learning_rate    = learning_rate
        self.discount_factor  = discount_factor
        self.epsilon          = epsilon
        self.action_space_size = action_space_size

    def choose_action(self, state, excluded=set()):
        q_vals = self.q_table[state, :].copy()
        for idx in excluded:
            q_vals[idx] = -999
        if np.random.uniform(0, 1) < self.epsilon:
            available = [i for i in range(self.action_space_size) if i not in excluded]
            return np.random.choice(available) if available else np.argmax(q_vals)
        return np.argmax(q_vals)

    def learn(self, state, action, reward, next_state, done):
        current_q  = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state, :])
        target_q   = reward + self.discount_factor * max_next_q * (1 - done)
        self.q_table[state, action] += self.learning_rate * (target_q - current_q)

# ─────────────────────────────────────────────
# 7. TOUR ENVIRONMENT
# ─────────────────────────────────────────────
class TourEnvironment:
    def __init__(self):
        self.current_user_idx  = None
        self.recommended_dests = set()

    def reset(self, user_idx=None):
        self.current_user_idx  = user_idx if user_idx is not None else np.random.randint(num_users)
        self.recommended_dests = set()
        return self.current_user_idx

    def step(self, action):
        dest_id  = idx_to_dest_id[action]
        user_id  = user_ids[self.current_user_idx]

        real_row = ratings_df[
            (ratings_df['User-ID'] == user_id) &
            (ratings_df['Dest-ID'] == dest_id)
        ]

        if not real_row.empty:
            rating = real_row.iloc[0]['Rating']
        else:
            dest_r = destinations_df[destinations_df['Dest-ID'] == dest_id]
            rating = float(dest_r.iloc[0]['Avg-Rating']) if not dest_r.empty else 5.0

        # Reward shaping
        if rating >= 8:   reward = 1.0
        elif rating >= 6: reward = 0.3
        elif rating >= 4: reward = -0.2
        else:             reward = -1.0

        if action in self.recommended_dests:
            reward -= 0.5
        self.recommended_dests.add(action)

        next_state = self.current_user_idx
        done       = len(self.recommended_dests) >= 10
        return next_state, reward, done, rating

# ─────────────────────────────────────────────
# 8. TRAIN Q-LEARNING AGENT
# ─────────────────────────────────────────────
print("\n🚀 Training Q-Learning Agent...")
print("─" * 45)

agent  = QLearningAgent(
    state_space_size  = num_users,
    action_space_size = num_dests,
    learning_rate     = 0.1,
    discount_factor   = 0.9,
    epsilon           = 0.1
)
env = TourEnvironment()

NUM_EPISODES   = 1000
total_rewards  = []
episode_log    = []

for episode in range(NUM_EPISODES):
    state          = env.reset()
    episode_reward = 0
    done           = False

    while not done:
        action                       = agent.choose_action(state, env.recommended_dests)
        next_state, reward, done, _  = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state          = next_state
        episode_reward += reward

    total_rewards.append(episode_reward)

    if (episode + 1) % 100 == 0:
        avg = np.mean(total_rewards[-100:])
        episode_log.append({
            'episode'   : episode + 1,
            'avg_reward': round(float(avg), 4)
        })
        print(f"  Episode {episode+1:4d} | Avg Reward (last 100): {avg:.4f}")

final_avg_reward = round(float(np.mean(total_rewards[-100:])), 4)
print(f"\n✅ Training complete! Final Avg Reward: {final_avg_reward}")

# ─────────────────────────────────────────────
# 9. GENERATE ALL RECOMMENDATIONS
# ─────────────────────────────────────────────
print("\n📦 Generating recommendations for all users...")

all_recommendations    = {}
all_rl_recommendations = {}

for user_id in user_ids:
    user_idx = user_id_to_idx[user_id]
    user_row = users_df[users_df['User-ID'] == user_id].iloc[0]

    already_rated = ratings_df[
        ratings_df['User-ID'] == user_id
    ]['Dest-ID'].tolist()

    already_rated_idx = {
        dest_id_to_idx[d] for d in already_rated if d in dest_id_to_idx
    }

    # ── Collaborative Filtering ──────────────
    sim_scores               = user_similarity[user_idx].copy()
    sim_scores[user_idx]     = -1
    top_similar_idxs         = np.argsort(sim_scores)[::-1][:5]
    top_similar_users        = [user_ids[i] for i in top_similar_idxs]
    top_sim_scores           = [round(float(sim_scores[i]), 4) for i in top_similar_idxs]

    collab_candidates = {}
    for sim_uid, sim_score in zip(top_similar_users, top_sim_scores):
        sim_ratings = ratings_df[
            (ratings_df['User-ID'] == sim_uid) &
            (ratings_df['Rating'] >= 7) &
            (~ratings_df['Dest-ID'].isin(already_rated))
        ]
        for _, row in sim_ratings.iterrows():
            did = row['Dest-ID']
            if did not in collab_candidates:
                collab_candidates[did] = {'rating_sum':0,'count':0,'sim_score':0}
            collab_candidates[did]['rating_sum'] += row['Rating']
            collab_candidates[did]['count']      += 1
            collab_candidates[did]['sim_score']  += sim_score

    collab_results = []
    for did, vals in collab_candidates.items():
        dest_rows = destinations_df[destinations_df['Dest-ID'] == did]
        if dest_rows.empty: continue
        d     = dest_rows.iloc[0]
        avg_r = vals['rating_sum'] / vals['count']
        score = round((0.5*(avg_r/10)) + (0.5*min(vals['sim_score'],1.0)), 4)
        collab_results.append({
            'dest_id'     : did,
            'name'        : str(d['Name']),
            'type'        : str(d['Type']),
            'division'    : str(d['Division']),
            'budget_level': int(d['Budget-Level']),
            'best_season' : str(d['Best-Season']),
            'avg_rating'  : float(d['Avg-Rating']),
            'score'       : score,
            'tags'        : str(d['Tags'])
        })
    collab_results = sorted(collab_results, key=lambda x: x['score'], reverse=True)[:10]

    # ── Content-Based Filtering ──────────────
    content_results = []
    for _, d in destinations_df.iterrows():
        if d['Dest-ID'] in already_rated: continue
        c_score = content_score(user_row, d)
        final   = round((0.5*c_score) + (0.3*float(d['Avg-Rating'])/10) + 0.2, 4)
        content_results.append({
            'dest_id'     : str(d['Dest-ID']),
            'name'        : str(d['Name']),
            'type'        : str(d['Type']),
            'division'    : str(d['Division']),
            'budget_level': int(d['Budget-Level']),
            'best_season' : str(d['Best-Season']),
            'avg_rating'  : float(d['Avg-Rating']),
            'score'       : final,
            'tags'        : str(d['Tags'])
        })
    content_results = sorted(content_results, key=lambda x: x['score'], reverse=True)[:10]

    # ── Similar Users Detail ─────────────────
    similar_users_detail = []
    for sim_uid, sim_sc in zip(top_similar_users, top_sim_scores):
        su = users_df[users_df['User-ID'] == sim_uid]
        if su.empty: continue
        su = su.iloc[0]
        similar_users_detail.append({
            'user_id'       : int(sim_uid),
            'similarity'    : sim_sc,
            'division'      : str(su['Division']),
            'travel_style'  : str(su['Travel-Style']),
            'preferred_type': str(su['Preferred-Type']),
            'budget_level'  : int(su['Budget-Level'])
        })

    all_recommendations[str(user_id)] = {
        'collaborative' : collab_results,
        'content_based' : content_results,
        'similar_users' : similar_users_detail
    }

    # ── Q-Learning Recommendations ───────────
    q_values = agent.q_table[user_idx, :].copy()
    for idx in already_rated_idx:
        q_values[idx] = -999

    top_rl_idxs = np.argsort(q_values)[::-1][:10]
    rl_results  = []

    for action_idx in top_rl_idxs:
        dest_id  = idx_to_dest_id[action_idx]
        dest_row = destinations_df[destinations_df['Dest-ID'] == dest_id]
        if dest_row.empty: continue
        d = dest_row.iloc[0]
        rl_results.append({
            'dest_id'     : dest_id,
            'name'        : str(d['Name']),
            'type'        : str(d['Type']),
            'division'    : str(d['Division']),
            'budget_level': int(d['Budget-Level']),
            'best_season' : str(d['Best-Season']),
            'avg_rating'  : float(d['Avg-Rating']),
            'q_value'     : round(float(q_values[action_idx]), 4),
            'tags'        : str(d['Tags'])
        })

    all_rl_recommendations[str(user_id)] = rl_results

print("✅ All recommendations generated!")

# ─────────────────────────────────────────────
# 10. EXPORT JSON FILES
# ─────────────────────────────────────────────

# users.json
users_out = []
for _, row in users_df.iterrows():
    users_out.append({
        'user_id'       : int(row['User-ID']),
        'division'      : str(row['Division']),
        'age'           : int(row['Age']),
        'travel_style'  : str(row['Travel-Style']),
        'preferred_type': str(row['Preferred-Type']),
        'budget_level'  : int(row['Budget-Level'])
    })
with open('users.json','w') as f:
    json.dump(users_out, f)
print("✅ users.json exported")

# destinations.json
dests_out = []
for _, row in destinations_df.iterrows():
    dests_out.append({
        'dest_id'     : str(row['Dest-ID']),
        'name'        : str(row['Name']),
        'division'    : str(row['Division']),
        'type'        : str(row['Type']),
        'tags'        : str(row['Tags']),
        'budget_level': int(row['Budget-Level']),
        'best_season' : str(row['Best-Season']),
        'avg_rating'  : float(row['Avg-Rating']),
        'description' : str(row['Description'])
    })
with open('destinations.json','w') as f:
    json.dump(dests_out, f)
print("✅ destinations.json exported")

# ratings.json
ratings_out = ratings_df[['User-ID','Dest-ID','Rating']].copy()
ratings_out.columns = ['user_id','dest_id','rating']
ratings_out['user_id'] = ratings_out['user_id'].astype(int)
ratings_out['rating']  = ratings_out['rating'].astype(int)
with open('ratings.json','w') as f:
    json.dump(ratings_out.to_dict(orient='records'), f)
print("✅ ratings.json exported")

# recommendations.json
with open('recommendations.json','w') as f:
    json.dump(all_recommendations, f)
print("✅ recommendations.json exported")

# rl_recommendations.json
with open('rl_recommendations.json','w') as f:
    json.dump(all_rl_recommendations, f)
print("✅ rl_recommendations.json exported")

# rl_training.json (training stats for dashboard chart)
rl_training = {
    'episodes'        : NUM_EPISODES,
    'final_avg_reward': final_avg_reward,
    'epsilon'         : agent.epsilon,
    'learning_rate'   : agent.learning_rate,
    'discount_factor' : agent.discount_factor,
    'episode_log'     : episode_log
}
with open('rl_training.json','w') as f:
    json.dump(rl_training, f)
print("✅ rl_training.json exported")

print("\n🎉 All 6 JSON files ready! Run: python -m http.server 8000")
