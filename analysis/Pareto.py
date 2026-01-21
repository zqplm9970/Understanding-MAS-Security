import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

data = {
    'Topology': ['Centralized', 'Centralized', 'Centralized', 'Decentralized', 'Decentralized', 'Decentralized',
                 'Layered', 'Layered', 'Layered', 'Decision_Center'],
    'Communication Paradigm': ['Competitive', 'Cooperation', 'Debate', 'Competitive', 'Cooperation', 'Debate',
                               'Competitive', 'Cooperation', 'Debate', 'Non'],
    'Hallucination Test Type': ['common-sense', 'common-sense', 'common-sense', 'common-sense', 'common-sense', 'common-sense',
                                'common-sense', 'common-sense', 'common-sense', 'common-sense'],
    'Accuracy (without decision intervention)': [0.792, 0.792, 0.816, 0.792, 0.808, 0.832, 0.808, 0.808, 0.800, 0.624],
    'Accuracy (Decision Intervention)': [0.832, 0.832, 0.872, 0.840, 0.816, 0.864, 0.864, 0.856, 0.872, 0.856],
    'Runtime (without decision intervention)': [23.55, 26.43, 28.70, 19.20, 19.18, 19.17, 15.36, 22.49, 14.37, 17.21],
    'Runtime (Decision Intervention)': [19.29, 19.41, 19.11, 11.34, 11.54, 19.02, 11.91, 12.09, 10.82, 5.30]
}

df = pd.DataFrame(data)

df['Accuracy Gain'] = (df['Accuracy (Decision Intervention)'] - df['Accuracy (without decision intervention)']) / \
                       df['Accuracy (without decision intervention)']

df['Time Cost'] = (df['Runtime (without decision intervention)'] - df['Runtime (Decision Intervention)']) / \
                  df['Runtime (without decision intervention)']

df['Accuracy Gain Normalized'] = (df['Accuracy Gain'] - df['Accuracy Gain'].min()) / \
                                  (df['Accuracy Gain'].max() - df['Accuracy Gain'].min())

df['Time Cost Normalized'] = (df['Time Cost'] - df['Time Cost'].min()) / \
                             (df['Time Cost'].max() - df['Time Cost'].min())

df['Efficiency Score'] = df['Accuracy Gain Normalized'] - df['Time Cost Normalized']

df['Configuration'] = df['Topology'] + ' - ' + df['Communication Paradigm']

pareto_front = []
for i in range(len(df)):
    is_dominated = False
    for j in range(len(df)):
        if i != j and df.iloc[j]['Accuracy Gain'] >= df.iloc[i]['Accuracy Gain'] and \
           df.iloc[j]['Time Cost'] <= df.iloc[i]['Time Cost']:
            is_dominated = True
            break
    if not is_dominated:
        pareto_front.append(df.iloc[i])


pareto_df = pd.DataFrame(pareto_front)

save_dir = 'analysis/results'
os.makedirs(save_dir, exist_ok=True)


df.to_csv(f'{save_dir}/complete_analysis_data.csv', index=False)


pareto_df.to_csv(f'{save_dir}/pareto_front_data.csv', index=False)


plt.figure(figsize=(12, 8))


scatter = plt.scatter(df['Time Cost'], df['Accuracy Gain'], 
                     c=df['Efficiency Score'], cmap='viridis', 
                     s=100, edgecolors="w", linewidth=0.5, 
                     label="All configurations")


plt.scatter(pareto_df['Time Cost'], pareto_df['Accuracy Gain'], 
           color='red', s=150, label='Pareto Front', 
           edgecolors="w", linewidth=2)

for i, txt in enumerate(df['Configuration']):
    plt.annotate(txt, (df['Time Cost'].iloc[i], df['Accuracy Gain'].iloc[i]), 
                fontsize=9, ha='center', va='bottom',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))


plt.title("Time-Accuracy Pareto Front Analysis", fontsize=16)
plt.xlabel("Time Cost (Reduction Ratio)", fontsize=12)
plt.ylabel("Accuracy Gain (Improvement Ratio)", fontsize=12)


cbar = plt.colorbar(scatter)
cbar.set_label("Efficiency Score", fontsize=12)


plt.grid(True, alpha=0.3)


plt.legend()

plt.tight_layout()

plt.savefig(f'{save_dir}/pareto_front_plot.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{save_dir}/pareto_front_plot.pdf', bbox_inches='tight')

plt.show()

efficiency_ranking = df.sort_values('Efficiency Score', ascending=False)[['Configuration', 'Efficiency Score']]
print(efficiency_ranking)