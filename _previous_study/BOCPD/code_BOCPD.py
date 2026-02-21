import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class BOCPD:
    def __init__(self, hazard=0.1, threshold=0.80):
        self.hazard = hazard  # Probabilité a priori d'un change point
        self.threshold = threshold  # Seuil de probabilité pour déclarer un change point
        self.run_length = 0  # Longueur du segment actuel
        self.R = [0]  # Liste des probabilités de run length
        self.change_points = []  # Liste des indices des change points
        self.data = []  # Liste des données observées
        self.mu = None  # Moyenne du segment actuel
        self.var = None  # Variance du segment actuel

    def update(self, value):
        self.data.append(value)
        self.run_length += 1

        # Mise à jour de la moyenne et de la variance du segment actuel
        if self.mu is None:
            self.mu = value
            self.var = 0
        else:
            old_mu = self.mu
            self.mu = old_mu + (value - old_mu) / self.run_length
            self.var = self.var + (value - old_mu) * (value - self.mu)

        # Calcul des probabilités de run length
        new_R = [0] * (self.run_length + 1)
        for r in range(self.run_length + 1):
            if r == 0:
                # Probabilité a priori d'un change point
                new_R[r] = self.R[0] * self.hazard
            else:
                # Vraisemblance des données depuis le dernier change point supposé
                if r <= len(self.data):
                    segment = self.data[-r:]
                    if len(segment) > 0:
                        mu_segment = np.mean(segment)
                        var_segment = np.var(segment) if len(segment) > 1 else 0
                        if var_segment > 0:
                            likelihood = np.prod([np.exp(-0.5 * ((x - mu_segment) ** 2) / var_segment) for x in segment])
                        else:
                            likelihood = 1
                    else:
                        likelihood = 1
                else:
                    likelihood = 1

                # Mise à jour de la probabilité
                if r < len(self.R):
                    new_R[r] = self.R[r] * (1 - self.hazard) * likelihood
                else:
                    new_R[r] = 0

        # Normalisation des probabilités
        sum_R = np.sum(new_R)
        if sum_R > 0:
            new_R = [r / sum_R for r in new_R]

        self.R = new_R

        # Détection d'un change point
        prob_change = self.R[0]
        if prob_change > self.threshold:
            self.change_points.append(len(self.data) - 1)
            # Réinitialisation pour le nouveau segment
            self.run_length = 0
            self.R = [0]
            self.mu = None
            self.var = None

# Exemple d'utilisation
df = pd.read_csv('donnée.csv',sep=';')  # Remplace par le chemin vers ton fichier
time = df['t'].values  # Remplace 'time' par le nom de ta colonne de temps
signal = df['nwl1'].values

bocpd = BOCPD(hazard=0.001, threshold=0.999)
for value in signal:
    bocpd.update(value)

# Affichage des résultats
plt.figure(figsize=(12, 6))
plt.plot(signal, label='Signal du capteur', color='blue')
for cp in bocpd.change_points:
    plt.axvline(x=cp, color='red', linestyle='--', label=f'Change point à t={cp}')
plt.xlabel('Temps')
plt.ylabel('Valeur du capteur')
plt.title('Détection de change points avec BOCPD')
plt.legend()
plt.grid(True)
plt.show()

print("Change points détectés aux instants :", bocpd.change_points)
