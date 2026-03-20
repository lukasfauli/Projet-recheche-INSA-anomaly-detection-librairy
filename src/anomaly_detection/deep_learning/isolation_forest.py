import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

class AnomalyDetectorIF:
    def __init__(self, contamination=0.05, random_state=42):
        """
        Initialise le modèle Isolation Forest.
        contamination : proportion attendue d'anomalies (ex: 0.05 pour 5%)
        """
        self.model = IsolationForest(contamination=contamination, random_state=random_state)
        self.features = None

    def fit(self, df, features):
        """Entraîne le modèle sur les données saines."""
        self.features = features
        self.model.fit(df[self.features])
        

    def predict_anomalies(self, df):
        """
        Prédit les anomalies. 
        Renvoie -1 pour une anomalie et 1 pour une donnée normale.
        """
        # On récupère les scores d'anomalie 
        df = df.copy()
        df['anomaly_score'] = self.model.decision_function(df[self.features])
        df['is_anomaly'] = self.model.predict(df[self.features])
        
        # Conversion pour plus de clarté : 1 = Anomalie, 0 = Normal
        df['anomaly_detected'] = df['is_anomaly'].apply(lambda x: 1 if x == -1 else 0)
        return df

    def plot_results(self, df_result, sensor_to_plot, time_col='time_real', ax=None):
        """
        Affiche le capteur choisi avec les zones d'anomalies en rouge.
        Supporte l'affichage individuel ou dans une grille (ax).
        """
        show_plot = False
        if ax is None:
            plt.figure(figsize=(15, 5))
            ax = plt.gca()
            show_plot = True

        # Signal original
        ax.plot(df_result[time_col], df_result[sensor_to_plot], 
                color='steelblue', label='Signal', alpha=0.7)
        
        # On surligne les anomalies
        anomalies = df_result[df_result['anomaly_detected'] == 1]
        ax.scatter(anomalies[time_col], anomalies[sensor_to_plot], 
                   color='red', label='Anomalie détectée', s=10, zorder=3)
        
        ax.set_title(f"{sensor_to_plot}", fontsize=10)
        
        # On n'affiche la légende que pour le premier graphique pour ne pas surcharger
        # ou si c'est un graphique isolé
        if show_plot:
            ax.legend()
            plt.show()