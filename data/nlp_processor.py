from transformers import pipeline
import torch

class StratumNLP:
    """
    Analizador de Lenguaje Natural para Variables Cualitativas.
    Mapea sentimiento geopolítico a la variable Phi (Φ).
    """
    def __init__(self):
        # Usamos un modelo optimizado para finanzas (FinBERT)
        self.analyzer = pipeline("sentiment-analysis", 
                                 model="ProsusAI/finbert", 
                                 device=0 if torch.cuda.is_available() else -1)

    def extract_phi_adjustment(self, text_list):
        """
        Analiza textos (noticias, posts, minutas) y devuelve un factor de fricción.
        """
        results = self.analyzer(text_list)
        
        # Lógica: Sentimiento Negativo + Mención de 'Disruption' = ↑ Phi
        sentiment_scores = [res['score'] if res['label'] == 'negative' else 0 for res in results]
        phi_bias = sum(sentiment_scores) / len(sentiment_scores)
        
        # El bias aumenta el índice de fricción física
        return 1.0 + phi_bias
