import torch
import torch.nn as nn

class StratumLSTM(nn.Module):
    """
    Arquitectura LSTM para predicción de Transición de Fase.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(StratumLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # Solo nos interesa el último estado temporal
        return self.sigmoid(out)

class NeuralInferenceEngine:
    def __init__(self, feature_count):
        self.model = StratumLSTM(input_size=feature_count).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()

    def predict_state(self, sequence):
        self.model.eval()
        with torch.no_grad():
            tensor_seq = torch.FloatTensor(sequence).cuda().unsqueeze(0)
            probability = self.model(tensor_seq)
        return probability.item()
