import torch
import torch.nn as nn

class CycloneGenesisLSTM(nn.Module):
    """
    Recurrent Neural Network (LSTM) for Tropical Cyclogenesis Prediction.
    Analyzes a sequence of historical meteorological observations
    (e.g., wind speed, central pressure, sea surface temperature, humidity, positional movement)
    to estimate the probability that a tropical disturbance will organize into a cyclone.
    """
    def __init__(self, input_size=7, hidden_size=32, num_layers=1):
        super(CycloneGenesisLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Recurrent layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Dense classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Sigmoid maps prediction to a probability (0.0 to 1.0)
        )

    def forward(self, x):
        # x shape: (Batch, Sequence_Length, Input_Size)
        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass through LSTM
        # out shape: (Batch, Sequence_Length, Hidden_Size)
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the output of the final time-step (many-to-one sequence prediction)
        final_timestep_out = out[:, -1, :]  # Shape: (Batch, Hidden_Size)
        
        # Compute probability
        probability = self.fc(final_timestep_out)  # Shape: (Batch, 1)
        return probability


if __name__ == "__main__":
    # Test compilation & forward pass with dummy tensor
    # 7 input features: [latitude, longitude, wind_speed, pressure, SST, humidity, cloud_coverage]
    # Sequence length: 5 hours/steps
    model = CycloneGenesisLSTM(input_size=7, hidden_size=32)
    dummy_sequence = torch.randn(2, 5, 7)  # Batch=2, Seq=5, Features=7
    prob = model(dummy_sequence)
    
    print("Genesis LSTM Compilation Test:")
    print(f"Input shape: {dummy_sequence.shape}")
    print(f"Probability output shape: {prob.shape} (Expected: [2, 1])")
    print(f"Sample probability output: {prob[0].item() * 100:.2f}%")
