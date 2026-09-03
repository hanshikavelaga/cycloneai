import torch
import torch.nn as nn

class CycloneTrackPredictorLSTM(nn.Module):
    """
    Sequence-to-Sequence LSTM Track Predictor.
    Takes a sequence of past observations: [latitude, longitude, wind_speed, pressure]
    and predicts future states [latitude, longitude, wind_speed, pressure] 
    for specified forecast intervals (+6h, +12h, +24h, +48h).
    """
    def __init__(self, input_size=4, hidden_size=64, num_forecast_steps=4):
        super(CycloneTrackPredictorLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_forecast_steps = num_forecast_steps
        self.input_size = input_size
        
        # Recurrent layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        
        # Regression mapping to output coordinates and wind parameters for all steps
        # Output shape: num_forecast_steps * input_size (e.g. 4 steps * 4 features = 16 values)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_forecast_steps * input_size)
        )

    def forward(self, x):
        # x shape: (Batch, Past_Steps, Features)
        
        # Initialize states
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Pull output from the final time-step
        final_out = lstm_out[:, -1, :]  # Shape: (Batch, Hidden_Size)
        
        # Map to regression dimensions
        flat_predictions = self.fc(final_out)  # Shape: (Batch, Num_Forecast_Steps * Input_Size)
        
        # Reshape to (Batch, Num_Forecast_Steps, Input_Size)
        # Reshaped output represents: +6h, +12h, +24h, +48h
        reshaped_predictions = flat_predictions.view(-1, self.num_forecast_steps, self.input_size)
        return reshaped_predictions


if __name__ == "__main__":
    # Test compilation & forward pass with dummy tensor
    # 4 input features: [latitude, longitude, wind_speed, pressure]
    # Sequence length: 4 steps (past 24h at 6h intervals)
    model = CycloneTrackPredictorLSTM(input_size=4, hidden_size=64, num_forecast_steps=4)
    dummy_input = torch.randn(2, 4, 4)  # Batch=2, Seq=4, Features=4
    predictions = model(dummy_input)
    
    print("Track Predictor LSTM Compilation Test:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output predictions shape: {predictions.shape} (Expected: [2, 4, 4])")
    print(f"Predicted values for Batch 1, Step 1 (+6h): {predictions[0, 0].tolist()}")
