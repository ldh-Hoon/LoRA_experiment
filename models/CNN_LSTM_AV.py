import torch
import torch.nn as nn
import torch.nn.functional as F
import os

COMBINE_SIZE = 512
LSTM_DIM = 64
N_MFCC = 60

class VideoCNN(nn.Module):
    def __init__(self):
        super(VideoCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.fc_video = nn.Linear(64 * 7 * 7, COMBINE_SIZE - 60)
        # frames * 224 * 224 -> 64 * 14 * 14
        # frames * 96 * 96 -> 64 * 6 * 6 

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.reshape(x.size(0), -1)
        x = self.fc_video(x)
        return x

class CNN_LSTM_AV(nn.Module):
    def __init__(self, num_labels=2, num_mfcc=60):
        super(CNN_LSTM_AV, self).__init__()

        self.video_cnn = VideoCNN()

        self.video_lstm = nn.LSTM(input_size=(COMBINE_SIZE), hidden_size=LSTM_DIM, num_layers=2, batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(LSTM_DIM * 2, 128)
        self.fc2 = nn.Linear(128, num_labels)
        self.dropout = nn.Dropout(0.3)

    def forward(self, video_input, audio_input):
        batch_size, frames, channels, height, width = video_input.size()
        x_video = video_input.reshape(batch_size * frames, channels, height, width)
        x_video = self.video_cnn(x_video)
        x_video = x_video.reshape(batch_size, frames, -1)

        audio_input = audio_input.view(batch_size, -1)  # (Batch, Features)

        num_mfcc_features = audio_input.size(1) // N_MFCC
        x_audio = audio_input.reshape(batch_size, num_mfcc_features, N_MFCC)  # (Batch, Frames, N_MFCC)

        x = torch.cat((x_video, x_audio), dim=-1)
        x_video_lstm, _ = self.video_lstm(x)

        x_video_last = x_video_lstm[:, -1, :]  # (batch_size, LSTM_DIM * 2)

        x = self.dropout(x_video_last)
        x = self.fc1(x)

        x = self.dropout(x)
        x = self.fc2(x)

        return x
    
    @classmethod
    def from_pretrained(cls, model_name: str, num_labels: int, pretrained_model_path: str = None):
        """
        사전 학습된 가중치를 로드하여 모델 인스턴스를 생성하는 클래스 메서드입니다.
        """
        print(f"Loading a new instance of {model_name} with {num_labels} labels.")
        
        model = cls(num_labels=num_labels)
        
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            print(f"Loading pretrained weights from {pretrained_model_path}")
            try:
                state_dict = torch.load(pretrained_model_path, map_location='cpu')
                model.load_state_dict(state_dict)
                print("Pretrained weights loaded successfully.")
            except Exception as e:
                print(f"Warning: Could not load pretrained weights. Initializing with random weights. Error: {e}")
        else:
            print("No valid pretrained weights path provided or file not found. Initializing with random weights.")
            
        return model