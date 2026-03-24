import torch
from torch import nn



class ACRNN(nn.Module):
    def __init__(
        self, reduce: int, k: int, num_channels: int = 32, num_timepoints: int = 384
    ):
        super().__init__()
        self.C = num_channels
        self.W = num_timepoints
        self.ratio = reduce
        self.k = k
        self.kernel_height = self.C
        self.kernel_width = 40
        self.kernel_stride = 1
        self.pooling_width = 75
        self.pooling_stride = 10
        self.hidden = 64
        self.hidden_attention = 512
        self.num_labels = 2

        # Compute LSTM input size dynamically from W and pooling params
        width_after_conv = self.W - self.kernel_width + 1
        width_after_pool = (
            width_after_conv - self.pooling_width
        ) // self.pooling_stride + 1
        self._lstm_input_size = self.k * width_after_pool

        self._build_channel_wise()
        self._build_cnn()
        self._build_lstm()
        self._build_attention()

        self.softmax = nn.Sequential(
            nn.Linear(self.hidden, self.num_labels),
            nn.Softmax(dim=1),
        )
        self.mean_pool = nn.AdaptiveAvgPool1d(1)

    def _build_channel_wise(self) -> None:
        reduced_channels = int(self.C / self.ratio)
        self.fc = nn.Sequential(
            nn.Linear(self.C, reduced_channels),
            nn.Tanh(),
            nn.Linear(reduced_channels, self.C),
        )
        self.softmax1 = nn.Softmax(dim=-1)

    def _build_cnn(self) -> None:
        self.conv = nn.Sequential(
            nn.Conv2d(
                1, self.k, (self.kernel_height, self.kernel_width), self.kernel_stride
            ),
            nn.BatchNorm2d(self.k),
            nn.ELU(),
            nn.MaxPool2d((1, self.pooling_width), self.pooling_stride),
            nn.Dropout(p=0.5),
        )

    def _build_lstm(self) -> None:
        self.lstm = nn.LSTM(
            input_size=self._lstm_input_size,
            hidden_size=self.hidden,
            num_layers=2,
            batch_first=True,
        )

    def _build_attention(self) -> None:
        self.W1 = nn.Parameter(torch.rand(size=(self.hidden, self.hidden_attention)))
        self.W2 = nn.Parameter(torch.rand(size=(self.hidden, self.hidden_attention)))
        self.b = nn.Parameter(torch.zeros(self.hidden_attention))
        self.activation = nn.Softmax(dim=-1)
        self.vector = nn.Linear(self.hidden, self.hidden)
        self.self_attention = nn.Sequential(
            nn.Linear(self.hidden_attention, self.hidden),
            nn.ELU(),
        )
        self.softmax2 = nn.Softmax(dim=2)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.mean_pool(x)
        x1 = x1.view(x.size(0), -1)

        feature_pre = self.fc(x1)
        v = self.softmax1(feature_pre)

        vr = v.unsqueeze(-1).repeat(1, 1, self.W)
        x = x * vr

        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.reshape(x.size(0), 1, -1)

        h, _ = self.lstm(x)
        y = self.vector(h)
        y = self.activation(
            torch.matmul(h, self.W1) + torch.matmul(y, self.W2) + self.b
        )

        z = self.self_attention(y)
        p = self.softmax2(z * h)

        attention_output = p * h
        attention_output = attention_output.reshape(-1, self.hidden)
        attention_output = self.dropout2(attention_output)

        return self.softmax(attention_output)