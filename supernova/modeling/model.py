"""Classifier architecture"""

import torch
from torch import nn
from dataclasses import dataclass

from supernova.config import N_BANDS


@dataclass
class SupernovaClassifierV1Config:
    metadata_input_size: int
    metadata_num_hidden_layers: int
    metadata_hidden_size: int
    metadata_output_size: int
    lightcurve_input_size: int
    lightcurve_num_hidden_layers: int
    lightcurve_hidden_size: int
    classifier_hidden_size: int
    classifier_num_hidden_layers: int
    num_classes: int
    dropout: float

    def __post_init__(self):
        assert self.metadata_input_size > 0
        assert self.metadata_num_hidden_layers > 0
        assert self.metadata_hidden_size > 0
        assert self.metadata_output_size > 0

        assert self.lightcurve_input_size > 0
        assert self.lightcurve_num_hidden_layers > 0
        assert self.lightcurve_hidden_size > 0

        assert self.classifier_hidden_size > 0
        assert self.classifier_num_hidden_layers > 0
        assert self.num_classes > 0

        assert 0 <= self.dropout < 1.0


class SupernovaClassifierV1(nn.Module):
    def __init__(self, config: SupernovaClassifierV1Config):
        super().__init__()
        self.config = config

        classifier_input_size = (
            config.metadata_output_size + N_BANDS * config.lightcurve_hidden_size
        )

        self.metadata_mlp = MLP(
            input_size=config.metadata_input_size,
            num_hidden_layers=config.metadata_num_hidden_layers,
            hidden_size=config.metadata_hidden_size,
            output_size=config.metadata_output_size,
            dropout=config.dropout,
        )
        self.lightcurve_lstm_modules = nn.ModuleList(
            [
                nn.LSTM(
                    input_size=config.lightcurve_input_size,
                    hidden_size=config.lightcurve_hidden_size,
                    num_layers=config.lightcurve_num_hidden_layers,
                    batch_first=True,
                    dropout=config.dropout,
                )
                for _ in range(N_BANDS)
            ]
        )
        self.classifier_mlp = MLP(
            input_size=classifier_input_size,
            num_hidden_layers=config.classifier_num_hidden_layers,
            hidden_size=config.classifier_hidden_size,
            output_size=config.num_classes,
            dropout=config.dropout,
        )

    def forward(self, metadata, sequences, lengths):
        """
        Args:
            metadata: tensor (batch_size, metadata_input_size)
            sequences: dict mapping band_id (0-5) to padded sequences (batch_size, max_seq_len, n_lightcurve_features)
            lengths: dict mapping band_id (0-5) to sequence lengths tensor (batch_size)

        Returns:
            logits: tensor of shape (batch_size, num_classes)
        """
        # Process metadata through MLP
        metadata_features = self.metadata_mlp(metadata)

        # Process each band's lightcurve through corresponding LSTM
        lightcurve_features = [
            self._process_lightcurve(band_id, sequences[band_id], lengths[band_id])
            for band_id in range(N_BANDS)
        ]

        # Concatenate all features
        combined_features = torch.cat([metadata_features] + lightcurve_features, dim=1)

        # Pass through classifier
        logits = self.classifier_mlp(combined_features)

        return logits

    def _process_lightcurve(
        self, band_id: int, sequence: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Process a single band's lightcurve through its LSTM module.

        Args:
            band_id: Band index
            sequence: Padded sequences (batch_size, max_seq_len, lightcurve_input_size)
            lengths: Sequence lengths tensor (batch_size)

        Returns:
            Final hidden state (batch_size, lightcurve_hidden_size)
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            sequence, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lightcurve_lstm_modules[band_id](packed)
        return hidden[-1]


class MLP(nn.Module):
    """Multi-Layer Perceptron"""

    def __init__(
        self,
        input_size: int,
        num_hidden_layers: int,
        hidden_size: int,
        output_size: int,
        dropout: float,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            *[
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)
                )
                for _ in range(num_hidden_layers)
            ],
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.network(x)
