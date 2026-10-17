"""
Torch device selection and the MPS LSTM workaround.
"""

import torch
import torch.nn as nn


def get_device() -> str:
    if torch.backends.mps.is_available():
        print("Device: Apple MPS (GPU) + CPU LSTM")
        return "mps"
    print("Device: CPU")
    return "cpu"


class _CPULSTMWrapper(nn.Module):
    """
    Forces an nn.LSTM to run on CPU while the surrounding model lives on MPS.

    PyTorch 2.1 MPS LSTM kernel crashes with MPSNDArrayDescriptor
    sliceDimension error when sequence_length >= lstm_hidden_size. Moving
    only the LSTM to CPU sidesteps the broken Metal kernel; the MLP layers
    stay on MPS and dominate compute, so throughput loss is small.

    Exposes all nn.LSTM attributes that sb3-contrib reads directly so the
    policy code sees a transparent drop-in replacement.

    Permanent fix: upgrade to PyTorch >= 2.3 where the MPS LSTM kernel was
    rewritten. This wrapper is safe to leave in place after upgrading — it
    just adds negligible CPU<->MPS transfer overhead.
    """
    def __init__(self, lstm: nn.LSTM):
        super().__init__()
        self.lstm = lstm.cpu()
        # Proxy every attribute sb3-contrib / PyTorch may read directly
        self.input_size    = lstm.input_size
        self.hidden_size   = lstm.hidden_size
        self.num_layers    = lstm.num_layers
        self.batch_first   = lstm.batch_first
        self.dropout       = lstm.dropout
        self.bidirectional = lstm.bidirectional

    def forward(self, x, hx=None):
        dev = x.device
        x_cpu = x.cpu()
        hx_cpu = (hx[0].cpu(), hx[1].cpu()) if hx is not None else None
        out, (h_n, c_n) = self.lstm(x_cpu, hx_cpu)
        return out.to(dev), (h_n.to(dev), c_n.to(dev))


def _patch_mps_lstm(policy) -> None:
    """
    Replace lstm_actor and lstm_critic with _CPULSTMWrapper instances.
    No-op if the attributes are already wrapped or do not exist.
    Only called when device == 'mps'.
    """
    for attr in ("lstm_actor", "lstm_critic"):
        lstm = getattr(policy, attr, None)
        if lstm is not None and isinstance(lstm, nn.LSTM):
            setattr(policy, attr, _CPULSTMWrapper(lstm))
            print(f"  [MPS fix] {attr} → CPU")
