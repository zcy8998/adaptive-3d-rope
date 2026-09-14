# DeepMIMO specialist-baseline provenance

The CsiNet, CsiNet-LSTM, and TransNet classes are self-contained PyTorch
adaptations to the common DeepMIMO tensor/mask interface. Their architecture
names refer to the corresponding published model families; they do not import
another server project. Beam-management MLP, CNN, LSTM, and codebook baselines
share the same observed Set-B beams and labels.

Any paper table will describe these as adapted/reimplemented baselines and will
report the common compression ratio, train/validation split, optimizer, seeds,
and parameter counts rather than implying byte-identical upstream execution.

For numerical stability on the normalized DeepMIMO grids, the recurrent and
Transformer feedback adapters use an observable per-sample maximum-amplitude
normalization and a bounded decoder output.  Beam RSRP auxiliary targets use a
1-dB minimum scale so nearly identical observed beams cannot create unbounded
regression targets.  These rules are applied to every rerun of the affected
model family and are recorded in each run manifest.
