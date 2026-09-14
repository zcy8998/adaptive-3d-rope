# DeepMIMO downstream provenance

This directory contains the local data adapter, masks, task metrics, and cache
format used for the DeepMIMO v4 downstream experiments. The underlying O1 and
ASU Campus tensors were exported from DeepMIMO/ray-tracing scenarios and remain
outside the code repository. They are simulation/ray-tracing data, not field
measurements.

- DeepMIMO project: https://www.deepmimo.net/
- DeepMIMO repository: https://github.com/DeepMIMO/DeepMIMO
- Tasks in this revision: beam management and CSI feedback/compression

The exported files contain CSI and beam/RSRP labels. They do not contain an
LOS/NLOS label, and the paper must not claim otherwise.
