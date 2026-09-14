# Optional Baseline Adapters

The core release contains the six positional-encoding variants used for the
shared-backbone comparisons. The historical HeterCSI implementation files are
not redistributed here because their upstream redistribution license was not
verified at release preparation time.

For Informer, LSTM, PAD/Prony, and LLM4CP comparisons, obtain the relevant
upstream implementation under its own license and use an adapter that maps
explicit CSI tensors and masks into the public data interface. The reference
release intentionally has no import-time dependency on another private or local
repository. Any future vendored baseline source must add its upstream commit,
license text, and local modification notice before distribution.
