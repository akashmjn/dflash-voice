"""Dataset loaders: each turns one source into a :class:`~dataprep.types.Segment` stream.

Everything dataset-shaped lives behind this seam -- Emilia rows are already one
utterance, Expresso rows hold several turns that its loader cuts apart -- so the
rest of dataprep never learns which dataset it is processing.

Import the submodules directly; ``dataprep.cli`` picks one per run, keeping the
imports lazy so an Expresso run never touches Emilia's gated-auth path.
"""
