# Asimov MuJoCo asset provenance

The XML, STL meshes, and walking reference in this directory come from
[`menloresearch/asimov-mjlab`](https://github.com/menloresearch/asimov-mjlab)
at commit `98870d12f079c0b6313bb0fe459aa591a5e7f251`, under its Apache 2.0
license (copied here as `LICENSE`). The XML has trailing whitespace removed;
the model values, meshes, and reference samples are unchanged.

The reference CSV has 1,000 samples: 25 repetitions of a 40-sample gait cycle.
The environment averages those repetitions to form one 40-sample cycle at
1.25 Hz. The CSV columns are mapped to MuJoCo joints by header name.
