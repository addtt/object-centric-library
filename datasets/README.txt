This folder contains repackaged versions of the following datasets:
- CLEVR (https://cs.stanford.edu/people/jcjohns/clevr/ -- but we 
  use the CLEVR-with-masks version of CLEVR provided at 
  https://github.com/deepmind/multi_object_datasets)
- Multi-dSprites (https://github.com/deepmind/multi_object_datasets)
- Objects Room (https://github.com/deepmind/multi_object_datasets)
- Shapestacks (https://ogroth.github.io/shapestacks/)
- Tetrominoes (https://github.com/deepmind/multi_object_datasets)

Each dataset has a .hdf5 file with name ending with "-full.hdf5", and 
an accompanying metadata file with an .npy extension. In addition,
a corresponding .hdf5 file contains the style transfer version of
each dataset. These need to be downloaded only if some transform
based on style transfer has to be applied.

CLEVR, Multi-dSprites, Objects Room, and Tetrominoes are distributed under the 
Apache 2.0 License. Shapestacks is distributed under the GPL 3.0 License.

The full texts of the licenses are included in LICENSE.

If using these versions of the datasets, please cite the original datasets
as well as our publication https://arxiv.org/abs/2107.00637.
