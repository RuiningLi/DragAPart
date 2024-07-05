# Drag-a-Move Dataset
This directory provides instructions on how to use the `Drag-a-Move' Dataset.

## Render Yourself!
We provide the rendering scripts in `render_tools` folder. To render the dataset, please first follow the instructions of [GAPartNet](https://pku-epic.github.io/GAPartNet/) to download the 3D models and part annotations and then specify the corresponding paths to `render_tools/utils/config_utils.py`. Then, use
```
python render.py
```
to render a single model. This script automatically saves the needed meta information to enable subsequent drag sampling (to sample drags _on the fly_, see [here](https://github.com/RuiningLi/DragAPart/blob/main/dataset.py)).

## Pre-Rendered Version.
We plan to release the pre-rendered version of Drag-a-Move, which we used to train DragAPart, *soon*!

# Citation
If you use the dataset in your work, please cite:
```
@article{li2024dragapart,
  title     = {DragAPart: Learning a Part-Level Motion Prior for Articulated Objects},
  author    = {Li, Ruining and Zheng, Chuanxia and Rupprecht, Christian and Vedaldi, Andrea},
  journal   = {arXiv preprint arXiv:2403.15382},
  year      = {2024}
}
```
