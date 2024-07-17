# DragAPart
Official implementation of 'DragAPart: Learning a Part-Level Motion Prior for Articulated Objects' (ECCV 2024)

<p align="center">
  [<a href="https://arxiv.org/abs/2403.15382"><strong>arXiv</strong></a>]
  [<a href="https://huggingface.co/spaces/rayli/DragAPart"><strong>Demo</strong></a>]
  [<a href="https://dragapart.github.io/"><strong>Project</strong></a>]
  [<a href="#citation"><strong>BibTeX</strong></a>]
</p>

![Teaser](https://dragapart.github.io/resources/teaser.png)

### Inference
Please refer to the [huggingface demo](https://huggingface.co/spaces/rayli/DragAPart/tree/main).

### Training
```
accelerate launch --multi_gpu --mixed_precision fp16 --num_processes 8 train.py --config configs/train-DragAPart.yaml --wandb
```

### Data
See the `Drag-a-Move` folder.

### TODO
- [x] Release inference code.
- [x] Release training code.
- [x] Release dataset downloading script and dataloader code.

### Citation

```
@article{li2024dragapart,
  title     = {DragAPart: Learning a Part-Level Motion Prior for Articulated Objects},
  author    = {Li, Ruining and Zheng, Chuanxia and Rupprecht, Christian and Vedaldi, Andrea},
  journal   = {arXiv preprint arXiv:2403.15382},
  year      = {2024}
}
```

### Acknowledgements
We would like to thank [Minghao Chen](https://silent-chen.github.io/), [Junyu Xie](https://scholar.google.com/citations?user=cDMqaTYAAAAJ&hl=en), and [Laurynas Karazija](https://karazijal.github.io/) for insightful discussions.
This work is in part supported by a Toshiba Research Studentship and ERC-CoG UNION 101001212.
