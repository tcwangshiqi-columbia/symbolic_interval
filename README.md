# Library of Symbolic Interval Analysis

## Introduction

Symbolic interval analysis is a formal analysis method for certifying the robustness of neural networks. Any safety properties of neural networks can be presented as a bounded input range, a targeted network, and a desired output behavior. Symbolic interval analysis will relax the network to an interval-based version such that it can directly take in arbitrary input interval and return an output interval. Such an analysis is sound, as it will always over-approximate the ground-truth output range of network within the given input range. Also, to make the estimations more accurate, the dependencies of the network inputs can be kept as a symbolic interval such that our interval-based network can propagate it layer by layer and return an output symbolic interval. The output interval/symbolic interval can then be used to verify the safety properties. The details of symbolic interval is first proposed in [ReluVal](https://arxiv.org/pdf/1804.10829.pdf) and further improved in [Neurify](https://arxiv.org/pdf/1809.08098.pdf). You can find simple examples in ReluVal papers.


## Applications of symbolic interval analysis
Symbolic interval analysis can be applied in various applications.

### Formal verification of neural networks
Symbolic interval analysis can be combined with iterative input bisections. We have presented [ReluVal](https://arxiv.org/pdf/1804.10829.pdf) (code available at https://github.com/tcwangshiqi-columbia/ReluVal) which is currently the state-of-the-art verifier for verifying small datasets like ACAS Xu.

On the other hand, symbolic interval analysis is an important part of [Neurify](https://arxiv.org/pdf/1809.08098.pdf) (code available at https://github.com/tcwangshiqi-columbia/Neurify), which is one of the state-of-the-art verifiers for verifying large convolutional networks (over 10,000 ReLUs) on various safety properties. Specifically, it can be used to identify key non-linear ReLUs and a linear solver can be further called to efficiently verify large networks.

### Training verifiable robust networks
To let the network learn to be robust and also to be easily verified, symbolic interval analysis can be incorporated into the training process. We present [MixTrain](https://arxiv.org/pdf/1811.02625.pdf) for efficiently improving the verifiable robustness of trained networks. MixTrain is now one of the state-of-the-art scalable certifiable training methods.

### Enhancing gradient-based attacks
Futhermore, we present [interval attack](https://arxiv.org/pdf/1906.02282.pdf) (code available at https://github.com/tcwangshiqi-columbia/Interval-Attack) by applying symbolic interval analysis to enhance the state-of-the-art gradient-based attacks like PGD or CW attacks. 


## Usage

### Prerequisites

The code is tested with python3 and PyTorch v1.0 with and without CUDA.

```
git clone https://github.com/tcwangshiqi-columbia/symbolic_interval
```

### Examples
One can run the test file by
```
cd symbolic_interval
python test.py
```

### APIs

```
from symbolic_interval.symbolic_network import Interval_network
from symbolic_interval.symbolic_network import sym_interval_analyze
from symbolic_interval.symbolic_network import naive_interval_analyze
```

## Reporting bugs

If you find any issues with the code or have any question about symbolic interval analysis, please contact [Shiqi Wang](https://www.cs.columbia.edu/~tcwangshiqi/) (tcwangshiqi@cs.columbia.edu).

## Citing ReluVal

```
@inproceedings {shiqi2018reluval,
	author = {Shiqi Wang and Kexin Pei and Justin Whitehouse and Junfeng Yang and Suman Jana},
	title = {Formal Security Analysis of Neural Networks using Symbolic Intervals},
	booktitle = {27th {USENIX} Security Symposium ({USENIX} Security 18)},
	year = {2018},
	address = {Baltimore, MD},
	url = {https://www.usenix.org/conference/usenixsecurity18/presentation/wang-shiqi},
	publisher = {{USENIX} Association},
}
@inproceedings{wang2018efficient,
 	title={Efficient formal safety analysis of neural networks},
 	author={Wang, Shiqi and Pei, Kexin and Whitehouse, Justin and Yang, Junfeng and Jana, Suman},
 	booktitle={Advances in Neural Information Processing Systems},
 	pages={6367--6377},
 	year={2018}
}
```


## Contributors

* [Shiqi Wang](https://sites.google.com/view/tcwangshiqi) - tcwangshiqi@cs.columbia.edu
* [Yizheng Chen](https://surrealyz.github.io/) - surrealyz@gmail.com
* [Kexin Pei](https://sites.google.com/site/kexinpeisite/) - kpei@cs.columbia.edu
* [Justin Whitehouse](https://www.college.columbia.edu/node/11475) - jaw2228@columbia.edu
* [Junfeng Yang](http://www.cs.columbia.edu/~junfeng/) - junfeng@cs.columbia.edu
* [Suman Jana](http://www.cs.columbia.edu/~suman/) - suman@cs.columbia.edu


## License
Copyright (C) 2018-2019 by its authors and contributors and their institutional affiliations under the terms of modified BSD license.

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
