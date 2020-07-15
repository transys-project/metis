# Metis: Interpreting Deep Learning-Based Networking Systems

Metis is an integrated explainer to provide post-hoc interpretation for different types of Deep Learning (DL)-based networked systems. We refer the readers to the [technical report](https://arxiv.org/pdf/1910.03835.pdf) for more details.

In the current stage, we provide the interpretation methods and implementations for three DL-based networking systems:

- Pensieve (`interpret-pensieve`) is an adaptive video streaming algorithm based on deep reinforcement learning. The source codes of Pensieve are available at [https://github.com/hongzimao/pensieve](https://github.com/hongzimao/pensieve).
- AuTO (`interpret-auto`) is an on-switch traffic scheduler in datacenters (under refactoring). The source codes of AuTO are available at [https://bitbucket.org/JustinasLingys/auto_sigcomm2018](https://bitbucket.org/JustinasLingys/auto_sigcomm2018).
- RouteNet (`interpret-routenet`) is an SDN traffic optimizer to find routes for all src-dst pairs. The source codes of RouteNet are available at [https://github.com/knowledgedefinednetworking/demo-routenet](https://github.com/knowledgedefinednetworking/demo-routenet).

We further provide four use cases of Metis:

- Metis helps network operators to redesign the DNN structure of Pensieve with a quality of experience (QoE) improvement by 5.1% on average (`case-design`). 
- Metis debugs the DNN in Pensieve and improves the average QoE by up to 4% with only decision trees (`case-debug`). 
- Metis enables a lightweight DL-based flow scheduler (AuTO) and a lightweight Pensieve with shorter decision latency by 27x and lower resource consumption by up to 156x (`case-deploy`).
- Metis helps network operators to adjust the routing paths of a DL-based routing optimizer (RouteNet) when ad-hoc adjustments are needed (`case-adjust`).

The running scripts for interpretation methods and use cases could be found in respective directories. Currently we are still working on documentating and refactoring the repository. Other codes will be released soon. Please stay tuned!

For any questions, please post an issue or send an email to [zilim@ieee.org](mailto:zilim@ieee.org).

## Citation

```
@inproceedings{meng2020interpreting,
  title={Interpreting Deep Learning-Based Networking Systems},
  author={Meng, Zili and Wang, Minhu and Bai, Jiasong and Xu, Mingwei and Mao, Hongzi and Hu, Hongxin},
  booktitle={Proc. ACM SIGCOMM},
  year={2020}
}
```
