
## Abstract

While many deep learning (DL)-based networking systems have demonstrated superior performance, the underlying Deep Neural Networks (DNNs) remain blackboxes and stay uninterpretable for network operators. The lack of interpretability makes DL-based networking systems prohibitive to deploy in practice. In this paper, we propose Metis, a framework that provides interpretability for two general categories of networking problems spanning local and global control. Accordingly, Metis introduces two different interpretation methods based on decision tree and hypergraph, where it converts DNN policies to interpretable rule-based controllers and highlight critical components based on analysis over hypergraph. We evaluate Metis over two categories of state-of-the-art DL-based networking systems and show that Metis provides human-readable interpretations while preserving nearly no degradation in performance. We further present four concrete use cases of Metis, showcasing how Metis helps network operators to design, debug, deploy, and ad-hoc adjust DL-based networking systems.

## Paper

### Interpreting Deep Learning-Based Networking Systems

Zili Meng, Minhu Wang, Jiasong Bai, Mingwei Xu, Hongzi Mao, Hongxin Hu<br>Proceedings of the 2020 ACM SIGCOMM Conference<br>[[PDF]](https://zilimeng.com/papers/metis-sigcomm20.pdf)

### Citation

```
@inproceedings{meng2020interpreting,
  title={Interpreting Deep Learning-Based Networking Systems},
  author={Meng, Zili and Wang, Minhu and Bai, Jiasong and Xu, Mingwei and Mao, Hongzi and Hu, Hongxin},
  booktitle={Proc. ACM SIGCOMM},
  year={2020}
}
```

## Videos

[SIGCOMM video (20min version, with subtitles)](https://youtu.be/J0QkGT4lrvI)<br>
[SIGCOMM video (10min version, with subtitles)](https://youtu.be/MJjX7oUpSkE)<br>
[Talk at the APNet Workshop 2020](https://youtu.be/5QvyweOzyro?t=23746)<br>
[Topic preview given by Keith Winstein](https://youtu.be/5VtWWG_a1sk?t=1793)

## Slides

[Presentation Slides (.pptx, better played with Windows and Office 2019+)](metis-sigcomm20-slides.pptx)<br>
[Presentation Slides (.pptx, compatible version for Mac users)](metis-sigcomm20-slides-compatible.pptx)

## Code

[GitHub](https://github.com/transys-project/metis/)

## FAQ

- **Q1: Does this mean that we could just directly use decision tree for the same problem, rather than using complex DNNs? Why is it easier to train a decision tree from a DNN than to train a decision tree directly?**<br>A1: Our experiences are that if we want to directly training a decision tree in the settings of, e.g., reinforcement learning, it would be much challenging to get the same level of results because they are non-parametric. We can dynamically and precisely calculate the gradients and update the neurons with mature solutions, but it may not be the case for decision trees. The main benefits is that using imitation learning, or a given finetuned DNN, the training of decision tree would be much like to a supervised learning with labels (correct actions) while direct training of decision tree in the reinforcement learning settings does not have a labelled groundtruth.
- **Q2: Is every DNN susceptible to being converted into a decision tree?**<br>A2: Basically, if DNNs are trained in the settings of supervised learning, then decision trees can also be directly trained, where the conversion might not bring additional benefits. But if the DNNs are trained in the settings of sequential decision-making process (e.g., reinforcement learning as many networking systems do), the conversion might be helpful. Networking problems are probably better suited for decision trees since it's sequential.
- **Q3: Does Metis try avoiding overfitted result?**<br>A3: In the local system cases, the design goal of Metis is trying to mimic the original DNN as precise as possible. Therefore the overfitting in DNN might also be mimicked by Metis (e.g., in the case of "Metis helps to debug"). Operators may use the interpretation results to overcome the overfitting of the original DNN. Metis does not support the automatic detection of overfitting. Operators might be aware of some strange behaviors, such as the "missing bitrates" in Pensieve, to discover the overfitting.
- **Q4: How do you ensure the conversion of decision trees does not suffer performance loss?**<br>A4: Metis in the current stage cannot guarantee the performance loss but trying to *minimize* the performance loss. Experiments show that the performance loss is quite acceptable (maybe because existing systems are quite simple in terms of DNN structural complexity).
- **Q5: Can we use other models such as random forests as the interpretation target?**<br>A5: Indeed, random forests or other models might be used as the interpretation target. Since our major goal is to help network operators to understand the decision-making policy, we decide to use decision tree, which resembles the existing representations of networking policies (e.g., routing policies).
- **Q6: Can we imagine a process of DL->decision tree -> modified decision tree -> New DL model, so that we can get a better model, with human insights?**<br>A6: We do have a case study in the paper where Metis helps to improve the design of Pensieve by adjusting the DNN structure. In short, by discovering the critical decision variables, we can give those variables more weights and simpler connections to help the DNN reinforce the training on that variable. You may want to find more details in the long video or the paper :)


## Supporters

The research is supported by the National Natural Science Foundation of China (No. 61625203 and 61832013), and the National Key R&D Program of China (No. 2017YFB0801701). 

## Contact
For any questions, please send an email to [zilim@ieee.org](mailto:zilim@ieee.org).