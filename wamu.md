## What is WAMU?

WAMU is a topic for probing the possiblities of multimodal.

1) Is finetuned World Action Models make the video generation model loss the capability of Semgmentation, Depth Estimation, Point tracking and so on.
2) Compare with baseline, video generation model has been proved that they integrated multi-modal ability.

## Related Work
（0）这篇文章应该是相关类型文章的起源：Video models are zero-shot learners and reasoners。首次发现了VGM可以直接做下游任务，但是主要是在google闭源模型Veo3做，实验相对比较toy。
https://arxiv.org/abs/2509.20328

（1） 一篇文章，两个版本，讲述的是同一个故事，他们发现了视频生成模型能够one-shot的完成下游视觉相关的任务。
Video Generation Models are General-Purpose Vision Learners
https://arxiv.org/pdf/2607.09024 \
THFM: A Unified Video Foundation Model for 4D Human Perception and Beyond
https://arxiv.org/pdf/2603.25892

（2）Vision Banana，这篇unified vision foundation model是之前这些文章验证版本，即我们能通过视频生成模型做下游任意的视觉任务。
https://deepmind.google/research/publications/240658/

（3）Gen4U: 深度分析了视频生成模型哪些层，哪些timestep的noise更具备特殊模态的下游能力（比如中间层更接近具备语义模态）。
https://arxiv.org/abs/2607.06856?utm_source=chatgpt.com

（4）这篇文章提出了，World Action Models消除了视频生成模型的先验。**与我们强相关**
https://arxiv.org/abs/2607.08127


## Key Research Points
（1）
与之前的工作不同，我们的目的除了要探究WAMs可能使得VGM丧失多模态的能力，之外，我们还需要探究是否生成其它模态的能力（因为有了action conditioned），同时，之前的实验主要集中在生成，对于时序的感知比较弱，缺乏4D模态的验证。最后，现有的大部分工作能否担当得起3D-WAM，geometry-wam和4D-WAM的能力？

（2）
还有一个问题，推理时不生成未来是否说明WAMs（例如fastwam的去除未来生成），预测未来已经不具备潜在的vision下游能力，导致预测未来作用不大（因为像素空间预测的未来 condition 到action的收益很小）

（3）除此之外，训练的架构是否会对模型产生不同的影响？比如传统的IDM，Joint Denoising and casual self-regression. 



| 模型 | 视频/理解骨干 | 动作侧 | 总参数 | 架构范式 |
|----|------|------|------|-----|
| RepWAM（小）|	1.3B（从零训练）|	350M	| ~1.65B|	Causal DiT, MoT|
| Faster-WAM	| 5B (Wan2.2) |	~50M	| ~5.05B	| DoT（轻量对接）|
| LingBot-VA v1	 | 5B (Wan2.2) |	~350M	| ~5.3B	| MoT双流|
| FastWAM	| 5B (Wan2.2)| 	1B	| ~6B	| MoT|
| MemoryWAM	| 5B (Wan2.2)|	1B|	~6B	| MoT + 持久记忆|
| RepWAM（大）	| 5B（从零训练）	| 350M	| ~5.35B	|Causal DiT, MoT
| Motus	| 5B(VGM)+2.13B(VLM)|	~895M|	~8B	| 三专家MoT|
| DreamZero	| Wan2.2-TI2V-5B	 |—	| ~14B	| 自回归视频扩散
| LingBot-VA 2.0	| 从零训练	|—	| 13B总/1.9B活跃|	稀疏激活
| LingBot-Video(骨干)|	—	| —	|30B总/3B活跃|	稀疏MoE DiT|



### Step 1: Reproduce VGM ability
To see wether video generation model has such scalable multimodal ability.

Semantic Mutual-KNN similarity
![Semantic Ability](artifacts/vatex1024/semantic_shared_y/figure3_mknn_shared_y.png)


DINO-v2-patch Mutual-KNN similarity
![DINO feature similarity](artifacts/vatex1024/dinov2_patch/figure4_dinov2.png)

DINO-v2-cls Mutual-KNN similarity
![DINO feature similarity](artifacts/vatex1024/dinov2_cls/figure4_dinov2.png)

VideoMAEv2 Mutual-KNN similarity
![Video MAEv2](artifacts/vatex1024/videomaev2/figure4_videomaev2.png)


### Pretrained checkpoints 

https://huggingface.co/OpenWAM/pretrain_ego_robot_cotrain_action_sees_video

pretrained with ego_robot for action and video


