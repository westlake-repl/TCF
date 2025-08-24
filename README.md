# TCF

Code for paper: 📖**Exploring the Upper Limits of Text-Based Collaborative Filtering Using Large Language Models: Discoveries and Insights**

This repository is intended for appendix right now. We provide some additional results, including all results for the NDCG metric, and two additional text recommendation paradigms (one using LLM as the recommendation backbone, and the other a ChatGPT-based recommendation model).

## More results on NDCG10

**Table 1: Accuracy (NDCG@10) comparison of IDCF and TCF using DSSM and SASRec. _FR_ represents using frozen LM, while _FT_  represents using fine-tuned LM.**
![ID vs TCF](https://github.com/anonymous-TCF/anonymous-TCF/blob/main/Figures/Table_IDvsLLM_NDCG1.jpg)

**Table 2: Warm item recommendation (NDCG@10). 20 means items < 20 interactions are removed. TCF\textsubscript{175B} uses the pre-extracted features from the 175B LM. Only SASRec backbone is reported.**
![Warm item](https://github.com/anonymous-TCF/anonymous-TCF/blob/main/Figures/Table_PLM_warm_NDCG.jpg)

**Table 3: TCF's results (NDCG@10)  with renowned text encoders in the last 10 years. Text encoders are frozen and the SASRec backbone is used. Notable  advances in NLP benefit RS.**
![TCF_last10years](https://github.com/anonymous-TCF/anonymous-TCF/blob/main/Figures/Table_LMcompare_NDCG.jpg)

**Figure 1: TCF’s performance (y-axis: NDCG@10(%)) with 9 text encoders of increasing size (x-axis). SASRec (upper three subfigures) and DSSM (bottom three subfigures**
![Scaling_up_ndcg](https://github.com/anonymous-TCF/anonymous-TCF/blob/main/Figures/scaling_up_ndcg.jpg)

**Figure 2: TCF with retrained LM vs frozen LM (y-axis: NDCG@10(%)), where only the top two layers are retrained. The 175B LM is not retrained due to its ultra-high computational cost.**
![FtvsFz_ndcg](https://github.com/anonymous-TCF/anonymous-TCF/blob/main/Figures/finetune_vs_freeze_ndcg.jpg)

## Results on Bili8M
**Figure 3: TCF’s performance (y-axis: HR@10(%) in left and NDCG@10(%) in right) of 3 item encoder with increased sizes (x-axis) on Bili8M. SASRec is used as the backbone. LLM is frozen.**
![TCFlargeBili8M](https://github.com/anonymous-TCF/anonymous-TCF/blob/main/Figures/bili8M_NDCG.jpg)

## Other paradigms for LLM-based recommender models


This paper primarily focuses on the TCF paradigm with LLMs as item encoders. However, apart from TCF, there are other paradigms for LLM-based recommendation models. Here, we briefly investigate two popular approaches, namely the GPT4Rec paradigm and ChatGPT4Rec paradigm.

### GPT4Rec
The GPT4Rec[1] paradigm (as illustrated in Figure 4) utilizes LLM as the backbone architecture rather than the item encoder. In this approach, the text of items clicked by users is concatenated and fed into the LLM to obtain user representations. Recommendations are then made by calculating the dot product between the user representation and the candidate item representations, which are also represented using LLM.

We conducted experiments using LLMs with 1.3B and 125M versions. As shown in Table 4, fine-tuning only the top-1 block resulted in significantly worse performance compared to full fine-tuning. Even the 1.3B version LLM performed substantially worse than the 125M version when fully fine-tuned. In fact, we have discovered that freezing the LLM or only fine-tuning the top-1 block makes it extremely challenging to provide effective recommendations using this approach.

Furthermore, the GPT4Rec paradigm necessitates significant computational resources and GPU memory. When dealing with longer user sequences, it is not practical to fully fine-tune a very large LLM. This limitation helps explain why the GPT4Rec paradigm has not yet employed very large LLMs as a backbone. Most of such papers used the LLM with a size smaller than 3B. 

**Figure 4: Architecture of GPT4Rec**
![GPT4Rec_Arc](https://github.com/anonymous-TCF/anonymous-TCF/blob/main/Figures/GPT4Rec_arc-cropped.jpg)

**Table 4: Results of GPT4Rec (HR@10(%)) paradigm. Even in 80G A100, we were not able to fully fine-tune 1.3B GPT4Rec. Note that this paradigm requires too much computation and memory when there are long-range item interactions.**
![GPT4Rec_Results](https://github.com/anonymous-TCF/anonymous-TCF/blob/main/Figures/Table_GPT4Rec_task.jpg)
### ChatGPT4Rec

Beyond the TCF paradigm, building text recommender models by leveraging prompt strategies is also becoming increasingly popular[4,5,6,9]. Recently, due to the tremendous success of ChatGPT, a number of preprint papers have explored the use of prompt engineering with ChatGPT for recommender systems[2,7,8,10]. Readers may be interested in whether prompt-based techniques on ChatGPT, referred to as ChatGPT4Rec(We use gpt-3.5-turbo API in https://platform.openai.com/docs/models/gpt-4), can outperform the classical TCF paradigm under the common recommendation setting. Do we still need the TCF paradigm in the ChatGPT era?

We randomly selected 1024 users from the testing sets of MIND, HM, and Bili, and created two tasks for ChatGPT. In the first task (Task 1 in Table 5), ChatGPT was asked to select the most preferred item from four candidates (one ground truth and three randomly selected items), given the user's historical interactions as a condition. The second task (Task 2 in Table 5 was to ask ChatGPT to rank the top-10 preferred items from 100 candidates (one ground truth and 99 randomly selected items, excluding all historical interactions), also provided with the user's historical interactions as input. 

The results are given in Table 5, which illustrate ChatGPT's poor performance compared to TCF in typical recommendation settings. 
Similar bad results have also been reported in [2,3]. Despite that, we believe with more finely-tuned prompts, ChatGPT may have the potential for certain recommendation scenarios. 
Another major drawback of ChatGPT is that it cannot generate recommendations from an item pool with millions of items due to limited memory. This limitation limits the use of ChatGPT as a re-ranking module in existing recommendation pipelines and prevents its use for recommendations from a huge pool of millions of closed-domain items.

In recent months, there has been a substantial increase in literature on recommender systems based on LLM and ChatGPT. It is challenging to thoroughly investigate and compare all these approaches within a single paper. However, we believe that these new paradigms (fine-tuning, prompt-tuning, instruct-tuning, adapter-tuning, etc.) have the potential to bring fresh insights to the recommender system community and may even surpass existing classical paradigms.

The primary contribution of this paper focuses on the performance of the TCF paradigm, which is defined to employ the LLM as the item encoder.

**Table 5: ChatGPT4Rec vs TCF. _FR_ & _FT_ means freezing and fine-tuning LM respectively.**
![ChatGPT4Rec_task](https://github.com/anonymous-TCF/anonymous-TCF/blob/main/Figures/Table_ChatGPT_task.jpg)

The output by ChatGPT in the figure below indicates that ChatGPT fully understands the recommendation request. 
**Figure 5: Verifying that ChatGPT understands the request.**
![Verify_ChatGPT4Rec](https://github.com/anonymous-TCF/anonymous-TCF/blob/main/Figures/ChatGPT4Rec_0-cropped.jpg)

We also show the prompts for ChatGPT on MIND, HM, and Bili respectively with the figures below.

**Figure 6: Example of Task 1 for MIND**
![Mind_ChatGPT4Rec1](https://github.com/anonymous-TCF/anonymous-TCF/blob/main/Figures/ChatGPT4Rec_MIND_4-cropped.jpg)
**Figure 7: Example of Task 2 for MIND**
![Mind_ChatGPT4Rec2](https://github.com/anonymous-TCF/anonymous-TCF/blob/main/Figures/ChatGPT4Rec_MIND_10-cropped.jpg)

**Figure 8: Example of Task 1 for HM**
![HM_ChatGPT4Rec1](https://github.com/anonymous-TCF/anonymous-TCF/blob/main/Figures/ChatGPT4Rec_HM_4-cropped.jpg)
**Figure 9: Example of Task 2 for HM**
![HM_ChatGPT4Rec2](https://github.com/anonymous-TCF/anonymous-TCF/blob/main/Figures/ChatGPT4Rec_HM_10-cropped.jpg)

**Figure 10: Example of Task 1 for Bili**
![Bili_ChatGPT4Rec1](https://github.com/anonymous-TCF/anonymous-TCF/blob/main/Figures/ChatGPT4Rec_Bili_4-cropped.jpg)
**Figure 11: Example of Task 2 for Bili**
![Bili_ChatGPT4Rec2](https://github.com/anonymous-TCF/anonymous-TCF/blob/main/Figures/ChatGPT4Rec_Bili_10-cropped.jpg)






## Reference
[1] Jinming Li, Wentao Zhang, Tian Wang, Guanglei Xiong, Alan Lu, and Gerard Medioni. 2023. GPT4Rec: A Generative Framework for Personalized Recommendation and User Interests Interpretation. _arXiv preprint arXiv:2304.03879 (2023)._

[2] Junling Liu, Chao Liu, Renjie Lv, Kang Zhou, and Yan Zhang. 2023. Is ChatGPT a Good Recommender? A Preliminary Study. _arXiv preprint arXiv:2304.10149 (2023)._

[3] Keqin Bao, Jizhi Zhang, Yang Zhang, Wenjie Wang, Fuli Feng, and Xiangnan He. 2023. TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation. arXiv:2305.00447 [cs.IR]

[4] Lei Li, Yongfeng Zhang, and Li Chen. 2023. Personalized prompt learning for explainable recommendation. _ACM Transactions on Information Systems 41,_ 4 (2023), 1–26.

[5] Lei Wang and Ee-Peng Lim. 2023. Zero-Shot Next-Item Recommendation using Large Pretrained Language Models. _arXiv preprint arXiv:2304.03153 (2023)._

[6] Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Ge, and Yongfeng Zhang. 2023. Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5). arXiv:2203.13366 [cs.IR]

[7] Sunhao Dai, Ninglu Shao, Haiyuan Zhao, Weijie Yu, Zihua Si, Chen Xu, Zhongxiang Sun, Xiao Zhang, and Jun Xu. 2023. Uncovering ChatGPT’s Capabilities in Recommender Systems. _arXiv preprint arXiv:2305.02182 (2023)._

[8] Yunfan Gao, Tao Sheng, Youlin Xiang, Yun Xiong, Haofen Wang, and Jiawei Zhang. 2023. Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System. _arXiv preprint arXiv:2303.14524 (2023)._

[9] Yuhui Zhang, Hao Ding, Zeren Shui, Yifei Ma, James Zou, Anoop Deoras, and Hao Wang. 2021. Language models as recommender systems: Evaluations and limitations. (2021).

[10] Wenjie Wang, Xinyu Lin, Fuli Feng, Xiangnan He, and Tat-Seng Chua. 2023. Generative Recommendation: Towards Next-generation Recommender Paradigm. _arXiv preprint arXiv:2304.03516 (2023)._
