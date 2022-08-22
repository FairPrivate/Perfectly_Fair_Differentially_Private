# Perfectly_Fair_Differentially_Private

The proposed selection procedures applied on two different datasets

## FICO Credit Scores

We investigated our proposed methods on these two different experiment settings on FICO credit score dataset

#### 1 ) White vs. Black
#### 2 ) White + Hispanic vs. Asian


## Adult Income 

We investigated our proposed methods on the following experiment setting on Adult income dataset

#### 1) White vs. Black

# File description

In the ["adversarial_debiasing.py"](https://github.com/FairPrivate/Perfectly_Fair_Differentially_Private/blob/main/adversarial_debiasing.py) we have changed the procedure for training the adversary so that it satisfies equal opportunity fairness notion. To do so, we adopted the code from ["adversarial_debiasing.py"](https://github.com/Trusted-AI/AIF360/blob/master/aif360/algorithms/inprocessing/adversarial_debiasing.py) in [FairAI360](https://github.com/Trusted-AI/AIF360) Github repository. Since we wanted to satisfy equal opportunity according to [[1]](#1) we limited the adversary to only be trained on the qualified applicants.


## References
<a id="1">[1]</a> 
Brian Hu Zhang, Blake Lemoine, and Margaret Mitchell. 
Mitigatingunwanted biases with adversarial learning. 
In Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society, pages 335–340, 2018.
