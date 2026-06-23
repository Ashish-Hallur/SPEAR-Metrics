# InterSpeech Rebuttal

## Concise 2000-character version:

R1 raises concerns about inferred attributes and dataset context. The dataset does not provide ground-truth demographics, so we use Vox-Profile’s public models to obtain model-predicted sex labels, age bins, and arousal/dominance scores for stratified distribution shifts (Sec. 2.4), treating them as stratifiers rather than population inference. We explicitly note this limitation. To address confidence/uncertainty (R1), we report Vox-Profile benchmark performance (sex classification: 97.7% accuracy, macro-F1 0.971; age-group prediction: 67.6% accuracy, macro-F1 0.624) and caution that age-conditioned findings are noisier. We also clarify that dominance is a direct output of Vox-Profile’s dimensional emotion model. Regarding dataset composition/heterogeneity (R1/R3), we summarize in Methods the Naturalistic/Improvised design and refer to the dataset paper for detailed scenarios/recording conditions; we additionally screened available subset/task metadata and found differences to be practically negligible (small effect sizes) relative to the dominant conditioning factors, so we pool Naturalistic and Improvised for compactness (R3) and state this rationale. For demographics (R1), we do not restrict age to 18–60: we analyze three ranges: 18–29/30–59/60+, and non-binary/undisclosed gender categories are too sparse and not represented in the model label space, so we acknowledge this as a limitation. R5 questions novelty, conditioning strength, and the link to practical evaluation, and notes that some directions are established; we agree that several effects are known, but the novelty is the large-scale, evaluation-oriented characterization of prosodic and temporal regimes jointly conditioned on speaker (sex/age) and interaction state (arousal/dominance), showing how existing pooled baselines can mask systematic shifts. We also clarify the use case: given an S2S output, compute the same metrics, select matched strata, and report percentile deviations/out-of-regime flags as interpretable diagnostics; this is intended to complement perceptual/HCI evaluation (not replace it) and does not assume “human-like \= natural,” with perceptual mapping left for future work. Finally, to address completeness (R5), we note we computed gender effects for speech rate/pause ratio but observed small effect sizes relative to state and age, and prioritized dominant drivers under the 4-page constraint, and we explicitly acknowledge single-corpus generalizability limits and improve Fig. 2–3 caption readability by removing inline bin-edge ranges.

## Revised 2000 character version:

R5 questions the distinct contribution, we clarify that our novelty is providing a large-scale, evaluation-oriented map of conversational prosodic & temporal distributions and showing how they systematically shift with who is speaking and interactional state. To make the practical use explicit (R5/R3), the intended application is a diagnostic for Speech-to-Speech outputs: compute the same metrics on an agent waveform, compare them to the corresponding reference distribution for the relevant speaker/state category, and report whether the output falls in typical percentiles. We do not assume human-like \= natural. R1 raises concerns about inferred attributes: the dataset provides no ground-truth demographics, so we use Vox-Profile’s public models to obtain predicted sex labels, age bins (18–29/30–59/60+), and arousal/dominance (direct outputs of the dimensional emotion model), treating them as stratifiers rather than population inference. We note this in the limitations. To address confidence/uncertainty (R1), we will summarize Vox-Profile benchmark performance (sex: 97.7% accuracy, 0.971 F1; age-group: 67.6% accuracy, 0.624 F1). Non-binary gender categories (R1) are not represented in the model label space and are too sparse for reliable analysis, we will state this as a limitation. Regarding dataset composition/heterogeneity (R1/R3), we will summarize in Methods and refer to the dataset paper for detailed scenarios/recording conditions. We also previously screened the available subset/task metadata and found that differences had small effect sizes relative to the dominant conditioning factors, so we pool Naturalistic & Improvised for compactness and will state this rationale. Finally, to address completeness (R5), we computed gender effects for speech rate/pause ratio but observed small effect sizes so we prioritized dominant drivers under the 4-page constraint. We also acknowledge single-corpus generalizability in limitations & simplify Fig. 2–3 captions (R3).

## In-depth version:

1. ### Speaker attributes (age, gender) inferred via a foundation model without validation against ground truth. (R1)

R1 notes that age/sex labels are inferred rather than ground-truth. Seamless Interaction does not provide demographic labels, so we use Vox-Profile’s publicly released models to obtain model-predicted age bins and sex labels for stratified analyses, which is now clarified in section 2.4, line 117\. We treat these annotations as stratifiers (not population inference).

2. ### No reporting of confidence or uncertainty in inferred attributes; potential bias propagation from the foundation model. (R1)

Vox-Profile reports strong benchmark performance for sex prediction (≈95%+ accuracy; WavLM Large 97.7% accuracy, macro-F1 0.971) and moderate performance for age-group prediction (WavLM Large 67.6% accuracy, macro-F1 0.624). We’ve added this in section 2.4, line 131\. We have already addressed model-level uncertainties and have explicitly noted possible bias propagation in Limitations.

3. ### Insufficient description of dataset composition and conversational context (types, domains, recording conditions). (R1)

We have already sufficiently discussed the dataset in the beginning of section 2.1, and now we have improved the wording for clarity. For a further detailed description, one must refer to the dataset paper that discusses these topics elaborately.

4. ### Overclaiming “general reference baselines” despite strong context dependence. (R1)

We agree that baselines are context dependent. The manuscript explicitly quantifies regime shifts under speaker and interaction factors (gender/age/arousal/dominance). Pooled regimes are reported only as coarse references when screened factors show negligible effect sizes for that metric, otherwise, we present stratified regimes. We will clarify this rationale in line 168 in the opening of section 3\.

5. ### Limited treatment of demographics (e.g., no non-binary gender, restricted age range). (R1)

Our stratifications follow the label space available from Vox-Profile (binary sex labels and age bins 18–29 / 30–59 / 60+), which matches the age ranges mentioned in the dataset. We do not restrict age to 18–60; we explicitly include a 60+ bin (Fig. 5). Non-binary/undisclosed categories are present in metadata but are too sparse and are not predicted by the Vox-Profile sex model, so we exclude them from analysis. We’ve added this as a limitation in section 4.3, line 250\.

6. ### Unclear computation of derived traits like dominance; possible mismatch with what the model actually provides. (R1)

We clarify that dominance is not an ad-hoc construct, because we use the Vox-Profile released dimensional emotion model, which outputs arousal and dominance scores in \[0,1\] as defined in Vox-Profile. This has already been made explicit in section 2.4.

7. ### Pooled Naturalistic and Improvised subsets without analyzing distributional differences. (R3)

We did compare the global prosodic and temporal metrics across these subsets and found differences to be practically negligible (small effect sizes), so we pooled them to keep the analysis compact and focused on the strongest conditioning factors. We have added a brief sentence clarifying this in the opening of section 3, line 171\.

8. ### Broader dataset heterogeneity not controlled (domains, interaction settings, recording conditions). (R1/R3 combined)

The dataset provides subset/task metadata. We screened available subgroup variables and found that they produced small effect sizes relative to the conditioning factors emphasized in the paper.

9. ### Lacks concrete proposals on implementing the alignment of S2S agents in practice. (R3)

We have explicitly rewritten section 4.2 to address this.

10. ### Figure 2 & 3 captions are clunky with inline ranges. (R3)

We simplified captions by removing inline bin-edge ranges and relying on axes/figure labeling.

11. ### Unclear contribution relative to prior literature; many findings appear confirmatory. (R5)

We agree that several directional effects are established. The novelty here is not the existence of these effects, but the large-scale, evaluation-oriented characterization of prosodic and temporal operating regimes and their conditioning on interaction state and speaker factors in a single corpus. We package these regimes as actionable reference targets intended for speech-native diagnostics of S2S conversational outputs. This matters because S2S systems can fall within pooled ranges yet still sound unnatural if they fail to produce state-appropriate expressivity and timing; our regimes provide interpretable diagnostics for that failure mode.

12. ### Weak case for conditioning effects (age, gender, etc.). (R5)

We agree that many studies analyze individual factors, but our contribution is an evaluation-oriented characterization that quantifies and visualizes how conversational regimes shift under a small set of dominant, operationally available stratifiers, which we justify in Section 2.4. Critically, these conditioning effects are not a minor detail, because pooled baselines can mask systematic shifts, so matched-stratum reference distributions are necessary for meaningful speech-native diagnostics of S2S outputs.

13. ### Unclear connection between analysis and practical use in naturalness evaluation; questions the assumption that human-like \= natural. (R5)

We agree that it might not be well established that naturalness means human-like and that user experience depends on expectations, task, persona, and context. Our claim is narrower, that the regimes we report are candidate behavioral plausibility references for speech-native cues that are often implicated when synthetic dialogue sounds unnatural. Practically, an S2S output can be evaluated by computing the same metrics and reporting percentile deviations relative to matched strata, yielding interpretable diagnostics. We present this as a complement to perceptual/HCI evaluation rather than a replacement, and we explicitly state that mapping deviations to user-rated naturalness is future work.

14. ### Missing gender-based analysis for pause and speech rate. (R5)

We computed gender-stratified speech rate and pause ratio, but observed small effect sizes relative to the dominant drivers (state and age). Given the 4-page constraint, we prioritized the strongest conditioning factors. We note this explicitly as a scope decision.

15. ### Limited generalizability due to reliance on a single dataset. (R5)

# Reviews

**Paper ID**  
2922  
**Paper Title**  
Distributional Baselines for Conversational Prosody and Rhythm  
**Track Name**  
Interspeech 2026 Main Track

## **Reviewer \#1**

### **Questions**

* **1\. I certify that this review complies with the ISCA Code of Conduct for Reviewers and Meta-Reviewers (https://isca-speech.org/https/www.isca-speech.org/Code-of-Conduct-for-Reviewers-and-Meta-Reviewers).**  
  * Agreement accepted  
* **2\. I certify that if I used Generative AI, I followed the ISCA policy Manuscripts or parts of the manuscripts under review must never be uploaded to public or commercial GenAI platforms that transmit data externally, retain inputs for training, or lack confidentiality agreements. Their use is permitted only on firewall-protected or institutionally approved systems that guarantee confidentiality. When GenAI is used, it should be under the following restrictions: \- GenAI cannot be used to assess the originality, novelty, or correctness of the manuscript. \- GenAI outputs cannot be used as authoritative scientific evidence. \- Peer reviewers should verify that GenAI outputs do not introduce or reinforce bias against methodological approaches, demographic groups, or research paradigms used in the manuscript. \- Any use of GenAI must be acknowledged in the review with a sentence summarizing how it was used. \- Peer reviewers will remain fully responsible for the accuracy, fairness, and professionalism of the review, regardless of GenAI assistance.**  
  * Agreement accepted  
* **3\. Introduction and agreement to ethical guidelines As a reviewer, your task is to assess (1) originality, (2) technical correctness, and (3) clarity. Please supply your detailed comments to back up your numerical score in each of these three dimensions. For more details, please refer to the Reviewer Handbook: https://wiki-is.isca-speech.org/en/Technical-Programme/Reviewer-handbook/Reviewing-instructions Your comments will be forwarded to the authors. They will also help Interspeech 2026 to decide the outcome of the paper, and justify the decision for the authors. If the paper is accepted, the comments should guide the authors to improve the presentation of their final manuscript. Interspeech 2026 is committed to fairness and confidentiality throughout the peer-review process. Please carefully review (and agree to) the following ethical guidelines: 1\. You are responsible for your review \- do not outsource your review to anyone else. Keep the contents confidential. 2\. Be constructive and avoid offensive language. 3\. Do not try to actively discover the identities or affiliations of the authors. If you accidentally discover or suspect who the authors are, indicate this under “Confidential comments”.**  
  * Agreement accepted  
* **6\. Type of scientific or technological contribution Please assess the type of contribution. Tick all applicable options. If the type of contribution is not listed, or if you are unable to determine it, please select the last option (only).**  
  * Case study to address a phenomenon in speech science  
* **7\. Originality (novelty) score**  
  * 2: Minor novelty  
* **8\. Technical or methodological correctness score**  
  * 3: Minor issues but credible results  
* **9\. Clarity of presentation score**  
  * 3\. Clear enough, could benefit from some revision  
* **10\. Overall recommendation**  
  * 4: Weak Accept: I am leaning to accept this paper  
* **11\. Feedback and justification of your numerical scores Please provide your comments to the authors here, including justification for your numerical scores. We recommend you to use the following template (copy & paste it to the text field). \------------------------------------------------------------------- Summary of the paper: Major strengths or weaknesses (taking into account originality, technical correctness, clarity): Minor issues: Justification for the overall recommendation: \-------------------------------------------------------------------**  
  * Summary of the paper:

    This paper analyzes a large-scale conversational speech corpus (approximately 4000 hours) to characterize statistical properties of prosodic and temporal features such as speech rate and fundamental frequency (F0). The study investigates how these features vary across speaker attributes, particularly age and gender, with the goal of establishing reference distributions that can be used for comparison, analysis, and evaluation of speech systems.

    Major strengths or weaknesses (taking into account originality, technical correctness, clarity):

    Strengths:

    \- The use of a large-scale dataset (4000 hours) is a major strength and enables robust empirical analysis.  
    \- The paper provides a systematic characterization of prosodic and temporal features, including summary statistics such as mean, standard deviation, and range.  
    \- The idea of building reference distributions conditioned on speaker attributes is valuable and has potential applications in speech synthesis evaluation and conversational analysis.  
    \- The work offers useful empirical observations about how speech characteristics may shift across demographic groups, which could inform future modeling and evaluation efforts.

    Weaknesses:

    \- A key limitation is that speaker attributes (age and gender) are entirely inferred using a foundational model, with no clear description of how these attributes are computed, validation against ground truth, or discussion of model bias or parameter choices.  
    This raises concerns about reliability, reproducibility, and potential bias propagation, especially since the paper positions these findings as reference baselines.

    \- The absence of human-verified labels or validation significantly weakens confidence in the reported demographic trends.

    \- The paper does not sufficiently describe the nature of the conversational data: What types of conversations are included? What domains or topics are covered?  
    Since prosodic features are highly context-dependent, this omission limits the interpretability and generalizability of the findings.  
    For the same reason, the claim that the resulting statistics can serve as general reference baselines is not fully justified without controlling for contextual variables such as Conversation type or Topic.

    \- The treatment of demographic attributes is limited, which possibly relates to the training data of the foundation model. No discussion of non-binary gender categories is provided, and the age range appears restricted (e.g., 18–60), with no explanation of how other groups are handled.

    \- The paper references constructs such as dominance derived from the foundation model, but it is unclear how these are computed based on the paper. This is potentially problematic if the underlying model does not explicitly provide such attributes (e.g., only expressiveness is available in the main paper \[1\])  
    \[1\] Feng et al., Vox-Profile: A Speech Foundation Model  
    Benchmark for Characterizing Diverse Speaker and Speech Traits. https://arxiv.org/pdf/2505.14648

    Minor issues:

    It would improve the paper if the authors could \-  
    \- clarify how speaker attributes are inferred, including model details and confidence levels.  
    \- provide more detailed descriptions of dataset composition (conversation types, recording conditions, domains).  
    \- clarify terminology around the different derived traits (e.g., dominance vs. expressiveness).

    Justification for the overall recommendation:  
    The paper presents an interesting and potentially useful large-scale analysis of prosodic and temporal speech features across demographic attributes. The dataset scale and the goal of building reference distributions are strong aspects of the work.

    However, the reliance on automatically inferred speaker attributes without validation, combined with limited description of the dataset and contextual factors, raises significant concerns about the reliability and generalizability of the findings. These limitations make it difficult to fully trust the proposed reference profiles as broadly applicable baselines.

    Overall, the work is a promising starting point, but additional validation, clearer methodology, and more nuanced analysis are needed to support its claims.

## **Reviewer \#3**

### **Questions**

* **1\. I certify that this review complies with the ISCA Code of Conduct for Reviewers and Meta-Reviewers (https://isca-speech.org/https/www.isca-speech.org/Code-of-Conduct-for-Reviewers-and-Meta-Reviewers).**  
  * Agreement accepted  
* **2\. I certify that if I used Generative AI, I followed the ISCA policy Manuscripts or parts of the manuscripts under review must never be uploaded to public or commercial GenAI platforms that transmit data externally, retain inputs for training, or lack confidentiality agreements. Their use is permitted only on firewall-protected or institutionally approved systems that guarantee confidentiality. When GenAI is used, it should be under the following restrictions: \- GenAI cannot be used to assess the originality, novelty, or correctness of the manuscript. \- GenAI outputs cannot be used as authoritative scientific evidence. \- Peer reviewers should verify that GenAI outputs do not introduce or reinforce bias against methodological approaches, demographic groups, or research paradigms used in the manuscript. \- Any use of GenAI must be acknowledged in the review with a sentence summarizing how it was used. \- Peer reviewers will remain fully responsible for the accuracy, fairness, and professionalism of the review, regardless of GenAI assistance.**  
  * Agreement accepted  
* **3\. Introduction and agreement to ethical guidelines As a reviewer, your task is to assess (1) originality, (2) technical correctness, and (3) clarity. Please supply your detailed comments to back up your numerical score in each of these three dimensions. For more details, please refer to the Reviewer Handbook: https://wiki-is.isca-speech.org/en/Technical-Programme/Reviewer-handbook/Reviewing-instructions Your comments will be forwarded to the authors. They will also help Interspeech 2026 to decide the outcome of the paper, and justify the decision for the authors. If the paper is accepted, the comments should guide the authors to improve the presentation of their final manuscript. Interspeech 2026 is committed to fairness and confidentiality throughout the peer-review process. Please carefully review (and agree to) the following ethical guidelines: 1\. You are responsible for your review \- do not outsource your review to anyone else. Keep the contents confidential. 2\. Be constructive and avoid offensive language. 3\. Do not try to actively discover the identities or affiliations of the authors. If you accidentally discover or suspect who the authors are, indicate this under “Confidential comments”.**  
  * Agreement accepted  
* **6\. Type of scientific or technological contribution Please assess the type of contribution. Tick all applicable options. If the type of contribution is not listed, or if you are unable to determine it, please select the last option (only).**  
  * New corpus, software toolkit, or experimental protocol  
* **7\. Originality (novelty) score**  
  * 3: Sufficiently novel  
* **8\. Technical or methodological correctness score**  
  * 3: Minor issues but credible results  
* **9\. Clarity of presentation score**  
  * 4\. Very well written  
* **10\. Overall recommendation**  
  * 4: Weak Accept: I am leaning to accept this paper  
* **11\. Feedback and justification of your numerical scores Please provide your comments to the authors here, including justification for your numerical scores. We recommend you to use the following template (copy & paste it to the text field). \------------------------------------------------------------------- Summary of the paper: Major strengths or weaknesses (taking into account originality, technical correctness, clarity): Minor issues: Justification for the overall recommendation: \-------------------------------------------------------------------**  
  * Summary of the paper:

    This paper analyzes descriptive characteristics of 4000+ hours of conversational speech from the Seamless Interaction dataset. It determines reference distributions for various aspects of speech that contribute to conversational dynamics and perceived naturalness, including prosodic (mostly related to F0), temporal (speech rate and pausing behavior), and speaker-related factors (arousal, dominance, and age). The authors study the distributional statistics of each of these aspects, quantifying how they are related to each other in real conversational speech settings. The work provides insights into how to set prosody and rhythm targets for speech-to-speech agents designed for dialogue, demonstrating potential for improving the naturalness of these agents in conversational settings.

    Major strengths or weaknesses (taking into account originality, technical correctness, clarity):

    Strengths:

    The paper is written quite well and does a good job of motivating and grounding the work. For each of the quantities that the authors measure and analyze, they provide connections to speech production theory or physiology that clearly motivate their inclusion and usage in the study. They also do a good job of describing each of the procedures they do in detail and providing the intuition and reasoning for why.

    The findings are also highly relevant to a hot research topic right now (naturalness of spoken dialogue agents), and the authors do well to outline the work’s boundaries and limitations in Section 4.3.

    Weaknesses:

    The Seamless Interaction dataset includes Naturalistic (untrained participants) and Improvised (voice actor) subsets. However, expressive speech datasets collected from voice actors often have different characteristics from real-world in-the-wild speech, even under improvised settings. It seems like the paper performed all of its analyses after pooling both subsets together, which could have effects on the resulting “reference distributions”. It might be meaningful to see if any such distributional differences exist between the two subsets in the paper’s analysis.

    The paper outlines distributions of speech characteristics with the goal of aligning S2S agents’ behavior against those distributions. While I recognize that this is a descriptive paper, I would have liked to also see some concrete proposals or discussion on how such alignment might actually be implemented in practice.

    Minor issues:

    The ranges given in the captions for Figures 2 and 3 are somewhat clunky to read inline. It might be better to format these differently or simply leave them out, since the x-axis already clearly displays the sextiles.

    Justification for the overall recommendation:

    This paper presents a well-motivated descriptive analysis of prosody and rhythm across a large dataset of conversational speech. It is well-written, and I believe the resulting distributions would be a useful addition to the current literature on S2S agent naturalness.

## **Reviewer \#5**

### **Questions**

* **1\. I certify that this review complies with the ISCA Code of Conduct for Reviewers and Meta-Reviewers (https://isca-speech.org/https/www.isca-speech.org/Code-of-Conduct-for-Reviewers-and-Meta-Reviewers).**  
  * Agreement accepted  
* **2\. I certify that if I used Generative AI, I followed the ISCA policy Manuscripts or parts of the manuscripts under review must never be uploaded to public or commercial GenAI platforms that transmit data externally, retain inputs for training, or lack confidentiality agreements. Their use is permitted only on firewall-protected or institutionally approved systems that guarantee confidentiality. When GenAI is used, it should be under the following restrictions: \- GenAI cannot be used to assess the originality, novelty, or correctness of the manuscript. \- GenAI outputs cannot be used as authoritative scientific evidence. \- Peer reviewers should verify that GenAI outputs do not introduce or reinforce bias against methodological approaches, demographic groups, or research paradigms used in the manuscript. \- Any use of GenAI must be acknowledged in the review with a sentence summarizing how it was used. \- Peer reviewers will remain fully responsible for the accuracy, fairness, and professionalism of the review, regardless of GenAI assistance.**  
  * Agreement accepted  
* **3\. Introduction and agreement to ethical guidelines As a reviewer, your task is to assess (1) originality, (2) technical correctness, and (3) clarity. Please supply your detailed comments to back up your numerical score in each of these three dimensions. For more details, please refer to the Reviewer Handbook: https://wiki-is.isca-speech.org/en/Technical-Programme/Reviewer-handbook/Reviewing-instructions Your comments will be forwarded to the authors. They will also help Interspeech 2026 to decide the outcome of the paper, and justify the decision for the authors. If the paper is accepted, the comments should guide the authors to improve the presentation of their final manuscript. Interspeech 2026 is committed to fairness and confidentiality throughout the peer-review process. Please carefully review (and agree to) the following ethical guidelines: 1\. You are responsible for your review \- do not outsource your review to anyone else. Keep the contents confidential. 2\. Be constructive and avoid offensive language. 3\. Do not try to actively discover the identities or affiliations of the authors. If you accidentally discover or suspect who the authors are, indicate this under “Confidential comments”.**  
  * Agreement accepted  
* **6\. Type of scientific or technological contribution Please assess the type of contribution. Tick all applicable options. If the type of contribution is not listed, or if you are unable to determine it, please select the last option (only).**  
  * Original theoretical or conceptual contribution  
* **7\. Originality (novelty) score**  
  * 2: Minor novelty  
* **8\. Technical or methodological correctness score**  
  * 4: Technically solid  
* **9\. Clarity of presentation score**  
  * 3\. Clear enough, could benefit from some revision  
* **10\. Overall recommendation**  
  * 2: Reject: I think this paper should be rejected  
* **11\. Feedback and justification of your numerical scores Please provide your comments to the authors here, including justification for your numerical scores. We recommend you to use the following template (copy & paste it to the text field). \------------------------------------------------------------------- Summary of the paper: Major strengths or weaknesses (taking into account originality, technical correctness, clarity): Minor issues: Justification for the overall recommendation: \-------------------------------------------------------------------**  
  * Summary of the paper:  
    This paper presents an analysis of a large dataset of conversations, from the Seamless Interaction dataset released by Meta. The motivation behind the work is that if you can characterise various working ranges for parameters (referred to as "distributional operating regimes") relating to conversational speech, then you can use that to make better speech-to-speech AI systems, in that you have a baseline to compare against. The authors hope this will "guide the development of multi-cue evaluation models grounded in conversational behaviour". The aim to fill a gap where prosody and timing are usually considered separately or at a smaller scale.

    Major strengths or weaknesses (taking into account originality, technical correctness, clarity):  
    The paper is mostly very clear and easy to read. The dataset is undoubtedly large at 4000+ hours, and offers potential to examine conversational properties with greater confidence, albeit within this single dataset.  
    My main concern with the paper is the actual contribution. I get that this is large scale, and that jointly considering the co-variation in various attributes offers potential. You admit it in your own final section on limitations, but I wanted to get more on the actual way in which you are proposing this information gets used. When you present the results on the various metrics, you are mostly saying that what you find agrees with what has already been demonstrated in work that you cite. If you consider any of your results new or more insightful that prior studies, you are not bringing that out very well. You perhaps are not making a strong enough case for the conditioning effects. Many studies have investigated various attributes of conversations across corpora such as Switchboard, Candor, AMI. Characteristics like FTO have been shown to differ online versus in-person (just an example, I know you didn't examine this). I am not sure what your data analysis is showing us beyond existing literature.

    Additionally, the connection between this analysis and practically how you would employ it within conversation naturalness evaluation for conversation with an AI agent was not clear. There's quite a jump from this analysis to that reality. Is it even well established that naturalness means human-like. There's a wealth of literature on this in the HCI community too.

    Minor issues:  
    You look at gender aspects for F0, but not for a metric like pause rate or speech rate. That surprised me as differences have been observed before. Work such as these may be of interest:  
    Kendall, Tyler. "Sociophonetics and speech rate and pause." The Routledge handbook of sociophonetics (2023):

    Binnenpoorte, Diana, Christophe Van Bael, Els den Os, and Lou Boves. "Gender in everyday speech and language: a corpus-based study." In INTERSPEECH, pp. 2213-2216. 2005\. 55-75.

    Justification for the overall recommendation:  
    Overall, I would need to be more strongly convinced of the distinct contribution of this paper, or potential for use of the results, before accepting it.