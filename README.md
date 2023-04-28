### Paper Reproduction: Knowledge-aware Assessment of Severity of Suicide Risk for Early Intervention Code Implementation 

Paper Link - [Knowledge-aware Assessment of Severity of Suicide Risk for Early Intervention Code Implementation](https://www.researchgate.net/publication/333076070_Knowledge-aware_Assessment_of_Severity_of_Suicide_Risk_for_Early_Intervention)

The research paper "Knowledge-aware Assessment of Severity of Suicide Risk for Early 
Intervention" introduces a novel approach to identifying suicidal tendencies by analyzing
Reddit data. Traditional methods for identifying suicidal behavior or ideation rely on surveys,
questionnaires or patients sharing their mental health struggle and are typically not sufficient for
quantitative analysis. This paper incorporates domain specific knowledge into prediction of
suicide risk. The paper provides an annotated dataset of 500 Reddit users that contains post
content from mental health subreddits and the respective label (suicidal ideation, behavior,
attempt). A suicide risk severity lexicon is developed for each label using medical knowledge
bases and suicide ontology to detect suicidal thoughts, behavior and action. The feature set is
embellished with the addition of external features such as AFINN, Language Assessment by
Mechanical Turk, etc. The paper discusses how CNNs emerge as the superior model for suicide
risk prediction over rule based and SVM-linear models based on 4 evaluation metrics, namely -
Graded recall, Confusion matrix, Ordinal error and Perceived risk measure.

#### Download Instruction for Data

The data/ folder contains the following files - 

1. 500_Reddit_users_posts_labels.csv - Annotated Reddit Post data
2. AFINN-en-165.txt - Dataset contains words along with their AFINN score (ranging from -5 to 5)
3. labMT - Language Assessment by Mechanical Turk - Dataset contains words, their
average happiness score (polarity), standard deviations, and rankings (Twitter, Google,
NYT, Lyrics).

Download the ConceptNet term vectors ("English-only") 
from [https://github.com/commonsense/conceptnet-numberbatch] - numberbatch-en-19.08.txt.gz
Unzip numberbatch-en-19.08.txt.gz to obtain numberbatch-en.txt. Place numberbatch-en.txt in 
the data/folder. 

4. numberbatch-en.txt - A set of semantic vectors (word embeddings) 
5. External_Features.csv - Dataset contains users along with AFINN score, labMT scores,
First Person Pronouns Ratio, Height of the dependency parse tree and other characteristic features. 
