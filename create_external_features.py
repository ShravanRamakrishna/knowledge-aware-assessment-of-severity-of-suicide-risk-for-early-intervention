# import statements
import re
import nltk
import spacy
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize

# download punkt
nltk.download('punkt')

# read csv files for reddit, afinn and labMT data
reddit = pd.read_csv('data/500_Reddit_users_posts_labels.csv')
afinn = pd.read_csv('data/AFINN-en-165.txt', sep="\t", header=None)
labMT = pd.read_csv("data/labMT")

# define dictionaries to access values for each word
afinn_dict = dict(zip(afinn[0], afinn[1]))
labMT_happiness_rank_dict = dict(zip(labMT['word'], labMT['happiness_rank']))
labMT_happiness_avg_dict = dict(zip(labMT['word'], labMT['happiness_average']))
labMT_happiness_standard_deviation_dict = dict(zip(labMT['word'], labMT['happiness_standard_deviation']))
labMT_twitter_rank_dict = dict(zip(labMT['word'], labMT['twitter_rank']))
labMT_google_rank_dict = dict(zip(labMT['word'], labMT['google_rank']))
labMT_nyt_rank_dict = dict(zip(labMT['word'], labMT['nyt_rank']))
labMT_lyrics_rank_dict = dict(zip(labMT['word'], labMT['lyrics_rank']))

# function to clean post data - removing URLs, HTML, non alpha numeric
def clean_string_data(post):

    post = post.lower()
    post = re.sub(r'http\S+', '', post)
    post = re.sub('<[^<]+?>', '', post)
    post = re.sub(r'[^a-zA-Z0-9 ]', ' ', post)
    return ' '.join(post.split())

# obtain average afinn score for the post
def get_afinn_score(post):
  
  post = clean_string_data(post)
  afinn_score = [afinn_dict[p] for p in post.split() if p in afinn_dict.keys()]
  if len(afinn_score) > 0:
    return np.mean(afinn_score)
  else:
    return 0

# obtain average happiness rank score for the post - labMT
def get_happiness_rank_score(post):

  post = clean_string_data(post)
  happiness_rank_score = [labMT_happiness_rank_dict[p] for p in post.split() if p in labMT_happiness_rank_dict.keys()]
  if len(happiness_rank_score) > 0:
    return np.mean(happiness_rank_score)
  else:
    return 0
  
# obtain average happiness score for the post - labMT
def get_happiness_average_score(post):

  post = clean_string_data(post)
  happiness_average_score = [labMT_happiness_avg_dict[p] for p in post.split() if p in labMT_happiness_avg_dict.keys()]
  if len(happiness_average_score) > 0:
    return np.mean(happiness_average_score)
  else:
    return 0

# obtain average happiness standard deviation score for the post - labMT 
def get_happiness_standard_deviation_score(post):

  post = clean_string_data(post)
  happiness_standard_deviation_score = [labMT_happiness_standard_deviation_dict[p] for p in post.split() if p in labMT_happiness_standard_deviation_dict.keys()]
  if len(happiness_standard_deviation_score) > 0:
    return np.mean(happiness_standard_deviation_score)
  else:
    return 0

# obtain average twitter rank score for the post - labMT  
def get_twitter_rank_score(post):
  
  post = clean_string_data(post)
  twitter_rank_score = [labMT_twitter_rank_dict[p] for p in post.split() if ( p in labMT_twitter_rank_dict.keys() and not pd.isna(labMT_twitter_rank_dict[p]))]
  if len(twitter_rank_score) > 0:
    return np.mean(twitter_rank_score)
  else:
    return 0

# obtain average google rank score for the post - labMT  
def get_google_rank_score(post):

  post = clean_string_data(post)
  google_rank_score = [labMT_google_rank_dict[p] for p in post.split() if ( p in labMT_google_rank_dict.keys() and not pd.isna(labMT_google_rank_dict[p]))]
  if len(google_rank_score) > 0:
    return np.mean(google_rank_score)
  else:
    return 0

# obtain average nyt rank score for the post - labMT  
def get_nyt_rank_score(post):

  post = clean_string_data(post)
  nyt_rank_score = [labMT_nyt_rank_dict[p] for p in post.split() if ( p in labMT_nyt_rank_dict.keys() and not pd.isna(labMT_nyt_rank_dict[p]))]
  if len(nyt_rank_score) > 0:
    return np.mean(nyt_rank_score)
  else:
    return 0

# obtain average lyrics rank score for the post - labMT  
def get_lyrics_rank_score(post):

  post = clean_string_data(post)
  lyrics_rank_score = [labMT_lyrics_rank_dict[p] for p in post.split() if ( p in labMT_lyrics_rank_dict.keys() and not pd.isna(labMT_lyrics_rank_dict[p]))]
  if len(lyrics_rank_score) > 0:
    return np.mean(lyrics_rank_score)
  else:
    return 0

# obtain ratio of first person pronouns to total pronoun count for the post
def get_first_person_pronoun_ratio(post):

  # list of pronouns
  first_person_pronouns = ["I", "me", "my", "mine", "we", "us", "our", "ours"]
  other_pronouns = ["he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs"]

  first_person_count, other_pronoun_count = 0, 0

  post = clean_string_data(post)

  for p in post.split():
    p_lower = p.lower()
    if p_lower in first_person_pronouns:
      first_person_count += 1
    elif p_lower in other_pronouns:
      other_pronoun_count += 1

  total_pronoun_count = first_person_count + other_pronoun_count
  
  if total_pronoun_count > 0:
    return (first_person_count/total_pronoun_count)*100
  else:
    return 0

# obtain number of sentences for the post
def get_number_of_sentences(post):

  number_of_sentences = sent_tokenize(post)
  return len(number_of_sentences)

# obtain number of definite articles for the post
def get_number_of_definite_articles(post):
  post_lower = post.lower()
  words = post_lower.split()
  number_of_definite_articles = sum(1 for word in words if word == "the")
  return number_of_definite_articles

# obtain the maximum verb phrase length for the post
def get_max_verb_phrase(post):
  
  nlp = spacy.load('en_core_web_sm')
  doc = nlp(post)
  max_length = 0
  current_length = 0
  
  for token in doc:
      if token.pos_ == 'VERB':
          # if the token is a verb, increment the current verb phrase length
          current_length += 1
      elif current_length > 0:
          # if the token is not a verb and the current verb phrase length is greater than 0,
          # update max verb phrase length and reset the current verb phrase length
          max_length = max(max_length, current_length)
          current_length = 0
  
  max_length = max(max_length, current_length)
  
  return max_length

# obtain the average height of the dependency parse tree for the post
def get_avg_tree_height(post):  
  
  nlp = spacy.load("en_core_web_sm", disable=['ner'])

  # recursively compute the height of dependency parse tree
  def tree_height(root):
      if not list(root.children):
          return 0
      else:
          return 1 + max(tree_height(x) for x in root.children)
  
  doc = nlp(post)
  roots = [sent.root for sent in doc.sents]
  heights = [tree_height(root) for root in roots]
  avg_height = sum(heights) / len(heights)
  return avg_height

# obtain first person pronoun count for the post
def get_first_person_pronoun_count(post):

  post = clean_string_data(post)  
  fpp = ["i", "me", "my", "mine", "we", "us", "our", "ours"]
  first_person = 0
  for word in post.split():
      if word in fpp:
          first_person += 1

  return first_person

# obtain other person pronoun count for the post
def get_other_person_pronoun_count(post):

  post = clean_string_data(post)  
  spp = ["you", "your", "yours"]
  tpp = ["he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs"]
  other_pp = ["it", "its", "who", "whom", "whose", "which", "what", "that"]
  other_person = 0
  for word in post.split():
      if ((word in spp) or (word in tpp) or (word in other_pp)):
          other_person += 1
  return other_person

# apply the functions on the post data
reddit["afinn_score"] = reddit["Post"].apply(get_afinn_score)
reddit['happiness_rank_score'] = reddit["Post"].apply(get_happiness_rank_score)
reddit['happiness_average_score'] = reddit["Post"].apply(get_happiness_average_score)
reddit['happiness_standard_deviation_score'] = reddit["Post"].apply(get_happiness_standard_deviation_score)
reddit['twitter_rank_score'] = reddit["Post"].apply(get_twitter_rank_score)
reddit['google_rank_score'] = reddit["Post"].apply(get_google_rank_score)
reddit['nyt_rank_score'] = reddit["Post"].apply(get_nyt_rank_score)
reddit['lyrics_rank_score'] = reddit["Post"].apply(get_lyrics_rank_score)
reddit["FPPR"] = reddit["Post"].apply(get_first_person_pronoun_ratio)
reddit['number_of_sentences'] = reddit['Post'].apply(get_number_of_sentences)
reddit['number_of_definite_articles'] = reddit['Post'].apply(get_number_of_definite_articles)
reddit['max_verb_phrase_length'] = reddit['Post'].apply(get_max_verb_phrase)
reddit['parse_tree_height'] = reddit['Post'].apply(get_avg_tree_height)
reddit['fpp_count'] = reddit['Post'].apply(get_first_person_pronoun_count)
reddit['other_pronoun_count'] = reddit['Post'].apply(get_other_person_pronoun_count)

# extract user and external features only
reddit_df_ef = reddit[["User", "afinn_score", "FPPR", "happiness_rank_score", "happiness_average_score",
                          "happiness_standard_deviation_score", "twitter_rank_score", "google_rank_score",
                          "nyt_rank_score", "lyrics_rank_score", "parse_tree_height", "max_verb_phrase_length", 
                          "fpp_count", "number_of_sentences", "number_of_definite_articles", "other_pronoun_count"]]

# save the dataframe to a csv file
reddit_df_ef.to_csv("data/External_Features.csv", index=False)