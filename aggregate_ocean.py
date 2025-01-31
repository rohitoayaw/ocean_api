import numpy as np
import pandas as pd
from ocean_score import score_personality

def statistically_scored_ocean(text):


  result = []

  thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
  min_matches = range(1, 7)

  for th in thresholds:
      min_match_found = False
      for min_match in min_matches:
          personality_scores = score_personality(text, threshold=th, min_matches=min_match)
          if not personality_scores:
              break


          result.append({"threshold": th, "min_match": min_match, "score": personality_scores})
          min_match_found = True
      if not min_match_found:
          break


  all_traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
  all_traits_set = set(all_traits)


  for item in result:
      score = item['score']
      # Only add missing traits
      missing_traits = all_traits_set - score.keys()
      for trait in missing_traits:
          score[trait] = 0

  result = np.round(np.array([list(d['score'].values()) for d in result]).mean(axis=0),2).tolist()

  mean_openness = max(3.0,result[0])
  mean_Conscientiousness = max(3.0,result[1])
  mean_Extraversion = max(3.0,result[2])
  mean_Agreeableness = max(3.0,result[3])
  mean_Neuroticism = max(3.0,result[4])

  return {"Openness":mean_openness, "Conscientiousness":mean_Conscientiousness, "Extraversion":mean_Extraversion, "Agreeableness":mean_Agreeableness, "Neuroticism":mean_Neuroticism}