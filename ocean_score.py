from wordllama import WordLlama
import numpy as np



def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


# Function to calculate the similarity and score the input for all traits with minimum matches
def score_personality(text, threshold=0.1, min_matches=1):
    # Define the personality levels for all traits
    personality_levels =  {
        "Openness": {
            1: ["Not curious", "Avoids new experiences", "Conservative", "Rigid in thinking", "Resistant to change", "Closed-minded", "Unimaginative", "Set in their ways", "Traditional", "Inflexible"],
            2: ["Occasionally curious", "Reluctant to explore new ideas", "Prefers familiar experiences", "Minimal interest in novelty", "Mildly open", "Can be open in specific areas", "Shows limited interest in change", "Slightly imaginative", "Somewhat flexible", "Conventional"],
            3: ["Moderately curious", "Willing to try new things with caution", "Some openness to diverse ideas", "Enjoys learning when comfortable", "Occasionally adventurous", "Moderate exploration", "Inquisitive but cautious", "Opens up slowly", "Able to consider new perspectives", "Occasionally creative"],
            4: ["Curious and open-minded", "Likes new experiences but with some reservations", "Intellectually engaged in new concepts", "Enjoys occasional novelty", "Open to growth", "Receptive to new ideas", "Explores at a moderate pace", "Shows interest in broadening views", "Intellectually flexible", "Reasonably adventurous"],
            5: ["Very curious", "Seeks out diverse experiences", "Engages with new ideas and perspectives", "Enjoys intellectual exploration", "Highly imaginative", "Explores regularly", "Open to unconventional ideas", "Seeks novelty", "Intellectually adventurous", "Constantly exploring"],
            6: ["Highly creative", "Embraces unconventional ideas", "Interested in abstract concepts and problem-solving", "Seeks novelty in both thought and experience", "Constantly innovating", "Embraces change", "Enjoys intellectual challenges", "Pioneering", "Always looking for fresh perspectives", "Unrestricted by norms"],
            7: ["Very open to new experiences", "Explores many ideas and activities", "Tolerates unpredictability and embraces novelty", "Constantly seeking new and challenging experiences", "Highly adaptable", "Eager to break traditions", "Looks for variety", "Experimentally engaged", "Unrestrained in exploration", "Challenging the status quo"],
            8: ["Extremely curious", "Passionately seeks out new and diverse experiences", "Engaged in abstract thinking and creative work", "Seeks understanding through exploration", "Creative and boundary-pushing", "Explores intensively", "Highly imaginative", "Relentless in seeking out the new", "Inquisitive beyond comfort zones", "Eager to redefine norms"],
            9: ["Innovative and boundary-pushing", "Regularly explores unconventional ideas", "Challenging traditional norms", "Extremely proactive in seeking change and innovation", "Fearlessly curious", "Groundbreaking in thought", "Redefines creativity", "Constantly evolving ideas", "Pushing boundaries", "Challenging assumptions"],
            10: ["Completely open", "Unfettered by tradition or routine", "Constantly innovating and exploring all aspects of life", "Living at the forefront of creativity and discovery", "Endlessly curious", "Truly unconventional", "Boundless exploration", "Fearlessly experimental", "Completely free from conventional limits", "A visionary"],
        },
        "Conscientiousness": {
            1: ["Disorganized", "Lacks follow-through", "Unreliable", "Avoids planning", "Impulsive", "Chaotic", "Unfocused", "Unmethodical", "Careless", "Sloppy"],
            2: ["Occasionally organized", "Forgetful, often procrastinates", "Needs constant reminders", "Struggles with maintaining structure", "Lacks consistency", "Somewhat disorganized", "Irregular in efforts", "Often distracted", "Somewhat scattered", "Can be unprepared"],
            3: ["Generally organized", "Can complete tasks but not always timely", "Some effort towards structure, but lacks consistency", "Handles tasks when necessary but may get distracted", "Moderately punctual", "Somewhat responsible", "Efforts are inconsistent", "Organized with occasional lapses", "Reasonably reliable", "May procrastinate occasionally"],
            4: ["Moderately organized", "Completes tasks on time but can be disorganized under pressure", "Sometimes struggles with prioritizing tasks", "Needs reminders but usually responsible", "Somewhat punctual", "Attempts to stay on track", "Approaches structure when needed", "Occasionally neglects planning", "Disorganized under stress", "Generally responsible"],
            5: ["Well-organized", "Efficient at time management", "Meets goals consistently", "Focused on achievement and performance", "Timely", "Systematic", "Good at task management", "Well-planned", "Disciplined", "Reliable"],
            6: ["Highly disciplined", "Plans ahead and avoids procrastination", "Highly efficient and focused", "Follows through with little supervision", "Focused and deliberate", "Highly structured", "Keeps track of every detail", "Self-motivated", "Goal-oriented", "Shows strong self-control"],
            7: ["Very reliable", "Strives for excellence", "High attention to detail", "Perform consistently at a high level", "Organized and driven", "Meets deadlines consistently", "Performs with diligence", "Responsible", "Highly committed", "Excellent at time management"],
            8: ["Extremely meticulous", "Sets high standards", "Rarely makes mistakes", "Approaches all tasks with a focus on precision and thoroughness", "Detail-oriented", "Impeccably organized", "Extremely reliable", "Constantly exceeds expectations", "Unfailingly accurate", "Efficient and structured"],
            9: ["Exceptionally dependable", "Exceeds expectations consistently", "Very proactive and structured", "Performs well under pressure and always meets deadlines", "Highly consistent", "Always prepared", "Exceptionally efficient", "Never misses a deadline", "Extraordinarily organized", "Exceeds high standards"],
            10: ["Completely structured", "Demonstrates flawless organization and responsibility", "Exemplifies highest standards of reliability and planning", "Can manage complex tasks with ease and excellence", "Fully accountable", "Totally dependable", "Exceptionally organized", "Unshakable reliability", "Impeccable time management", "Exceeds expectations without fail"],
        },
        "Extraversion": {
            1: ["Introverted", "Avoids social interactions", "Prefers solitude", "Reserved, socially withdrawn", "Quiet", "Shy", "Private", "Reticent", "Solitary", "Unsocial"],
            2: ["Somewhat introverted", "Reluctant to engage in social activities", "Feels more comfortable in quiet, small groups", "Enjoys solitude more than socializing", "Tends to avoid crowds", "Prefers calm environments", "Reticent in social settings", "Withdrawn", "Low-energy in group settings", "Introverted tendencies"],
            3: ["Reserved but occasionally social", "Enjoys spending time with a small group of friends", "Prefers quiet environments", "Occasionally enjoys social interactions but at a lower intensity", "Comfortable with close friends", "Introverted but adaptable", "Occasionally outgoing", "Neutral in group settings", "Can enjoy social settings occasionally", "Low-key social interactions"],
            4: ["Sociable", "Can enjoy social settings but not always outgoing", "Comfortable in moderate social interactions", "Enjoys conversation with familiar people", "Friendly", "Can mingle with acquaintances", "Gives attention to others", "Tends to join social gatherings", "Not overly enthusiastic but enjoys company", "Comfortable in social situations"],
            5: ["Outgoing", "Engages with others easily", "Energized by social interactions", "Enjoys large social gatherings and new acquaintances", "Sociable and extroverted", "Comfortable initiating interactions", "Warm and approachable", "Talkative", "Very social", "Naturally outgoing"],
            6: ["Very energetic", "Loves being in social settings", "Thrives on engaging with new people", "Feels confident in large, energetic groups", "Highly expressive", "Outgoing", "Spontaneous", "Lively", "Radiates energy", "Enthusiastic in social situations"],
            7: ["Extremely talkative", "Has a wide social network", "Excels in initiating conversations", "Enjoys taking charge in social situations", "Eager to meet new people", "Charismatic", "Fluent communicator", "Infectious energy", "Always the center of attention", "Highly influential in social circles"],
            8: ["Highly outgoing", "Seeks new social experiences", "Has a large circle of friends", "Thrives in both large and small social groups", "Loves social adventures", "Quick to socialize", "Energetic and engaging", "Social magnet", "Eager to connect", "Proactively builds connections"],
            9: ["Highly charismatic", "Leadership-driven", "Easily gains influence in social situations", "Can inspire and motivate others effortlessly", "Exudes charm", "Commanding presence", "Easily connects with others", "Compelling", "Has a strong influence", "Highly persuasive"],
            10: ["Exceptionally outgoing", "Constantly seeks new social experiences", "Socially magnetic, attracts attention effortlessly", "Highly engaging and energetic in all social situations", "Unstoppable in social engagement", "Makes connections effortlessly", "Dominates social scenes", "Highly energetic in all situations", "Larger-than-life presence", "Totally sociable and engaging"],
        },
        "Agreeableness": {
            1: ["Uncooperative", "Self-centered", "Lacks empathy", "Argumentative", "Disregards others' needs", "Critical in discussions", "Unpleasant", "Rude", "Aloof", "Indifferent"],
            2: ["Somewhat self-interested", "Rarely compromises", "Argues often", "Occasionally shows empathy but not consistently", "Defensive", "Tends to be blunt", "Not always considerate", "Emotionally distant", "Can be indifferent", "Self-absorbed"],
            3: ["Occasionally cooperative", "Can be kind but self-focused at times", "Prefers to avoid conflict but may not always be accommodating", "Not always empathetic", "Tends to keep to themselves", "Somewhat unaccommodating", "Can be tough", "Occasionally warm", "At times reluctant to compromise", "May be dismissive"],
            4: ["Generally friendly", "Tends to compromise when necessary", "Values harmony but can stand firm in disagreements", "Sometimes empathetic, especially in close relationships", "Moderately kind", "Willing to support", "Fairly balanced", "Friendly but assertive", "Diplomatic", "Generally agreeable"],
            5: ["Considerate", "Cooperative and understanding", "Values others' viewpoints", "Avoids unnecessary conflict and seeks harmony", "Kind", "Empathetic", "Acts selflessly", "Generally supportive", "Fair-minded", "Emotionally intelligent"],
            6: ["Highly empathetic", "Cares deeply for others' feelings", "Seeks to support and help others", "Trustworthy and honest in relationships", "Very kind", "Sensitive to emotions", "Acts with compassion", "Puts others first", "Caring", "Strong moral compass"],
            7: ["Extremely compassionate", "Cooperates willingly", "Always puts others' needs before their own", "Seeks peaceful resolutions and values unity", "Extremely understanding", "Highly generous", "Always thinking of others", "Highly cooperative", "Nurturing", "Deeply caring"],
            8: ["Deeply empathetic", "Highly generous and caring", "Actively nurtures relationships", "Extremely sensitive to the well-being of others", "Very nurturing", "Unfailingly considerate", "Exemplifies emotional intelligence", "Always available for others", "Highly supportive", "Caring in every way"],
            9: ["Exceptionally warm and considerate", "Puts others' needs ahead of their own", "Always ready to offer support and care", "Actively seeks to uplift those around them", "Sacrificial", "Selfless", "Uplifting", "Kind beyond measure", "Embraces others' feelings", "Ever-ready to help"],
            10: ["Exemplary kindness", "Always goes out of their way to help others", "Embodies compassion and selflessness in every action", "Lives to serve and care for others with utmost sincerity", "Uncompromising kindness", "Always supportive", "Noble-hearted", "Deeply altruistic", "Unwavering in compassion", "Exemplifies selflessness"],
        },
      "Neuroticism":{
            1: ["Emotionally stable", "Remains calm under pressure", "Resilient in stressful situations", "Rarely experiences anxiety or frustration", "Even-tempered", "Consistently composed", "Balanced emotions", "Imperturbable", "Emotionally steady", "Grounded"],
            2: ["Generally calm", "Occasionally feels anxious but manages stress well", "Can handle stress with moderate difficulty", "Rarely experiences emotional instability", "Steady", "Emotionally resilient", "Calm under mild pressure", "Handles stress relatively well", "Mildly anxious", "Can overcome stress easily"],
            3: ["Moderately emotionally reactive", "Tends to experience stress occasionally", "May be susceptible to mood fluctuations", "Generally handles pressure, but may become tense in difficult situations", "Occasionally anxious", "Susceptible to mood swings", "Emotionally fluctuating", "Becomes stressed under certain pressures", "Handles moderate stress well", "Slightly volatile"],
            4: ["Occasionally anxious", "Sensitive to criticism", "Experiences mild emotional turbulence", "Can be overwhelmed by stress but recovers relatively quickly", "Mildly unstable", "Occasionally prone to stress", "May become upset at times", "Moderately emotionally reactive", "Emotionally fragile at times", "Sensitive to external stress"],
            5: ["Emotionally reactive", "Sometimes overwhelmed by stress", "Frequently experiences mood swings", "Can become easily upset in high-pressure situations", "Mood fluctuates", "Struggles under stress", "Nervous in challenging circumstances", "Frequent anxiety", "Sensitive to criticism", "Easily overwhelmed"],
            6: ["Frequently anxious", "Struggles with stress management", "Emotional fluctuations noticeable", "Can become easily irritated or agitated under pressure", "Highly sensitive", "Emotional instability", "Frequent anxiety episodes", "Feels overwhelmed easily", "Subject to emotional extremes", "Prone to worry"],
            7: ["Highly emotional", "Prone to anxiety and worry", "Easily overwhelmed by stress", "Struggles with emotional regulation", "Very sensitive", "Emotionally fragile", "High emotional reactivity"],
            8: ["Very anxious", "Mood swings can interfere with daily functioning", "Difficulty managing stress and emotions", "May experience ongoing uncertainty or nervousness", "Severely stressed", "Emotionally unstable", "Prone to emotional upheaval"],
            9: ["Extremely volatile", "Intense emotional reactions to stress", "Struggles to manage emotions effectively", "Frequent feelings of overwhelm and instability", "Severe mood swings", "Constantly stressed", "Emotional rollercoaster"],
            10: ["Severely neurotic", "Constantly stressed", "anxious", "and emotionally unstable", "Struggles to manage emotions in almost every situation", "Feels consistently out of control in emotional responses", "Deeply anxious", "Emotionally fragile", "Perpetually on edge"]
      }
    }

    # Load the default WordLlama model
    model = WordLlama.load(trunc_dim=64)

    scores = {}

    # Encode the input text
    input_embedding = model.embed(text)[0]


    for trait, levels in personality_levels.items():
        level_matches = 0
        level_score = 0

        for level, keywords in levels.items():
            for keyword in keywords:
            # Encode the keywords for the current level
              level_embeddings = model.embed(keyword)[0]

              # Calculate cosine similarity between input and keywords
              similarities = cosine_similarity(input_embedding, level_embeddings)

              # Count matches exceeding the threshold
              matches = (similarities > threshold).sum().item()

              # If matches exceed the threshold and meet min_matches, assign level score
              if matches >= min_matches:
                  level_matches += 1
                  level_score = level

        if level_matches > 0:
          scores[trait] = float(level_score)

    return scores
