# nlp_module.py

import random
import string
from itertools import combinations


# === Developer A === #
# 🧼 Utility-style minimalist coder — prioritizes performance, clarity, and brevity
def normalize(text):
    return text.strip().lower().translate(str.maketrans('', '', string.punctuation))


# === Developer B === #
# 📚 Loves structure, verbose explanations, and explicit design
class TextResponder:
    """Handles pattern-based response generation using intents and keyword-based matching."""

    def __init__(self):
        self.intents = {
            "greeting": {
                "patterns": ["muraho", "hello", "mwiriwe", "wiriweho"],
                "responses": [
                    "Muraho neza! Nishimiye kukubona hano.",
                    "Mwiriwe! Ufite ikibazo ushaka kumbaza?"
                ]
            },
            "ask_news": {
                "patterns": ["amakuru", "amakuru yawe", "amakuru y'umunsi", "amakuru y'icyumweru", "amakuru ya mugitondo", "amakuru ya nijoro"],
                "responses": [
                    "Amakuru ni meza cyane, urakoze kubaza!",
                    "Umunsi wagenze neza, ndashimira Imana. Wowe se?",
                    "Icyumweru cyagenze neza rwose, wowe ho?",
                    "Mugitondo wagenze neza cyane! Wowe uko byagenze?",
                    "Nijoro wagenze neza cyane, ndabashimiye!",
                ]
            },
            "gratitude": {
                "patterns": ["urakoze", "murakoze", "murakoze cyane", "ndashimira", "turabashimira"],
                "responses": [
                    "Murakoze cyane! Imana ibahe umugisha.",
                    "Twishimiye ko dukorana neza. Urakoze cyane!"
                ]
            },
            "default": {
                "patterns": [],
                "responses": [
                    "Ndababarira, sinashoboye kukumva neza. Ongera ugerageze.",
                    "Sinabyumvise neza, ushobora gusubiramo ikibazo?",
                ]
            }
        }

        self.qa = {
            "ni iki gituma u rwanda rwitwa igihugu cy'imisozi igihumbi":
                "Kubera ko gifite imisozi myinshi cyane itatse igihugu cyose.",
            "umusozi muremure mu rwanda ni uwuhe":
                "Ni Karisimbi, ufite uburebure bwa metero 4,507.",
            "ni irihe shyamba rinini riboneka mu rwanda":
                "Ni ishyamba rya Nyungwe.",
            "ni izihe ndimi zikoreshwa cyane mu rwanda":
                "Ikinyarwanda, Icyongereza, Igifaransa, n'Igiswahili.",
            "ni iki cyihariye ku muco nyarwanda":
                "Gukunda igihugu, gusabana, kubaha abakuru, nimigenzo nk'igisabo."
        }

        self.keyword_map = self._create_keyword_map()

    def _create_keyword_map(self):
        """Generates a mapping from keyword combinations to known questions for fuzzy matching."""
        keyword_dict = {}
        for q in self.qa:
            words = normalize(q).split()
            for n in [2, 3]:
                if len(words) >= n:
                    for combo in combinations(words, n):
                        key = " ".join(sorted(combo))
                        keyword_dict.setdefault(key, []).append(q)
        return keyword_dict

    def _match_keywords(self, normalized_input):
        """Attempts to match input text with QA dictionary using keyword overlap."""
        words = normalized_input.split()

        # First: Try full match
        for question in self.qa:
            if normalize(question) == normalized_input:
                return question

        # Then: Try keyword overlap
        candidate_scores = {}
        for n in [3, 2]:
            if len(words) >= n:
                for combo in combinations(words, n):
                    key = " ".join(sorted(combo))
                    for q in self.keyword_map.get(key, []):
                        candidate_scores[q] = candidate_scores.get(q, 0) + n

        if candidate_scores:
            return max(candidate_scores.items(), key=lambda item: item[1])[0]
        return None

    def _detect_intent(self, normalized_input):
        """Identifies intent from user text by comparing against known patterns."""
        for intent, data in self.intents.items():
            if any(normalize(p) in normalized_input for p in data["patterns"]):
                return intent
        return "default"

    def reply(self, user_message):
        """Main interface: takes input, normalizes it, and returns best-matched response."""
        norm_text = normalize(user_message)

        match = self._match_keywords(norm_text)
        if match:
            return self.qa[match]

        detected_intent = self._detect_intent(norm_text)
        return random.choice(self.intents[detected_intent]["responses"])


# === Developer A === #
# 🎯 Direct input/output function for chatbot pipeline
class Hypothesis:
    def __init__(self, text):
        self.text = text


# Singleton responder to avoid reinitializing in every call
_bot = TextResponder()

def get_response(user_input):
    if isinstance(user_input, Hypothesis):
        user_input = user_input.text
    return _bot.reply(user_input)
