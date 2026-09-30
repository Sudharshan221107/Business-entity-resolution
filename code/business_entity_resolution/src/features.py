"""
Feature Engineering Module for Business Entity Resolution.
Implements matching features strictly using techniques explicitly mentioned in the challenge:
- Jaccard similarity (token & char n-gram)
- Levenshtein similarity (name, address, token-sorted)
- TF-IDF cosine similarity (word & char n-grams)
- Name and address structural similarities (exact match, length ratios, address numbers)
- Country information without hard-coding US/India (open-set consistency)
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compute_levenshtein_distance(s1: str, s2: str) -> int:
    """Computes exact Levenshtein edit distance with O(min(m, n)) space."""
    if s1 == s2:
        return 0
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1] + [0] * len(s2)
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row[j + 1] = min(insertions, deletions, substitutions)
        previous_row = current_row

    return previous_row[-1]

def normalized_levenshtein_sim(s1: str, s2: str) -> float:
    """Normalized Levenshtein similarity in [0.0, 1.0]."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    dist = compute_levenshtein_distance(s1, s2)
    return max(0.0, 1.0 - (dist / max_len))

def jaccard_similarity(set1: set, set2: set) -> float:
    """Computes Jaccard similarity |A & B| / |A | B|."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    union = len(set1 | set2)
    if union == 0:
        return 0.0
    return len(set1 & set2) / union

def get_char_ngrams(text: str, n=3) -> set:
    """Extract character n-grams from string."""
    clean = text.replace(' ', '')
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[i:i+n] for i in range(len(clean) - n + 1)}

class FeatureExtractor:
    def __init__(self):
        self.name_word_tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1)
        self.name_char_tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=1)
        self.addr_word_tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1)
        self.addr_char_tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=1)
        self.is_fitted = False

        self.feature_names = [
            "name_token_jaccard",
            "name_char3_jaccard",
            "addr_token_jaccard",
            "addr_char3_jaccard",
            "name_levenshtein",
            "addr_levenshtein",
            "name_token_sort_levenshtein",
            "name_exact_match",
            "addr_exact_match",
            "name_len_ratio",
            "addr_len_ratio",
            "addr_number_jaccard",
            "addr_number_mismatch",
            "country_match",
            "name_tfidf_word_cosine",
            "name_tfidf_char_cosine",
            "addr_tfidf_word_cosine",
            "addr_tfidf_char_cosine",
        ]

    def fit(self, all_records: list[dict]):
        """Fit TF-IDF vectorizers on the full vocabulary of names and addresses."""
        names = [r['norm_name'] for r in all_records if r['norm_name']]
        addrs = [r['norm_address'] for r in all_records if r['norm_address']]

        if not names:
            names = ["default business name"]
        if not addrs:
            addrs = ["default business address"]

        self.name_word_tfidf.fit(names)
        self.name_char_tfidf.fit(names)
        self.addr_word_tfidf.fit(addrs)
        self.addr_char_tfidf.fit(addrs)
        self.is_fitted = True

    def extract_pair_features(self, r1: dict, r2: dict) -> list[float]:
        """
        Extract feature vector for a pair of records (r1, r2).
        """
        n1 = r1['norm_name']
        n2 = r2['norm_name']
        a1 = r1['norm_address']
        a2 = r2['norm_address']
        c1 = r1['norm_country']
        c2 = r2['norm_country']

        # 1. Jaccard similarities
        t_jaccard_name = jaccard_similarity(set(r1['name_tokens']), set(r2['name_tokens']))
        c_jaccard_name = jaccard_similarity(get_char_ngrams(n1, 3), get_char_ngrams(n2, 3))
        t_jaccard_addr = jaccard_similarity(set(r1['addr_tokens']), set(r2['addr_tokens']))
        c_jaccard_addr = jaccard_similarity(get_char_ngrams(a1, 3), get_char_ngrams(a2, 3))

        # 2. Levenshtein similarities
        lev_name = normalized_levenshtein_sim(n1, n2)
        lev_addr = normalized_levenshtein_sim(a1, a2)

        # Token-sorted Levenshtein
        n1_sort = ' '.join(sorted(r1['name_tokens']))
        n2_sort = ' '.join(sorted(r2['name_tokens']))
        lev_token_sort_name = normalized_levenshtein_sim(n1_sort, n2_sort)

        # 3. Exact match and length ratios
        exact_name = 1.0 if (n1 and n1 == n2) else 0.0
        exact_addr = 1.0 if (a1 and a1 == a2) else 0.0

        max_nlen = max(len(n1), len(n2), 1)
        name_len_ratio = 1.0 - abs(len(n1) - len(n2)) / max_nlen

        max_alen = max(len(a1), len(a2), 1)
        addr_len_ratio = 1.0 - abs(len(a1) - len(a2)) / max_alen

        # 4. Address number signals
        num1 = r1['addr_numbers']
        num2 = r2['addr_numbers']
        addr_num_jaccard = jaccard_similarity(num1, num2)
        addr_num_mismatch = 1.0 if (num1 and num2 and len(num1 & num2) == 0) else 0.0

        # 5. Open-set Country feature (no hardcoding of US/India/France)
        country_match = 1.0 if (c1 and c2 and c1 == c2) else 0.0

        # 6. TF-IDF Cosine similarities
        if self.is_fitted:
            try:
                # Name TF-IDF
                v1_nw = self.name_word_tfidf.transform([n1 or ""])
                v2_nw = self.name_word_tfidf.transform([n2 or ""])
                sim_nw = float(cosine_similarity(v1_nw, v2_nw)[0][0])

                v1_nc = self.name_char_tfidf.transform([n1 or ""])
                v2_nc = self.name_char_tfidf.transform([n2 or ""])
                sim_nc = float(cosine_similarity(v1_nc, v2_nc)[0][0])

                # Address TF-IDF
                v1_aw = self.addr_word_tfidf.transform([a1 or ""])
                v2_aw = self.addr_word_tfidf.transform([a2 or ""])
                sim_aw = float(cosine_similarity(v1_aw, v2_aw)[0][0])

                v1_ac = self.addr_char_tfidf.transform([a1 or ""])
                v2_ac = self.addr_char_tfidf.transform([a2 or ""])
                sim_ac = float(cosine_similarity(v1_ac, v2_ac)[0][0])
            except Exception:
                sim_nw, sim_nc, sim_aw, sim_ac = 0.0, 0.0, 0.0, 0.0
        else:
            sim_nw, sim_nc, sim_aw, sim_ac = 0.0, 0.0, 0.0, 0.0

        return [
            t_jaccard_name,
            c_jaccard_name,
            t_jaccard_addr,
            c_jaccard_addr,
            lev_name,
            lev_addr,
            lev_token_sort_name,
            exact_name,
            exact_addr,
            name_len_ratio,
            addr_len_ratio,
            addr_num_jaccard,
            addr_num_mismatch,
            country_match,
            sim_nw,
            sim_nc,
            sim_aw,
            sim_ac,
        ]

    def extract_matrix(self, pairs: list[tuple[dict, dict]]) -> np.ndarray:
        """Extract feature matrix for a list of (rec1, rec2) pairs."""
        if not pairs:
            return np.empty((0, len(self.feature_names)), dtype=np.float32)
        rows = [self.extract_pair_features(r1, r2) for r1, r2 in pairs]
        return np.array(rows, dtype=np.float32)
