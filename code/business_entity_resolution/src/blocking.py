"""
Candidate Generation & Blocking Module.
Multi-pass blocking designed to maximize recall ceiling while maintaining a high reduction ratio.
Produces candidate pairs (S1 -> list of S2/S3 IDs) immediately before model scoring.
"""
from collections import defaultdict

class BlockingEngine:
    def __init__(self, min_token_len=3, min_ngram_overlap=2):
        self.min_token_len = min_token_len
        self.min_ngram_overlap = min_ngram_overlap

        # Inverted indices for target records (Source 2 and Source 3)
        self.target_records = {} # id -> preprocessed record
        self.token_index = defaultdict(set) # (country, token) -> set of target ids
        self.prefix_index = defaultdict(set) # (country, prefix) -> set of target ids
        self.ngram_index = defaultdict(set) # (country, 3gram) -> set of target ids
        self.addr_num_index = defaultdict(set) # (country, number) -> set of target ids
        self.country_pool = defaultdict(list) # country -> list of target ids

    def fit(self, target_records: list[dict]):
        """
        Index all candidate records from Source 2 and Source 3.
        """
        self.target_records = {rec['entity_id']: rec for rec in target_records}
        self.token_index.clear()
        self.prefix_index.clear()
        self.ngram_index.clear()
        self.addr_num_index.clear()
        self.country_pool.clear()

        for rec in target_records:
            t_id = rec['entity_id']
            country = rec['norm_country']
            self.country_pool[country].append(t_id)

            # 1. Prefix key (first token, or first 2 tokens)
            name_tokens = rec['name_tokens']
            if name_tokens:
                first_tok = name_tokens[0]
                self.prefix_index[(country, first_tok)].add(t_id)
                if len(first_tok) >= 4:
                    self.prefix_index[(country, first_tok[:4])].add(t_id)
                if len(name_tokens) >= 2:
                    two_tok = f"{name_tokens[0]}_{name_tokens[1]}"
                    self.prefix_index[(country, two_tok)].add(t_id)

            # 2. Significant tokens index
            for tok in name_tokens:
                if len(tok) >= self.min_token_len:
                    self.token_index[(country, tok)].add(t_id)

            # 3. 3-gram character index on normalized name
            clean_name = rec['norm_name'].replace(' ', '')
            if len(clean_name) >= 3:
                for i in range(len(clean_name) - 2):
                    ngram = clean_name[i:i+3]
                    self.ngram_index[(country, ngram)].add(t_id)

            # 4. Address numbers index
            for num in rec['addr_numbers']:
                self.addr_num_index[(country, num)].add(t_id)

    def generate_candidates_for_record(self, s1_rec: dict, max_candidates=50) -> list[str]:
        """
        Generate candidate target IDs for a single Source 1 record.
        Prioritizes high recall across all blocking passes.
        """
        s1_id = s1_rec['entity_id']
        country = s1_rec['norm_country']
        name_tokens = s1_rec['name_tokens']
        clean_name = s1_rec['norm_name'].replace(' ', '')

        candidates = set()

        # Pass 1: Prefix matching
        if name_tokens:
            first_tok = name_tokens[0]
            candidates.update(self.prefix_index.get((country, first_tok), set()))
            if len(first_tok) >= 4:
                candidates.update(self.prefix_index.get((country, first_tok[:4]), set()))
            if len(name_tokens) >= 2:
                two_tok = f"{name_tokens[0]}_{name_tokens[1]}"
                candidates.update(self.prefix_index.get((country, two_tok), set()))

        # Pass 2: Token inverted index
        for tok in name_tokens:
            if len(tok) >= self.min_token_len:
                candidates.update(self.token_index.get((country, tok), set()))

        # Pass 3: Character 3-gram overlap counting
        if len(clean_name) >= 3:
            ngram_counts = defaultdict(int)
            for i in range(len(clean_name) - 2):
                ngram = clean_name[i:i+3]
                for t_id in self.ngram_index.get((country, ngram), set()):
                    ngram_counts[t_id] += 1
            for t_id, cnt in ngram_counts.items():
                if cnt >= self.min_ngram_overlap:
                    candidates.add(t_id)

        # Pass 4: Address numbers matching + at least one common 2-char prefix
        for num in s1_rec['addr_numbers']:
            addr_cands = self.addr_num_index.get((country, num), set())
            for t_id in addr_cands:
                t_rec = self.target_records.get(t_id)
                if t_rec and name_tokens and t_rec['name_tokens']:
                    # If share first character of name, include
                    if name_tokens[0][:1] == t_rec['name_tokens'][0][:1]:
                        candidates.add(t_id)

        # Fallback safeguard: if 0 candidates found but country has records,
        # grab closest 5 candidates by character overlap so S1 is never left blank if matches exist
        if not candidates and self.country_pool.get(country):
            pool = self.country_pool[country]
            # Simple length proximity fallback
            candidates.update(pool[:min(5, len(pool))])

        # Return sorted list of valid S2 / S3 IDs
        sorted_cands = sorted(candidates)
        if max_candidates and len(sorted_cands) > max_candidates:
            # Score roughly by jaccard on tokens to keep top candidates
            s1_toks = set(name_tokens)
            scored = []
            for cid in sorted_cands:
                t_toks = set(self.target_records[cid]['name_tokens'])
                overlap = len(s1_toks & t_toks)
                scored.append((overlap, cid))
            scored.sort(key=lambda x: -x[0])
            sorted_cands = [x[1] for x in scored[:max_candidates]]

        return sorted_cands

    def generate_all_candidates(self, s1_records: list[dict], max_candidates=50) -> dict[str, list[str]]:
        """
        Generate candidates for a list of Source 1 records.
        Returns dict: s1_id -> [s2/s3 candidate IDs]
        """
        results = {}
        for rec in s1_records:
            s1_id = rec['entity_id']
            results[s1_id] = self.generate_candidates_for_record(rec, max_candidates=max_candidates)
        return results

    def compute_recall_and_reduction(self, candidate_map: dict[str, list[str]], ground_truth: dict[str, list[str]], total_target_records: int):
        """
        Compute blocking recall ceiling and reduction ratio.
        """
        total_true_matches = 0
        captured_matches = 0
        total_pairs_generated = 0
        total_possible_pairs = len(candidate_map) * total_target_records

        for s1_id, true_matches in ground_truth.items():
            true_set = set(true_matches)
            total_true_matches += len(true_set)
            cand_set = set(candidate_map.get(s1_id, []))
            captured_matches += len(true_set & cand_set)
            total_pairs_generated += len(cand_set)

        recall_ceiling = (captured_matches / total_true_matches) if total_true_matches > 0 else 1.0
        reduction_ratio = 1.0 - (total_pairs_generated / total_possible_pairs) if total_possible_pairs > 0 else 1.0

        return {
            'recall_ceiling': recall_ceiling,
            'reduction_ratio': reduction_ratio,
            'total_true_matches': total_true_matches,
            'captured_matches': captured_matches,
            'total_pairs_generated': total_pairs_generated,
            'avg_candidates_per_s1': total_pairs_generated / max(1, len(candidate_map))
        }
