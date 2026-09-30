"""
Data Preprocessing Module for Business Entity Resolution.
Normalizes business names, addresses, and country labels while preserving distinctive signals.
"""
import re
import unicodedata

# Common business abbreviation expansions / normalizations
LEGAL_SUFFIXES = [
    (r'\bprivate limited\b', 'pvt ltd'),
    (r'\bpvt\b\.?\s*\bltd\b\.?', 'pvt ltd'),
    (r'\blimited\b', 'ltd'),
    (r'\bltd\b\.?', 'ltd'),
    (r'\bcorporation\b', 'corp'),
    (r'\bcorp\b\.?', 'corp'),
    (r'\bincorporated\b', 'inc'),
    (r'\binc\b\.?', 'inc'),
    (r'\bcompany\b', 'co'),
    (r'\bco\b\.?', 'co'),
    (r'\bholding\b', 'hldg'),
    (r'\bholdings\b', 'hldg'),
    (r'\bs\.?a\.?r\.?l\.?\b', 'sarl'),
    (r'\bs\.?a\.?s\.?\b', 'sas'),
    (r'\bs\.?a\.?\b', 'sa'),
    (r'\bgmbh\b', 'gmbh'),
    (r'\bllc\b', 'llc'),
    (r'\bl\.l\.c\b', 'llc'),
    (r'\bllp\b', 'llp'),
    (r'\bl\.l\.p\b', 'llp'),
]

BUSINESS_TERMS = [
    (r'\b&\b', 'and'),
    (r'\btech\b', 'technology'),
    (r'\btechnologies\b', 'technology'),
    (r'\bmfg\b', 'manufacturing'),
    (r'\bmfg\.\b', 'manufacturing'),
    (r'\bengg\b', 'engineering'),
    (r'\bsvcs\b', 'services'),
    (r'\bservice\b', 'services'),
    (r'\bmgmt\b', 'management'),
    (r'\bsys\b', 'systems'),
    (r'\bsystem\b', 'systems'),
    (r'\bintl\b', 'international'),
    (r'\bglbl\b', 'global'),
    (r'\bhosp\b', 'hospital'),
    (r'\bpharm\b', 'pharmaceuticals'),
    (r'\bpharma\b', 'pharmaceuticals'),
]

ADDRESS_ABBREVIATIONS = [
    (r'\bstreet\b', 'st'),
    (r'\broad\b', 'rd'),
    (r'\bavenue\b', 'ave'),
    (r'\bboulevard\b', 'blvd'),
    (r'\bdrive\b', 'dr'),
    (r'\blane\b', 'ln'),
    (r'\bsuite\b', 'ste'),
    (r'\bfloor\b', 'fl'),
    (r'\bbuilding\b', 'bldg'),
    (r'\bnorth\b', 'n'),
    (r'\bsouth\b', 's'),
    (r'\beast\b', 'e'),
    (r'\bwest\b', 'w'),
    (r'\bopposite\b', 'opp'),
    (r'\bopp\b\.?', 'opp'),
    (r'\bnear\b', 'nr'),
    (r'\bnational highway\b', 'nh'),
    (r'\bhighway\b', 'hwy'),
    (r'\bindustrial\b', 'ind'),
    (r'\bphase\b', 'ph'),
    (r'\bplot no\b\.?', 'plot'),
    (r'\bsector\b', 'sec'),
]

def strip_accents(text: str) -> str:
    """Normalize unicode characters and strip accents (e.g. for French names/addresses)."""
    if not text:
        return ""
    normalized = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in normalized if not unicodedata.combining(c))

def clean_text_base(text: str) -> str:
    """Basic lowercasing, accent stripping, and whitespace collapse."""
    if not text or not isinstance(text, str):
        return ""
    text = strip_accents(text).lower()
    text = text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
    return text.strip()

def preprocess_business_name(name: str) -> str:
    """Preprocess and normalize a business name."""
    text = clean_text_base(name)
    if not text:
        return ""

    # Replace punctuation with spaces except hyphens inside words
    text = re.sub(r'[\/\\,\.\(\)\[\]\{\};:"\?!\*\+\=\_]', ' ', text)

    # Standardize business terms
    for pat, rep in BUSINESS_TERMS:
        text = re.sub(pat, rep, text)

    # Standardize legal suffixes
    for pat, rep in LEGAL_SUFFIXES:
        text = re.sub(pat, rep, text)

    # Collapse multiple whitespaces
    tokens = [t.strip() for t in text.split() if t.strip()]
    return ' '.join(tokens)

def preprocess_business_address(address: str) -> str:
    """Preprocess and normalize a business address."""
    text = clean_text_base(address)
    if not text:
        return ""

    # Replace noise punctuation
    text = re.sub(r'[\/\\,\.\(\)\[\]\{\};:"\?!\*\+\=\_#]', ' ', text)

    # Standardize address terms
    for pat, rep in ADDRESS_ABBREVIATIONS:
        text = re.sub(pat, rep, text)

    # Remove extra landmark noise indicators like "opp", "nr", "behind"
    text = re.sub(r'\b(opp|nr|behind|beside)\b', ' ', text)

    tokens = [t.strip() for t in text.split() if t.strip()]
    return ' '.join(tokens)

def extract_numbers(text: str) -> list[str]:
    """Extract sequences of digits (building numbers, postal codes, suite numbers)."""
    if not text:
        return []
    return re.findall(r'\b\d+\b', text)

def preprocess_country(country: str) -> str:
    """Open-set country normalization (do not hardcode US/India/France)."""
    text = clean_text_base(country)
    # Remove all punctuation, strip whitespace
    text = re.sub(r'[^a-z0-9]', '', text)
    return text

def preprocess_record(record: dict) -> dict:
    """
    Preprocess a raw record containing entity_id, business_name, business_address, country.
    Returns an enriched record dict with normalized fields and extracted signals.
    """
    raw_name = record.get('business_name', '')
    raw_addr = record.get('business_address', '')
    raw_country = record.get('country', '')

    norm_name = preprocess_business_name(raw_name)
    norm_addr = preprocess_business_address(raw_addr)
    norm_country = preprocess_country(raw_country)

    return {
        'entity_id': record.get('entity_id', '').strip(),
        'raw_name': raw_name,
        'raw_address': raw_addr,
        'raw_country': raw_country,
        'norm_name': norm_name,
        'norm_address': norm_addr,
        'norm_country': norm_country,
        'name_tokens': norm_name.split(),
        'addr_tokens': norm_addr.split(),
        'addr_numbers': set(extract_numbers(raw_addr)),
    }
