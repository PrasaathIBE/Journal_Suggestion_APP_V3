import re
import math
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from dateutil import parser as date_parser


# =========================
# Utilities
# =========================
def normalize_text(x: Optional[str]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_key(x: Optional[str]) -> str:
    s = normalize_text(x).casefold()
    s = re.sub(r"[\u2010-\u2015]", "-", s)
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def to_yes_no(x: Optional[str]) -> str:
    s = normalize_key(x)
    if s in {"yes", "y", "true", "1", "open", "soon"}:
        return "Yes"
    if s in {"no", "n", "false", "0", "closed"}:
        return "No"
    return "Unknown"


def parse_deadline(x: Optional[str]):
    s = normalize_text(x)
    if not s:
        return pd.NaT
    try:
        return date_parser.parse(s, fuzzy=True).date()
    except Exception:
        return pd.NaT


def safe_join_keywords(series: pd.Series) -> str:
    vals = []
    seen = set()
    for v in series.fillna("").astype(str).tolist():
        t = normalize_text(v)
        if t and t not in seen:
            seen.add(t)
            vals.append(t)
    return " | ".join(vals)


# =========================
# Aim & Scope helpers
# =========================
AIM_SCOPE_MAX_CHARS = 1800

AIM_SCOPE_BOILERPLATE_PATTERNS = [
    r"\bthe aim of (this|the) journal is to\b",
    r"\bthe aims of (this|the) journal are to\b",
    r"\bthe scope of (this|the) journal includes\b",
    r"\bthe journal aims to\b",
    r"\bthe journal publishes\b",
    r"\bthis journal publishes\b",
    r"\bthis journal is a peer reviewed journal\b",
    r"\bthis is a peer reviewed journal\b",
    r"\bpeer reviewed journal\b",
    r"\binclude[s]?, but (is|are) not limited to\b",
]


def clean_aim_scope(text: Optional[str], max_chars: int = AIM_SCOPE_MAX_CHARS) -> str:
    s = normalize_text(text)
    if not s:
        return ""
    s = re.sub(r"【[^】]+】", " ", s)
    s = s.replace("•", " | ").replace("●", " | ").replace("▪", " | ")
    s = s.replace(";", " | ")
    for pat in AIM_SCOPE_BOILERPLATE_PATTERNS:
        s = re.sub(pat, " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*\|\s*", " | ", s)
    s = re.sub(r"\s+", " ", s).strip(" |")
    if len(s) > max_chars:
        s = s[:max_chars].rsplit(" ", 1)[0].strip()
        s = s.rstrip(" ,;:|.-")
    return s


def merge_keywords_and_scope(keywords: Optional[str], aim_scope: Optional[str]) -> str:
    kw = normalize_text(keywords)
    scope = clean_aim_scope(aim_scope)
    if not kw and not scope:
        return ""
    if kw and not scope:
        return kw
    if scope and not kw:
        return scope
    kw_parts = [normalize_text(p) for p in kw.split("|")]
    scope_parts = [normalize_text(p) for p in scope.split("|")]
    out = []
    seen = set()
    for part in kw_parts + scope_parts:
        if not part:
            continue
        k = normalize_key(part)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(part)
    return " | ".join(out)


# =========================
# Domain seeds (keyword fallback)
# =========================
DOMAIN_SEEDS = {
    "AI/ML & Data Science": [
        "machine learning", "deep learning", "neural network", "ai", "artificial intelligence",
        "ml", "nlp", "llm", "transformer", "computer vision", "classification", "prediction",
        "regression", "clustering", "cnn", "convolutional", "segmentation", "feature extraction",
        "image recognition", "object detection", "data mining", "knowledge discovery",
        "federated learning", "reinforcement learning", "generative ai", "large language model",
        "multimodal", "hybrid model", "explainable ai", "xai",
    ],
    "IoT & Embedded": [
        "iot", "internet of things", "embedded", "sensor", "wearable", "edge", "microcontroller",
        "raspberry pi", "arduino", "smart device", "iomt", "edge computing", "edge intelligence",
        "smart home", "industrial iot", "fanet", "ad hoc network",
    ],
    "Cybersecurity & Privacy": [
        "security", "cybersecurity", "privacy", "encryption", "blockchain", "authentication",
        "intrusion", "malware", "attack", "forensics", "differential privacy", "federated privacy",
        "access control", "secure", "chebyshev", "data encryption", "smart city security",
    ],
    "Networking & Communications": [
        "5g", "6g", "network", "routing", "wireless", "communication", "spectrum", "qos",
        "latency", "throughput", "fanet", "flying ad hoc", "fibre optic", "optical sensing",
        "distribution network", "signal transmission", "clustering protocol",
    ],
    "Robotics & Automation": [
        "robot", "robotics", "automation", "autonomous", "drone", "uav", "control",
        "path planning", "manipulator", "robotic arm", "grasping", "sorting robot",
        "logistics robot", "target recognition", "trajectory",
    ],
    "Healthcare & Biomedical": [
        "health", "medical", "biomedical", "clinical", "disease", "diagnosis", "patient",
        "hospital", "ecg", "mri", "ct", "heart", "cancer", "tumor", "oncology", "genomics",
        "biomarker", "prognosis", "immune", "gene", "prostate", "survival",
        "drug rehabilitation", "substance abuse", "relapse", "sports injury", "injury risk",
        "brain network", "alzheimer", "neurodegenerative", "dental", "fluoride", "caries",
        "toothpaste", "mouthwash", "azoospermia", "reproductive", "cognitive load",
        "rehabilitation monitoring", "personalized feedback", "health monitoring",
        "pickleball", "athletic injury", "geriatric", "elderly health",
    ],
    "Energy & Environment": [
        "energy", "renewable", "solar", "wind", "battery", "grid", "carbon", "climate",
        "environment", "sustainability", "microgrid", "hydrogen", "photovoltaic", "power grid",
        "smart grid", "dual carbon", "power system", "energy storage", "frequency resilience",
        "energy conversion", "power supervision", "distribution network inspection",
        "carbon emission", "net zero", "energy management", "smart energy",
    ],
    "Business & Management": [
        "business", "management", "supply chain", "marketing", "hr",
        "human resource", "strategy", "enterprise", "supply chain finance",
        "entrepreneurship accelerator", "venture", "startup", "organizational",
        "operations management", "corporate", "hotel booking", "hospitality management",
        "chatbot business", "ai chatbot trust", "user acceptance",
    ],
    "Education & Learning": [
        "education", "learning", "teaching", "curriculum", "pedagogy", "university",
        "student", "teacher", "school", "course", "competency", "digital literacy",
        "higher education", "e-learning", "mooc", "career guidance", "career development",
        "student performance", "gamification", "mobile learning", "adaptive learning",
        "personalized learning", "learning analytics", "educational assessment",
        "martial arts training", "physical training feedback", "stem education",
        "gender inclusivity", "educational data", "intelligent assessment",
        "barrier free", "elderly digital", "ui ux education", "academic performance",
        "college student", "ideological education",
    ],
    "Mathematics & Statistics": [
        "statistics", "probability", "bayesian", "clustering algorithm",
        "optimization", "mathematical model", "differential equation",
        "statistical modeling", "time series", "econometrics", "regression model",
        "multi collinearity", "dimensionality reduction", "predictive accuracy",
        "big data analytics", "stochastic", "numerical method",
    ],
    "Chemical & Materials": [
        "materials", "alloy", "composite", "polymer", "nanomaterial",
        "corrosion", "electrocatalysis", "coating", "ceramic", "metallurgy",
        "thin film", "crystal", "synthesis", "nanoparticle", "tungsten",
        "hydrogen production", "electrodeposition", "fiber reinforced",
        "rare earth", "ni-mo alloy", "interpenetrating network", "aerospace material",
        "aluminum composite", "a356", "electrocatalytic", "amorphous",
    ],
    "Civil & Structural": [
        "concrete", "structural", "bridge", "dam", "geotechnical",
        "construction", "infrastructure", "pavement", "seismic",
        "reinforced concrete", "fiber content", "water absorption",
        "civil engineering", "foundation", "masonry", "brick", "mesoscopic",
        "damage mechanics", "palf", "sorptivity", "chemical resistance",
    ],
    "Mechanical & Manufacturing": [
        "mechanical", "manufacturing", "machining", "fatigue",
        "heat transfer", "additive manufacturing", "3d printing",
        "turbine", "annealing", "titanium alloy", "selective laser",
        "deformation", "fracture", "tribology", "mechanical anisotropy",
        "supercritical", "ti-6al-4v", "laser melting",
    ],
    "Media & Communication Studies": [
        "media", "journalism", "communication", "social media",
        "discourse", "narrative", "broadcasting", "digital media",
        "content analysis", "multimodal", "semiotics", "puppetry",
        "short video", "platform", "self-presentation", "douyin",
        "shadow puppetry", "chinese media", "visual art", "female representation",
        "critical discourse", "liminal", "body image",
    ],
    "Social Sciences & Humanities": [
        "sociology", "anthropology", "psychology", "culture",
        "society", "qualitative", "ethnography", "humanities",
        "political", "governance", "cross-cultural", "behavioural",
        "envy", "social networking", "identity", "social network envy",
        "business anthropology", "behavioural strategy", "sns",
    ],
    "Transportation & Logistics": [
        "transportation", "logistics", "traffic", "railway",
        "highway", "aviation", "shipping", "mobility", "freight",
        "routing", "urban rail", "signal system", "transit",
        "digital twin rail", "rail monitoring", "signal optimization",
    ],
    "Agriculture & Food Science": [
        "agriculture", "crop", "soil", "irrigation", "livestock",
        "food safety", "agronomy", "horticulture", "fertilizer",
        "food processing", "plant disease", "farm", "harvest",
        "crop disease detection", "agricultural blockchain",
    ],
    "Finance & Economics": [
        "economics", "financial", "investment", "market", "banking",
        "monetary", "fiscal", "gdp", "trade", "economic growth",
        "debt", "equity", "entrepreneurship", "venture capital",
        "government debt", "local government", "supply chain finance",
        "time series econometrics", "financial forecasting",
    ],
    "Law & Policy": [
        "law", "legal", "policy", "regulation", "governance",
        "legislation", "compliance", "rights", "judicial",
        "public policy", "political party", "reform", "funding",
        "party funding", "post-war", "european politics",
    ],
    "Earth & Environmental Sciences": [
        "geology", "hydrology", "earthquake", "geomorphology",
        "oceanography", "remote sensing", "landslide", "rockfall",
        "erosion", "geohazard", "seismic", "flood", "terrain",
        "rockfall susceptibility", "ahp", "highway geohazard",
        "slope stability", "geological hazard",
    ],
}


def extract_concepts_from_title(title: str) -> List[str]:
    t = normalize_key(title)
    tokens = [w for w in t.split() if len(w) >= 3]
    seen = set()
    out = []
    for w in tokens:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out[:25]


def score_domains(title: str) -> Tuple[str, List[Tuple[str, float]]]:
    t = normalize_key(title)
    scores = []
    for dom, seeds in DOMAIN_SEEDS.items():
        sc = 0.0
        for s in seeds:
            sk = normalize_key(s)
            if sk and sk in t:
                sc += 1.0
        scores.append((dom, sc))
    scores.sort(key=lambda x: x[1], reverse=True)
    top1, top1_score = scores[0]
    top2_score = scores[1][1] if len(scores) > 1 else 0.0
    if top1_score == 0:
        return "General / Unknown", scores[:3]
    if top1_score >= 2 and (top1_score - top2_score) >= 1:
        return top1, scores[:3]
    return top1, scores[:3]


def score_domains_text(title: str, abstract: Optional[str]) -> Tuple[str, List[Tuple[str, float]]]:
    t = normalize_key(title)
    a = normalize_key(abstract) if abstract else ""
    combined = (t + " " + a).strip()
    scores = []
    for dom, seeds in DOMAIN_SEEDS.items():
        sc = 0.0
        for s in seeds:
            sk = normalize_key(s)
            if sk and sk in combined:
                sc += 1.0
        scores.append((dom, sc))
    scores.sort(key=lambda x: x[1], reverse=True)
    top1, top1_score = scores[0]
    top2_score = scores[1][1] if len(scores) > 1 else 0.0
    if top1_score == 0:
        return "General / Unknown", scores[:3]
    if top1_score >= 2 and (top1_score - top2_score) >= 1:
        return top1, scores[:3]
    return top1, scores[:3]


def build_query_text(title: str, domains_top3: List[Tuple[str, float]]) -> str:
    concepts = extract_concepts_from_title(title)
    doms = [d for d, s in domains_top3 if s > 0][:3]
    parts = [normalize_text(title)]
    if doms:
        parts.append("Domains: " + ", ".join(doms))
    if concepts:
        parts.append("Concepts: " + ", ".join(concepts[:12]))
    return " | ".join(parts)


def build_query_text_v2(title: str, abstract: Optional[str], domains_top3: List[Tuple[str, float]]) -> str:
    base = build_query_text(title, domains_top3)
    abs_norm = normalize_text(abstract)
    if len(abs_norm) > 1200:
        abs_norm = abs_norm[:1200] + "..."
    return base + " | Abstract: " + abs_norm


# =========================
# Semantic domain scoring
# =========================
DOMAIN_DESCRIPTIONS = {
    "AI/ML & Data Science": (
        "Machine learning and artificial intelligence research. Deep learning, neural networks, "
        "convolutional neural networks CNN, transformers, image segmentation, object detection, "
        "computer vision, natural language processing NLP, large language models LLM, "
        "feature extraction, classification, regression, clustering, predictive modeling, "
        "data science, pattern recognition, UNet, focal transformer, semantic segmentation, "
        "data mining, knowledge discovery, federated learning, reinforcement learning, "
        "explainable AI, hybrid intelligence, generative AI, multimodal learning, "
        "student performance prediction using machine learning, career guidance AI systems, "
        "cognitive load prediction, document classification deep learning."
    ),
    "IoT & Embedded": (
        "The Internet of Things IoT involves a network of interrelated devices embedded with sensors "
        "and software. Smart devices, edge computing, microcontrollers, wearable sensors, industrial "
        "monitoring, autonomous vehicles, firmware design, Internet of Medical Things IoMT, "
        "green IoT, edge intelligence, smart grid IoT, flying ad hoc networks FANET, "
        "mobile gas recognition, IoT-driven energy management."
    ),
    "Cybersecurity & Privacy": (
        "Cybersecurity research: protecting computer systems, networks and data from cyberthreats. "
        "Encryption, authentication, malware detection, intrusion detection, privacy-preserving "
        "methods, blockchain security, ransomware, forensics, access control, differential privacy, "
        "federated learning privacy, Chebyshev encryption, smart city security, "
        "secure decentralized systems, homomorphic encryption, data protection frameworks. "
        "Note: blockchain used purely as a security or privacy mechanism belongs here. "
        "Blockchain applied to healthcare data security and medical record privacy also belongs here."
    ),
    "Networking & Communications": (
        "Networking and communications: wired and wireless networks, communication protocols, "
        "5G, 6G, routing, spectrum management, latency, throughput, quality of service QoS, "
        "network design, mobile communications, signal processing, fibre optic sensing, "
        "distribution network inspection, flying ad hoc networks FANET, UAV communications, "
        "DBSCAN clustering for network topology, ad hoc network protocols."
    ),
    "Robotics & Automation": (
        "Robotics: designing and programming machines to replicate human actions. Autonomous systems, "
        "path planning, drone UAV, industrial automation, manipulation, human-robot interaction, "
        "control systems, robotic perception, logistics sorting robotic arm, grasping optimization, "
        "target recognition robotics, trajectory planning, UAV measurement and control."
    ),
    "Healthcare & Biomedical": (
        "Healthcare and biomedical research covers a wide range of medical and health topics. "
        "This includes: cancer, tumor, oncology, prostate cancer, genomics, bioinformatics, "
        "biomarkers, prognosis, survival prediction, immune infiltration, gene expression, "
        "clinical diagnosis, medical imaging MRI CT scan, patient outcomes, Cox regression, "
        "immunotherapy, disease mechanisms, clinical data analysis. "
        "It also includes: drug rehabilitation monitoring, substance abuse relapse prevention, "
        "personalized feedback systems for patient care, cloud-based health monitoring, "
        "sports injury risk prediction, injury prevention in athletics, pickleball injury, "
        "brain network analysis, Alzheimer disease, neurodegenerative disease, complex network "
        "analysis of brain activity, azoospermia diagnosis, reproductive medicine AI diagnostics, "
        "dental health, fluoride toothpaste efficacy, dental caries prevention, mouthwash review, "
        "geriatric health, elderly care, cognitive load in healthcare interfaces. "
        "Papers that apply AI or blockchain specifically to improve healthcare outcomes, "
        "medical record retrieval, or patient monitoring belong in Healthcare, not in AI/ML or Cybersecurity."
    ),
    "Energy & Environment": (
        "Renewable energy technologies, solar, wind, battery systems, smart grids, carbon reduction, "
        "climate modelling, environmental sustainability, pollution control, energy storage, "
        "microgrid frequency resilience, hydrogen energy systems, photovoltaic systems, "
        "power grid management, dual-carbon target policy, power supervision intelligent systems, "
        "supply chain coordination for energy grids, energy conversion efficiency, "
        "solar-wind-hydrogen hybrid systems, energy security, smart energy management IoT."
    ),
    "Business & Management": (
        "Business and management research: organizational planning, supply chain management, "
        "finance, marketing, human resources, business analytics, enterprise strategy, "
        "operations management, supply chain finance, entrepreneurship accelerator, "
        "startup funding, resource-based view, hospitality management, hotel AI chatbots, "
        "AI chatbot user acceptance in business, trust modelling in commercial AI systems, "
        "privacy-preserving chatbots for hotel booking."
    ),
    "Electrical & Electronics": (
        "Electrical and electronic engineering: circuits, power systems, signal processing, "
        "semiconductor devices, control systems, electromagnetic theory, VLSI design, "
        "power electronics, electric motors, actuators."
    ),
    "Mechanical & Manufacturing": (
        "Mechanical engineering and manufacturing: design, motion, energy, force, materials processing, "
        "production methods, automation, aerospace, automotive, thermal systems, "
        "titanium alloy mechanical anisotropy, selective laser melting, supercritical annealing, "
        "Ti-6Al-4V alloy, additive manufacturing microstructure, fatigue, fracture mechanics."
    ),
    "Civil & Structural": (
        "Civil engineering: structural works, bridges, dams, highways, geotechnical engineering, "
        "construction management, infrastructure development, concrete, steel structures, "
        "PALF fiber reinforced concrete, durability evaluation, water absorption, sorptivity, "
        "brick-concrete mesoscopic damage mechanics, RBCA replacement ratio, fiber content effects."
    ),
    "Chemical & Materials": (
        "Chemical engineering and materials science: polymers, nanomaterials, composites, catalysis, "
        "process engineering, molecular transformations, drug delivery, biomaterials, "
        "tungsten-based rare earth alloys, hydrogen production catalysis, Ni-Mo alloy electrodeposition, "
        "amorphous nanocrystalline electrocatalysis, three-dimensional interpenetrating network composites, "
        "A356 aluminum matrix composite, aerospace lightweight structural materials, "
        "electrocatalytic hydrogen evolution, rare earth materials."
    ),
    "Mathematics & Statistics": (
        "Mathematics and statistics: algebra, calculus, probability, optimization, statistical modeling, "
        "differential equations, Bayesian inference, clustering algorithms, numerical methods, "
        "data analysis, survival analysis, Cox regression models, multi-collinearity, "
        "time series analysis, econometric modeling, dimensionality reduction, "
        "predictive accuracy in big data, statistical inference, WGHRU, GRU networks for prediction."
    ),
    "Agriculture & Food Science": (
        "Agriculture and food science: crop production, livestock, food safety, agronomy, "
        "sustainable agriculture, food biotechnology, soil science, agricultural economics, "
        "crop disease detection using federated learning, blockchain for agricultural traceability, "
        "plant disease classification, smart farming."
    ),
    "Earth & Environmental Sciences": (
        "Earth and environmental sciences: geology, hydrology, oceanography, climate science, "
        "ecology, pollution control, natural resource management, glaciology, snow and ice science, "
        "rockfall susceptibility assessment, geohazard analysis along highways, "
        "AHP analytical hierarchy process for geological risk, slope stability, "
        "landslide prediction, Duhok Kurdistan geohazard, multi-parametric rockfall assessment."
    ),
    "Education & Learning": (
        "Education research covers all aspects of learning, teaching, and educational systems. "
        "This includes: pedagogy, curriculum design, teacher professional development, "
        "digital competency, higher education, university teachers, e-learning, EdTech, "
        "digital transformation in education, competency frameworks, DigComp, learning outcomes, "
        "physical education, sports curriculum, school evaluation standards. "
        "It also includes: MOOC platforms and user profiling, career guidance systems for college students, "
        "student performance prediction in educational context, adaptive and personalized learning, "
        "gamification in education, mobile learning systems, learning analytics, "
        "intelligent assessment models for educational data, language feature extraction in education, "
        "martial arts training with adaptive feedback systems, student career development suggestion, "
        "STEM education gender inclusivity, barrier-free UI/UX for elderly digital engagement, "
        "ideological education for students, independent variable structuring for student guidance. "
        "Papers that apply AI or machine learning specifically to improve educational outcomes, "
        "student assessment, or learning systems belong in Education, not in AI/ML."
    ),
    "Social Sciences & Humanities": (
        "Social science and humanities: sociology, psychology, cultural studies, ethics, linguistics, "
        "anthropology, political science, history, philosophy, media studies, "
        "managing envy on social networking sites, cross-cultural behavioural strategies, "
        "business anthropology, SNS user behaviour, social comparison theory."
    ),
    "Transportation & Logistics": (
        "Transportation engineering and logistics: traffic modelling, supply chain, routing, "
        "transportation networks, logistics optimization, smart mobility, "
        "urban rail signal system monitoring, digital twin for rail systems, "
        "railway signal optimization, transit safety management."
    ),
    "Law & Policy": (
        "Law and public policy: legal systems, regulatory frameworks, policy analysis, "
        "human rights, governance, compliance, public administration, "
        "political party funding, post-war European political reforms, "
        "party finance regulation, democratic governance."
    ),
    "Media & Communication Studies": (
        "Media studies and communication: journalism, digital media, advertising, public relations, "
        "media effects, social media, broadcast, mass communication, "
        "Chinese shadow puppetry in digital media, cultural heritage reconfiguration, "
        "multimodal critical discourse analysis, female self-presentation on Douyin, "
        "short video platforms, visual performing arts in digital context, "
        "fragmented bodies liminal desire, digital cultural transformation."
    ),
    "Astronomy & Space Sciences": (
        "Astronomy and space sciences: astrophysics, cosmology, planetary science, "
        "telescopes, dark matter, space exploration, stellar physics."
    ),
    "Finance & Economics": (
        "Finance and economics: financial markets, banking, accounting, risk management, "
        "microeconomics, macroeconomics, economic policy, investment, corporate finance, "
        "local government debt governance, corporate AI adoption measurement, "
        "supply chain finance game theory, entrepreneurship accelerator financing, "
        "time series econometrics for financial data."
    ),
    "Interdisciplinary & General": (
        "Interdisciplinary research combining multiple academic disciplines. "
        "Cross-field studies, mixed methods, general science journals."
    ),
}


def _l2_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v) + 1e-12)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (_l2_norm(a) * _l2_norm(b)))


def score_domains_semantic(
    paper_embedding: np.ndarray,
    domain_embeddings: Dict[str, np.ndarray],
    topn: int = 3
) -> Tuple[str, List[Tuple[str, float]]]:
    scores: List[Tuple[str, float]] = []
    for domain, dvec in domain_embeddings.items():
        scores.append((domain, cosine_sim(paper_embedding, dvec)))
    scores.sort(key=lambda x: x[1], reverse=True)
    top_domain = scores[0][0] if scores else "Interdisciplinary & General"
    return top_domain, scores[:max(1, int(topn))]


# =========================
# Ensemble domain resolver — margin-based confidence (V7 fix)
# =========================
def resolve_domain_ensemble(
    title_domain: str,
    title_scores: List[Tuple[str, float]],
    combined_domain: str,
    combined_scores: List[Tuple[str, float]],
) -> Tuple[str, List[Tuple[str, float]], float]:
    """
    Resolves the final primary domain by ensembling title-only and combined signals.

    Confidence uses margin between top-1 and top-2 domain scores — not raw score.
    This fixes the always-1.0 problem where mpnet scores (0.2-0.6) * 5.0 always
    exceeded 1.0, making low_confidence_warning never fire.

    Margin interpretation:
      margin >= 0.10  → confidence ~1.0   (domain clearly dominant)
      margin ~  0.05  → confidence ~0.6   (reasonable confidence)
      margin <  0.04  → confidence < 0.5  → low_confidence_warning fires in main.py
      margin <  0.02  → confidence ~0.24  → fallback widens search
    """
    title_map    = {d: s for d, s in title_scores}
    combined_map = {d: s for d, s in combined_scores}

    sorted_combined = sorted(combined_scores, key=lambda x: x[1], reverse=True)
    top1_score      = sorted_combined[0][1] if len(sorted_combined) > 0 else 0.0
    top2_score      = sorted_combined[1][1] if len(sorted_combined) > 1 else 0.0
    combined_margin = top1_score - top2_score

    if title_domain == combined_domain:
        final_domain = title_domain
        confidence   = min(1.0, combined_margin * 12.0)
    else:
        if combined_margin >= 0.03:
            final_domain = combined_domain
            confidence   = min(1.0, combined_margin * 12.0)
        else:
            sorted_title = sorted(title_scores, key=lambda x: x[1], reverse=True)
            t1           = sorted_title[0][1] if len(sorted_title) > 0 else 0.0
            t2           = sorted_title[1][1] if len(sorted_title) > 1 else 0.0
            title_margin = t1 - t2
            final_domain = title_domain
            confidence   = min(1.0, title_margin * 12.0)

    all_domains = set(title_map.keys()) | set(combined_map.keys())
    merged = []
    for d in all_domains:
        t_score = title_map.get(d, 0.0)
        c_score = combined_map.get(d, 0.0)
        merged.append((d, (t_score + c_score) / 2.0))
    merged.sort(key=lambda x: x[1], reverse=True)

    return final_domain, merged, round(confidence, 4)


# =========================
# Data prep
# =========================
def prepare_primary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ren = {}
    if "Journal" in df.columns and "Journal_Name" not in df.columns:
        ren["Journal"] = "Journal_Name"
    if "Special Issue" in df.columns and "Special_Issue_Name" not in df.columns:
        ren["Special Issue"] = "Special_Issue_Name"
    df = df.rename(columns=ren)
    for col in ["Journal_Name", "Special_Issue_Name", "Special_Issue_keywords"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].map(normalize_text)
    if "Aim_and_Scope" not in df.columns:
        df["Aim_and_Scope"] = ""
    df["Aim_and_Scope"] = df["Aim_and_Scope"].map(normalize_text)
    df["Special_Issue_keywords"] = df.apply(
        lambda row: merge_keywords_and_scope(
            row.get("Special_Issue_keywords", ""),
            row.get("Aim_and_Scope", "")
        ),
        axis=1
    )
    df["Journal_Name_norm"] = df["Journal_Name"].map(normalize_key)
    df["Special_Issue_Name_norm"] = df["Special_Issue_Name"].map(normalize_key)
    if "SI_Open" not in df.columns:
        df["SI_Open"] = "Unknown"
    df["SI_Open_std"] = df["SI_Open"].map(to_yes_no)
    if "Deadline" not in df.columns:
        df["Deadline"] = ""
    df["Deadline_parsed"] = df["Deadline"].map(parse_deadline)
    df = df[
        (df["SI_Open_std"] == "Yes") &
        (df["Journal_Name"] != "") &
        (df["Special_Issue_Name"] != "")
    ]
    df["dedupe_key"] = df["Journal_Name_norm"] + "||" + df["Special_Issue_Name_norm"]
    df = df.drop_duplicates("dedupe_key", keep="first").drop(columns=["dedupe_key"])
    return df


def prepare_fallback(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Journal" in df.columns and "Journal_Name" not in df.columns:
        df = df.rename(columns={"Journal": "Journal_Name"})
    if "Special_Issue_keywords" not in df.columns:
        if "Keywords" in df.columns:
            df["Special_Issue_keywords"] = df["Keywords"]
        else:
            df["Special_Issue_keywords"] = ""
    if "Aim_and_Scope" not in df.columns:
        df["Aim_and_Scope"] = ""
    for col in ["Journal_Website", "Index", "Journal_Login_Status", "APC", "Publisher"]:
        if col not in df.columns:
            df[col] = ""
    df["Journal_Name"] = df["Journal_Name"].map(normalize_text)
    df["Journal_Name_norm"] = df["Journal_Name"].map(normalize_key)
    df["Special_Issue_keywords"] = df["Special_Issue_keywords"].map(normalize_text)
    df["Aim_and_Scope"] = df["Aim_and_Scope"].map(normalize_text)
    df["Special_Issue_keywords"] = df.apply(
        lambda row: merge_keywords_and_scope(
            row.get("Special_Issue_keywords", ""),
            row.get("Aim_and_Scope", "")
        ),
        axis=1
    )
    df = df[df["Journal_Name"] != ""]
    agg = df.groupby(
        ["Journal_Name", "Journal_Name_norm"],
        as_index=False
    ).agg(
        Special_Issue_keywords=("Special_Issue_keywords", safe_join_keywords),
        Journal_Website=("Journal_Website", "first"),
        Index=("Index", "first"),
        Journal_Login_Status=("Journal_Login_Status", "first"),
        APC=("APC", "first"),
        Publisher=("Publisher", "first"),
    )
    agg["Special_Issue_Name"] = ""
    agg["Special_Issue_Name_norm"] = ""
    return agg


# =========================
# Embedding strings
# =========================
def embed_text_primary(row: pd.Series) -> str:
    j  = normalize_text(row.get("Journal_Name", ""))
    si = normalize_text(row.get("Special_Issue_Name", ""))
    kw = normalize_text(row.get("Special_Issue_keywords", ""))
    return f"{j} | {si} | {kw}".strip(" |")


def embed_text_fallback(row: pd.Series) -> str:
    j  = normalize_text(row.get("Journal_Name", ""))
    kw = normalize_text(row.get("Special_Issue_keywords", ""))
    return f"{j} | {kw}".strip(" |")


# =========================
# History scoring — with net-zero guard
# =========================
def add_history_scores_from_aggregates(
    cand_df: pd.DataFrame,
    pub_j: pd.DataFrame, rej_j: pd.DataFrame,
    pub_si: pd.DataFrame, rej_si: pd.DataFrame,
    title: str,
    title_domain: str,
    domain_weights: Optional[List[Tuple[str, float]]] = None,
) -> pd.DataFrame:
    df = cand_df.copy()
    concepts = extract_concepts_from_title(title)

    def concept_fit(text: str) -> float:
        t = normalize_key(text)
        if not t:
            return 0.0
        hits = sum(1 for c in concepts if c in t)
        return min(1.0, hits / max(6, len(concepts)))

    if domain_weights and len(domain_weights) > 0:
        top_domains = domain_weights[:3]
        total_w     = sum(w for _, w in top_domains) + 1e-9

        def domain_fit(text: str) -> float:
            t = normalize_key(text)
            if not t:
                return 0.0
            score = 0.0
            for dom, w in top_domains:
                dom_tokens = normalize_key(dom).split()
                hits  = sum(1 for tok in dom_tokens if tok and tok in t)
                frac  = min(1.0, hits / max(3, len(dom_tokens)))
                score += (w / total_w) * frac
            return min(1.0, score)
    else:
        dom_tokens = normalize_key(title_domain).split()

        def domain_fit(text: str) -> float:
            t = normalize_key(text)
            if not t or title_domain == "General / Unknown":
                return 0.0
            hits = sum(1 for w in dom_tokens if w and w in t)
            return min(1.0, hits / max(3, len(dom_tokens)))

    df["concept_fit"] = df["candidate_text"].map(concept_fit)
    df["domain_fit"]  = df["candidate_text"].map(domain_fit)

    df = df.merge(pub_j, on="Journal_Name_norm", how="left").merge(
             rej_j, on="Journal_Name_norm", how="left")
    df = df.merge(pub_si, on=["Journal_Name_norm", "Special_Issue_Name_norm"], how="left").merge(
             rej_si, on=["Journal_Name_norm", "Special_Issue_Name_norm"], how="left")

    for c in ["pub_count_j", "rej_count_j", "pub_count_si", "rej_count_si"]:
        df[c] = df[c].fillna(0).astype(int)

    def pub_boost(row):
        if row["sim"] < 0.45 or row["concept_fit"] <= 0:
            return 0.0
        return 0.06 * math.log1p(row["pub_count_j"]) + 0.10 * math.log1p(row["pub_count_si"])

    def rej_penalty(row):
        if row["sim"] < 0.45 or row["concept_fit"] <= 0:
            return 0.0
        return 0.06 * math.log1p(row["rej_count_j"]) + 0.10 * math.log1p(row["rej_count_si"])

    df["pub_boost"]   = df.apply(pub_boost, axis=1)
    df["rej_penalty"] = df.apply(rej_penalty, axis=1)

    net_zero_mask = df["pub_boost"] == df["rej_penalty"]
    df.loc[net_zero_mask, "pub_boost"]   = 0.0
    df.loc[net_zero_mask, "rej_penalty"] = 0.0

    df["final_score"] = (
        df["sim"]
        + 0.30 * df["concept_fit"]
        + 0.05 * df["domain_fit"]
        + df["pub_boost"]
        - df["rej_penalty"]
    )

    return df.sort_values("final_score", ascending=False)