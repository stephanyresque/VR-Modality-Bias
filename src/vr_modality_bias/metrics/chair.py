"""CHAIR — Caption Hallucination Assessment with Image Relevance.

Reference: Rohrbach et al. 2018, "Object Hallucination in Image Captioning".

We compute two scores:

    CHAIR_i = (# hallucinated object mentions) / (# total object mentions)
              — proportion of mentioned objects that are not in the ground
              truth. Sensitive to "wrong things named", not to how many
              objects the model talks about.

    CHAIR_s = (# captions with ≥1 hallucination) / (# captions)
              — proportion of captions polluted by any hallucination.

For each caption:
    1. Extract the set of COCO-80 categories mentioned (synonym match).
    2. Compare to the ground-truth set of categories present in the image.
    3. Hallucinated = mentioned − ground_truth.

Synonym map
-----------
The COCO-80 synonym list below mirrors the one published with the original
CHAIR paper (Rohrbach 2018, ``utils/synonyms.txt`` in the Hallucination
repo), augmented with common plurals and a few colloquial variants. Whole-
word matching is case-insensitive after punctuation is stripped; multi-word
entries (``hot dog``, ``fire hydrant``) are matched as space-padded
substrings so ``"hot dog"`` is found but ``"red"`` is not found inside
``"reduced"``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

__all__ = [
    "COCO_CATEGORIES",
    "COCO_SYNONYMS",
    "extract_mentioned_objects",
    "chair_per_caption",
    "compute_chair_aggregate",
    "load_ground_truth_objects",
]


COCO_CATEGORIES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)


COCO_SYNONYMS: dict[str, tuple[str, ...]] = {
    "person": (
        "man", "woman", "boy", "girl", "child", "kid", "baby", "adult",
        "lady", "gentleman", "guy", "people", "men", "women", "children",
        "persons", "kids", "boys", "girls", "someone", "player", "players",
        "rider", "skier", "snowboarder", "surfer", "skateboarder",
        "pedestrian", "pedestrians", "athlete", "athletes",
    ),
    "bicycle": ("bike", "bicycles", "bikes", "cyclist", "cyclists"),
    "car": ("automobile", "automobiles", "vehicle", "vehicles", "sedan",
            "taxi", "cab", "cars", "suv", "minivan"),
    "motorcycle": ("motorbike", "motorcycles", "motorbikes", "scooter", "moped"),
    "airplane": ("plane", "aircraft", "jet", "airplanes", "planes", "jets",
                 "airliner", "airliners"),
    "bus": ("buses",),
    "train": ("locomotive", "trains", "tram", "subway", "railroad", "railway"),
    "truck": ("lorry", "trucks", "pickup"),
    "boat": ("ship", "sailboat", "canoe", "kayak", "boats", "ships",
             "yacht", "raft"),
    "traffic light": ("traffic lights", "stoplight", "stoplights",
                      "traffic signal", "traffic signals"),
    "fire hydrant": ("hydrant", "hydrants", "fire hydrants"),
    "stop sign": ("stop signs",),
    "parking meter": ("parking meters",),
    "bench": ("benches",),
    "bird": ("birds", "sparrow", "crow", "pigeon", "duck", "ducks",
             "goose", "geese", "seagull", "seagulls", "parrot", "eagle",
             "owl", "chicken", "chickens", "hen", "rooster"),
    "cat": ("cats", "kitten", "kitty", "kittens", "feline", "felines"),
    "dog": ("dogs", "puppy", "puppies", "canine", "canines"),
    "horse": ("horses", "pony", "ponies", "stallion", "mare", "foal"),
    "sheep": ("lamb", "lambs", "ram"),
    "cow": ("cows", "cattle", "bull", "bulls", "calf", "calves", "ox", "oxen"),
    "elephant": ("elephants",),
    "bear": ("bears", "cub", "cubs", "grizzly", "polar bear"),
    "zebra": ("zebras",),
    "giraffe": ("giraffes",),
    "backpack": ("backpacks", "rucksack", "knapsack"),
    "umbrella": ("umbrellas", "parasol"),
    "handbag": ("handbags", "purse", "purses", "pocketbook"),
    "tie": ("ties", "necktie", "neckties", "bowtie", "bow tie"),
    "suitcase": ("suitcases", "luggage"),
    "frisbee": ("frisbees", "disc"),
    "skis": ("ski",),
    "snowboard": ("snowboards",),
    "sports ball": ("soccer ball", "basketball", "football", "baseball",
                    "tennis ball", "volleyball"),
    "kite": ("kites",),
    "baseball bat": ("bat", "bats"),
    "baseball glove": ("mitt", "mitts"),
    "skateboard": ("skateboards",),
    "surfboard": ("surfboards",),
    "tennis racket": ("racket", "rackets", "racquet", "racquets"),
    "bottle": ("bottles",),
    "wine glass": ("wine glasses", "wineglass", "wineglasses"),
    "cup": ("cups", "mug", "mugs"),
    "fork": ("forks",),
    "knife": ("knives",),
    "spoon": ("spoons",),
    "bowl": ("bowls",),
    "banana": ("bananas",),
    "apple": ("apples",),
    "sandwich": ("sandwiches", "burger", "burgers", "hamburger",
                 "hamburgers", "cheeseburger", "cheeseburgers"),
    "orange": ("oranges",),
    "broccoli": ("broccolis",),
    "carrot": ("carrots",),
    "hot dog": ("hotdog", "hotdogs", "hot dogs"),
    "pizza": ("pizzas",),
    "donut": ("donuts", "doughnut", "doughnuts"),
    "cake": ("cakes", "cupcake", "cupcakes"),
    "chair": ("chairs", "armchair", "armchairs", "recliner"),
    "couch": ("sofa", "sofas", "couches"),
    "potted plant": ("houseplant", "houseplants", "potted plants"),
    "bed": ("beds",),
    "dining table": ("table", "tables", "desk", "desks", "tabletop"),
    "toilet": ("toilets", "urinal"),
    "tv": ("television", "televisions", "tvs", "screen", "screens"),
    "laptop": ("laptops", "notebook", "notebooks"),
    "mouse": ("mice",),
    "remote": ("remotes", "remote control"),
    "keyboard": ("keyboards",),
    "cell phone": ("cellphone", "cellphones", "phone", "phones",
                   "smartphone", "smartphones", "iphone", "iphones",
                   "mobile phone", "mobile phones"),
    "microwave": ("microwaves", "microwave oven"),
    "oven": ("ovens", "stove", "stoves", "range"),
    "toaster": ("toasters",),
    "sink": ("sinks", "basin", "basins"),
    "refrigerator": ("fridge", "refrigerators", "fridges"),
    "book": ("books",),
    "clock": ("clocks",),
    "vase": ("vases",),
    "scissors": ("scissor", "shears"),
    "teddy bear": ("teddy", "teddy bears", "stuffed bear", "stuffed animal",
                   "plush bear"),
    "hair drier": ("hair dryer", "hairdryer", "blow dryer", "hair driers"),
    "toothbrush": ("toothbrushes",),
}


# Pre-compile a single regex that recognises any synonym (including the
# canonical name) and gives us back which category it mapped to. Built
# lazily on first use so import is cheap.
_SYNONYM_TO_CATEGORY: dict[str, str] | None = None


def _build_synonym_index() -> dict[str, str]:
    """{synonym_lowercased: category} for fast membership lookup."""
    out: dict[str, str] = {}
    for cat in COCO_CATEGORIES:
        out[cat] = cat
        for syn in COCO_SYNONYMS.get(cat, ()):
            out[syn] = cat
    return out


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace, pad with spaces.

    The leading/trailing spaces let us do whole-word substring matches like
    ``" cat " in text`` without false hits on ``"cattle"``.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return f" {text} "


def extract_mentioned_objects(
    caption: str,
    synonyms: dict[str, tuple[str, ...] | list[str]] | None = None,
) -> set[str]:
    """Return the COCO-80 categories mentioned in ``caption``.

    Multi-word synonyms (``hot dog``, ``fire hydrant``) match correctly
    because we use space-padded substring search on the normalised text.
    Single-word synonyms also need surrounding spaces, so ``"cat"`` does
    NOT match inside ``"category"``.
    """
    global _SYNONYM_TO_CATEGORY
    if synonyms is None:
        if _SYNONYM_TO_CATEGORY is None:
            _SYNONYM_TO_CATEGORY = _build_synonym_index()
        syn_index = _SYNONYM_TO_CATEGORY
    else:
        # Custom synonym dict (mostly for tests).
        syn_index = {}
        for cat, syns in synonyms.items():
            syn_index[cat] = cat
            for s in syns:
                syn_index[s] = cat

    text = _normalise(caption)
    mentioned: set[str] = set()
    for syn, cat in syn_index.items():
        if cat in mentioned:
            continue
        if f" {syn} " in text:
            mentioned.add(cat)
    return mentioned


def chair_per_caption(
    caption: str,
    ground_truth_objects: set[str],
    synonyms: dict | None = None,
) -> dict:
    """Per-caption decomposition.

    Returns a dict with:
        mentioned        : set of COCO categories named in the caption
        hallucinated     : mentioned − ground_truth
        n_mentioned      : len(mentioned)
        n_hallucinated   : len(hallucinated)
        has_hallucination: bool, True iff any object is hallucinated
    """
    mentioned = extract_mentioned_objects(caption, synonyms)
    hallucinated = mentioned - set(ground_truth_objects)
    return {
        "mentioned": mentioned,
        "hallucinated": hallucinated,
        "n_mentioned": len(mentioned),
        "n_hallucinated": len(hallucinated),
        "has_hallucination": len(hallucinated) > 0,
    }


def compute_chair_aggregate(per_caption_results: list[dict]) -> dict:
    """CHAIR_i and CHAIR_s aggregated across a set of per-caption results.

    Returns NaN scores on an empty input (no captions to score).
    A caption that mentions ZERO objects counts as 0/0 for CHAIR_i (we just
    don't add to the numerator/denominator), and counts as 0-hallucination
    for CHAIR_s.
    """
    n = len(per_caption_results)
    if n == 0:
        return {
            "chair_i": float("nan"),
            "chair_s": float("nan"),
            "n_captions": 0,
            "n_captions_with_hallucination": 0,
            "total_mentioned": 0,
            "total_hallucinated": 0,
        }
    total_mentioned = sum(int(r["n_mentioned"]) for r in per_caption_results)
    total_hallucinated = sum(int(r["n_hallucinated"]) for r in per_caption_results)
    n_with_halluc = sum(1 for r in per_caption_results if r["has_hallucination"])
    chair_i = (total_hallucinated / total_mentioned) if total_mentioned > 0 else 0.0
    chair_s = n_with_halluc / n
    return {
        "chair_i": chair_i,
        "chair_s": chair_s,
        "n_captions": n,
        "n_captions_with_hallucination": n_with_halluc,
        "total_mentioned": total_mentioned,
        "total_hallucinated": total_hallucinated,
    }


def load_ground_truth_objects(instances_path: Path) -> dict[str, set[str]]:
    """Read ``instances_val2017.json`` → ``{image_id_str: set(category_name)}``.

    image_id is returned as the **zero-padded 12-digit string** matching the
    MSCOCO file-naming convention (e.g. ``"000000000139"``) so it can be
    keyed directly off the file stem.
    """
    with Path(instances_path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    cat_id_to_name = {int(c["id"]): str(c["name"]) for c in data["categories"]}

    image_to_objects: dict[str, set[str]] = {}
    for ann in data["annotations"]:
        img_id_str = f"{int(ann['image_id']):012d}"
        cat_name = cat_id_to_name[int(ann["category_id"])]
        image_to_objects.setdefault(img_id_str, set()).add(cat_name)

    # Also make sure every image in the index has an entry (possibly empty)
    # so look-ups for images-without-annotations don't raise KeyError.
    for img in data.get("images", []):
        img_id_str = f"{int(img['id']):012d}"
        image_to_objects.setdefault(img_id_str, set())

    return image_to_objects
