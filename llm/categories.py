"""The drink categories, shared by the order parser (text) and the image classifier (VLM).

Each category has two descriptions:
  - "order":  how people ask for it, for mapping a spoken/typed request.
  - "visual": what the can or bottle looks like, for classifying a camera crop.

Adding a product = add its brand to the right category here (both descriptions
and BRAND_KEYWORDS). Both models then pick it up.
"""

import re

NOT_SUPPORTED = "not_supported"

CATEGORIES: dict[str, dict[str, str]] = {
    "coke": {
        "order": "regular Coca-Cola only (also 'coke', 'coca cola', 'classic coke')",
        "visual": "Regular Coca-Cola. RED can or bottle label with the white cursive "
                  "'Coca-Cola' script and white wave ribbon. NOT silver, NOT black.",
    },
    "diet_coke": {
        "order": "Diet Coke or Coke Zero (also 'diet coke', 'coke zero', 'sugar-free coke')",
        "visual": "Diet Coke: SILVER/grey can with the word 'Diet' and red 'Coke' lettering. "
                  "Coke Zero: BLACK can or label with red 'Coca-Cola' and 'Zero Sugar'.",
    },
    "water": {
        "order": "plain / still bottled water (also 'water', 'bottle of water', 'water bottle'; "
                 "e.g. Dasani, Aquafina, Fiji, Evian, Smartwater, Poland Spring)",
        "visual": "Still (non-carbonated) water: clear or light-blue plastic bottle of clear "
                  "liquid with a narrow label, e.g. Dasani, Aquafina, Fiji (square bottle, "
                  "pink hibiscus), Evian, Smartwater, Poland Spring. No 'sparkling' wording.",
    },
    "sparkling_water": {
        "order": "sparkling / seltzer / carbonated water of any flavor "
                 "(e.g. LaCroix, Bubly, Perrier, Topo Chico, 'lime sparkling water')",
        "visual": "Sparkling water / seltzer in any flavor. Usually a slim or standard can "
                  "with bright pastel or neon graphics and a fruit name, e.g. LaCroix "
                  "(colorful splash pattern), Bubly (bold solid color, smiling 'bubly' logo), "
                  "Spindrift, Waterloo, AHA, Polar; or a green glass Perrier bottle, "
                  "Topo Chico, San Pellegrino. Words like 'sparkling', 'seltzer', 'carbonated'.",
    },
    "energy_drink": {
        "order": "energy drinks (e.g. Red Bull, Monster, Celsius, Rockstar, Bang)",
        "visual": "Energy drink: tall slim or 16 oz can, often dark or metallic with aggressive "
                  "branding, e.g. Red Bull (blue/silver with two red bulls), Monster (black "
                  "with green claw 'M'), Celsius, Rockstar, Bang, Reign, Alani Nu, Ghost. "
                  "Words like 'energy', 'caffeine'.",
    },
    "coconut_water": {
        "order": "coconut water (e.g. Vita Coco, Zico)",
        "visual": "Coconut water: carton (Tetra Pak) or can showing a coconut or palm tree "
                  "and the words 'coconut water', e.g. Vita Coco (white/green), Zico, Harmless "
                  "Harvest (clear bottle, pink-tinted liquid).",
    },
    "general_soda": {
        "order": "any other soda / pop that is not Coca-Cola or Diet Coke "
                 "(e.g. Sprite, Pepsi, Fanta, Dr Pepper, Canada Dry ginger ale, root beer)",
        "visual": "Any other soft drink that is not a Coca-Cola product above, e.g. Pepsi "
                  "(blue with red/white/blue globe), Sprite (green), Fanta (orange), Dr Pepper "
                  "(maroon), Canada Dry ginger ale (green/white with gold crest), 7UP, "
                  "Mountain Dew, A&W root beer, Mug, Schweppes.",
    },
}

LABELS = list(CATEGORIES)


def catalog(kind: str) -> str:
    """Bulleted list of categories with the 'order' or 'visual' description."""
    return "\n".join(f"- {name}: {c[kind]}" for name, c in CATEGORIES.items())


# Brand/product words, checked in this order (first match wins): unsupported
# drinks first, coconut before water, other soda brands before diet_coke (so
# "Diet Pepsi" is general_soda), diet_coke before coke, generic words last.
# Used on the request text (order parser) and on the text the VLM reads off the
# container. Small models get the words right more often than the label, so
# a keyword match overrides the model's label.
BRAND_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    (NOT_SUPPORTED, ("juice", "lemonade", "milk", "coffee", "tea", "beer", "wine")),
    ("coconut_water", ("coconut", "vita coco", "zico")),
    ("sparkling_water", ("sparkling", "seltzer", "carbonated", "fizzy", "soda water",
                         "club soda", "lacroix", "la croix", "bubly", "perrier", "topo chico",
                         "pellegrino", "spindrift", "waterloo", "polar")),
    ("general_soda", ("pepsi", "sprite", "fanta", "dr pepper", "dr. pepper", "canada dry",
                      "7up", "7 up", "mountain dew", "a&w", "schweppes")),
    ("diet_coke", ("diet coke", "coke zero", "zero sugar", "coca-cola zero", "coca cola zero",
                   "diet")),
    ("coke", ("coca-cola", "coca cola", "coke")),
    ("energy_drink", ("red bull", "monster", "celsius", "rockstar", "bang", "reign",
                      "alani", "ghost", "energy")),
    ("water", ("dasani", "aquafina", "fiji", "evian", "smartwater", "poland spring",
               "deer park", "voss", "essentia", "water")),
    ("general_soda", ("ginger ale", "root beer", "cola", "soda", "pop")),
]
# Sodas despite the name; replaced before matching so "beer" doesn't catch them.
_SODA_ALIASES = {"root beer": "root_soda", "ginger beer": "ginger_soda"}


def label_from_text(text: str) -> str | None:
    """Category implied by product words in `text` (may be NOT_SUPPORTED), or None."""
    text = text.lower()
    for alias, soda in _SODA_ALIASES.items():
        text = text.replace(alias, f"{soda} soda")
    for label, words in BRAND_KEYWORDS:
        # Match at a word start: "waters" hits "water", "steam" doesn't hit "tea".
        if any(re.search(r"\b" + re.escape(w), text) for w in words):
            return label
    return None
