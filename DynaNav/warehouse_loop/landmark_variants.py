"""Evaluation-only appearance interventions; never imported by perception."""

from PIL import Image, ImageDraw, ImageFont

VARIANTS = (
    "original",
    "color",
    "layout",
    "surface",
    "lighting",
    "occlusion",
    "development",
    "combined",
)


def label_for(index, row, variant):
    # Novel strings unavailable to perception. Repeat ordinary text deliberately.
    if index == 4 and row in (4, 5):
        return "CAUTION"
    if index == 3 and row == 3:
        return ""
    return (
        f"{70 + index:02d}/{row}R"
        if variant in ("original", "development")
        else f"K{row}-{70 + index}"
    )


def texture(path, label, variant):
    bg, fg = ("white", (175, 20, 35))
    if variant in ("color", "combined"):
        bg, fg = ((20, 70, 150), "white")
    elif variant == "surface":
        bg, fg = ((145, 150, 148), (15, 20, 25))
    im = Image.new("RGB", (1024, 512), bg)
    draw = ImageDraw.Draw(im)
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    if variant in ("layout", "combined"):
        font = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
    if variant in ("original", "development") and "/" in label:
        draw.rectangle((580, 0, 1023, 511), fill=fg)
        left, right = label.split("/")
        draw.text(
            (280, 236), left, anchor="mm", font=ImageFont.truetype(font, 300), fill=fg
        )
        draw.text(
            (802, 250),
            "/" + right,
            anchor="mm",
            font=ImageFont.truetype(font, 176),
            fill="white",
        )
    else:
        draw.text(
            (512, 256),
            label,
            anchor="mm",
            font=ImageFont.truetype(font, 190 if len(label) < 7 else 145),
            fill=fg,
        )
    if variant == "occlusion":
        draw.rectangle((610, 0, 720, 512), fill=(70, 70, 70))
    im.save(path)
