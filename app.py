import cv2
import urllib.request
import urllib.parse
import numpy as np
import gradio as gr
from inference_sdk import InferenceHTTPClient
from groq import Groq
import os
import json
import datetime
from collections import Counter
from gradio.themes.base import Base
from gradio.themes.utils import colors

CUSTOM_CSS = """
    /* Global dark override */
    body, .gradio-container {
        background:
            radial-gradient(ellipse 80% 50% at 10% 0%,   rgba(59,130,246,0.07) 0%, transparent 60%),
            radial-gradient(ellipse 60% 40% at 90% 100%, rgba(139,92,246,0.07) 0%, transparent 60%),
            radial-gradient(ellipse 50% 60% at 50% 50%,  rgba(200,146,42,0.04) 0%, transparent 70%),
            #0a0a0a !important;
        background-attachment: fixed !important;
    }
    .tabs > .tab-nav { border-bottom: 1px solid #2a2518 !important; }
    .tabs > .tab-nav > button {
        font-family: 'Cinzel', serif !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.06em !important;
        color: #666 !important;
        border-radius: 0 !important;
        padding: 10px 20px !important;
        transition: color 0.2s !important;
    }
    .tabs > .tab-nav > button.selected {
        color: #c8922a !important;
        border-bottom: 2px solid #c8922a !important;
        background: transparent !important;
    }
    .tabs > .tab-nav > button:hover { color: #e0c080 !important; }
    .block { border-radius: 14px !important; border-color: #2a2518 !important; }
    label { color: #888 !important; font-size: 0.75rem !important; letter-spacing: 0.05em !important; text-transform: uppercase !important; }
    textarea, input[type=text], input[type=number] {
        background: #111 !important;
        border-color: #2a2518 !important;
        color: #C5C7C4 !important;
        border-radius: 10px !important;
    }
    .svelte-1ipelgc { color: #e0c080 !important; }
    button.primary { border-radius: 10px !important; font-weight: 700 !important; letter-spacing: 0.04em !important; }
    button.secondary { background: #1a1a1a !important; border-color: #333 !important; color: #888 !important; border-radius: 10px !important; }
    button.stop { background: #1a0a0a !important; border-color: #ef444433 !important; color: #ef4444 !important; border-radius: 10px !important; }
"""

custom_theme = Base(
    primary_hue=colors.amber,
    secondary_hue=colors.stone,
    neutral_hue=colors.stone,
).set(
    body_background_fill="#0a0a0a",
    body_background_fill_dark="#0a0a0a",
    block_background_fill="#111111",
    block_background_fill_dark="#111111",
    button_primary_background_fill="#c8922a",
    button_primary_background_fill_hover="#e0a832",
    button_primary_text_color="#0a0a0a",
    block_label_text_color="#888",
    block_title_text_color="#e0c080",
    input_background_fill="#1a1a1a",
    input_background_fill_dark="#1a1a1a",
    input_border_color="#333",
    body_text_color="#C5C7C4",
    body_text_color_subdued="#888",
    shadow_drop="0 8px 32px rgba(200,146,42,0.15)",
    border_color_primary="#333",
    color_accent="#c8922a",
    color_accent_soft="#2a1f0a",
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
CLIENT = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key="cXQEyLSlDBcL2Yz0VbYX")
MODEL_ID = "fossil-scanner-v2-ncp2c-oon07/1"
CATALOG_FILE = "fossil_catalog.json"


def load_catalog():
    if os.path.exists(CATALOG_FILE):
        with open(CATALOG_FILE, "r") as f:
            return json.load(f)
    return []


def save_catalog(catalog):
    with open(CATALOG_FILE, "w") as f:
        json.dump(catalog, f, indent=2)


def add_to_catalog(label, confidence, description, image_path):
    catalog = load_catalog()
    entry = {
        "id": len(catalog) + 1,
        "name": label.title(),
        "confidence": round(confidence * 100, 2),
        "description": description,
        "image_path": image_path,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "verified": None,
    }
    catalog.append(entry)
    save_catalog(catalog)


def mark_entry(entry_id, verified):
    catalog = load_catalog()
    for entry in catalog:
        if entry["id"] == int(entry_id):
            entry["verified"] = verified
            break
    save_catalog(catalog)



def get_reference_image(fossil_name):
    """Fetch a reference image URL using Wikimedia search API."""
    headers = {
        "User-Agent": "FossilScanner/1.0 (educational fossil identification tool; contact@fossilscanner.app)",
        "Accept": "application/json",
    }

    try:
        query = urllib.parse.quote(fossil_name + " fossil")
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json&srlimit=1"
        req = urllib.request.Request(search_url, headers=headers)
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode())
            results = data.get("query", {}).get("search", [])
            if results:
                page_title = urllib.parse.quote(results[0]["title"].replace(" ", "_"))
                img_url2 = f"https://en.wikipedia.org/w/api.php?action=query&titles={page_title}&prop=pageimages&format=json&pithumbsize=320"
                req2 = urllib.request.Request(img_url2, headers=headers)
                with urllib.request.urlopen(req2, timeout=6) as resp2:
                    idata = json.loads(resp2.read().decode())
                    pages = idata.get("query", {}).get("pages", {})
                    for page in pages.values():
                        thumb = page.get("thumbnail", {})
                        src = thumb.get("source", "")
                        if src:
                            return src
    except Exception:
        pass
    try:
        query2 = urllib.parse.quote(fossil_name)
        commons_url = (
            f"https://commons.wikimedia.org/w/api.php?action=query&list=search"
            f"&srsearch={query2}&srnamespace=6&format=json&srlimit=1"
        )
        req = urllib.request.Request(commons_url, headers=headers)
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode())
            results = data.get("query", {}).get("search", [])
            if results:
                title = results[0]["title"]
                title_enc = urllib.parse.quote(title.replace(" ", "_"))
                info_url = (
                    f"https://commons.wikimedia.org/w/api.php?action=query"
                    f"&titles={title_enc}&prop=imageinfo&iiprop=url&format=json"
                )
                req2 = urllib.request.Request(info_url, headers=headers)
                with urllib.request.urlopen(req2, timeout=6) as resp2:
                    idata = json.loads(resp2.read().decode())
                    pages = idata.get("query", {}).get("pages", {})
                    for page in pages.values():
                        infos = page.get("imageinfo", [])
                        if infos:
                            return infos[0].get("url", "")
    except Exception:
        pass

    return ""


def search_fossil_by_name(fossil_name):
    if not fossil_name or not fossil_name.strip():
        return SEARCH_PLACEHOLDER_HTML

    try:
        interpret = client.chat.completions.create(
            messages=[{"role": "user", "content": (
                f'The user is searching for a fossil and typed: "{fossil_name}". '
                f"This may contain typos, abbreviations, or be loosely worded. "
                f"Identify the single most likely fossil or prehistoric artifact they are looking for. "
                f"Reply with ONLY the correct fossil name, nothing else. No punctuation, no explanation."
            )}],
            model="llama-3.1-8b-instant",
        )
        resolved_name = interpret.choices[0].message.content.strip().strip(".")
    except Exception as e:
        return f"<div style='color:#ef4444;padding:1rem;'>Failed to interpret search: {e}</div>"

    prompt = (
        f"Give a thorough description on {resolved_name}. "
        f"Put it in the format: a general one-paragraph description, "
        f"then a description of physical characteristics and composition, "
        f"followed by a list of uses and significance of the artifact. "
        f"(Don't include sources)"
    )
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
        )
        description = chat_completion.choices[0].message.content
    except Exception as e:
        return f"<div style='color:#ef4444;padding:1rem;'>Failed to generate description: {e}</div>"

    paragraphs = description.strip().split("\n")
    formatted = ""
    in_list = False
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if p.startswith("-") or p.startswith("*") or p.startswith("•"):
            if not in_list:
                formatted += "<ul style='padding-left:1.4rem;margin:0 0 1rem 0;'>"
                in_list = True
            formatted += f"<li style='margin-bottom:6px;color:#C5C7C4;'>{p[1:].strip()}</li>"
        else:
            if in_list:
                formatted += "</ul>"
                in_list = False
            formatted += f"<p style='margin:0 0 1rem 0;line-height:1.8;color:#C5C7C4;'>{p}</p>"
    if in_list:
        formatted += "</ul>"

    correction_note = ""
    if resolved_name.lower() != fossil_name.strip().lower():
        correction_note = f'<div class="correction-note">Showing results for: <span style="color:#c8922a;font-weight:700;">{resolved_name}</span></div>'
    img_url = get_reference_image(resolved_name)
    if img_url:
        img_html = f"""
        <div style="margin-top:1.5rem;">
            <div style="font-size:0.65rem;letter-spacing:0.14em;text-transform:uppercase;
                        color:#c8922a;font-weight:700;margin-bottom:0.75rem;
                        display:flex;align-items:center;gap:8px;">
                Reference Image
                <span style="flex:1;height:1px;background:linear-gradient(90deg,#c8922a55,transparent);display:block;"></span>
            </div>
            <div style="border-radius:12px;overflow:hidden;border:1px solid #2a2518;
                        max-width:480px;position:relative;">
                <img src="{img_url}"
                     alt="{resolved_name}"
                     referrerpolicy="no-referrer"
                     style="width:100%;max-height:320px;object-fit:cover;display:block;
                            filter:sepia(0.15) contrast(1.05);transition:transform 0.4s,filter 0.4s;"
                     onmouseover="this.style.transform='scale(1.03)';this.style.filter='none'"
                     onmouseout="this.style.transform='scale(1)';this.style.filter='sepia(0.15) contrast(1.05)'"
                />
                <div style="position:absolute;bottom:0;left:0;right:0;
                            background:linear-gradient(transparent,rgba(0,0,0,0.75));
                            padding:0.5rem 0.8rem;font-size:0.68rem;color:#aaa;font-style:italic;">
                    Reference image via Wikimedia
                </div>
            </div>
        </div>"""
    else:
        img_html = ""

    return f"""
    <style>
    @keyframes slideUp {{
        from {{ opacity:0; transform:translateY(24px); }}
        to   {{ opacity:1; transform:translateY(0); }}
    }}
    @keyframes shimmer {{
        0%   {{ background-position: -200% center; }}
        100% {{ background-position:  200% center; }}
    }}
    .fossil-result {{ animation: slideUp 0.5s cubic-bezier(.16,1,.3,1) both; }}
    .fossil-title {{
        font-size:1.6rem; font-weight:800; letter-spacing:-0.02em;
        background: linear-gradient(90deg, #e0c080, #c8922a, #e0c080);
        background-size: 200% auto;
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
        animation: shimmer 3s linear infinite;
        margin-bottom:0.25rem;
    }}
    .correction-note {{
        font-size:0.75rem; color:#666; margin-bottom:1rem;
        padding:4px 10px; background:#1a1a0a; border-radius:99px;
        display:inline-block; border:1px solid #333;
    }}
    .section-heading {{
        font-size:0.7rem; letter-spacing:0.12em; text-transform:uppercase;
        color:#c8922a; font-weight:700; margin:1.5rem 0 0.5rem;
        display:flex; align-items:center; gap:8px;
    }}
    .section-heading::after {{
        content:''; flex:1; height:1px; background:linear-gradient(90deg,#c8922a33,transparent);
    }}
    </style>
    <div class="fossil-result" style="background:linear-gradient(135deg,#141410,#0f0f0f);
         border:1px solid #2a2518; border-radius:16px; padding:1.75rem;
         font-family:'Georgia',serif; box-shadow:0 20px 60px rgba(0,0,0,0.5),
         inset 0 1px 0 rgba(200,146,42,0.1);">
        <div class="fossil-title">{resolved_name}</div>
        {correction_note}
        <div style="font-size:0.85rem; line-height:1.8;">{formatted}</div>
        {img_html}
    </div>
    """


SEARCH_PLACEHOLDER_HTML = """
<div style="text-align:center;padding:3rem 1rem;font-family:sans-serif;">
    <div style="font-size:2.5rem;margin-bottom:1rem;opacity:0.3;">🦕</div>
    <div style="color:#555;font-size:0.9rem;">Enter a fossil name above to generate a detailed description</div>
</div>
"""


def draw_fixed_label(img, label, confidence):
    label_text = label.title()
    conf_text = f"{confidence*100:.2f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_scale, conf_scale = 0.9, 0.7
    label_thick, conf_thick = 2, 1
    (lw, lh), _ = cv2.getTextSize(label_text, font, label_scale, label_thick)
    (cw, ch), _ = cv2.getTextSize(conf_text, font, conf_scale, conf_thick)
    pad = 12
    x, y = 35, 65
    box_w = max(lw, cw) + pad * 2
    box_h = lh + ch + pad * 3
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y - lh - pad), (x + box_w, y + box_h - lh), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    cv2.putText(img, label_text, (x + pad, y), font, label_scale, (255, 255, 255), label_thick, cv2.LINE_AA)
    cv2.putText(img, conf_text, (x + pad, y + lh + pad), font, conf_scale, (144, 238, 144), conf_thick, cv2.LINE_AA)
    cv2.rectangle(img, (x + pad, y + lh + pad + 5),
                  (x + pad + int(confidence * (box_w - pad * 2)), y + lh + pad + 8), (0, 200, 80), -1)
    return img


def process_image(frame):
    if frame is None:
        return None, "No image uploaded."
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite("temp.jpg", img)
    try:
        result = CLIENT.infer("temp.jpg", model_id=MODEL_ID)
    except Exception as e:
        return None, f"Error: {e}"
    predictions = result.get("predictions", [])
    if not predictions:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "No fossils detected."
    overlay = img.copy()
    if "x" in predictions[0] and "y" in predictions[0]:
        for pred in predictions:
            x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
            x1, y1 = max(0, x - w // 2), max(0, y - h // 2)
            x2, y2 = min(img.shape[1], x + w // 2), min(img.shape[0], y + h // 2)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 80), 3)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 5000:
                cv2.drawContours(overlay, [cnt], -1, (0, 200, 80), 3)
    merged = cv2.addWeighted(overlay, 0.8, img, 0.2, 0)
    label, confidence = predictions[0]["class"], predictions[0]["confidence"]
    user = (f"Give a thorough description on {label}. Put it in the format: "
            f"a general one-paragraph description, then a description of physical "
            f"characteristics and composition, followed by a list of uses and "
            f"significance of the artifact. (Don't include sources)")
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": user}],
            model="llama-3.1-8b-instant",
        )
        response = chat_completion.choices[0].message.content
    except Exception as e:
        response = f"Failed to generate description: {e}"
    final_img = draw_fixed_label(merged.copy(), label, confidence)
    output_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    catalog_img_path = f"catalog_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{label}.jpg"
    cv2.imwrite(catalog_img_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
    add_to_catalog(label, confidence, response, catalog_img_path)
    info = f"**{label.title()}** — {confidence*100:.2f}% Confidence\n\n**Fossil Description:**\n{response}"
    return output_img, info


def build_histogram_html(catalog):
    counts = Counter(e["name"] for e in catalog)
    if not counts:
        return ""
    max_val = max(counts.values())
    bar_colors = ["#c8922a", "#22c55e", "#3b82f6", "#8b5cf6", "#06b6d4", "#f97316", "#ec4899", "#ef4444"]
    bars = ""
    for i, (sp, val) in enumerate(sorted(counts.items(), key=lambda x: -x[1])):
        pct = int((val / max_val) * 100)
        color = bar_colors[i % len(bar_colors)]
        bars += f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
            <div style="width:140px;font-size:0.78rem;color:#aaa;text-align:right;
                        white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="{sp}">{sp}</div>
            <div style="flex:1;background:#1e1e1e;border-radius:4px;height:20px;overflow:hidden;position:relative;">
                <div style="width:{pct}%;background:linear-gradient(90deg,{color},{color}aa);
                             height:100%;border-radius:4px;
                             transition:width 0.8s cubic-bezier(.16,1,.3,1);
                             box-shadow:0 0 8px {color}66;"></div>
            </div>
            <div style="width:24px;font-size:0.82rem;font-weight:700;color:{color};">{val}</div>
        </div>"""
    return f"""
    <div style="background:linear-gradient(135deg,#141410,#0f0f0f);border:1px solid #2a2518;
                border-radius:14px;padding:1.25rem;margin-bottom:1.5rem;">
        <div style="font-size:0.65rem;letter-spacing:0.15em;text-transform:uppercase;
                    color:#c8922a;font-weight:700;margin-bottom:1rem;">Detections by Species</div>
        {bars}
    </div>"""


def get_stats_html():
    catalog = load_catalog()
    if not catalog:
        return """<div style='text-align:center;padding:2rem;color:#444;font-family:sans-serif;'>
            <div style='font-size:2rem;margin-bottom:0.5rem;'>🦴</div>
            No fossils cataloged yet. Scan one to get started.
        </div>"""
    total = len(catalog)
    unique = len(set(e["name"] for e in catalog))
    avg_conf = round(sum(e["confidence"] for e in catalog) / total, 1)
    reviewed = [e for e in catalog if e.get("verified") is not None]
    correct = [e for e in catalog if e.get("verified") is True]
    if reviewed:
        acc = round(len(correct) / len(reviewed) * 100, 1)
        acc_val, acc_sub = f"{acc}%", f"{len(correct)}/{len(reviewed)} correct"
    else:
        acc_val, acc_sub = "—", "mark entries below"
    histogram = build_histogram_html(catalog)

    def stat_card(value, label, color, bg):
        return f"""
        <div style="background:{bg};border:1px solid {color}33;border-radius:14px;padding:1.1rem;
                    text-align:center;position:relative;overflow:hidden;
                    box-shadow:0 4px 20px {color}11;">
            <div style="position:absolute;top:-10px;right:-10px;width:60px;height:60px;
                        background:{color}11;border-radius:50%;"></div>
            <div style="font-size:1.7rem;font-weight:800;color:{color};font-family:monospace;
                        letter-spacing:-0.03em;">{value}</div>
            <div style="font-size:0.72rem;color:#666;margin-top:3px;text-transform:uppercase;
                        letter-spacing:0.08em;">{label}</div>
        </div>"""

    return f"""
    <div style="font-family:sans-serif;padding:0.5rem 0;">
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:1.5rem;">
            {stat_card(total, "Total Scans", "#3b82f6", "#0a0f1a")}
            {stat_card(unique, "Unique Species", "#22c55e", "#0a1a0f")}
            {stat_card(f"{avg_conf}%", "Avg Confidence", "#c8922a", "#1a1208")}
            {stat_card(acc_val, "Verified Accuracy", "#8b5cf6", "#120a1a")}
        </div>
        {histogram}
    </div>"""


def build_entries_html(search_query=""):
    catalog = load_catalog()
    if not catalog:
        return ""
    query = search_query.strip().lower()
    filtered = [e for e in catalog if query in e["name"].lower()] if query else catalog
    if not filtered:
        return f"<div style='text-align:center;padding:1.5rem;color:#555;font-family:sans-serif;'>No results for &quot;{search_query}&quot;</div>"
    rows = ""
    for i, entry in enumerate(reversed(filtered)):
        conf_color = "#22c55e" if entry["confidence"] >= 75 else "#c8922a" if entry["confidence"] >= 50 else "#ef4444"
        desc_preview = entry["description"][:280] + "..." if len(entry["description"]) > 280 else entry["description"]
        verified = entry.get("verified")
        if verified is True:
            badge = '<span style="background:#0a1a0f;color:#22c55e;font-size:0.72rem;font-weight:700;padding:3px 10px;border-radius:99px;border:1px solid #22c55e44;">✓ Correct</span>'
        elif verified is False:
            badge = '<span style="background:#1a0a0a;color:#ef4444;font-size:0.72rem;font-weight:700;padding:3px 10px;border-radius:99px;border:1px solid #ef444444;">✗ Incorrect</span>'
        else:
            badge = '<span style="background:#1a1a1a;color:#666;font-size:0.72rem;font-weight:700;padding:3px 10px;border-radius:99px;border:1px solid #333;">? Unreviewed</span>'

        delay = i * 0.05
        rows += f"""
        <div style="background:linear-gradient(135deg,#141410,#0f0f0f);border:1px solid #2a2518;
                    border-radius:14px;padding:1.25rem;margin-bottom:0.75rem;
                    animation:entryFadeIn 0.4s {delay}s both cubic-bezier(.16,1,.3,1);
                    transition:border-color 0.2s,box-shadow 0.2s;"
             onmouseover="this.style.borderColor='#c8922a44';this.style.boxShadow='0 8px 32px rgba(200,146,42,0.1)'"
             onmouseout="this.style.borderColor='#2a2518';this.style.boxShadow='none'">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;margin-bottom:0.6rem;">
                <div>
                    <span style="font-size:1rem;font-weight:700;color:#e0c080;font-family:Georgia,serif;">
                        #{entry['id']} — {entry['name']}
                    </span>
                    <div style="font-size:0.72rem;color:#555;margin-top:2px;font-family:monospace;">{entry['timestamp']}</div>
                </div>
                <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;">
                    <span style="background:{conf_color}18;color:{conf_color};font-size:0.72rem;font-weight:700;
                                 padding:3px 10px;border-radius:99px;border:1px solid {conf_color}44;">
                        {entry['confidence']}%</span>
                    {badge}
                </div>
            </div>
            <p style="font-size:0.82rem;color:#888;line-height:1.65;margin:0;font-family:Georgia,serif;">{desc_preview}</p>
        </div>"""
    return f"""
    <style>
    @keyframes entryFadeIn {{
        from {{ opacity:0; transform:translateY(12px); }}
        to   {{ opacity:1; transform:translateY(0); }}
    }}
    </style>
    <div style="font-family:sans-serif;">{rows}</div>"""

HEADER_HTML = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@700;900&family=Crimson+Pro:ital,wght@0,300;0,400;1,300&display=swap');

* { box-sizing: border-box; }

#fossil-header {
    position: relative;
    width: 100%;
    height: 220px;
    overflow: hidden;
    border-radius: 18px;
    margin-bottom: 1.5rem;
    border: 1px solid #2a2518;
}

#particle-canvas {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
}

.header-content {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    z-index: 2;
}

.header-title {
    font-family: 'Cinzel', serif;
    font-size: 2.6rem;
    font-weight: 900;
    letter-spacing: 0.08em;
    background: linear-gradient(180deg, #f5e090 0%, #c8922a 50%, #8a5c12 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: none;
    margin: 0;
    line-height: 1;
}

.header-sub {
    font-family: 'Crimson Pro', serif;
    font-style: italic;
    font-size: 1rem;
    color: #888;
    margin-top: 0.4rem;
    letter-spacing: 0.04em;
}

.scan-line {
    position: absolute;
    left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #c8922a88, transparent);
    animation: scanDown 4s ease-in-out infinite;
    z-index: 3;
}

@keyframes scanDown {
    0%   { top: 0%;   opacity: 0; }
    10%  { opacity: 1; }
    90%  { opacity: 1; }
    100% { top: 100%; opacity: 0; }
}

.orb-container {
    position: absolute;
    right: 60px;
    top: 50%;
    transform: translateY(-50%);
    width: 120px;
    height: 120px;
    perspective: 400px;
}

.orb {
    width: 100%;
    height: 100%;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 35%,
        #3a2a10 0%, #1a1008 40%, #0a0804 100%);
    border: 1px solid #c8922a44;
    box-shadow:
        0 0 40px rgba(200,146,42,0.2),
        0 0 80px rgba(200,146,42,0.05),
        inset 0 0 30px rgba(0,0,0,0.8);
    animation: orbSpin 12s linear infinite, orbPulse 3s ease-in-out infinite;
    position: relative;
    transform-style: preserve-3d;
}

.orb::before {
    content: '';
    position: absolute;
    inset: 8px;
    border-radius: 50%;
    border: 1px solid #c8922a22;
    animation: orbSpin 8s linear infinite reverse;
}

.orb::after {
    content: '🦕';
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2.8rem;
    filter: drop-shadow(0 0 12px rgba(200,146,42,0.6));
    animation: orbSpin 12s linear infinite reverse;
}

@keyframes orbSpin {
    from { transform: rotateY(0deg) rotateX(10deg); }
    to   { transform: rotateY(360deg) rotateX(10deg); }
}

@keyframes orbPulse {
    0%, 100% { box-shadow: 0 0 40px rgba(200,146,42,0.2), 0 0 80px rgba(200,146,42,0.05), inset 0 0 30px rgba(0,0,0,0.8); }
    50%       { box-shadow: 0 0 60px rgba(200,146,42,0.35), 0 0 120px rgba(200,146,42,0.1), inset 0 0 30px rgba(0,0,0,0.8); }
}

/* Ring around orb */
.orb-ring {
    position: absolute;
    inset: -15px;
    border-radius: 50%;
    border: 1px solid transparent;
    border-top-color: #c8922a66;
    border-bottom-color: #c8922a22;
    animation: ringRotate 6s linear infinite;
}

.orb-ring-2 {
    position: absolute;
    inset: -25px;
    border-radius: 50%;
    border: 1px dashed #c8922a22;
    animation: ringRotate 10s linear infinite reverse;
}

@keyframes ringRotate {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}

/* Corner decorations */
.corner {
    position: absolute;
    width: 40px;
    height: 40px;
    opacity: 0.4;
}
.corner-tl { top: 16px; left: 16px; border-top: 2px solid #c8922a; border-left: 2px solid #c8922a; }
.corner-tr { top: 16px; right: 16px; border-top: 2px solid #c8922a; border-right: 2px solid #c8922a; }
.corner-bl { bottom: 16px; left: 16px; border-bottom: 2px solid #c8922a; border-left: 2px solid #c8922a; }
.corner-br { bottom: 16px; right: 16px; border-bottom: 2px solid #c8922a; border-right: 2px solid #c8922a; }
</style>

<div id="fossil-header">
    <canvas id="particle-canvas"></canvas>
    <div class="scan-line"></div>
    <div class="corner corner-tl"></div>
    <div class="corner corner-tr"></div>
    <div class="corner corner-bl"></div>
    <div class="corner corner-br"></div>

    <div class="header-content">
        <h1 class="header-title">FOSSIL SCANNER</h1>
        <p class="header-sub">Identify, search, and catalog fossils in this three-in-one multi-use tool.</p>
    </div>

    <div class="orb-container">
        <div class="orb-ring"></div>
        <div class="orb-ring-2"></div>
        <div class="orb"></div>
    </div>
</div>

"""

# The particle <script> can't live inside gr.HTML (browsers don't run scripts
# injected via innerHTML), so it's passed to gr.HTML(js_on_load=...) instead,
# which Gradio executes when the component renders.
HEADER_JS = """
(function() {
    const canvas = document.getElementById('particle-canvas');
    if (!canvas) return;
    if (window.__fossilParticlesRaf) cancelAnimationFrame(window.__fossilParticlesRaf);
    const ctx = canvas.getContext('2d');

    function resize() {
        canvas.width  = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
    }
    resize();
    window.onresize = resize;

    // Particles
    const particles = [];
    const N = 80;
    for (let i = 0; i < N; i++) {
        particles.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            r: Math.random() * 1.5 + 0.3,
            vx: (Math.random() - 0.5) * 0.3,
            vy: (Math.random() - 0.5) * 0.3,
            alpha: Math.random() * 0.5 + 0.1,
            color: Math.random() > 0.5 ? '#c8922a' : '#e0c080',
        });
    }

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Background gradient
        const grad = ctx.createRadialGradient(
            canvas.width * 0.3, canvas.height * 0.5, 0,
            canvas.width * 0.3, canvas.height * 0.5, canvas.width * 0.6
        );
        grad.addColorStop(0, '#1a1208');
        grad.addColorStop(1, '#080808');
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw connections
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 80) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(200,146,42,${0.08 * (1 - dist / 80)})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }

        // Draw particles
        particles.forEach(p => {
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = p.color;
            ctx.globalAlpha = p.alpha;
            ctx.fill();
            ctx.globalAlpha = 1;

            p.x += p.vx;
            p.y += p.vy;
            if (p.x < 0 || p.x > canvas.width)  p.vx *= -1;
            if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
        });

        window.__fossilParticlesRaf = requestAnimationFrame(draw);
    }
    draw();
})();
"""


def get_stats_html():
    catalog = load_catalog()
    if not catalog:
        return """<div style='text-align:center;padding:2rem;color:#444;font-family:sans-serif;'>
            <div style='font-size:2rem;margin-bottom:0.5rem;'>🦴</div>
            No fossils cataloged yet. Scan one to get started.
        </div>"""
    total = len(catalog)
    unique = len(set(e["name"] for e in catalog))
    avg_conf = round(sum(e["confidence"] for e in catalog) / total, 1)
    reviewed = [e for e in catalog if e.get("verified") is not None]
    correct = [e for e in catalog if e.get("verified") is True]
    if reviewed:
        acc = round(len(correct) / len(reviewed) * 100, 1)
        acc_val, acc_sub = f"{acc}%", f"{len(correct)}/{len(reviewed)} correct"
    else:
        acc_val, acc_sub = "—", "mark entries below"
    histogram = build_histogram_html(catalog)

    def stat_card(value, label, color, bg):
        return f"""
        <div style="background:{bg};border:1px solid {color}33;border-radius:14px;padding:1.1rem;
                    text-align:center;position:relative;overflow:hidden;">
            <div style="position:absolute;top:-10px;right:-10px;width:60px;height:60px;
                        background:{color}0a;border-radius:50%;"></div>
            <div style="font-size:1.7rem;font-weight:800;color:{color};font-family:monospace;">{value}</div>
            <div style="font-size:0.65rem;color:#555;margin-top:3px;text-transform:uppercase;letter-spacing:0.1em;">{label}</div>
        </div>"""

    return f"""
    <div style="font-family:sans-serif;padding:0.5rem 0;">
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:1.5rem;">
            {stat_card(total, "Total Scans", "#3b82f6", "#0a0f1a")}
            {stat_card(unique, "Unique Species", "#22c55e", "#0a1a0f")}
            {stat_card(f"{avg_conf}%", "Avg Confidence", "#c8922a", "#1a1208")}
            {stat_card(acc_val, "Verified Accuracy", "#8b5cf6", "#120a1a")}
        </div>
        {histogram}
    </div>"""


with gr.Blocks(
    title="Fossil Scanner"
) as app:

    gr.HTML(HEADER_HTML, js_on_load=HEADER_JS)

    with gr.Tabs():
        with gr.Tab("⬡ Scanner"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="numpy", label="Upload Fossil Image")
                    scan_btn = gr.Button("⬡  Scan Fossil", variant="primary")
                with gr.Column():
                    image_output = gr.Image(type="numpy", label="Detection Result")
                    info_output = gr.Markdown(label="Analysis")
            scan_btn.click(fn=process_image, inputs=image_input, outputs=[image_output, info_output])

        with gr.Tab("Search"):
            gr.Markdown("Fossil Base")
            gr.Markdown("Type the name of any fossil and get a description and reference image of said fossil.")
            with gr.Row():
                fossil_search_input = gr.Textbox(
                    placeholder="e.g.  ammonite, coral",
                    label="Search Query",
                    scale=5,
                )
                fossil_search_btn = gr.Button("Search", variant="primary", scale=1)
            fossil_search_output = gr.HTML(value=SEARCH_PLACEHOLDER_HTML)
            fossil_search_btn.click(fn=search_fossil_by_name, inputs=fossil_search_input, outputs=fossil_search_output)
            fossil_search_input.submit(fn=search_fossil_by_name, inputs=fossil_search_input, outputs=fossil_search_output)

        with gr.Tab("⬡ Catalog"):
            gr.Markdown("Scan History")
            with gr.Row():
                refresh_btn = gr.Button("↺  Refresh", variant="secondary")
                clear_btn   = gr.Button("⊘  Clear All", variant="stop")

            stats_display = gr.HTML(value=get_stats_html())

            gr.Markdown("Filter Entries")
            with gr.Row():
                search_input = gr.Textbox(placeholder="Filter by name...", label="Search Catalog", scale=5)
                search_btn   = gr.Button("Filter", variant="primary", scale=1)

            entries_html = gr.HTML(value=build_entries_html())

            gr.Markdown("Verify Prediction")
            gr.Markdown("Enter the entry ID and verify whether the AI identified it correctly.")
            with gr.Row():
                entry_id_input = gr.Number(label="Entry ID", precision=0, minimum=1)
                correct_btn    = gr.Button("✓  Correct", variant="primary")
                incorrect_btn  = gr.Button("✗  Incorrect", variant="stop")

            def do_correct(eid, query=""):
                if eid is None: return get_stats_html(), build_entries_html(query)
                mark_entry(int(eid), True)
                return get_stats_html(), build_entries_html(query)

            def do_incorrect(eid, query=""):
                if eid is None: return get_stats_html(), build_entries_html(query)
                mark_entry(int(eid), False)
                return get_stats_html(), build_entries_html(query)

            def do_refresh(query=""):
                return get_stats_html(), build_entries_html(query)

            def do_clear():
                save_catalog([])
                return get_stats_html(), build_entries_html()

            def do_search(query):
                return build_entries_html(query)

            correct_btn.click(fn=do_correct,   inputs=[entry_id_input, search_input], outputs=[stats_display, entries_html])
            incorrect_btn.click(fn=do_incorrect, inputs=[entry_id_input, search_input], outputs=[stats_display, entries_html])
            refresh_btn.click(fn=lambda q: do_refresh(q), inputs=search_input, outputs=[stats_display, entries_html])
            clear_btn.click(fn=do_clear, outputs=[stats_display, entries_html])
            search_input.change(fn=do_search, inputs=search_input, outputs=entries_html)
            search_btn.click(fn=do_search,  inputs=search_input, outputs=entries_html)

        scan_btn.click(fn=lambda: do_refresh(""), outputs=[stats_display, entries_html])

if __name__ == "__main__":
    # ssr_mode=False avoids the experimental SSR path (and its teardown log
    # noise) on Hugging Face Spaces. In Gradio 6, theme/css are launch() args.
    app.launch(show_error=True, share=False, ssr_mode=False, theme=custom_theme, css=CUSTOM_CSS)