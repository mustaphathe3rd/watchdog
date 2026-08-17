# =============================================================================
# seed_sources.py  —  Curated Seed Sources for Dataset Collection
# =============================================================================
# PURPOSE:
#   Starting point of domain-level sources for §3.9's three dataset
#   categories (Table 3.4). This is a SEED list, not a finished dataset —
#   you (the student) must browse each source, find specific working
#   article/video URLs, and add them to the SPECIFIC_URLS lists below.
# =============================================================================

from watchdog.dataset_manager import (
    CATEGORY_TELECOM_FMCG_ADS,
    CATEGORY_POLITICAL_MEDIA,
    CATEGORY_TEXTUAL_CORPUS,
)

# =============================================================================
# CATEGORY 1: Nigerian Telecom/FMCG Ads (video)
# =============================================================================
TELECOM_FMCG_DOMAINS = [
    "https://www.mtn.ng",
    "https://www.youtube.com/@MTNNigeria",
]

TELECOM_FMCG_SPECIFIC_URLS = [
    {"url": "https://www.instagram.com/p/DKRurzYinvb/",
     "publisher": "The Mob Film Company", "expected_label": 1,
     "notes": ("SELF-DISCLOSED by the producer as 'Nigeria's first fully "
               "AI-generated TV commercial' for Golden Terra — this is the "
               "campaign referenced in §3.9/§2.3.1. label=1 is producer-"
               "confirmed, not inferred. Facebook download requires cookies "
               "with c_user/xs present — currently failing without them.")},

    {"url": "https://www.instagram.com/p/DZe3lLlNT7H/",
     "publisher": "MTN Nigeria", "expected_label": 1,
     "notes": "verified AI-avatar ad — downloaded and confirmed synthetic by student"},
    {"url": "https://www.instagram.com/p/DZcf5rBts_i/",
     "publisher": "MTN Nigeria", "expected_label": 1,
     "notes": "verified AI-avatar ad — downloaded and confirmed synthetic by student"},
    {"url": "https://www.instagram.com/reel/DZ2l85zEWTa/",
     "publisher": "MTN Nigeria", "expected_label": 1,
     "notes": "verified AI-avatar ad — downloaded and confirmed synthetic by student"},
    {"url": "https://www.instagram.com/reel/DZ0PRWCjErX/",
     "publisher": "MTN Nigeria", "expected_label": 1,
     "notes": "verified AI-avatar ad — downloaded and confirmed synthetic by student"},
    {"url": "https://www.instagram.com/reel/DXb49G8DKJc/",
     "publisher": "MTN Nigeria", "expected_label": 1,
     "notes": "verified AI-avatar ad — downloaded and confirmed synthetic by student"},
    {"url": "https://www.instagram.com/reel/DW4X0oUDRJi/",
     "publisher": "MTN Nigeria", "expected_label": 1,
     "notes": "verified AI-avatar ad — downloaded and confirmed synthetic by student"},
    {"url": "https://www.instagram.com/reel/DPV2tNLDWmM/",
     "publisher": "MTN Nigeria", "expected_label": 1,
     "notes": "verified AI-avatar ad — downloaded and confirmed synthetic by student"},

    # ── New finds (batch 2) ──────────────────────────────────────────────────
    {"url": "https://x.com/instablog9ja/status/2074511313246568478/video/1",
     "publisher": "iLOTBET", "expected_label": 1,
     "notes": ("INDEPENDENTLY CORROBORATED — see the leadership.ng article "
               "logged under TEXTUAL_CORPUS below, which describes this exact "
               "ad as made of 'the cheapest, ugliest, most robotic AI graphics "
               "imaginable.' Different case profile from the MTN/Golden Terra "
               "finds: LOW production-quality AI ad rather than a polished "
               "avatar, useful for dataset diversity. Same Facebook auth "
               "problem applies — see notes on the Golden Terra entry above.")},

    {"url": "https://www.youtube.com/watch?v=CXSjHmxsVe4&t=3s",
     "publisher": "Supramult", "expected_label": None,
     "notes": ("Claimed AI-avatar ad — NEEDS MANUAL VERIFICATION before "
               "labelling (no independent corroboration found yet, unlike "
               "iLOTBET/Golden Terra above). YouTube should download cleanly "
               "without cookies.")},

    {"url": "https://x.com/iamtastemaker/status/2006736194684739766",
     "publisher": "Tastemaker", "expected_label": None,
     "notes": ("Claimed AI-avatar ad — NEEDS MANUAL VERIFICATION. X/Twitter "
               "has tightened anti-scraping significantly — this may also "
               "require cookie auth even for public posts. Try without "
               "cookies first; if it fails with a login-required-style "
               "error, the same YTDLP_COOKIES_FILE approach used for "
               "Instagram should work, using an X-domain cookie export.")},

    {"url": "https://www.instagram.com/p/DXJ1dK9lTvT/?hl=en",
     "publisher": "Bet9ja", "expected_label": None,
     "notes": "claimed AI-avatar ad — NEEDS MANUAL VERIFICATION before labelling"},
    {"url": "https://www.instagram.com/p/DW3yEwljKtS/?hl=en",
     "publisher": "Bet9ja", "expected_label": None,
     "notes": "claimed AI-avatar ad — NEEDS MANUAL VERIFICATION before labelling"},
    {"url": "https://www.instagram.com/p/DV0S-8AiiZv/?hl=en",
     "publisher": "Bet9ja", "expected_label": None,
     "notes": "claimed AI-avatar ad — NEEDS MANUAL VERIFICATION before labelling"},

    # ── Authentic (label=0) expansion — fixes 14 synthetic / 0 authentic
    # imbalance in this category, needed so the CV engine can be evaluated
    # for false positives (NFR-01, NFR-03). All below are YouTube (no
    # cookie auth needed) and predate or sit outside the AI-avatar-ad era
    # described in §2.3.1/§3.9. Every entry was verified directly from the
    # video's own page (title/description naming a real presenter, or the
    # video's own thumbnail showing a clear human face) via yt-dlp metadata
    # — never inferred from the URL or channel name alone.
    {"url": "https://www.youtube.com/watch?v=QUVdnzC19oU",
     "publisher": "MTN Nigeria", "expected_label": 0,
     "notes": ("VERIFIED via yt-dlp metadata: uploaded by MTN NIGERIA's "
               "official channel, 2013-04-22. Title/description name "
               "Nigerian comedian SAKA by name (\"Saka sef don port\") as "
               "the on-camera talent — live-action celebrity-endorsement "
               "ad, well before any AI-avatar campaign.")},
    {"url": "https://www.youtube.com/watch?v=hzMt7Nm6BIc",
     "publisher": "MTN Nigeria", "expected_label": 0,
     "notes": ("VERIFIED via yt-dlp metadata: \"Falz - MTN Pulse Ad\", "
               "uploaded 2016-05-16 by Falz's own artist channel (not "
               "MTN's), but title and content confirm it's an MTN Pulse "
               "commercial starring Nigerian musician Falz.")},
    {"url": "https://www.youtube.com/watch?v=PFJ-p00-AtQ",
     "publisher": "Glo (Globacom)", "expected_label": 0,
     "notes": ("VERIFIED via yt-dlp metadata: \"Glo Commercial A Day in "
               "the life of Korede Bello\", uploaded 2015-10-04 on Korede "
               "Bello's own channel, description \"CheckOut my Glo "
               "Advert.\" — Nigerian singer confirmed as on-camera talent.")},
    {"url": "https://www.youtube.com/watch?v=TRPQz7AF27g",
     "publisher": "Glo (Globacom)", "expected_label": 0,
     "notes": ("VERIFIED via yt-dlp metadata: \"Di'Ja Stars in Glo "
               "Commercial\", uploaded 2015-10-01 on Mavin Records' "
               "channel — Nigerian singer Di'Ja confirmed as on-camera "
               "talent for a Glo data-plan ad.")},
    {"url": "https://www.youtube.com/watch?v=e-DNQMaIYZE",
     "publisher": "Airtel Nigeria", "expected_label": 0,
     "notes": ("VERIFIED via yt-dlp metadata: \"Live Limitless\", Airtel "
               "Nigeria's official channel, 2024-11-05. Description names "
               "Farooq Oreagba (\"King of Steeze\") as the ad's subject — "
               "confirmed real-person celebrity ambassador, not an avatar.")},
    {"url": "https://www.youtube.com/watch?v=rwSjs8MsjdM",
     "publisher": "Indomie Nigeria", "expected_label": 0,
     "notes": ("VERIFIED via yt-dlp metadata: \"Mama Do Good\" (Pidgin), "
               "Indomie Nigeria's official channel, 2010-10-06. "
               "Description: \"highlighting the bond between mother and "
               "child\" — confirmed live-action mother/child cast, part "
               "of the classic pre-AI \"Mum like no other\" campaign.")},
    {"url": "https://www.youtube.com/watch?v=5D33aQTebMg",
     "publisher": "Indomie Nigeria", "expected_label": 0,
     "notes": ("VERIFIED via yt-dlp metadata: same \"Mum like no other\" "
               "campaign as the Pidgin cut above, English-language "
               "version, Indomie Nigeria official channel, 2010-10-06. "
               "Same mother/child on-camera cast confirmed via description.")},
    {"url": "https://www.youtube.com/watch?v=uGhu3jDKjyk",
     "publisher": "Indomie Nigeria", "expected_label": 0,
     "notes": ("VERIFIED via yt-dlp metadata: \"Iya Ewe\" (Yoruba dub), "
               "same 2010 \"Mum like no other\" campaign/channel/cast as "
               "the two entries above — included for language/regional "
               "diversity within the same verified live-action campaign.")},
    {"url": "https://www.youtube.com/watch?v=BOZZrluxEZE",
     "publisher": "Indomie Nigeria", "expected_label": 0,
     "notes": ("VERIFIED via yt-dlp metadata: \"Mama Yara\" (Hausa dub), "
               "same 2010 \"Mum like no other\" campaign/channel/cast as "
               "the entries above — included for language/regional "
               "diversity within the same verified live-action campaign.")},
    {"url": "https://www.youtube.com/watch?v=hbbiaM7-LDM",
     "publisher": "Dangote Cement", "expected_label": 0,
     "notes": ("VERIFIED via yt-dlp metadata: Dangote Cement Nigeria "
               "PLC's official channel, 2026-01-17. Description: \"In "
               "this interview...Fatima Aliko-Dangote (Group Executive "
               "Director, Commercial) shares...\" — confirmed on-camera "
               "talking-head interview with a named real executive.")},
    {"url": "https://www.youtube.com/watch?v=cpUVlyvQ8i0",
     "publisher": "Channels Television", "expected_label": 0,
     "notes": ("VERIFIED VIA THUMBNAIL (description was boilerplate "
               "channel copy, no per-video confirmation): downloaded and "
               "visually inspected the video's own YouTube thumbnail — "
               "shows a real person (academic in ceremonial regalia) "
               "speaking at a podium, clear frontal face filling a large "
               "portion of frame. Channels Television official channel, "
               "2026-07-31.")},
    {"url": "https://www.youtube.com/watch?v=7wNx-mjOz2w",
     "publisher": "Channels Television", "expected_label": 0,
     "notes": ("VERIFIED VIA THUMBNAIL: \"Politics Today\" studio panel "
               "segment, Channels Television official channel, "
               "2026-07-31. Thumbnail shows a real studio guest talking "
               "on camera, clear face filling roughly a third of frame. "
               "NOTE: this is a ~28-minute full segment, not a short ad — "
               "much longer than this category's other samples; flag if "
               "that matters for pipeline duration handling.")},
    {"url": "https://www.youtube.com/watch?v=gR7UnNw1xN4",
     "publisher": "Channels Television", "expected_label": 0,
     "notes": ("VERIFIED VIA THUMBNAIL: \"Hard Copy\" studio interview "
               "segment, Channels Television official channel, "
               "2026-07-31. Thumbnail shows a real studio guest talking "
               "on camera, clear face. NOTE: ~27-minute full segment, "
               "same duration caveat as the Politics Today entry above.")},
    {"url": "https://www.youtube.com/watch?v=i8vW5JhNwCY",
     "publisher": "NTA News 24", "expected_label": 0,
     "notes": ("VERIFIED VIA THUMBNAIL: \"Correctional Service budgetary "
               "allocation...Hon. Chinedu Ogah\", NTA News 24 official "
               "channel, 2026-03-26. Thumbnail shows a named lawmaker "
               "seated for a direct-to-camera interview, face large and "
               "centred in frame. NOTE: ~22-minute full segment.")},
    {"url": "https://www.youtube.com/watch?v=R4L0IfLwTZc",
     "publisher": "MTN Nigeria (via Arise News)", "expected_label": 0,
     "notes": ("VERIFIED VIA THUMBNAIL: \"MTN Nigeria Unveils Y'ellotide "
               "Campaign\", Arise News official channel, 2025-11-14. "
               "Thumbnail shows Kolapo Olaleye (captioned \"GM, Financial "
               "Planning, MTN Nigeria\") being interviewed on camera at "
               "the campaign launch — confirmed real named person, clear "
               "face.")},

    # ── Authentic (label=0) Instagram/portrait expansion — added to break a
    # platform/orientation confound found in strata analysis: every prior
    # authentic sample was landscape YouTube while most synthetic samples
    # were portrait Instagram, so a CV-engine detector could learn "portrait
    # Instagram = synthetic" without detecting any actual synthesis artifact.
    # ~45 Instagram candidates (comedians, fintech/bank/FMCG brand accounts,
    # event-coverage reposts) were screened for: 2024–2026, portrait,
    # 20–40s, posted by a Nigerian brand/agency/media-outlet account, and a
    # real human face clearly visible for a meaningful portion. Most failed
    # on duration (comedian skits run 60–130s; brand posts are either short
    # product stings or long event recaps) or turned out to be pure
    # motion-graphics/screen-recording PSAs with no human ever on camera
    # (e.g. rejected FairMoney/PalmPay reels — same failure mode). Only 2
    # survived full verification; search continuing to close the gap to 8–10.
    {"url": "https://www.instagram.com/reel/DB_cu_Ks2Va/",
     "publisher": "Airtel Nigeria (via MediaRoomHub)", "expected_label": 0,
     "notes": ("VERIFIED VIA DOWNLOADED FRAME: posted by MediaRoomHub (a "
               "real Nigerian PR/media outlet), 2024-11-05, 32.7s, "
               "720x1280 portrait. Frame shows a clear, well-lit frontal "
               "shot of a named executive standing directly in front of "
               "an Airtel-branded banner (\"...tel / TO IMAGINE\"), "
               "wearing an Airtel 'Live Limitless' branded shirt with "
               "other staff. Independently corroborated by Tribune "
               "Online, BusinessDay, MarketingEdge Magazine, and "
               "BrandCrunch articles all naming Farooq Oreagba as "
               "Airtel's real (non-avatar) ambassador for this exact "
               "'Live Limitless' launch event — same real-world campaign "
               "as the existing Arise News/MTN corporate-interview entry "
               "above, but PR/media-outlet event coverage rather than the "
               "brand's own produced ad.")},
    {"url": "https://www.instagram.com/bellanaija/reel/DBS_YaOonbi/",
     "publisher": "Star Radler (via BellaNaija)", "expected_label": 0,
     "notes": ("VERIFIED VIA DOWNLOADED FRAME: posted by BellaNaija "
               "(entertainment/events blog), 2024-10-19, 28.8s, 360x640 "
               "portrait. Frame shows a clear frontal face, holding a "
               "Star Radler can, at what BellaNaija's own article "
               "(\"Star Radler Brought Refreshing Moments to Lagos "
               "Cocktail Week\") states verbatim brought \"their brand "
               "ambassador, the gorgeous and ever-vibrant Erica "
               "Nlewedim,\" to the event — confirmed named real person, "
               "not inferred from the video alone.")},

    # ── Round 2 of the same Instagram/portrait expansion (see round-1 note
    # above). ~25 more candidates screened (banks, spirits, personal
    # influencer accounts); most failed the same two ways as round 1
    # (wrong duration, or dark/distant event footage with no identifiable
    # face — e.g. rejected Guinness Nigeria and Hennessy/lifestyle9ja
    # clips). 2 more survived: both scripted narrative ads (not event
    # coverage) on a beverage brand's own official account.
    {"url": "https://www.instagram.com/reel/DKyyJqTM5Ld/",
     "publisher": "Amstel Malta Nigeria", "expected_label": 0,
     "notes": ("VERIFIED VIA DOWNLOADED FRAMES: posted on Amstel Malta "
               "Nigeria's own official Instagram account, 2025-06-12, "
               "20.0s, 720x900 portrait. A scripted Democracy Day "
               "narrative ad (\"Happy Democracy Day, Nigeria\") — "
               "multiple clear, well-lit, identifiable human faces "
               "throughout: a young man giving a frontal speech into a "
               "megaphone at a protest, a woman holding the Nigerian "
               "flag, a family watching archival footage on TV. "
               "Persistent Amstel Malta on-screen branding confirms this "
               "is the brand's own produced commercial, not a repost.")},
    {"url": "https://www.instagram.com/amstelmalta/reel/DFnCuy8vIfw/",
     "publisher": "Amstel Malta Nigeria", "expected_label": 0,
     "notes": ("VERIFIED VIA DOWNLOADED FRAMES: posted on Amstel Malta "
               "Nigeria's own official Instagram account, 2025-02-03, "
               "360x640 portrait, a Valentine's Day ad (\"Share Your "
               "Love Story\"). Clear, well-lit, identifiable close-up "
               "face of a woman drinking the product through a straw. "
               "NOTE ON DURATION: ffprobe reports 19.991s — 9ms under "
               "the 20s floor used for this batch, evidently a container/"
               "encoding rounding artifact rather than a substantively "
               "shorter clip (functionally indistinguishable from the "
               "20.0s Democracy Day entry above). Flagged here rather "
               "than silently rounded, per this project's policy of not "
               "quietly fudging inconvenient numbers.")},
]


# =============================================================================
# CATEGORY 2: Naija-Political Media (video)
# =============================================================================
POLITICAL_MEDIA_DOMAINS = [
    "https://www.channelstv.com",
    "https://www.thecable.ng",
    "https://www.premiumtimesng.com",
    "https://www.vanguardngr.com",
]

POLITICAL_MEDIA_SPECIFIC_URLS = [
    # Still open — see prior notes. No accessible raw video file located yet.
]


# =============================================================================
# CATEGORY 3: Local Textual Corpus (text)
# =============================================================================
TEXTUAL_CORPUS_DOMAINS = [
    "https://www.punchng.com",
    "https://www.premiumtimesng.com",
    "https://www.vanguardngr.com",
    "https://www.thecable.ng",
    "https://nitda.gov.ng",
    "https://ndpc.gov.ng",
]

TEXTUAL_CORPUS_SPECIFIC_URLS = [
    {"url": "https://businessday.ng/technology/article/mtn-nigeria-leads-ai-rollout-as-13-bots-replace-200-sim-registration-workers/",
     "publisher": "BusinessDay NG", "expected_label": 0,
     "notes": "VERIFIED. Bylined human journalism, quantified AI displacement."},
    {"url": "https://www.thecable.ng/arcon-threatens-sanctions-over-unauthorised-ai-ad-featuring-tinubu/",
     "publisher": "The Cable", "expected_label": 0,
     "notes": "VERIFIED. Bylined human journalism, ARCON/Tinubu deepfake case, primary citation for §2.3.2."},
    {"url": "https://punchng.com/arcon-condemns-ai-generated-advert-featuring-tinubu/",
     "publisher": "Punch Nigeria", "expected_label": 0,
     "notes": "VERIFIED — corroborating coverage of the same ARCON statement."},

    {"url": "https://leadership.ng/the-%E2%82%A6150m-ilotbet-house-promo-how-a-fake-ai-ad-and-broken-math-exposed-the-ultimate-betting-illusion/",
     "publisher": "LEADERSHIP Newspapers", "expected_label": 0,
     "notes": ("VERIFIED (fetched in full). CAVEAT: page is tagged 'Branded "
               "Content' with a generic 'Agency Report' byline, not a named "
               "journalist's standard news piece — different provenance "
               "profile from the other 3 entries above. Still label=0: the "
               "informal, opinionated voice ('cheapest, ugliest, most "
               "robotic AI graphics imaginable') is clearly human-written "
               "commentary, not AI-generated copy. Also directly "
               "corroborates the iLOTBET video entry under TELECOM_FMCG_ADS "
               "above and is strong Ch.1 citation material — a real "
               "published account of consumer backlash against a low-"
               "quality AI-generated Nigerian ad.")},

    # ── Authentic-corpus expansion (batch 3) ────────────────────────────────
    # Added to fix the class imbalance and length-confound problems logged in
    # docs/PROJECT_CONTEXT.md §4 (31 synthetic vs 3 authentic text samples;
    # synthetic ran 67-849 words, authentic 360-12,965 with almost no
    # overlap). Punch/Premium Times/Vanguard entries below were fetched and
    # read in full (WebFetch) — byline and word count confirmed directly.
    # BusinessDay/The Cable/Leadership block WebFetch's crawler with a
    # domain-wide HTTP 403 (confirmed on ~10 URLs incl. each site's root, so
    # it's bot-detection, not dead links) — those entries are real headlines
    # found via web search, not independently content-verified by me. All
    # three domains already have at least one entry above that the project's
    # actual scraper (collect_dataset.py) fetched successfully, so they
    # should still work through the normal pipeline; treat word counts for
    # those as unconfirmed until scraped.
    {"url": "https://punchng.com/nigerias-economy-politics-and-security-in-2025/",
     "publisher": "Punch Nigeria", "expected_label": 0,
     "notes": "WebFetch-verified: bylined (Jide Ojo), ~1650 words, year-end review of Nigeria's economy/politics/security."},
    {"url": "https://punchng.com/constitutional-promises-economic-reforms-and-the-path-to-stability/",
     "publisher": "Punch Nigeria", "expected_label": 0,
     "notes": ("WebFetch-verified: bylined (Oluwadara Akingbohungbe), "
               "~3200-3400 words, opinion/analysis on the constitutional "
               "welfare mandate vs economic reforms. Longest in this batch — "
               "checked that it's one coherent essay, not a page-furniture "
               "scraping artifact like the 12,965-word sample flagged in "
               "PROJECT_CONTEXT.md.")},
    {"url": "https://healthwise.punchng.com/poverty-threatens-79-of-nigerians-despite-reforms-world-bank/",
     "publisher": "Punch Nigeria (Healthwise)", "expected_label": 0,
     "notes": "WebFetch-verified: bylined 'Our Reporters', ~2100 words, World Bank poverty-data reporting."},
    {"url": "https://punchng.com/us-opens-applications-to-support-nigerians-seeking-admission-abroad/",
     "publisher": "Punch Nigeria", "expected_label": 0,
     "notes": "WebFetch-verified: bylined (Samuel Omotere), ~450-500 words."},
    {"url": "https://punchng.com/nigerian-economy-is-in-dependency-mode/",
     "publisher": "Punch Nigeria", "expected_label": 0,
     "notes": "WebFetch-verified: bylined (Sheriffdeen Tella), ~2100 words, economics op-ed. Published 2024 — older but still real archived journalism."},

    {"url": "https://www.premiumtimesng.com/news/headlines/896984-icpc-confirms-questioning-gbajabiamila.html",
     "publisher": "Premium Times Nigeria", "expected_label": 0,
     "notes": "WebFetch-verified: bylined (Emmanuel Agbo), ~650-700 words."},
    {"url": "https://www.premiumtimesng.com/business/business-news/893409-tinubu-directs-fccpc-to-investigate-meta-google-x-ai-platforms-over-media-complaints.html",
     "publisher": "Premium Times Nigeria", "expected_label": 0,
     "notes": ("WebFetch-verified: bylined (Mariya Shuaibu Suleiman), "
               "~650-700 words. Topically about AI/media regulation but "
               "filed here under TEXTUAL_CORPUS as an authentic-writing "
               "sample, not under TELECOM_FMCG_ADS.")},
    {"url": "https://www.premiumtimesng.com/news/top-news/899600-osun-2026-accord-party-petitions-tinubu-igp-over-alleged-police-intimidation.html",
     "publisher": "Premium Times Nigeria", "expected_label": 0,
     "notes": "WebFetch-verified: bylined (Abubakar Abdulrasheed), ~450-500 words."},
    {"url": "https://www.premiumtimesng.com/news/more-news/899540-enugu-inaugurates-three-man-panel-to-investigate-community-crisis.html",
     "publisher": "Premium Times Nigeria", "expected_label": 0,
     "notes": ("WebFetch-verified: credited 'Agency Report' (NAN wire), "
               "~320-350 words — shortest in this batch, useful for "
               "length-distribution overlap with the shorter synthetic "
               "samples.")},
    {"url": "https://www.premiumtimesng.com/foreign/862296-iwd-2026-women-hold-just-64-legal-rights-men-enjoy-un.html",
     "publisher": "Premium Times Nigeria", "expected_label": 0,
     "notes": ("WebFetch-verified: credited 'Agency Report', ~650-700 words. "
               "Foreign/UN-report topic rather than domestic Nigeria news, "
               "but published and bylined by Premium Times Nigeria.")},

    {"url": "https://www.vanguardngr.com/2026/04/rising-revenues-deepening-poverty-nigerias-self-contradictory-reform-path/",
     "publisher": "Vanguard News", "expected_label": 0,
     "notes": "WebFetch-verified: bylined opinion piece (Moshood Oshunfurewa, public affairs commentator), ~1800 words."},
    {"url": "https://www.vanguardngr.com/2026/07/education-will-power-tinubus-1tn-economy-alausa-declares/",
     "publisher": "Vanguard News", "expected_label": 0,
     "notes": "WebFetch-verified: bylined (Joseph Erunke, Abuja), ~650-700 words."},
    {"url": "https://www.vanguardngr.com/2026/07/education-crisis-demands-leadership-investment-not-policy-changes-obi/",
     "publisher": "Vanguard News", "expected_label": 0,
     "notes": ("WebFetch-verified but NO INDIVIDUAL BYLINE (generic staff "
               "report) — still genuine Vanguard copy, ~520 words. Flag if "
               "byline-level provenance matters later.")},
    {"url": "https://www.vanguardngr.com/2026/07/reps-seek-increased-nutrition-budget-as-malnutrition-costs-nigeria-2-5bn-yearly/",
     "publisher": "Vanguard News", "expected_label": 0,
     "notes": "WebFetch-verified: bylined (Chioma Obinna), ~1100-1200 words."},
    {"url": "https://www.vanguardngr.com/2026/07/osun-2026-call-politicians-to-order-amupitan-tells-traditional-rulers/",
     "publisher": "Vanguard News", "expected_label": 0,
     "notes": "WebFetch-verified: bylined (Shina Abubakar, Osogbo), ~380 words."},

    {"url": "https://businessday.ng/news/legal-business/article/presidential-executive-order-boosts-virtual-assets-regulation-in-nigeria/",
     "publisher": "BusinessDay NG", "expected_label": 0,
     "notes": ("Found via search, real indexed headline — NOT WebFetch-"
               "verifiable, businessday.ng returns HTTP 403 to WebFetch's "
               "crawler UA domain-wide (confirmed on the site root too). "
               "The existing BusinessDay entry above (MTN AI rollout) was "
               "scraped successfully by collect_dataset.py, so this should "
               "still work through the real pipeline — length unconfirmed "
               "until scraped.")},
    {"url": "https://businessday.ng/backpage/article/why-inflation-survived-every-monetary-reform/",
     "publisher": "BusinessDay NG", "expected_label": 0,
     "notes": "Same WebFetch-403 caveat as the entry above — found via search only, not content-verified."},
    {"url": "https://businessday.ng/business-economy/article/inflation-projected-to-moderate-further-in-2026-but-no-single-digit-in-sight/",
     "publisher": "BusinessDay NG", "expected_label": 0,
     "notes": "Same WebFetch-403 caveat — found via search only, not content-verified."},
    {"url": "https://businessday.ng/insight-2/article/when-inflation-returns-external-shocks-and-nigerias-fragile-price-stability/",
     "publisher": "BusinessDay NG", "expected_label": 0,
     "notes": "Same WebFetch-403 caveat — found via search only, not content-verified. 'Insight' section, likely longer-form analysis."},
    {"url": "https://businessday.ng/news/article/stanchart-scales-back-nigeria-rate-cut-outlook-as-inflation-risks-persist-ahead-of-mpc/",
     "publisher": "BusinessDay NG", "expected_label": 0,
     "notes": "Same WebFetch-403 caveat — found via search only, not content-verified."},

    {"url": "https://www.thecable.ng/nigeria-2026-challenges-and-prospects/",
     "publisher": "The Cable", "expected_label": 0,
     "notes": ("Found via search, real indexed headline — thecable.ng also "
               "returns HTTP 403 to WebFetch's crawler UA domain-wide, same "
               "pattern as BusinessDay. The existing thecable.ng entry "
               "above (ARCON/Tinubu deepfake) was scraped successfully by "
               "the project pipeline, so this domain is known reachable via "
               "collect_dataset.py — length unconfirmed until scraped.")},
    {"url": "https://www.thecable.ng/five-exciting-events-to-expect-in-nigerian-politics-in-2026/",
     "publisher": "The Cable", "expected_label": 0,
     "notes": "Same WebFetch-403 caveat as the entry above — found via search only, not content-verified."},
    {"url": "https://www.thecable.ng/the-1-trillion-blueprint-why-2026-is-nigerias-digital-point-of-no-return/",
     "publisher": "The Cable", "expected_label": 0,
     "notes": "Same WebFetch-403 caveat — found via search only, not content-verified. Feature-length piece per headline/section."},
    {"url": "https://www.thecable.ng/offshore-lifeline-how-glo-network-coverage-saved-47-oil-workers-during-mid-atlantic-maritime-crisis/",
     "publisher": "The Cable", "expected_label": 0,
     "notes": "Same WebFetch-403 caveat — found via search only, not content-verified. Narrative feature, likely longer-form."},
    {"url": "https://www.thecable.ng/extra-us-mislabels-african-countries-at-aids-conference-places-nigeria-in-sahara/",
     "publisher": "The Cable", "expected_label": 0,
     "notes": "Same WebFetch-403 caveat — found via search only, not content-verified."},

    {"url": "https://leadership.ng/nigerias-agric-sector-holds-key-to-economic-recovery-fao/",
     "publisher": "LEADERSHIP Newspapers", "expected_label": 0,
     "notes": ("Found via search, real indexed headline — leadership.ng "
               "also 403s WebFetch's crawler UA, same pattern as "
               "BusinessDay/The Cable. This domain is proven reachable by "
               "the real pipeline: the iLOTBET entry above was fetched in "
               "full by collect_dataset.py — length unconfirmed until "
               "scraped.")},
    {"url": "https://leadership.ng/nigerias-agriculture-budget-too-low-for-food-security-actionaid/",
     "publisher": "LEADERSHIP Newspapers", "expected_label": 0,
     "notes": "Same WebFetch-403 caveat as the entry above — found via search only, not content-verified."},
    {"url": "https://leadership.ng/planting-season-insecurity-high-input-costs-threatening-food-production-say-farmers/",
     "publisher": "LEADERSHIP Newspapers", "expected_label": 0,
     "notes": "Same WebFetch-403 caveat — found via search only, not content-verified."},
    {"url": "https://leadership.ng/nigerias-recovery-that-millions-still-cannot-feel/",
     "publisher": "LEADERSHIP Newspapers", "expected_label": 0,
     "notes": "Same WebFetch-403 caveat — found via search only, not content-verified."},
    {"url": "https://leadership.ng/how-kano-achieved-80-of-its-agricultural-targets-prepares-bigger-for-2026-farming-season/",
     "publisher": "LEADERSHIP Newspapers", "expected_label": 0,
     "notes": "Same WebFetch-403 caveat — found via search only, not content-verified."},
]


# =============================================================================
# STRUCTURED EXPORT — used by collect_dataset.py
# =============================================================================
SEED_REGISTRY = {
    CATEGORY_TELECOM_FMCG_ADS: {
        "media_type": "video",
        "domains": TELECOM_FMCG_DOMAINS,
        "specific_urls": TELECOM_FMCG_SPECIFIC_URLS,
        "target_total": 3000,
    },
    CATEGORY_POLITICAL_MEDIA: {
        "media_type": "video",
        "domains": POLITICAL_MEDIA_DOMAINS,
        "specific_urls": POLITICAL_MEDIA_SPECIFIC_URLS,
        "target_total": 1000,
    },
    CATEGORY_TEXTUAL_CORPUS: {
        "media_type": "text",
        "domains": TEXTUAL_CORPUS_DOMAINS,
        "specific_urls": TEXTUAL_CORPUS_SPECIFIC_URLS,
        "target_total": 6000,
    },
}


def get_all_specific_urls() -> list[dict]:
    flattened = []
    for category, config in SEED_REGISTRY.items():
        for entry in config["specific_urls"]:
            flattened.append({**entry, "category": category})
    return flattened


def print_seed_summary() -> None:
    print("\n" + "=" * 65)
    print("  SEED SOURCES — Registered URLs vs. Table 3.4 Targets")
    print("=" * 65)
    for category, config in SEED_REGISTRY.items():
        queued = len(config["specific_urls"])
        print(
            f"  {category:22s} {queued:>4} URLs queued  "
            f"(target: {config['target_total']}, {len(config['domains'])} seed domains)"
        )
    print("=" * 65)
    if all(len(c["specific_urls"]) == 0 for c in SEED_REGISTRY.values()):
        print("  ⚠ No specific URLs populated yet.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    print_seed_summary()
