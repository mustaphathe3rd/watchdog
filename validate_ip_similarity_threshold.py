# =============================================================================
# validate_ip_similarity_threshold.py  —  Is IP_SIMILARITY_THRESHOLD=0.82 real?
# =============================================================================
# PURPOSE:
#   config.IP_SIMILARITY_THRESHOLD (0.82) was guessed before any data existed
#   (see config.py's PROVISIONAL — UNVALIDATED comment). Two direct tests
#   against the real 34-work protected corpus (watchdog/ip_similarity.py),
#   run before this threshold is trusted for anything:
#
#   1. NEGATIVE CONTROL (leave-one-out). Every one of the 34 protected works
#      is a DISTINCT article by a DIFFERENT publisher on a mostly different
#      topic — none of them should look like plagiarism of another. For each
#      work, its best-match similarity against the OTHER 33 (itself
#      excluded) is computed. If unrelated Nigerian journalism scores near
#      0.82 here, the threshold is too low and ordinary articles will get
#      flagged as infringing each other.
#
#   2. POSITIVE CONTROL (substantial paraphrase). Four of the 34 protected
#      works were paraphrased by hand — same facts, entities, and structure
#      of argument, but heavily reworded sentence-by-sentence (see
#      PARAPHRASES below) — simulating the actual threat model this module
#      exists for: an LLM rewriting a Nigerian article to evade a naive
#      exact-match or n-gram plagiarism check. If these score BELOW 0.82,
#      the threshold is too strict for the thing it's supposed to catch.
#
# WHAT THIS DOES NOT DO:
#   Change config.IP_SIMILARITY_THRESHOLD. Per CLAUDE.md rule 7 and the
#   explicit instruction this script was written under: measure first, then
#   report plainly whether 0.82 is defensible. If it isn't, this script
#   proposes a value WITH the evidence behind it — changing config.py is a
#   separate, deliberate step, not something to do silently inside a
#   validation run.
#
# USAGE:
#   python validate_ip_similarity_threshold.py
#
# OUTPUT:
#   - Both distributions printed to the terminal, verdict stated plainly
#   - results/ip_similarity_negative_control.csv
#   - results/ip_similarity_positive_control.csv
# =============================================================================

import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Hand-written substantial paraphrases of four protected works — different
# publishers, a spread of lengths (293-996 words originals). Facts, named
# entities, and factual claims preserved; wording, sentence structure, and
# ordering substantially changed throughout, matching how an LLM asked to
# "rewrite this article" would actually behave, not a token-level shuffle.
PARAPHRASES = [
    {
        "original_sample_id": "0e36a519-9b82-4cf3-b283-789d7898fb1f",  # TheCable, ARCON/Tinubu ad
        "original_publisher": "TheCable",
        "text": (
            "Nigeria's advertising watchdog has come down hard on whoever is responsible for a "
            "bogus, AI-produced commercial that puts words and a likeness resembling President "
            "Bola Tinubu behind an investment pitch. The Advertising Regulatory Council of Nigeria "
            "(ARCON) called the clip, circulating on Facebook, an attempt to swindle ordinary "
            "citizens. Director-General Olalekan Fadolapo, in a Monday statement, said early "
            "findings point to the footage being entirely machine-generated and that it was never "
            "cleared for release inside the country. According to Fadolapo, Tinubu never sat for "
            "the ad, never endorsed the scheme it promotes, and never gave anyone permission to use "
            "his likeness or voice. 'Our initial review indicates the material was assembled by AI "
            "with the express aim of deceiving Nigerians into a scam,' the statement read, adding "
            "that the piece bypassed the mandatory review process required under the ARCON Act of "
            "2022 and was never signed off by the Advertising Standards Panel. Fadolapo used the "
            "episode to restate why every advert aimed at Nigerian audiences must pass pre-broadcast "
            "vetting: to keep the airwaves free of dishonest, misleading, or unverified claims. He "
            "put broadcasters, newspapers, and online platforms on notice that Nigerian law obliges "
            "them to secure an approval certificate before running any commercial. He closed by "
            "asking everyone working in the advertising space to hold the line on honesty, "
            "legality, and consumer protection as the industry matures."
        ),
    },
    {
        "original_sample_id": "9f4a0ed3-3401-47c6-842d-9311d3815a3c",  # Punch, ARCON/Tinubu ad (Ponzi angle)
        "original_publisher": "Punch Newspapers",
        "text": (
            "The country's ad-standards body has slammed a deceptive AI-made commercial that "
            "borrows President Bola Tinubu's image to push what appears to be a Ponzi-style "
            "investment offer. ARCON's director-general, Dr Olalekan Fadolapo, branded the video a "
            "calculated attempt to con unsuspecting members of the public. He explained that a "
            "first-look investigation found the advert to be a product of computer-generated "
            "imagery built specifically to trick Nigerians, stressing that the president neither "
            "appeared in nor sanctioned any such promotion. Fadolapo pointed out that the commercial "
            "skipped the compulsory clearance process spelled out in the ARCON Act No. 23 of 2022 "
            "and was never reviewed or approved by the Advertising Standards Panel. He reiterated "
            "that pre-broadcast vetting exists precisely so that every advert reaching Nigerian "
            "audiences is honest, lawful, decent, socially conscious, and free of misleading claims. "
            "The DG cautioned all broadcasters — whether traditional stations, digital outlets, or "
            "social platforms — that airing unvetted content is a breach of the law, and warned that "
            "anyone found in violation of the Act would face regulatory penalties. He appealed to "
            "everyone in the advertising and marketing sector to recommit to honest and responsible "
            "practices as ARCON works to strengthen consumer protections across the industry."
        ),
    },
    {
        "original_sample_id": "967728cf-cf1d-45e4-8c77-2c4a36b13df9",  # BusinessDay, MTN AI SIM registration
        "original_publisher": "Businessday NG",
        "text": (
            "MTN Nigeria has become a proving ground for its parent group's push into artificial "
            "intelligence, with the operator confirming that a set of 13 AI agents has taken over "
            "work once handled by a team of more than 200 staff responsible for manually checking "
            "SIM registration records. This sits inside MTN Group's wider ambition to unlock roughly "
            "R30 billion in value from AI within the next three to five years, largely through "
            "leaner operations, revenue growth, and new digital ventures spanning the continent. "
            "Charles Molapisi, the group's chief technology and information officer, told the "
            "company's Capital Markets Day audience that the Nigerian operation now ranks among the "
            "group's most sophisticated AI implementations. Molapisi noted that verifying SIM "
            "registrations used to demand over 200 people cross-checking biometric details against "
            "national records by hand — a slow process prone to mistakes. Today, 13 computer-vision-"
            "powered AI agents carry out most of that verification without human intervention. "
            "'These agents work faster and with fewer errors than the old manual approach, and we've "
            "kept oversight mechanisms in place,' he said. Buoyed by the Nigerian results, MTN plans "
            "to roll the same approach out to Cameroon, Côte d'Ivoire, Eswatini, Ghana, and Zambia. "
            "The shift underlines just how central AI has become to Africa's biggest telecom group, "
            "especially in functions involving heavy customer administration and regulatory demands. "
            "MTN's AI ambitions reach well past registration checks — network and IT spending alone "
            "make up roughly 55% of operating costs and close to 80% of capital spending, so "
            "squeezing efficiency from those areas is a top priority. The company has rolled out AI "
            "tools that manage energy use at cell towers, flag network faults automatically, spot "
            "fibre breaks ahead of outages, and guide infrastructure placement decisions. It's also "
            "using AI-based revenue-assurance software that scans over 10,000 data points across "
            "billions of transactions daily to catch irregularities humans would likely miss. "
            "Nigeria additionally features among six countries running MTN's AI customer-engagement "
            "system, NBx 2.0, which tailors offers using behavioural data and now reaches about 44 "
            "million users across Nigeria, South Africa, Ghana, Cameroon, Uganda, and Zambia. The "
            "group has also introduced Telco GPT to assist call-centre staff, alongside its Zigi "
            "chatbot, which now handles both voice and text in local languages. MTN frames its AI "
            "push around three pillars — internal efficiency, consumer-facing improvements, and "
            "enterprise offerings such as GPU and edge computing — but is prioritising the internal-"
            "efficiency track first because it offers the fastest, most reliable payoff. 'We're "
            "leaning into AI Inside first, even as we keep B2C and B2B moving — it's mostly a matter "
            "of sequencing and where we put capital,' Molapisi said. MTN is targeting 2028 for "
            "roughly 80% of AI's financial benefit across its six biggest markets to come from "
            "internal operational gains, with Nigeria's SIM-verification overhaul serving as an "
            "early signal of the deeper changes AI could bring to how telecom services are delivered "
            "and regulated continent-wide."
        ),
    },
    {
        "original_sample_id": "49d54fcd-b33c-4f1e-8cbd-1438894649aa",  # LEADERSHIP, Kano agriculture
        "original_publisher": "LEADERSHIP Newspapers",
        "text": (
            "Kano State says its multi-pronged push into irrigation, farm mechanisation and support "
            "for growers is starting to pay off, with officials confirming the state has hit roughly "
            "four-fifths of the goals set out in its agricultural development plan. Given how "
            "central farming is to the state's economy and household incomes, officials are treating "
            "the milestone as a meaningful win for food security, poverty reduction and rural "
            "livelihoods. Dr Yakubu Danjuma Mahmud, the Commissioner for Agriculture and Natural "
            "Resources, credited the gains to deliberate policy choices made since Governor Abba "
            "Kabir Yusuf took office in 2023. He said the administration's first move was building a "
            "full database of farmers statewide, so that planning could rest on real numbers rather "
            "than guesswork. 'You can't plan properly without statistics — that's why the baseline "
            "database came first,' Mahmud told reporters in Kano, explaining that the database now "
            "lets the government and partner agencies pinpoint farmers, map their plots, and target "
            "support more precisely. The commissioner acknowledged the administration inherited a "
            "sector in poor shape but said real headway has since been made rebuilding "
            "infrastructure and growing output capacity. Among the wins he listed: roughly 1,230 "
            "hectares of irrigable farmland recovered and repaired statewide, plus the return of the "
            "Kadawa irrigation scheme to state control. To ease the long-running farmer-herder "
            "conflicts, the state has also mapped out about 1,300 kilometres of cattle routes, which "
            "Mahmud said has 'genuinely helped calm tensions' between the two groups. The centrepiece "
            "of Kano's current agricultural push is a large new dam under construction at Dantsoshiya "
            "Forest — by Mahmud's account, among the largest irrigation investments the state has "
            "ever undertaken. 'It runs close to two kilometres and will hold up to three billion "
            "cubic litres, irrigating around 1,250 hectares once finished,' he said, noting the dam "
            "is part of a bigger plan to revive neglected irrigation sites and open new ones. The "
            "project is well underway, he added, alongside a 14-kilometre access road meant to help "
            "farmers move produce out of the area. Beyond the flagship dam, the state has built 11 "
            "smaller earth dams across local government areas such as Albasu, Shanono, Gari, "
            "Dawakin Kudu, Rogo, Ajingi and Bichi, together capable of irrigating about 1,200 "
            "hectares — a figure Mahmud expects to climb toward 2,000 hectares as newer designs come "
            "online. On inputs, Kano has moved more than 623 truckloads of fertiliser to farmers over "
            "the past three years, drawing on both state resources and federal support, with some "
            "supplied free and others at subsidised cost. With the 2026 wet-season planting "
            "approaching, the government is introducing fresh support measures to reassure farmers "
            "rattled by unstable prices and rising costs — many of whom, Mahmud admitted, lost money "
            "last season to poor prices and erratic rainfall. Governor Yusuf has approved subsidised "
            "fertiliser purchases to help growers recoup some of those losses, according to the "
            "commissioner. Kano is also pushing mechanisation hard, with three Agricultural "
            "Mechanisation Centres now running in Gaya, Kadawa and Dambatta, stocked with tractors, "
            "combine harvesters, planters, transplanters, sprayers and threshers that farmers can "
            "access at reduced rates through public-private arrangements. 'The direction is clearly "
            "toward heavily mechanised farming, with equipment available to growers at a discount,' "
            "Mahmud said. The state is additionally promoting climate-resilient farming through "
            "drought-tolerant, fast-maturing seed varieties, and plans to distribute solar-powered "
            "irrigation pumps to farming cooperatives to cut reliance on expensive fuel-driven "
            "systems and extend dry-season cultivation. Mahmud pointed to an independent assessment "
            "— built on the blueprint's own milestones and performance indicators — that rated the "
            "ministry's progress at 80 percent as validation of the administration's approach. "
            "Between the irrigation investment, mechanisation drive, farmer-support schemes and "
            "climate-smart initiatives, Kano is positioning itself to build on these gains and "
            "expand food output in the years ahead, with the state government signalling "
            "determination to consolidate its standing as one of Nigeria's premier agricultural "
            "economies heading into the 2026 planting season."
        ),
    },
]


# Two root causes were found (by inspection, after the raw negative-control
# run below flagged them) for LOO scores landing near/above 0.82 despite
# these being genuinely distinct, non-infringing articles:
#   - SAME-EVENT CONVERGENT JOURNALISM: TheCable (0e36a519) and Punch
#     (9f4a0ed3) both cover the identical ARCON/Tinubu fake-ad story and
#     both quote director-general Fadolapo's statement VERBATIM — expected,
#     ethical journalism practice (quoting an official press statement),
#     not one outlet copying the other.
#   - SAME-PUBLISHER BEAT CLUSTERING: three BusinessDay NG pieces
#     (36d27a86, a18af7c8, 9a5eac81) are all inflation/monetary-policy
#     commentary from what looks like the same economics desk — shared
#     topic, house style, and framing inflate similarity without either
#     being derived from the other.
# A raw whole-document cosine-similarity threshold structurally CANNOT tell
# these apart from actual copying — that needs a different signal entirely
# (e.g. detecting attributed/quoted material, or an explicit same-live-event
# check), which is out of scope for this module. Flagged as a genuine §1.7
# limitation below, not solved by picking a different threshold number.
KNOWN_CONFOUND_SAMPLE_IDS = {
    "0e36a519-9b82-4cf3-b283-789d7898fb1f", "9f4a0ed3-3401-47c6-842d-9311d3815a3c",  # same-event
    "36d27a86-8046-421e-af2b-da7c165189c7", "a18af7c8-cecd-43fc-a362-0b8d036e1c42",  # same-beat
    "9a5eac81-7bfc-4735-8822-a3c8d2c136fb",                                          # same-beat
}


def run_negative_control(index: dict) -> "list[dict]":
    import numpy as np

    doc_embeddings = index["doc_embeddings"]  # (n, d), L2-normalised
    sim_matrix = doc_embeddings @ doc_embeddings.T
    np.fill_diagonal(sim_matrix, -1.0)  # exclude self-match

    rows = []
    for i in range(len(index["sample_ids"])):
        best_j = int(np.argmax(sim_matrix[i]))
        sample_id = index["sample_ids"][i]
        rows.append({
            "sample_id": sample_id[:8],
            "publisher": index["publisher_names"][i],
            "best_match_sample_id": index["sample_ids"][best_j][:8],
            "best_match_publisher": index["publisher_names"][best_j],
            "loo_best_similarity": float(sim_matrix[i, best_j]),
            "known_confound": sample_id in KNOWN_CONFOUND_SAMPLE_IDS,
        })
    return rows


def run_positive_control(index: dict) -> "list[dict]":
    from watchdog.ip_similarity import check_similarity

    rows = []
    for p in PARAPHRASES:
        result = check_similarity(p["text"], top_k=3, index=index)
        top = result.matches[0] if result.matches else None
        correctly_attributed = (top is not None) and (top.sample_id == p["original_sample_id"])
        rows.append({
            "original_sample_id": p["original_sample_id"][:8],
            "original_publisher": p["original_publisher"],
            "top_match_sample_id": top.sample_id[:8] if top else None,
            "top_match_publisher": top.publisher_name if top else None,
            "correctly_attributed": correctly_attributed,
            "similarity": top.similarity if top else None,
            "flagged_at_0.82": (top.similarity >= 0.82) if top else False,
        })
    return rows


def run_validation():
    import pandas as pd
    from config import IP_SIMILARITY_THRESHOLD
    from watchdog.ip_similarity import load_embeddings_index, build_embeddings_index

    index = load_embeddings_index()
    if index is None:
        logger.info("No cached embeddings index found — building it now.")
        index = build_embeddings_index()
    if index is None:
        logger.error("Protected corpus is empty — cannot validate. Run build_protected_corpus_index() first.")
        return

    n = len(index["sample_ids"])
    logger.info(f"Validating against {n}-work protected corpus. Current IP_SIMILARITY_THRESHOLD={IP_SIMILARITY_THRESHOLD}")

    # =========================================================================
    # 1. Negative control
    # =========================================================================
    print("\n" + "=" * 100)
    print("  1. NEGATIVE CONTROL — leave-one-out best-match similarity among the 34 protected works")
    print("  These are DISTINCT articles by different publishers. Scores near or above the threshold")
    print("  here mean unrelated Nigerian journalism looks like plagiarism to this matcher.")
    print("=" * 100)

    neg_rows = run_negative_control(index)
    neg_df = pd.DataFrame(neg_rows)
    sims = neg_df["loo_best_similarity"]
    print(f"\n  n={len(neg_df)}")
    print(f"  mean={sims.mean():.4f}  std={sims.std():.4f}  min={sims.min():.4f}  "
          f"median={sims.median():.4f}  max={sims.max():.4f}")
    n_over = int((sims >= IP_SIMILARITY_THRESHOLD).sum())
    print(f"  samples with LOO best-match >= {IP_SIMILARITY_THRESHOLD}: {n_over}/{len(neg_df)}")
    print(f"\n  Top 5 highest LOO best-matches (closest false-positive risk):")
    print(neg_df.sort_values("loo_best_similarity", ascending=False).head(5).to_string(index=False))

    neg_path = Path("results") / "ip_similarity_negative_control.csv"
    neg_path.parent.mkdir(exist_ok=True)
    neg_df.to_csv(neg_path, index=False)

    # =========================================================================
    # 2. Positive control
    # =========================================================================
    print("\n" + "=" * 100)
    print("  2. POSITIVE CONTROL — substantially paraphrased versions of 4 protected works")
    print("  Same facts/entities, heavily reworded — simulates an LLM asked to rewrite an article")
    print("  to evade detection. Scores BELOW the threshold here mean 0.82 is too strict to catch")
    print("  the actual threat model this module exists for.")
    print("=" * 100)

    pos_rows = run_positive_control(index)
    pos_df = pd.DataFrame(pos_rows)
    print()
    print(pos_df.to_string(index=False))
    pos_sims = pos_df["similarity"]
    print(f"\n  mean={pos_sims.mean():.4f}  min={pos_sims.min():.4f}  max={pos_sims.max():.4f}")
    n_correct = int(pos_df["correctly_attributed"].sum())
    n_flagged = int(pos_df["flagged_at_0.82"].sum())
    print(f"  correctly attributed to source article: {n_correct}/{len(pos_df)}")
    print(f"  flagged at current threshold ({IP_SIMILARITY_THRESHOLD}): {n_flagged}/{len(pos_df)}")

    pos_path = Path("results") / "ip_similarity_positive_control.csv"
    pos_df.to_csv(pos_path, index=False)

    # =========================================================================
    # 3. Verdict
    # =========================================================================
    print("\n" + "=" * 100)
    print("  VERDICT")
    print("=" * 100)
    neg_max = sims.max()
    pos_min = pos_sims.min()
    separation = pos_min - neg_max
    print(f"\n  Negative control ceiling (highest unrelated-article similarity): {neg_max:.4f}")
    print(f"  Positive control floor (lowest paraphrase similarity):            {pos_min:.4f}")
    print(f"  Separation margin:                                                {separation:+.4f}")

    if n_over > 0:
        print(f"\n  *** {n_over} unrelated article pair(s) already exceed {IP_SIMILARITY_THRESHOLD} — "
              f"threshold is too LOW for false-positive safety. ***")
    if n_flagged < len(pos_df):
        print(f"\n  *** {len(pos_df) - n_flagged}/{len(pos_df)} paraphrases were NOT flagged at "
              f"{IP_SIMILARITY_THRESHOLD} — threshold is too HIGH to catch the actual threat model. ***")

    if n_over == 0 and n_flagged == len(pos_df) and separation > 0:
        midpoint = (neg_max + pos_min) / 2
        print(f"\n  0.82 sits inside the separation margin [{neg_max:.4f}, {pos_min:.4f}] and is "
              f"DEFENSIBLE as measured: it clears every unrelated-article pair and catches every "
              f"paraphrase in this test. (For reference, the data-driven midpoint of the observed "
              f"margin would be {midpoint:.4f}.)")
    else:
        print(f"\n  0.82 is NOT cleanly defensible against both controls as measured above — see the "
              f"flagged issue(s). This script does not change config.py; that is a separate, "
              f"deliberate decision to make with this evidence in hand.")

    print(f"\nSaved negative control to {neg_path}")
    print(f"Saved positive control to {pos_path}\n")


if __name__ == "__main__":
    run_validation()
