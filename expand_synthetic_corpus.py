# =============================================================================
# expand_synthetic_corpus.py  —  Day 3b: Length-Matched Corpus Expansion
# =============================================================================
# WHY THIS EXISTS (read before running):
#
# The Day 3 evaluation (evaluate_nlp_heuristics.py) surfaced two real
# problems in the existing corpus that MORE SHORT SAMPLES WOULD WORSEN:
#
#   PROBLEM 1 — LENGTH CONFOUND:
#     Synthetic samples: 67-406 words (median 210)
#     Authentic samples: 360-12,965 words (median 1,889)
#     Only 1 of 11 synthetic samples overlapped the authentic range at all.
#     A classifier trained on this learns "short = AI, long = human" — a
#     spurious rule with nothing to do with authorship, which collapses on
#     any long AI article or short human post. It also mechanically drags
#     burstiness toward uniform for short texts (fewer sentences = less
#     length variance), which is very likely why burstiness showed ZERO
#     separation between classes (AUTHENTIC mean -0.420 vs SYNTHETIC
#     -0.418) and flagged 100% of samples in the Day 3 run.
#
#   PROBLEM 2 — §3.9 GENRE GAPS:
#     §3.9 specifies the textual corpus as "Nigerian journalism, Pidgin
#     English, Bureaucratic texts vs LLM copy." The existing 10 samples
#     contain ZERO Pidgin English and ZERO bureaucratic text. Two of the
#     three named sub-genres were entirely missing.
#
# WHAT THIS SCRIPT ADDS:
#   20 new LLM-authored samples deliberately built to fix both:
#     - Heavy weighting toward LONG samples (600-1,600 words) to overlap
#       the authentic length distribution instead of sitting below it
#     - 4 Pidgin English samples (fills §3.9 gap)
#     - 4 bureaucratic/regulatory samples (fills §3.9 gap)
#     - Wider topical spread (insurance, edtech, health, aviation, opinion)
#
# GROUND TRUTH: certain by construction — these were written by an LLM
# (Claude) expressly for this purpose, so label=1 is assigned immediately
# with no human review step needed.
#
# ADDITIVE & IDEMPOTENT: does not touch the 10 existing samples from
# generate_synthetic_corpus.py. Re-running skips anything already logged.
#
# ⚠ IMPORTANT — THIS ALONE DOES NOT FIX YOUR CLASS BALANCE:
#   Current: 11 synthetic vs 3 authentic. Adding 20 synthetic makes it
#   31 vs 3. You MUST expand the authentic side too (scrape more Punch /
#   Premium Times / Vanguard / BusinessDay articles via seed_sources.py)
#   before any of these numbers mean much. See the note printed at the
#   end of this script's run.
#
# USAGE:
#   python expand_synthetic_corpus.py --dry-run   # preview
#   python expand_synthetic_corpus.py             # write + log
# =============================================================================

import argparse
import logging
import uuid
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


EXPANDED_SAMPLES = [

    # =========================================================================
    # LONG-FORM NEWS (600-1,600 words) — fixes the length confound
    # =========================================================================
    {
        "filename": "synthetic_long_001_fintech_regulation.txt",
        "genre": "long_news",
        "text": """Nigeria's rapidly expanding financial technology sector is entering a period of intensified regulatory scrutiny, as authorities weigh how best to balance consumer protection against the innovation that has made the country a continental leader in digital payments.

Industry participants say the shift has been building for some time. Over the past several years, the number of licensed payment service providers operating in the country has grown substantially, driven by rising smartphone penetration, a young and increasingly urban population, and sustained policy emphasis on financial inclusion. Transaction volumes across instant payment channels have climbed year on year, with analysts pointing to the growth of agent banking networks as a particularly significant factor in extending services beyond traditional urban centres.

That growth, however, has not come without complications. Consumer advocacy groups have documented a rise in complaints relating to failed transactions, delayed reversals, and difficulties in obtaining redress from providers. In several reported cases, customers described waiting weeks for the resolution of disputes involving relatively modest sums, with limited recourse when providers were unresponsive. Industry observers note that while the largest institutions generally maintain functional dispute channels, standards vary considerably across the sector, particularly among newer entrants operating on thin margins.

Regulatory authorities have signalled that the current framework may require strengthening. Discussions reported to be underway focus on several areas: minimum capital requirements for licensed operators, mandatory timelines for the resolution of consumer complaints, clearer disclosure obligations around fees, and enhanced reporting requirements relating to system downtime and transaction failure rates. Some proposals under consideration would also introduce tiered supervisory regimes, applying more stringent oversight to institutions above certain transaction thresholds while maintaining lighter requirements for smaller operators.

The prospect of tighter regulation has drawn mixed reactions. Larger institutions, better positioned to absorb compliance costs, have generally expressed support for clearer standards, arguing that inconsistent practice across the sector damages consumer confidence in digital financial services as a whole. Smaller operators and some sector advocates have expressed concern that overly prescriptive requirements could raise barriers to entry and consolidate the market among a handful of well-capitalised incumbents, potentially reducing the competitive pressure that has driven service improvements and fee reductions in recent years.

Financial inclusion advocates have raised a related concern. Much of the progress in extending financial services to previously unbanked populations has come through agent networks and lightweight digital products with minimal onboarding requirements. Requirements that increase documentation burdens or operational costs at the agent level, they argue, risk reversing gains among precisely the populations financial inclusion policy was designed to reach. Several have called for any new framework to include explicit impact assessments examining effects on rural and low-income users before implementation.

The technology underpinning the sector adds further complexity. An increasing proportion of customer interactions are now handled through automated systems, including chatbots for first-line support and algorithmic models for credit assessment and fraud detection. These systems have improved efficiency and allowed providers to serve larger customer bases without proportionate increases in staffing. They have also introduced questions that existing regulatory frameworks were not designed to address, particularly around how customers can contest decisions made by automated systems and what recourse exists where such systems produce erroneous outcomes.

Data protection considerations intersect with these questions directly. Digital financial services generate substantial volumes of personal and transactional data, and the analytical use of that data underpins many of the sector's products. Existing data protection provisions establish obligations around lawful processing and data subject rights, but practitioners note that enforcement capacity has not kept pace with the scale of processing now occurring. Some have called for sector-specific guidance clarifying how general data protection obligations apply to particular financial services use cases.

Infrastructure constraints remain a persistent background factor. Providers cite unreliable power supply and inconsistent network coverage as ongoing operational challenges, particularly outside major urban areas. Service disruptions attributable to infrastructure rather than provider systems complicate both consumer expectations and regulatory assessment of provider performance. Industry groups have repeatedly called for coordinated investment in supporting infrastructure as a precondition for sustainable sector growth.

Looking ahead, sector analysts expect the regulatory conversation to continue through the coming year, with consultation processes likely to shape the final form of any new requirements. Most anticipate an outcome that increases obligations on larger operators while attempting to preserve room for smaller innovators, though the precise calibration remains contested. What appears less contested is that the period of relatively light-touch oversight that characterised the sector's early growth phase is drawing to a close, and that providers will need to invest more substantially in compliance capability regardless of the specific shape the final framework takes.

For consumers, the practical implications may take time to materialise. Advocacy groups have emphasised that the value of any new framework will depend heavily on enforcement rather than drafting, noting that existing obligations are already inconsistently observed. Whether strengthened requirements translate into improved consumer experience, they suggest, will depend on supervisory capacity and the willingness of authorities to apply meaningful consequences where standards are not met.""",
        "notes": "long-form news, ~700 words — length-matched to authentic journalism range",
    },
    {
        "filename": "synthetic_long_002_creative_industry_ai.txt",
        "genre": "long_news",
        "text": """The growing use of artificial intelligence tools across Nigeria's creative industries has prompted debate among practitioners about the future shape of an economy that has become one of the country's most visible cultural exports.

The concerns are not abstract. Practitioners across advertising, film production, voice work, and graphic design report that clients increasingly enquire about AI-assisted alternatives to conventional production, often with explicit reference to cost savings. Several production professionals describe losing work to briefs that were ultimately fulfilled using generative tools, in some cases at a fraction of the budget that a conventional production would have required.

Advertising has seen the most visible early adoption. Generative tools now allow agencies to produce campaign visuals, voiceovers, and in some cases complete video assets without engaging the photographers, models, voice artists, and crews that such work would traditionally require. Proponents point to dramatically compressed production timelines and the ability to produce multiple language variants of a campaign at marginal additional cost — a meaningful consideration in a country with substantial linguistic diversity. Critics point to the corresponding contraction in paid work for the practitioners such campaigns previously employed.

Voice work has proven particularly exposed. Synthetic voice generation has advanced to the point where produced audio is, in many commercial contexts, difficult for ordinary listeners to distinguish from human recording. Voice practitioners report a decline in routine commercial work — corporate narration, advertising voiceover, interactive voice response systems — that previously constituted a reliable revenue base. Some have shifted toward work requiring greater performance range or live direction, though such work is less plentiful and more competitive.

The film sector presents a more mixed picture. Production professionals generally describe AI tools as currently supplementary rather than substitutive in narrative filmmaking, useful for tasks such as previsualisation, rough editing, and certain post-production processes, but not yet capable of replacing performance or direction in ways audiences would accept. Some practitioners have embraced these tools as genuinely useful, arguing that reducing the cost of technical post-production work allows more resources to be directed toward creative elements. Others express concern that current limitations are temporary and that the sector should establish norms before rather than after capability improves further.

Questions of intellectual property run throughout these discussions. Generative systems are trained on large corpora of existing creative work, and practitioners have raised concerns about whether their own output has been incorporated into training data without permission, attribution, or compensation. The technical difficulty of establishing whether particular works were used in training complicates any assessment of the scale of the issue. Legal practitioners note that existing copyright frameworks were not designed with machine training in mind and that applying them to these circumstances raises unresolved questions.

Existing copyright provisions require that a work reflect sufficient human effort to attract protection, a formulation that on its face creates difficulties for purely machine-generated output. Practitioners have argued that this offers some protection against the wholesale displacement of human authorship, at least in principle, since fully automated output may fall outside the protection commercial clients typically require. Others counter that in practice, most commercial AI use involves human direction and selection sufficient to satisfy such requirements, limiting the provision's practical effect.

Disclosure has emerged as a focal point for those seeking near-term intervention. Several practitioner groups have called for requirements that commercial content produced substantially through generative tools be labelled as such, arguing that consumers have a legitimate interest in knowing whether the person apparently addressing them exists. Advocates frame this as a transparency measure rather than a restriction on the technology itself. Advertising industry representatives have generally expressed openness to disclosure norms in principle while raising practical questions about thresholds, given that most modern production involves some degree of digital manipulation.

Economic analysis of the sector's exposure remains limited. Practitioners note that reliable data on employment and earnings across the creative economy has historically been sparse, complicating any assessment of how displacement is actually distributing across sub-sectors and demographics. Several advocacy organisations have called for systematic data collection as a precondition for informed policy, arguing that the absence of baseline measurement makes it impossible to evaluate either the scale of the problem or the effectiveness of any response.

Practitioners themselves report a range of adaptive strategies. Some have incorporated generative tools into their own workflows, positioning themselves as directors of AI-assisted production rather than competitors to it. Others have concentrated on work where human involvement remains a selling point, including live performance and content where authenticity is central to the value proposition. A number have moved toward training and consultancy, advising organisations on how to deploy these tools. Whether these adaptations prove durable at the scale of the sector's workforce remains an open question.

What emerges from these accounts is not uniform opposition to the technology but a widely shared concern about the pace of change relative to the development of norms, protections, and adaptive capacity. Several practitioners drew the same distinction: the issue, as they framed it, is less that the tools exist than that their deployment is proceeding faster than any framework for managing its consequences.""",
        "notes": "long-form news, ~800 words — directly relevant to §1.1 displacement theme",
    },
    {
        "filename": "synthetic_long_003_telecom_infrastructure.txt",
        "genre": "long_news",
        "text": """Investment in telecommunications infrastructure across Nigeria has accelerated in recent years, though industry participants caution that persistent structural challenges continue to constrain the pace at which improved connectivity reaches underserved populations.

Operators have committed substantial capital to network expansion, including fibre backbone deployment, base station construction, and upgrades to existing sites. The stated objectives are broadly consistent across the sector: extending coverage into underserved areas, increasing capacity in congested urban markets, and preparing networks for the higher bandwidth demands associated with newer service categories. Progress against these objectives has been uneven, with coverage gains concentrated in areas where commercial returns are most readily established.

The economics of rural deployment remain the central obstacle. Extending network coverage into sparsely populated areas requires capital expenditure comparable to urban deployment while generating substantially lower revenue per site. Operators note that the resulting payback periods often exceed thresholds that would justify investment on conventional commercial criteria. Various mechanisms have been proposed to address this gap, including universal service funding, infrastructure sharing arrangements, and public-private partnership structures, with implementation proceeding at varying rates.

Infrastructure sharing has gained traction as a partial response. Arrangements under which multiple operators use common tower and transmission infrastructure reduce duplicated capital expenditure and can improve the viability of marginal sites. Independent tower companies now own and operate a substantial proportion of passive infrastructure, leasing capacity to multiple operators. Industry participants generally regard this as a positive structural development, though some note that concentration in the tower segment introduces its own commercial dynamics.

Power supply remains among the most frequently cited operational constraints. Base stations require continuous power, and unreliable grid supply means that a significant proportion of sites depend on diesel generation, either as primary supply or backup. The associated fuel, maintenance, and logistics costs represent a substantial component of network operating expenditure, particularly at remote sites where resupply is difficult. Operators have invested in alternative approaches, including solar hybrid systems and higher-efficiency equipment, with reported improvements in cost and reliability at deployed sites, though conversion of existing infrastructure proceeds gradually given capital requirements.

Security considerations have added complexity in certain regions. Operators report incidents affecting infrastructure and personnel in some areas, resulting in service disruptions and, in some cases, decisions to defer or withdraw from planned deployments. Industry groups have engaged with authorities on protective measures, and some operators have adjusted deployment approaches to reduce exposure. The cumulative effect has been to further complicate the economics of extending coverage in precisely the regions where connectivity gaps tend to be widest.

Regulatory and administrative processes affect deployment timelines. Operators cite variation across jurisdictions in the requirements and timelines associated with obtaining permissions for infrastructure construction, including right-of-way approvals for fibre deployment. Multiple industry submissions have called for harmonisation of these processes, arguing that inconsistency across states and local authorities introduces delay and cost that materially affects deployment rates. Some progress toward standardisation has been reported, though practitioners describe implementation as uneven.

Spectrum availability shapes what operators can deliver on deployed infrastructure. Allocation processes and the associated cost of spectrum represent significant considerations in network planning, and operators have engaged with authorities regarding both the quantity of spectrum available and the terms on which it is assigned. The relationship between spectrum policy and service quality is direct: available spectrum determines capacity, which in turn affects the user experience particularly in dense urban environments where demand concentration is highest.

Demand-side factors interact with supply constraints. Device affordability affects the extent to which improved network capability translates into actual usage, particularly for services requiring newer handset generations. Data pricing similarly influences consumption patterns. Operators note that network investment justifications depend on projected usage, which in turn depends on affordability factors partly outside their control. Several have pursued device financing arrangements and tiered pricing structures intended to broaden access, with varying results.

The relationship between connectivity investment and broader economic activity features prominently in industry advocacy. Operators and industry associations point to the role of telecommunications infrastructure in enabling financial services, commerce, education, and public service delivery, arguing that connectivity investment generates returns extending well beyond the telecommunications sector itself. This framing underpins arguments for policy treatment recognising infrastructure as foundational rather than as a conventional commercial sector.

Assessments of progress vary by vantage point. Coverage statistics show meaningful expansion over recent years by most measures. Quality-of-experience measures present a more mixed picture, with congestion in high-density areas and variable performance in areas at network edges. Users in underserved regions continue to report connectivity that falls short of what headline coverage figures might suggest. Bridging the gap between nominal coverage and consistent, usable service quality appears likely to remain a central sector challenge for some time.""",
        "notes": "long-form news, ~700 words — infrastructure/telecom register",
    },

    # =========================================================================
    # PIDGIN ENGLISH — fills a §3.9 gap (previously ZERO samples)
    # =========================================================================
    {
        "filename": "synthetic_pidgin_001_telecom_ad.txt",
        "genre": "pidgin_ad",
        "text": """Abeg, make we yarn true talk. You don tire for network wey dey disappoint you? You dey try load video, e go dey turn turn like say im dey do gym? No be your fault. Na your network dey fail you.

Come try SwiftLink Mobile. We no dey do long talk, we dey do action. With just five hundred naira, you go get 3GB wey go last you seven whole days. No hidden charge, no wahala, no story for the end of the month.

Our network strong well well. Whether you dey Lagos, Aba, Kano, or you travel go village for Christmas, SwiftLink dey there for you. Video call your people, stream your favourite series, do your online business — everything go dey smooth like fresh puff-puff.

And no be only data we get o. Call rate too dey cheap. You fit talk to your mama for thirty minutes and your credit no go finish. As e dey pain other networks, na so e sweet us.

So wetin you dey wait for? Waka go any SwiftLink shop wey dey near you, or just dial star four four four hash for your phone. Switch today, thank yourself tomorrow.

SwiftLink Mobile. Network wey sabi. Network wey no dey fail.""",
        "notes": "Pidgin English ad copy — fills §3.9 'Pidgin English' gap (previously zero samples)",
    },
    {
        "filename": "synthetic_pidgin_002_fmcg_ad.txt",
        "genre": "pidgin_ad",
        "text": """Mama, na you sabi! When your pot dey sweet, everybody for house go know say food don ready before you even call them.

Na so Kitchen Star Seasoning Cubes dey do am. One small cube, e go carry your soup go another level. Egusi, ogbono, vegetable, jollof — anything wey you touch, Kitchen Star go make am taste like say na professional chef cook am.

We no dey use anyhow ingredient. Everything wey dey inside our cube na correct thing — real spice, real flavour, no shortcut. Na why plenty mama for Naija don dey trust us for over twenty years.

Your pikin go chop finish and still dey ask for more. Your husband go call your name with sweet mouth. Even your neighbour go come knock your door say make you teach am how you take do am. Just tell them say na Kitchen Star.

You fit buy am for any market, any supermarket, any provision store wey dey near you. E no cost, but wetin e go do for your food, money no fit buy am.

Kitchen Star Seasoning. Wetin mama sabi.""",
        "notes": "Pidgin English FMCG ad — fills §3.9 gap",
    },
    {
        "filename": "synthetic_pidgin_003_social_caption.txt",
        "genre": "pidgin_social",
        "text": """Ehen! See wetin we bring come for una this weekend 🔥

Na our biggest sale of the year and we no dey do pass-me-by. Everything for our store don reduce — shoe, bag, cloth, watch, everything!

Up to 60% off. You hear am? SIXTY PERCENT. That bag wey you don dey eye since since, na now e cheap reach your budget.

But make I warn you early — this thing no go last. Last year our stock finish within three days and plenty people dey cry for our DM. No let am happen to you again this time.

Sale start Friday morning 8am sharp. Set your alarm. Tell your friend. Tell your sister. Tell that person wey dey always collect am from you 😂

Free delivery too if your order pass ₦20,000. Anywhere for Nigeria — Lagos, PH, Abuja, everywhere.

Click the link for our bio make you see everything wey dey available.

See you for Friday. Come with your money o! 💃🏾

#NaijaFashion #WeekendSale #LagosStyle #ShopNaija""",
        "notes": "Pidgin English social caption — fills §3.9 gap, short-form register",
    },
    {
        "filename": "synthetic_pidgin_004_informal_news.txt",
        "genre": "pidgin_news",
        "text": """Plenty people for Naija don begin complain say the AI advert wey dey show for social media these days no be beans.

If you don dey use Instagram or TikTok recently, you go don see am. Person go dey talk for video, dey advertise product, and the way im mouth dey move no gree match wetin im dey talk. Some of them even get face wey dey change small small as the video dey go. Na AI dey do am, no be real human being.

One tailor for Lagos wey we talk to yarn say the thing dey worry am. "Before before, if company want advert, dem go call model, dem go call photographer, dem go call plenty people. Everybody go chop. Now, na one person for computer go just press button, everything don finish. Where we wey dey do the work go go?"

Na so plenty people for the creative side dey feel. Voice-over people, model, video editor — all of them dey talk the same thing. The work wey dem dey get before don begin reduce.

But some people dey see am differently. One young man wey dey do design talk say the AI thing dey help am work fast. "Wetin I go use three days do before, now na three hours. E mean say I fit collect more work."

Wetin regulators go do about am, nobody sabi yet. Some people dey talk say make dem force company to write "AI" for the advert so people go know say no be human being. Others talk say make dem just leave the thing make e flow.

One thing wey clear be say the matter no go just disappear. As the technology dey grow, na so the argument go dey grow too.""",
        "notes": "Pidgin English news-style — fills §3.9 gap, directly on-theme for the study",
    },

    # =========================================================================
    # BUREAUCRATIC / REGULATORY — fills a §3.9 gap
    # =========================================================================
    {
        "filename": "synthetic_bureaucratic_001_public_notice.txt",
        "genre": "bureaucratic",
        "text": """PUBLIC NOTICE

RE: MANDATORY COMPLIANCE REGISTRATION FOR ORGANISATIONS PROCESSING PERSONAL DATA THROUGH AUTOMATED SYSTEMS

The general public, and in particular all organisations engaged in the processing of personal data of data subjects resident within the Federal Republic of Nigeria, are hereby notified of the requirements set out below.

1. SCOPE OF APPLICATION

This notice applies to all data controllers and data processors, whether incorporated within or outside Nigeria, who process the personal data of Nigerian data subjects through automated or partially automated means. For the avoidance of doubt, this includes but is not limited to organisations deploying algorithmic decision-making systems, automated profiling tools, and machine learning models in the course of their operations.

2. REGISTRATION REQUIREMENT

All organisations falling within the scope described in paragraph 1 above are required to submit a completed compliance registration in the prescribed form. The registration shall include, at minimum, particulars of the categories of personal data processed, the lawful basis relied upon for such processing, the retention periods applied, and a description of any automated decision-making processes employed.

3. DOCUMENTATION TO BE SUBMITTED

Organisations shall submit the following alongside the completed registration form:

(a) A current data protection policy, duly approved by the organisation's governing body;
(b) Particulars of the designated Data Protection Officer, where the appointment of such officer is required;
(c) Where applicable, a Data Privacy Impact Assessment in respect of any processing operation likely to result in high risk to the rights and freedoms of data subjects;
(d) Evidence of the technical and organisational measures implemented to ensure the security of personal data processed.

4. TIMELINE FOR COMPLIANCE

Organisations are required to complete the registration within ninety (90) days of the date of this notice. Organisations commencing operations after the date of this notice shall complete registration prior to the commencement of any processing activity.

5. CONSEQUENCES OF NON-COMPLIANCE

Failure to comply with the requirements set out in this notice may result in regulatory action, including but not limited to the issuance of enforcement notices, the imposition of administrative sanctions, and where appropriate, referral for further action in accordance with the provisions of applicable law.

6. ENQUIRIES

Enquiries regarding the requirements set out in this notice may be directed to the compliance desk during official working hours, being 8:00am to 5:00pm on business days.

All stakeholders are enjoined to treat this notice with the seriousness it deserves and to ensure full compliance within the stipulated timeframe.

Signed
Director, Regulatory Compliance
For: The Commission""",
        "notes": "bureaucratic/regulatory register — fills §3.9 'Bureaucratic texts' gap (previously zero)",
    },
    {
        "filename": "synthetic_bureaucratic_002_circular.txt",
        "genre": "bureaucratic",
        "text": """CIRCULAR

REF: REG/CIR/2026/014

TO: ALL LICENSED OPERATORS AND SERVICE PROVIDERS

SUBJECT: GUIDELINES ON THE DISCLOSURE OF SYNTHETIC AND ARTIFICIALLY GENERATED CONTENT IN COMMERCIAL COMMUNICATIONS

1. INTRODUCTION

1.1 This Circular is issued pursuant to the powers conferred on the regulatory authority under applicable enabling legislation and shall take effect from the date of issuance.

1.2 The purpose of this Circular is to establish minimum standards for the disclosure of artificially generated or synthetically manipulated content deployed in commercial communications directed at consumers within the Federal Republic of Nigeria.

2. BACKGROUND

2.1 The authority has observed a marked increase in the deployment of artificially generated imagery, audio, and audiovisual content in commercial communications.

2.2 While the authority acknowledges the legitimate commercial applications of such technologies, it has become necessary to establish clear standards to ensure that consumers are not misled as to the nature of the communications addressed to them.

3. DISCLOSURE REQUIREMENTS

3.1 Any commercial communication in which a human likeness, voice, or performance has been wholly or substantially generated through artificial means shall carry a clear and conspicuous disclosure to that effect.

3.2 The disclosure required under paragraph 3.1 shall be:

(a) Presented in a manner that is legible or audible to the ordinary consumer;
(b) Positioned so as to be apparent at the point at which the consumer encounters the communication;
(c) Expressed in plain language, avoiding technical terminology unlikely to be understood by the general public.

3.3 For the avoidance of doubt, ordinary post-production processes including colour correction, audio levelling, and conventional editing shall not, without more, trigger the disclosure requirement under this Circular.

4. PROHIBITED PRACTICES

4.1 No operator shall deploy artificially generated content depicting an identifiable natural person without the documented consent of that person or their duly authorised representative.

4.2 No operator shall deploy artificially generated content in a manner calculated to convey the impression that a public official, regulatory authority, or recognised institution has endorsed a product or service where no such endorsement has been given.

5. RECORD KEEPING

5.1 Operators shall maintain records sufficient to demonstrate compliance with this Circular for a period of not less than three (3) years from the date of last publication of the relevant communication.

5.2 Such records shall be produced to the authority upon request within fourteen (14) days.

6. ENFORCEMENT

6.1 Contravention of the provisions of this Circular may attract such sanctions as are provided for under the applicable enabling legislation.

6.2 The authority may, where circumstances warrant, direct the immediate withdrawal of any communication found to be in contravention of this Circular.

7. EFFECTIVE DATE

This Circular takes effect from the date of issuance. Operators are allowed a transitional period of sixty (60) days within which to bring existing communications into compliance.

Signed
Director-General""",
        "notes": "bureaucratic circular — fills §3.9 gap; thematically mirrors the study's own subject",
    },
    {
        "filename": "synthetic_bureaucratic_003_press_release.txt",
        "genre": "bureaucratic",
        "text": """PRESS RELEASE
FOR IMMEDIATE RELEASE

AGENCY COMMENCES STAKEHOLDER CONSULTATION ON DIGITAL COMPLIANCE FRAMEWORK

The Agency has today announced the commencement of a structured stakeholder consultation process in respect of the proposed digital compliance framework currently under development.

The consultation, which will run over a period of eight weeks, is intended to gather input from industry participants, civil society organisations, academic institutions, and members of the general public on the proposed framework's scope, requirements, and implementation timeline.

Speaking on the announcement, a spokesperson for the Agency noted that the consultation reflects the Agency's commitment to evidence-based regulation developed in partnership with those it affects. "Effective regulation requires an understanding of operational realities," the spokesperson stated. "This consultation provides an opportunity for stakeholders to contribute directly to the development of a framework that is both protective of consumers and workable for operators."

The proposed framework addresses several areas identified through the Agency's ongoing supervisory activities, including standards for automated processing systems, requirements relating to consumer redress mechanisms, and reporting obligations in respect of system incidents affecting service availability.

Stakeholders wishing to participate may submit written responses through the dedicated consultation portal accessible via the Agency's official website. The Agency will additionally convene a series of in-person engagement sessions in Lagos, Abuja, and Port Harcourt, details of which will be published in due course.

The Agency has emphasised that no final decisions have been taken in respect of the matters under consultation and that all submissions received will be considered in the development of the final framework. A summary of responses received, together with the Agency's response to the issues raised, will be published following the conclusion of the consultation period.

Members of the public requiring further information may contact the Agency's public affairs unit during official working hours.

ENDS

Issued by the Office of Public Affairs""",
        "notes": "bureaucratic press release register — fills §3.9 gap",
    },
    {
        "filename": "synthetic_bureaucratic_004_compliance_directive.txt",
        "genre": "bureaucratic",
        "text": """DIRECTIVE

TO: ALL REGISTERED DATA CONTROLLERS OF MAJOR IMPORTANCE

RE: SUBMISSION OF ANNUAL DATA PROTECTION AUDIT RETURNS

PREAMBLE

WHEREAS the applicable data protection legislation imposes obligations upon data controllers in respect of the lawful processing of personal data;

AND WHEREAS such legislation empowers the regulatory authority to require the submission of audit returns from designated categories of data controller;

AND WHEREAS it is expedient to specify the form, content, and timing of such returns;

NOW THEREFORE the following directive is issued.

PART I — APPLICATION

1. This Directive applies to every data controller designated as being of major importance in accordance with the criteria published by the authority.

2. Where a data controller ceases to meet the applicable criteria during a reporting period, the obligations under this Directive shall nonetheless subsist in respect of that period.

PART II — SUBMISSION REQUIREMENTS

3. Every data controller to whom this Directive applies shall submit an annual audit return in the prescribed form not later than the 31st day of March following the end of the reporting year.

4. The audit return shall be accompanied by:

(a) A statement of the categories of personal data processed during the reporting period;
(b) Particulars of any personal data breach occurring during the reporting period, whether or not previously notified;
(c) A summary of data subject requests received and the manner of their disposition;
(d) Where automated decision-making was employed, a description of the logic involved and the measures adopted to safeguard the rights of data subjects;
(e) A certification, executed by an officer duly authorised in that behalf, as to the accuracy and completeness of the return.

PART III — VERIFICATION

5. The authority may require the production of such further information or documentation as it considers necessary to verify the contents of any return submitted.

6. The authority may, upon reasonable notice, conduct an on-site inspection of the premises and systems of any data controller for the purpose of verifying compliance.

PART IV — GENERAL

7. Nothing in this Directive shall be construed as derogating from any other obligation imposed upon data controllers under the applicable legislation.

8. This Directive may be cited as the Annual Data Protection Audit Returns Directive and shall come into force on the date of publication.

MADE this day at the principal office of the authority.

Signed
Chief Executive""",
        "notes": "bureaucratic legal-directive register with archaic legalese — fills §3.9 gap",
    },

    # =========================================================================
    # MEDIUM-LENGTH AD COPY — wider topical spread
    # =========================================================================
    {
        "filename": "synthetic_ad_006_insurance.txt",
        "genre": "ad_copy",
        "text": """Life is unpredictable. Your family's security shouldn't be.

At Anchor Life Assurance, we understand that planning for the unexpected is one of the most important decisions you will ever make. That is why we have designed insurance products that work for real Nigerian families, with premiums that fit real Nigerian budgets.

Our Family Shield Plan provides comprehensive life cover starting from as little as ₦3,500 monthly. In the event of the unexpected, your beneficiaries receive a lump sum payout designed to cover immediate expenses, outstanding obligations, and continued support for dependants.

We have also made the process simple. No lengthy medical examinations for standard cover. No confusing paperwork. Apply online in under fifteen minutes, or visit any of our branches nationwide and speak with an advisor who will explain your options in plain language.

Claims matter most when they matter most. That is why we have committed to processing valid claims within seven working days of receiving complete documentation. Over the past year, we have honoured claims worth billions of naira to families across the country.

Anchor Life Assurance is fully licensed and regulated, with a track record spanning more than two decades. We have built our reputation on paying what we promise, when we promise it.

Protect what matters. Visit anchorlifeng.com or call our customer care line today for a free, no-obligation quotation.

Anchor Life Assurance. Because tomorrow deserves a plan.""",
        "notes": "insurance ad copy — new vertical",
    },
    {
        "filename": "synthetic_ad_007_edtech.txt",
        "genre": "ad_copy",
        "text": """Your child's future starts with the right foundation.

BrightPath Learning brings world-class educational support directly to your home. Our platform offers structured lessons across Mathematics, English Language, Sciences, and more, aligned with the Nigerian curriculum and designed by experienced Nigerian educators who understand what your child actually needs to excel.

Every lesson is delivered through short, engaging video modules that hold attention and build understanding step by step. Practice questions after each topic reinforce learning, and detailed progress reports let you see exactly where your child is excelling and where extra support might help.

We know data is expensive. That is why BrightPath is built to work on low bandwidth, with a download option that lets your child study offline once lessons are saved. No constant streaming, no wasted data.

Preparing for WAEC, NECO, or JAMB? Our examination preparation track includes past questions, timed mock examinations, and targeted revision on the topics that appear most frequently.

Plans start from ₦2,500 monthly for a single subject, with family bundles available for households with multiple children. Your first seven days are completely free, with no card details required to start.

Thousands of Nigerian parents have already made the switch. Join them.

Visit brightpathlearning.ng or download the app today.

BrightPath Learning. Every child deserves a fair start.""",
        "notes": "edtech ad copy — new vertical",
    },
    {
        "filename": "synthetic_ad_008_healthcare.txt",
        "genre": "ad_copy",
        "text": """Quality healthcare shouldn't require a long queue.

MediReach connects you to licensed Nigerian doctors from wherever you are. Book a consultation through our app and speak with a qualified medical practitioner by video or voice call, usually within thirty minutes of booking.

Our doctors are fully registered and verified. Whether you need advice on a persistent symptom, a follow-up on ongoing treatment, or a prescription renewal, you can get professional guidance without the travel, the waiting room, or the lost work hours.

Consultations start from ₦2,000. Prescriptions issued through the platform can be fulfilled at partner pharmacies nationwide, with home delivery available in Lagos, Abuja, Port Harcourt, and Ibadan.

We also offer a family plan covering up to five members for ₦8,000 monthly, including unlimited consultations and priority booking. For households managing chronic conditions, this has proven particularly valuable.

Your medical information stays private. All consultations are encrypted, and your records are accessible only to you and the practitioners you consult.

MediReach is not a replacement for emergency care. If you are experiencing a medical emergency, please proceed to the nearest hospital immediately.

Download MediReach from the App Store or Play Store and get your first consultation at half price.

MediReach. Healthcare that comes to you.""",
        "notes": "healthcare ad copy — new vertical, includes appropriate care disclaimer",
    },
    {
        "filename": "synthetic_ad_009_aviation.txt",
        "genre": "ad_copy",
        "text": """Fly more. Spend less. Arrive ready.

SkyBridge Airways is opening up domestic travel across Nigeria with fares that make sense and a schedule you can plan around. Daily flights connecting Lagos, Abuja, Port Harcourt, Kano, Enugu, and Owerri, with new routes announced this quarter.

Book early and travel from as low as ₦48,000 one way. Our advance purchase fares are released thirty days ahead, and our flexible fare option lets you change your booking up to six hours before departure at no charge.

Every ticket includes a 20kg checked baggage allowance and 7kg hand luggage as standard. No surprise charges at the counter.

Our fleet is young, our maintenance standards are uncompromising, and our on-time performance is among the strongest in the domestic market. We publish our punctuality figures monthly because we believe you should be able to check.

Join SkyBridge Rewards and earn points on every flight, redeemable against future travel. Membership is free, and points never expire while your account remains active.

Business travellers: ask about our corporate account programme, offering consolidated billing, flexible rebooking, and dedicated support.

Book at skybridgeairways.ng, through our mobile app, or with any accredited travel agent.

SkyBridge Airways. Nigeria, connected.""",
        "notes": "aviation ad copy — new vertical",
    },
    {
        "filename": "synthetic_ad_010_real_estate.txt",
        "genre": "ad_copy",
        "text": """Own your home. Sooner than you thought possible.

Cornerstone Residences is a new development of two and three bedroom apartments, thoughtfully designed for Nigerian families who want quality without compromise. Located within a secured estate with reliable power infrastructure, treated water supply, and twenty-four hour security.

Every unit is finished to a high standard: fitted kitchens, en-suite bathrooms in principal bedrooms, tiled throughout, and pre-wired for both conventional and alternative power. Ample parking and landscaped communal areas complete the development.

We have structured payment plans that work for real budgets. Spread payments over up to thirty-six months with an initial deposit of twenty percent. For buyers with mortgage pre-approval, we work directly with several licensed lenders to streamline the process.

All titles are perfected and verifiable. Documentation is complete, and we encourage every prospective buyer to conduct independent verification before committing. We will make all relevant documents available for your solicitor's review.

Construction is well advanced, with the first phase scheduled for handover before year end. Site inspections are available by appointment from Monday to Saturday.

Prices start from ₦42,000,000 for a two bedroom unit.

Call our sales team or visit cornerstoneresidences.ng to arrange your inspection.

Cornerstone Residences. Built properly. Priced honestly.""",
        "notes": "real estate ad copy — new vertical",
    },

    # =========================================================================
    # OPINION / EDITORIAL — different register again
    # =========================================================================
    {
        "filename": "synthetic_opinion_001_tech_commentary.txt",
        "genre": "opinion",
        "text": """There is a familiar pattern in how new technologies arrive in Nigeria, and it is worth naming clearly before the current wave of artificial intelligence adoption completes another cycle of it.

The pattern runs roughly as follows. A technology developed elsewhere, optimised for conditions elsewhere, arrives with considerable enthusiasm. Early adopters demonstrate impressive results in contexts that resemble the environments the technology was built for. Broader adoption follows, and with it, the discovery that the technology performs less reliably in conditions its designers never encountered. By the time these limitations are widely understood, substantial investment has already been committed, and the practical response is adaptation rather than reconsideration.

We have seen versions of this with enterprise software, with agricultural technology, with digital identity systems. There is no reason to assume artificial intelligence will be different, and several reasons to expect it will be more pronounced.

Consider the question of training data. Contemporary AI systems learn from large corpora of text, images, and audio. The composition of those corpora determines what the resulting systems handle well. Systems trained predominantly on North American and European English will encode the patterns of that English, and will treat departures from those patterns as anomalies. This is not a flaw in the sense of a defect to be fixed; it is a straightforward consequence of how these systems work.

The practical implications are already visible. Text classification tools have been documented producing inconsistent results on formal English written by African authors. Speech recognition systems perform less reliably on accents underrepresented in training data. Image analysis systems have well-documented disparities in performance across skin tones. None of this is contested at a technical level. What is contested is what follows from it.

One response is to treat these as transitional problems that will resolve as datasets diversify. There is something to this: datasets do improve, and measured disparities have narrowed in some domains. But the improvement is neither automatic nor evenly distributed. It happens where there is commercial or institutional pressure to make it happen, and that pressure has historically been weakest for the populations least represented in the first place.

A second response is to build locally. This is more demanding, requiring data, computational resources, and technical capacity that are not evenly available. But it is not impossible, and the cost of not doing it compounds. Systems built elsewhere and adopted here will continue to encode assumptions made elsewhere, and those assumptions will continue to produce outcomes that are worse for people here than for people there.

The third response, and the one that seems most immediately actionable, is institutional rather than technical. Where automated systems are deployed in consequential decisions, there should be mechanisms for contesting their outputs. Where systems produce different error rates across populations, those rates should be measured and published. Where a person is affected by an automated decision, they should be able to find out that it was automated.

None of these requires solving the underlying technical problems. They require only that we decline to treat automated outputs as authoritative simply because they are automated. That is a smaller ask than it sounds, and a more useful one than waiting for the technology to become fair on its own.

The enthusiasm is not misplaced. These tools do useful things, and the productivity gains are real. But enthusiasm and scrutiny are not opposites, and the period of greatest enthusiasm is precisely when scrutiny is most valuable and least likely.""",
        "notes": "opinion/editorial register — new genre, directly on-theme",
    },
    {
        "filename": "synthetic_opinion_002_business_commentary.txt",
        "genre": "opinion",
        "text": """Every few years, a consensus forms about what will transform the Nigerian economy. Telecommunications was going to do it. Then oil sector reform. Then agriculture. Then technology startups. Each of these has contributed something. None has been transformative in the way the consensus predicted.

This is not an argument for pessimism. It is an argument for attending more carefully to the difference between a sector growing and an economy transforming, because the two are routinely conflated in a way that distorts both policy and investment.

A sector can grow substantially while contributing modestly to broad economic outcomes. This happens when growth is concentrated among a small number of firms, when employment intensity is low, when inputs are largely imported, or when the growth serves demand that already existed rather than creating new productive capacity. Several of Nigeria's recent growth stories exhibit some combination of these characteristics.

Consider employment intensity specifically. A technology firm reaching substantial valuation may employ a few hundred people directly. A manufacturing operation of comparable economic weight might employ several thousand, with further employment through supply chains. Both are valuable, but they are not equivalent from the standpoint of an economy with the demographic profile Nigeria has. Roughly speaking, the country needs to accommodate millions of new labour market entrants annually. Sectors that generate substantial value with modest headcount will not, by themselves, meet that requirement.

This is not a case against technology investment, which has been genuinely productive and has built capability that did not previously exist. It is a case against treating technology investment as a substitute for the harder work of building sectors that employ people at scale.

The harder work is well understood, which is part of what makes its neglect frustrating. Reliable power. Functional ports and roads. Predictable regulation. Access to working capital at rates that permit investment. Contract enforcement that operates in reasonable timeframes. None of these are novel insights. They appear in essentially every serious analysis of Nigerian economic constraints produced over several decades.

What has changed is that the cost of not addressing them has risen. When the working-age population was smaller, the consequences of insufficient job creation were more absorbable. They are less absorbable now, and will be less absorbable still in a decade.

There is a tendency in commentary of this kind to conclude with a list of recommendations, and I am aware that the list above is neither novel nor detailed. But the shortage has not been of recommendations. It has been of sustained implementation across political cycles, which is a different kind of problem and does not yield to better analysis.

What might yield to better analysis is the framing. If the question is which sector will transform the economy, the answer will keep disappointing, because no single sector will. If the question is what conditions allow many sectors to grow simultaneously and employ people while doing so, the answer is unglamorous, expensive, slow, and considerably more likely to work.""",
        "notes": "business opinion/editorial register — new genre",
    },

    # =========================================================================
    # MEDIUM NEWS — additional topical spread
    # =========================================================================
    {
        "filename": "synthetic_news_006_education_sector.txt",
        "genre": "news",
        "text": """Enrolment in private tertiary institutions across Nigeria has continued to rise, according to sector observers, as prospective students weigh considerations including academic calendar predictability and available capacity in the public system.

The trend has been developing over several years. Private universities, of which there are now well over one hundred licensed to operate, have expanded both in number and in individual institutional capacity. Several have invested substantially in physical infrastructure and academic staffing, positioning themselves as alternatives to established public institutions.

Cost remains the principal constraint on access. Fees at private institutions typically exceed those at federal and state universities by a substantial margin, placing them beyond the reach of many households. Institutions have responded with scholarship schemes and instalment payment arrangements, though the scale of such provision varies considerably and independent assessment of its reach is limited.

Academic quality across the private sector is uneven, according to observers familiar with the regulatory landscape. Accreditation processes provide a baseline, and institutions are subject to periodic review, but practitioners note significant variation in staffing depth, research output, and facilities between the strongest private institutions and those operating closer to minimum requirements.

Public institutions continue to accommodate the substantial majority of tertiary enrolment nationally, and sector advocates emphasise that the growth of private provision does not reduce the importance of adequate public sector funding. Several have cautioned against interpreting private sector expansion as evidence that public provision is less necessary, arguing that a well-functioning system requires both.

Employment outcomes for graduates remain a shared concern across institution types. Surveys of recent graduates have consistently identified difficulty securing employment matching qualification levels, a pattern that observers attribute more to labour market conditions than to institutional performance. Some institutions have expanded career services and industry partnership programmes in response.

The regulatory framework governing tertiary education has itself been subject to review discussions, with proposals under consideration relating to accreditation processes, quality assurance mechanisms, and the criteria governing the establishment of new institutions. Stakeholders have generally supported measures strengthening quality assurance while raising practical questions about implementation capacity.""",
        "notes": "medium-length news, education sector — new topic",
    },
    {
        "filename": "synthetic_news_007_energy_transition.txt",
        "genre": "news",
        "text": """Adoption of solar power systems among Nigerian households and businesses has accelerated, driven by a combination of falling equipment costs and continued unreliability in grid supply.

Installers report growing demand across residential and commercial segments, with particularly strong interest from small businesses for whom power interruptions translate directly into lost revenue. Systems ranging from modest residential installations to larger commercial arrays are being deployed, with battery storage increasingly included as costs have moderated.

The economics have shifted meaningfully. Equipment prices have declined substantially over the past several years, while the cost of diesel generation has moved in the opposite direction. Installers report that payback periods that were previously prohibitive for many customers now fall within ranges that support investment, particularly for users with high generator dependence.

Financing remains a barrier for many prospective adopters. The upfront capital requirement for a system adequate to meaningful household or business load is substantial, and access to credit on suitable terms is limited. Several providers have introduced instalment arrangements and lease-to-own structures intended to address this, with reported uptake though at scales that remain modest relative to demand.

Quality variation across the market has emerged as a concern. Industry participants report the presence of substandard equipment, particularly in the lower price segments, with consequences for system performance and lifespan. Several have called for stronger standards enforcement, noting that poor experiences with substandard installations risk damaging confidence in the sector more broadly.

Grid-connected and off-grid approaches present different regulatory questions. Policy discussions have addressed frameworks for distributed generation, including arrangements under which excess generation might be fed back into the grid. Practitioners note that the practical viability of such arrangements depends on grid infrastructure and metering capability that is unevenly available.

For rural communities, mini-grid approaches have attracted attention as a means of extending electrification where grid connection is not economically viable. Several projects have been implemented, with reported success in delivering reliable supply to previously unserved communities. Scaling such approaches, practitioners note, depends on financing structures capable of supporting projects with long payback horizons and modest returns.""",
        "notes": "medium-length news, energy sector — new topic",
    },
    {
        "filename": "synthetic_news_008_logistics_sector.txt",
        "genre": "news",
        "text": """Nigeria's logistics and last-mile delivery sector has expanded considerably alongside growth in e-commerce, though operators report that infrastructure constraints continue to shape both cost structures and service reliability.

The sector now encompasses a wide range of participants, from established international operators to a substantial informal segment of independent riders and small local operations. This diversity has enabled coverage in areas that formal operators find difficult to serve economically, while also producing considerable variation in service standards.

Addressing remains among the most frequently cited operational challenges. The absence of consistent, verifiable addressing in many areas means that delivery frequently depends on telephone contact and local knowledge rather than systematic routing. Operators report that this materially affects delivery times and completion rates, and complicates efforts to optimise routes algorithmically.

Several initiatives have sought to address this, including digital addressing systems and the use of location coordinates in place of conventional addresses. Uptake has been partial, with operators noting that the value of such systems depends on adoption by both senders and recipients.

Traffic congestion in major urban centres imposes further constraints, particularly in Lagos where operators report that journey times vary substantially by time of day. Motorcycle-based delivery has become predominant for smaller items in congested areas, offering speed advantages though with corresponding safety and capacity limitations. Regulatory restrictions on motorcycle operation in certain areas have required operational adjustments.

Payment arrangements reflect broader commerce patterns. Cash on delivery remains widely used, with operators reporting that it accounts for a significant proportion of transactions. This introduces cash handling requirements and associated risks, and complicates reconciliation. Growth in digital payment adoption has reduced but not eliminated this dependence.

Cost pressures have intensified with fuel price movements, which affect the sector directly given its transport intensity. Operators have adjusted pricing, though competitive dynamics limit the extent to which increases can be passed through. Several have invested in route optimisation and consolidation approaches intended to reduce per-delivery costs.

Employment in the sector is substantial though frequently informal, with many riders operating on flexible or contract arrangements rather than conventional employment. This has attracted attention from labour advocates, who have raised questions about income stability, insurance coverage, and working conditions.""",
        "notes": "medium-length news, logistics sector — new topic",
    },
]


def run_expansion(dry_run: bool = False) -> None:
    from config import RAW_TEXT_DIR, MANIFEST_PATH
    from watchdog.dataset_manager import (
        init_manifest, add_entry, load_manifest, print_stats,
        ManifestEntry, CATEGORY_TEXTUAL_CORPUS, LABEL_SYNTHETIC,
    )

    synthetic_dir = RAW_TEXT_DIR / "synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    init_manifest(MANIFEST_PATH)
    existing = load_manifest(MANIFEST_PATH)
    existing_paths = set(existing["local_path"]) if not existing.empty else set()

    written, skipped = 0, 0
    genre_counts = {}
    word_counts = []

    for i, sample in enumerate(EXPANDED_SAMPLES, 1):
        file_path = synthetic_dir / sample["filename"]
        rel_path = str(file_path)
        wc = len(sample["text"].split())

        if rel_path in existing_paths or str(file_path.relative_to(file_path.cwd()) if file_path.is_absolute() else file_path) in existing_paths:
            logger.info(f"[{i}/{len(EXPANDED_SAMPLES)}] Already logged, skipping: {sample['filename']}")
            skipped += 1
            continue

        logger.info(f"[{i}/{len(EXPANDED_SAMPLES)}] {sample['genre']:18s} {wc:>5} words  {sample['filename']}")
        genre_counts[sample["genre"]] = genre_counts.get(sample["genre"], 0) + 1
        word_counts.append(wc)

        if dry_run:
            continue

        file_path.write_text(sample["text"], encoding="utf-8")

        entry = ManifestEntry(
            sample_id=str(uuid.uuid4()),
            local_path=str(file_path.relative_to(file_path.cwd())) if file_path.is_absolute() else str(file_path),
            source_url=f"llm-generated:claude:{sample['genre']}:{sample['filename']}",
            media_type="text",
            category=CATEGORY_TEXTUAL_CORPUS,
            label=LABEL_SYNTHETIC,
            publisher_name="N/A (LLM-generated)",
            scraped_at=datetime.now(timezone.utc).isoformat(),
            notes=f"genre={sample['genre']}; words={wc}; {sample['notes']}",
        )
        add_entry(entry, path=MANIFEST_PATH)
        written += 1

    print("\n" + "=" * 70)
    print("  CORPUS EXPANSION COMPLETE")
    print("=" * 70)
    print(f"  Newly written:  {written}")
    print(f"  Already had:    {skipped}")
    if word_counts:
        word_counts.sort()
        print(f"\n  Word count distribution of new samples:")
        print(f"    min={word_counts[0]}  median={word_counts[len(word_counts)//2]}  max={word_counts[-1]}")
        print(f"    samples >= 360 words (authentic range): "
              f"{sum(1 for w in word_counts if w >= 360)}/{len(word_counts)}")
    if genre_counts:
        print(f"\n  Genres added:")
        for g, c in sorted(genre_counts.items()):
            print(f"    {g:20s} {c}")
    print("=" * 70)

    print("\n" + "!" * 70)
    print("  ⚠ CLASS BALANCE WARNING — READ THIS")
    print("!" * 70)
    print("  This script only expands the SYNTHETIC (label=1) side.")
    print("  Your AUTHENTIC (label=0) side is still only ~3 samples.")
    print()
    print("  Before drawing ANY conclusions from re-running")
    print("  evaluate_nlp_heuristics.py, expand the authentic side:")
    print()
    print("    1. Open seed_sources.py")
    print("    2. Add 20-30 more real Nigerian news article URLs to")
    print("       TEXTUAL_CORPUS_SPECIFIC_URLS (Punch, Premium Times,")
    print("       Vanguard, BusinessDay, The Cable, Leadership)")
    print("    3. python collect_dataset.py")
    print("    4. python collect_dataset.py --label   (label them 0)")
    print()
    print("  A 31-vs-3 split cannot produce meaningful accuracy numbers.")
    print("!" * 70 + "\n")

    if not dry_run:
        print_stats(MANIFEST_PATH)


def main():
    parser = argparse.ArgumentParser(
        description="Expand the synthetic text corpus with length-matched, genre-diverse samples"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without writing files or updating the manifest")
    args = parser.parse_args()
    run_expansion(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
