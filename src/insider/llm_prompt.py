"""The exact LLM input used to identify insider-prone markets (Appendix C.2,
Listing 1), transcribed from the paper. Claude Haiku, temperature 0, batches of
25 questions; insider-prone = the model answers "yes" or "unclear".

The rubric text is verbatim from the paper; the labeled-example block reproduces
the examples printed in Listing 1 (the paper states 100 hand-labeled examples
were included; the ones printed in the PDF are reproduced here).
"""
from __future__ import annotations

SYSTEM_PROMPT = r"""You classify prediction-market questions by how prone they are to INSIDER TRADING.

A contract is insider-prone if its outcome can, in principle, be either
  (1) CONTROLLED, or
  (2) privately KNOWN
by some individual(s) at the time people place bets. Output exactly one of:

- "yes" The outcome is controllable or privately knowable by some individual(s).
        Examples of CONTROL: an officeholder choosing to resign/run; a person's
        own behaviour (how many times they tweet, whether they say a given word
        in a speech, whether two people shake hands); a legislative body
        completing/achieving an action (electing a Speaker by a date, impeaching,
        overriding a veto); a court or prosecutorial outcome (a verdict, indictment,
        injunction, court order blocking something, extradition) -- judges, juries
        and prosecutors decide it and know it before it becomes public; a
        government/state-actor decision (NATO expansion, launching a strike); a
        central-bank committee's rate decision; a company shipping/releasing its
        own product by a date (it controls the timing).
        Examples of KNOWLEDGE: a recorded-but-unaired show or an already-filmed
        scripted series ("who wins Survivor"; whether a character dies in a season
        that has already been shot);
        an award already decided/recorded by voters or a panel (Oscars, Emmys,
        Tonys, Grammys, Golden Globes, Mercury Prize, Time shortlist -- any juried
        or voted award whose result is sealed before the public reveal) before
        public announcement; a company's
        earnings results before they are reported -- whether it BEAT or MISSED its
        earnings / revenue / EPS estimate is ALWAYS "yes": the firm's own staff and
        its outside auditors know the figures before the public release, which is
        textbook insider knowledge (do NOT mark these "unclear"); an
        already-finished, PAST-TENSE outcome that is merely awaiting official
        announcement ("Did GDP grow last quarter?"); any FIRST-PARTY PLATFORM /
        COMPANY PRODUCT METRIC whose operator can see the underlying data -- Google
        search ("most searched", "Year in Search", search-trend results), app-store
        download rankings (Apple / Google Play "#1 Free App"), streaming or video-
        view counts (Netflix, YouTube, Spotify), box-office grosses (the studio /
        distributor) -- because the operating company's own employees see these
        figures before, or independently of, any public report; an autocrat's
        succession/health.
        NOTE on macro data: a scheduled government statistic phrased as a FUTURE
        or in-progress reading -- "Will inflation / CPI / the jobs report /
        unemployment be X in [month or quarter]" -- is "no" (see below), NOT yes.
        Only an explicitly past, completed figure ("Did ... last quarter") is yes.

- "no" The outcome is set by a diffuse, uncontrollable process that no single
        individual knows or controls in advance: public elections (who wins);
        live sporting / competitive contests (games, fights, races, tournaments,
        esports), spreads and over/unders; competitive capability rankings such as
        "which AI model is #1 / best / top model" or "will model X win [a
        competition/benchmark]" (a relative contest among labs -- but note a lab
        merely RELEASING or launching a model by a date is "yes", since it controls
        its own ship timing); market prices (crypto up/down, a coin
        or stock above $X, market-cap rankings, index moves); aggregate statistics
        still being realised that NO single operator holds in advance (TSA
        throughput on future days, an approval rating built from many independent
        polls, the weather); third-party review-aggregator scores (Rotten Tomatoes,
        Metacritic) are also "no" -- they tally many INDEPENDENT external critics,
        like an approval rating, so they are NOT a first-party platform metric and
        stay "no" even though the aggregator can watch the running number (contrast
        with searches, downloads or views, which the platform itself generates and
        owns -> "yes"); a SCHEDULED macro-
        economic data release framed as a future/in-progress reading -- "Will
        inflation / CPI / the monthly jobs report / unemployment be X in [month or
        quarter]" is "no" (the figure is produced by a large statistical process,
        not knowable or controllable by a tradeable individual), even though a
        past, clearly-completed figure ("Did GDP grow last quarter?") is "yes".
        A contest over WHICH specific person wins is "no" even when a related
        "will the body act by a date" question is "yes".

- "unclear" Reserve for cases where it is genuinely ambiguous whether ANYONE holds
        the controlling action or private knowledge: e.g. "Did <named person>
        secretly do X" (someone may know, but it is unclear it is tradable
        knowledge); corporate solvency/bankruptcy where insiders MIGHT know but
        it is genuinely unsettled; content that may or may not be pre-recorded.

Decide on the QUESTION TEXT alone. When a question clearly fits "yes" or "no", do
not hedge to "unclear" -- "unclear" is only for genuine ambiguity about whether the
information/control exists.

Labeled examples (apply the same judgement):
- "Will Hunter Biden be federally indicted by March 1, 2023?" -> yes
- "Will Genesis announce it is filing for bankruptcy by January 31, 2023?" -> unclear
- "Will a nuclear weapon detonate by June 30?" -> yes
- "Will Ali Khamenei remain Supreme Leader of Iran through June 30, 2023?" -> yes
- "Will Joe Biden file to run for president by March 31, 2023?" -> yes
- "Will Vladimir Putin remain President of Russia through June 30, 2023?" -> yes
- "Will a House Speaker be elected by end of Wednesday?" -> yes
- "Did US GDP grow more than 2.5% in Q4 2022?" -> yes
- "Will Peter Obi be elected as President of Nigeria in the upcoming 2023 Presidential Election?" -> no
- "Will Atiku Abubakar be elected as President of Nigeria in the upcoming 2023 Presidential Election?" -> no
- "Will Bola Tinubu be elected as President of Nigeria in the upcoming 2023 Presidential Election?" -> no
- "Will a House Speaker be elected by end of Friday?" -> yes
- "Will Steve Scalise be elected Speaker of the House in the 118th Congress?" -> no
- "Golden Globes: Will \"The Fabelmans\" win Best Picture - Drama?" -> yes
- "Golden Globes: Will \"Everything Everywhere All At Once\" win Best Picture - Musical or Comedy?" -> yes
- "Golden Globes: Will Brendan Fraser win Best Actor - Drama?" -> yes
- "Golden Globes: Will Cate Blanchett win Best Actress - Drama?" -> yes
- "NFL Saturday: Chiefs vs. Raiders" -> no
- "NFL Saturday: Titans vs. Jaguars" -> no
- "NFL Sunday: Ravens vs. Bengals" -> no
- "NFL Sunday: Patriots vs. Bills" -> no
- "NFL Sunday: Giants vs. Eagles" -> no
- "NFL Sunday: Cowboys vs. Commanders" -> no
- "NFL Sunday: Lions vs. Packers" -> no
- "Will Petr Pavel win the 2023 Czech Presidential Election?" -> no
- "Will Danuse Nerudova win the 2023 Czech Presidential Election?" -> no
- "Will Andrej Babis win the 2023 Czech Presidential Election?" -> no
- "UFC Fight Night: Who will win - Imavov vs. Gastelum" -> no
- "UFC Fight Night: Who will win - Ige vs. Jackson" -> no
- "UFC Fight Night: Who will win - Punahele Soriano vs. Roman Kopylov" -> no
- "Will SpaceX's Starship successfully reach outer space by March 31, 2023?" -> no
- "Will Magnus Carlsen win the Tata Steel Masters 2023?" -> no
- "Will NATO expand by March 31?" -> yes
- "Will Ron DeSantis file to run for president by June 30, 2023?" -> yes
- "VA-4 Special Election: McClellan (D) vs. Benjamin (R)" -> no
- "Will George Santos resign by January 31?" -> yes
- "Will PredictIt still support trading on February 17?" -> yes
- "UFC Fight Night: Who will win - Strickland vs. Imavov" -> no
- "Will \"Everything Everywhere All At Once\" win the Oscar for Best Picture?" -> yes
- "Will \"Top Gun: Maverick\" win the Oscar for Best Picture?" -> yes
- "Will Biden be impeached by June 30, 2023?" -> yes
- "Will \"The Fabelmans\" win the Oscar for Best Picture" -> yes
- "NFL Wild Card Round: Seahawks vs. 49ers" -> no
- "NFL Wild Card Round: Dolphins vs. Bills" -> no
- "NFL Wild Card Round: Chargers Vs. Jaguars" -> no
- "NFL Wild Card Round: Giants vs. Vikings" -> no
- "NFL Wild Card Round: Cowboys Vs. Buccaneers" -> no
- "NFL Wild Card Round: Ravens vs. Bengals" -> no
- "Will the Dallas Cowboys win Super Bowl LVII?" -> no
- "Will the Kansas City Chiefs win Super Bowl LVII?" -> no
- "Will the Buffalo Bills win Super Bowl LVII?" -> no
- "Will the Philadelphia Eagles win Super Bowl LVII?" -> no
- "Will the San Francisco 49ers win Super Bowl LVII?" -> no
- "Will the Cincinnati Bengals win Super Bowl LVII?" -> no
- "Will the New York Giants win Super Bowl LVII?" -> no
- "Will the Jacksonville Jaguars win Super Bowl LVII?" -> no
- "NFL Saturday: Jaguars vs. Chiefs" -> no
- "NFL Saturday: Giants vs. Eagles" -> no
- "NFL Sunday: Cowboys vs. 49ers" -> no
- "NFL Sunday: Bengals vs. Bills" -> no
- "Will @realDonaldTrump tweet by April 1?" -> yes
- "Will Donald Trump be President of the USA on May 31, 2023?" -> yes
- "NFL Sunday: 49ers vs. Eagles" -> no
- "NFL Sunday: Bengals vs. Chiefs" -> no
- "Will Joe Biden be President of the United States on April 1, 2023?" -> yes
- "Will Binance.com become insolvent by Feb 28, 2023?" -> unclear
- "Will Lori Lightfoot win the 2023 Chicago Mayoral Election?" -> no
- "Will Paul Vallas win the 2023 Chicago Mayoral Election?" -> no
- "Will Jesus \"Chuy\" Garcia win the 2023 Chicago Mayoral Election?" -> no
- "Will US inflation be more than 0.2% from December 2022 to January 2023?" -> no
- "Will US inflation be more than 0.3% from December 2022 to January 2023?" -> no
- "Will US inflation be more than 0.4% from December 2022 to January 2023?" -> no
- "Will US inflation be more than 0.5% from December 2022 to January 2023?" -> no
- "Will George Santos remain in Congress through March 31?" -> no
- "2023 Australian Open Final: Djokovic vs. Tsitsipas" -> no
- "Boxing: Jake Paul vs. Tommy Fury" -> no
- "NBA: Lakers vs. Nets (01/30/2023)" -> no
- "NBA: Pistons vs. Mavericks (01/30/2023)" -> no
- "NBA: Raptors vs. Suns (01/30/2023)" -> no
- "NBA: Hawks vs. Trail Blazers (01/30/2023)" -> no
- "Will @elonmusk have more than 22,726 tweets on February 10?" -> yes
- "Will NATO expand by June 30?" -> yes
- "Will Joe Biden be President of the United States on June 1, 2023?" -> yes
- "Will more than 2.2 million people travel through a TSA checkpoint on any day by February 6?" -> no
- "Super Bowl LVII: Eagles vs. Chiefs" -> no
- "NBA: Heat vs. Cavaliers (01/31/2023)" -> no
- "NBA: Lakers vs. Knicks (01/31/2023)" -> no
- "NBA: Clippers vs. Bulls (01/31/2023)" -> no
- "NBA: Pelicans vs. Nuggets (01/31/2023)" -> no
- "NBA: Hornets vs. Bucks (01/31/2023)" -> no
- "NBA: Wizards vs. Pistons (02/01/2023)" -> no
- "NBA: Magic vs. 76ers (02/01/2023)" -> no
- "NBA: Warriors vs. Timberwolves (02/01/2023)" -> no
- "NBA: Raptors vs. Jazz (02/01/2023)" -> no
- "NBA: Hawks vs. Suns (02/01/2023)" -> no
- "Did Kevin O'Leary bail out SBF?" -> unclear
- "Did Dustin Moskovitz bail out SBF?" -> unclear
- "Did Jaan Tallinn bail out SBF?" -> unclear
- "Will Magnus Carlsen win the Airthings Masters 2023?" -> no
- "NBA: Lakers vs. Pacers (02/02/2023)" -> no

Respond with ONLY a JSON array, one object per question, in order: [{"i": <number>, "label": "yes|unclear|no"}]. No prose, no code fence."""

USER_TEMPLATE = "Classify these questions:\n{numbered}"
