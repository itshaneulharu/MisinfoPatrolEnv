"""
MisinfoPatrolEnv — Task Definitions
Three tasks ranging from easy (classic myth) → medium (mixed facts) → hard (misleading stats).
Each task defines a viral social media post, its embedded claims, correct verdicts, and overall label.
"""

TASKS = [
    {
        "id": "easy_brain_myth",
        "difficulty": "easy",
        "post_text": (
            "🚨 BREAKING — Harvard scientists just CONFIRMED what we always suspected: "
            "humans only use 10% of their brains! 🧠 This is why meditation and special "
            "brain supplements can UNLOCK the other 90%! Big Pharma doesn't want you to "
            "know this. Share before it gets CENSORED! 🔥 #MindUnlocked #BigPharmaLies"
        ),
        "claims": [
            "humans only use 10% of their brains",
            "Harvard scientists recently confirmed the 10% brain myth",
            "meditation and supplements can unlock unused brain capacity",
            "this information is being censored by Big Pharma",
        ],
        "claim_verdicts": ["false", "false", "misleading", "false"],
        "overall_label": "misinformation",
        "explanation": (
            "The '10% brain myth' is thoroughly debunked by neuroscience. Brain imaging (fMRI/PET) "
            "shows virtually all brain regions are active. No Harvard study supports this. "
            "The censorship framing is a manipulation tactic with no evidence."
        ),
    },
    {
        "id": "medium_mixed_facts",
        "difficulty": "medium",
        "post_text": (
            "🌍 Mind-blowing science facts your teacher never told you!\n"
            "✅ The Great Wall of China is visible from space with the naked eye\n"
            "✅ Water covers about 71% of Earth's surface\n"
            "✅ Lightning never strikes the same place twice\n"
            "✅ The Sahara Desert is the world's largest desert\n"
            "Science is amazing! Like & share! 🔬"
        ),
        "claims": [
            "The Great Wall of China is visible from space with the naked eye",
            "Water covers about 71% of Earth's surface",
            "Lightning never strikes the same place twice",
            "The Sahara Desert is the world's largest desert",
        ],
        "claim_verdicts": ["false", "true", "false", "false"],
        "overall_label": "misinformation",
        "explanation": (
            "Multiple false claims embedded among a true one: (1) The Great Wall is too narrow "
            "to see from orbit — confirmed false by astronauts including Chinese astronaut Yang Liwei. "
            "(2) ~71% water coverage is accurate. (3) Lightning absolutely strikes the same place "
            "multiple times — the Empire State Building is hit ~20-25x/year. (4) Antarctica is the "
            "world's largest desert by area (~14.2M km²), larger than the Sahara (~9.2M km²)."
        ),
    },
    {
        "id": "hard_misleading_vaers",
        "difficulty": "hard",
        "post_text": (
            "🚨 They can't hide this anymore. The CDC's own VAERS database recorded "
            "OVER 900,000 adverse event reports following COVID-19 vaccination — nearly "
            "a million people harmed! The FDA approved these vaccines in just 8 months "
            "versus the usual 10+ years. The mRNA technology had NEVER been used in a "
            "human vaccine before 2021. Why is mainstream media silent? Do your own "
            "research. 🧵 #VaccineInjury #DoYourResearch"
        ),
        "claims": [
            "VAERS recorded over 900,000 adverse event reports after COVID vaccines",
            "nearly a million people were harmed by COVID vaccines",
            "COVID vaccines were FDA approved in 8 months vs usual 10+ years",
            "mRNA technology had never been used in a human vaccine before 2021",
            "mainstream media is silent about vaccine harms",
        ],
        "claim_verdicts": ["true", "false", "misleading", "misleading", "false"],
        "overall_label": "misleading",
        "explanation": (
            "Classic misleading-statistics pattern: (1) VAERS report count is technically accurate "
            "but VAERS is a passive, unverified self-reporting system — anyone can submit a report, "
            "reports do not imply causation or confirmation of harm. (2) Conflating 'reports' with "
            "'people harmed' is false. (3) The 8-month timeline omits that platform technology, "
            "regulatory frameworks, and manufacturing were pre-established; trials ran in parallel, "
            "not sequentially. (4) mRNA research has been ongoing since the 1990s and was tested in "
            "cancer vaccine trials before COVID. (5) Mainstream media extensively covered vaccine "
            "side effects and VAERS data."
        ),
    },
]
