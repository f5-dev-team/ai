LABELS = [
    "ACADEMIC",
    "FINANCIAL",
    "PERSONAL_ELIGIBILITY",
    "ACTIVITY",
    "ADMINISTRATIVE",
    "PERIOD",
    "OTHER",
]

LABEL_TO_ID = {
    label: index
    for index, label in enumerate(LABELS)
}

ID_TO_LABEL = {
    index: label
    for label, index in LABEL_TO_ID.items()
}