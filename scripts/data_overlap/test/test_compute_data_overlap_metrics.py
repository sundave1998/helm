from typing import List

from compute_data_overlap_metrics import (
    compute_scenario_document_data_overlap,
    create_all_data_overlap_stats,
    create_ngram_index,
    EntryDataOverlapKey,
    Ngram,
    NgramIndex,
    AllDataOverlapStats,
)
from light_scenario import LightScenario, LightInstance, LightScenarioKey
from light_tokenizer import LightTokenizer, DefaultTokenizer
from data_overlap_stats import (
    DataOverlapStats,
    DataOverlapStatsKey,
    PART_INPUT,
    PART_REF,
)
from common.general import asdict_without_nones

N_VALUES = [5, 13]

TEST_DOCUMENT: str = (
    "The Center for Research on Foundation Models (CRFM) is "
    "an interdisciplinary initiative born out of the Stanford "
    "Institute for Human-Centered Artificial Intelligence (HAI) "
    "that aims to make fundamental advances in the study, development, "
    "and deployment of foundation models."
)

TEST_TOKENS_SPLIT_BY_SPACE: List[str] = [
    "The",
    "Center",
    "for",
    "Research",
    "on",
    "Foundation",
    "Models",
    "(CRFM)",
    "is",
    "an",
    "interdisciplinary",
    "initiative",
    "born",
    "out",
    "of",
    "the",
    "Stanford",
    "Institute",
    "for",
    "Human-Centered",
    "Artificial",
    "Intelligence",
    "(HAI)",
    "that",
    "aims",
    "to",
    "make",
    "fundamental",
    "advances",
    "in",
    "the",
    "study,",
    "development,",
    "and",
    "deployment",
    "of",
    "foundation",
    "models.",
]

TEST_TOKENS_BY_DEFAULT_TOKENIZER: List[str] = [
    "the",
    "center",
    "for",
    "research",
    "on",
    "foundation",
    "models",
    "crfm",
    "is",
    "an",
    "interdisciplinary",
    "initiative",
    "born",
    "out",
    "of",
    "the",
    "stanford",
    "institute",
    "for",
    "human",
    "centered",
    "artificial",
    "intelligence",
    "hai",
    "that",
    "aims",
    "to",
    "make",
    "fundamental",
    "advances",
    "in",
    "the",
    "study",
    "development",
    "and",
    "deployment",
    "of",
    "foundation",
    "models",
    "",
]

TEST_SCENARIO_1 = LightScenario(
    light_scenario_key=LightScenarioKey(metadata={"name": "TEST_SCENARIO_1"}),
    light_instances=[
        LightInstance(input="Center for Research on Foundation", references=["bar", "baz"]),
        LightInstance(input="bar bar", references=["foo", "baz"]),
    ],
)
TEST_SCENARIO_2 = LightScenario(
    light_scenario_key=LightScenarioKey(metadata={"name": "TEST_SCENARIO_2"}),
    light_instances=[LightInstance(input=TEST_DOCUMENT, references=[TEST_DOCUMENT, TEST_DOCUMENT])],
)


def test_light_tokenizer():
    light_tokenizer = LightTokenizer()
    default_tokenizer = DefaultTokenizer()

    assert light_tokenizer.tokenize(TEST_DOCUMENT) == TEST_TOKENS_SPLIT_BY_SPACE
    assert default_tokenizer.tokenize(TEST_DOCUMENT) == TEST_TOKENS_BY_DEFAULT_TOKENIZER

    simple_tokenization_test_case: str = "THis ,,iS a SiMPlE t-EsT cAsE"
    assert light_tokenizer.tokenize(simple_tokenization_test_case) == ["THis", ",,iS", "a", "SiMPlE", "t-EsT", "cAsE"]
    assert default_tokenizer.tokenize(simple_tokenization_test_case) == [
        "this",
        "is",
        "a",
        "simple",
        "t",
        "est",
        "case",
    ]


def test_create_overlap_stats():
    scenarios = [TEST_SCENARIO_1, TEST_SCENARIO_2]
    all_overlap_stats: AllDataOverlapStats
    all_overlap_stats = create_all_data_overlap_stats(light_scenarios=scenarios, n_values=N_VALUES)

    stats_1_key, stats_2_key, stats_3_key = (
        DataOverlapStatsKey(metadata={"light_scenario_key": TEST_SCENARIO_1.light_scenario_key, "N": 5}),
        DataOverlapStatsKey(metadata={"light_scenario_key": TEST_SCENARIO_2.light_scenario_key, "N": 5}),
        DataOverlapStatsKey(metadata={"light_scenario_key": TEST_SCENARIO_2.light_scenario_key, "N": 13}),
    )

    assert stats_1_key in all_overlap_stats and stats_2_key in all_overlap_stats and stats_3_key in all_overlap_stats

    stats_1: DataOverlapStats
    stats_2: DataOverlapStats
    stats_3: DataOverlapStats
    stats_1, stats_2, stats_3 = (
        all_overlap_stats[stats_1_key],
        all_overlap_stats[stats_2_key],
        all_overlap_stats[stats_3_key],
    )
    assert stats_1.num_instances == 2 and stats_2.num_instances == 1 and stats_3.num_instances == 1


def test_create_ngram_index():
    tokenizer = LightTokenizer()
    scenarios = [TEST_SCENARIO_1, TEST_SCENARIO_2]
    ngram_index: NgramIndex
    ngram_index = create_ngram_index(light_scenarios=scenarios, n_values=N_VALUES, tokenizer=tokenizer)

    stats_1_key, stats_2_key, stats_3_key = (
        DataOverlapStatsKey(metadata={"light_scenario_key": TEST_SCENARIO_1.light_scenario_key, "N": 5}),
        DataOverlapStatsKey(metadata={"light_scenario_key": TEST_SCENARIO_2.light_scenario_key, "N": 5}),
        DataOverlapStatsKey(metadata={"light_scenario_key": TEST_SCENARIO_2.light_scenario_key, "N": 13}),
    )

    test_5_gram: Ngram = ("Center", "for", "Research", "on", "Foundation")
    test_13_gram: Ngram = (
        "The",
        "Center",
        "for",
        "Research",
        "on",
        "Foundation",
        "Models",
        "(CRFM)",
        "is",
        "an",
        "interdisciplinary",
        "initiative",
        "born",
    )

    assert ngram_index[5][test_5_gram] == set(
        [
            EntryDataOverlapKey(stats_key=stats_1_key, instance_id=0, part=PART_INPUT),
            EntryDataOverlapKey(stats_key=stats_2_key, instance_id=0, part=PART_INPUT),
            EntryDataOverlapKey(stats_key=stats_2_key, instance_id=0, part=PART_REF),
        ]
    )
    assert ngram_index[13][test_13_gram] == set(
        [
            EntryDataOverlapKey(stats_key=stats_3_key, instance_id=0, part=PART_INPUT),
            EntryDataOverlapKey(stats_key=stats_3_key, instance_id=0, part=PART_REF),
        ]
    )


def test_compute_scenario_document_overlap():
    tokenizer = LightTokenizer()
    scenarios = [TEST_SCENARIO_1, TEST_SCENARIO_2]
    all_overlap_stats: AllDataOverlapStats
    ngram_index: NgramIndex
    all_overlap_stats = create_all_data_overlap_stats(light_scenarios=scenarios, n_values=N_VALUES)
    ngram_index = create_ngram_index(light_scenarios=scenarios, n_values=N_VALUES, tokenizer=tokenizer)

    compute_scenario_document_data_overlap(
        document=TEST_DOCUMENT,
        ngram_index=ngram_index,
        all_overlap_stats=all_overlap_stats,
        tokenizer=tokenizer,
    )

    stats_1_key, stats_2_key, stats_3_key = (
        DataOverlapStatsKey(metadata={"light_scenario_key": TEST_SCENARIO_1.light_scenario_key, "N": 5}),
        DataOverlapStatsKey(metadata={"light_scenario_key": TEST_SCENARIO_2.light_scenario_key, "N": 5}),
        DataOverlapStatsKey(metadata={"light_scenario_key": TEST_SCENARIO_2.light_scenario_key, "N": 13}),
    )
    stats_1, stats_2, stats_3 = (
        all_overlap_stats[stats_1_key],
        all_overlap_stats[stats_2_key],
        all_overlap_stats[stats_3_key],
    )

    assert (
        stats_1._input_bits[0] == 1
        and stats_1.overlapping_input_fraction == 0.5
        and stats_1.overlapping_reference_fraction == 0
    )
    assert stats_2.overlapping_input_fraction == 1 and stats_2.overlapping_reference_fraction == 1
    assert stats_3.overlapping_input_fraction == 1 and stats_3.overlapping_reference_fraction == 1

    assert stats_1.generate_summary() == {
        "setting": {"stats_key": asdict_without_nones(stats_1_key)},
        "num_instances": 2,
        "num_instances_with_overlapping_input": 1,
        "num_instances_with_overlapping_reference": 0,
        "overlapping_input_fraction": 0.5,
        "overlapping_reference_fraction": 0,
    }
