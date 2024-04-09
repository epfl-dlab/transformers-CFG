from unittest import TestCase

import logging
from transformers_cfg.parser import parse_ebnf
from transformers_cfg.recognizer import StringRecognizer
from dataclasses import dataclass


@dataclass
class PDDLTestCase:
    name: str
    PDDL: str


valid_blocks_sentences = [
    PDDLTestCase("simple_operator", "(put-down c)"),
    PDDLTestCase("two_actions", "(put-down a) (put-down c)"),
    PDDLTestCase(
        "multiple_actions",
        "(pick-up-and-stack b a) (pick-up-and-stack c b) (pick-up-and-stack d c)",
    ),
    PDDLTestCase(
        "complex_only",
        "(unstack-and-stack c b d) (pick-up-and-stack b c) (pick-up-and-stack a b)",
    ),
]

valid_blocks_prefixes = [
    PDDLTestCase("empty_string", ""),
    PDDLTestCase("one_action_spaced", "(pick-up-and-stack b a) "),
    PDDLTestCase("r_unbalanced_paranthesis", "(pick-up c"),
]

invalid_blocks_sentences = [
    PDDLTestCase("undefined_block", "(pick-up z)"),
    PDDLTestCase("wrong_number_of_arguments", "(pick-up c c)"),
    PDDLTestCase("l_unbalanced_paranthesis", "(pick-up c))"),
    PDDLTestCase("unexisitng_operator", "(unstack-and-pick-up c b d)"),
    PDDLTestCase("empty_paranthesis", "()"),
]


valid_depot_sentences = [
    PDDLTestCase("simplest_operator", "(drive truck0 depot0 distributor0)"),
    PDDLTestCase("lift", "(lift hoist0 crate0 pallet0 distributor0)"),
    PDDLTestCase(
        "drive_and_lift",
        "(drive-and-lift truck0 hoist0 crate0 pallet0 distributor0)",
    ),
    PDDLTestCase(
        "lift_and_drive", "(lift-and-drive truck0 hoist0 crate0 pallet0 depot0 depot1)"
    ),
    PDDLTestCase(
        "long_realistic",
        "(lift-and-drive truck0 hoist0 crate0 pallet0 depot0 depot0) (lift hoist2 crate2 crate1 distributor1) "
        + "(drive truck0 depot0 distributor0) (drive-and-lift truck0 hoist1 crate1 pallet2 distributor0) "
        + "(drop hoist1 crate1 crate3 distributor0) (drive-and-load truck1 hoist0 crate0 depot0) "
        + "(drive-and-unload truck1 hoist0 crate0 pallet2 depot0) (drive truck1 depot0 distributor1) "
        + "(drive-and-load truck1 hoist2 crate2 distributor1) (drive-and-unload truck1 hoist2 crate2 pallet0 distributor1)",
    ),
    PDDLTestCase(
        "long_real",
        "(lift-and-drive truck1 hoist0 crate5 pallet0 depot0 depot0) (drive-and-load truck1 hoist0 crate5 depot0) "
        + "(drive-andlift truck0 hoist2 crate4 crate0 distributor1) (lift hoist1 crate0 pallet4 distributor0) "
        + "(drive-and-lift truck1 hoist0 crate1 pallet5 depot0) (drive-and-load truck1 hoist0 crate1 depot0) "
        + "(drive-and-lift truck1 hoist0 crate3 crate2 depot0) (drive-and-load truck1 hoist0 crate3 depot0) "
        + "(drive-and-unload truck1 hoist0 crate3 pallet1 depot0) (drop hoist2 crate4 pallet5 distributor1) "
        + "(drive-and-unload truck1 hoist0 crate1 crate2 depot0) (drive-and-lift truck0 hoist2 crate1 crate2 distributor1) "
        + "(drop hoist2 crate1 crate4 distributor1) (drive-and-unload truck1 hoist0 crate5 crate1 depot0) "
        + "(drop hoist1 crate0 pallet3 distributor0)",
    ),
]

valid_depot_prefixes = [
    PDDLTestCase("empty_string", ""),
    PDDLTestCase("one_action_spaced", "(load hoist0 crate0 pallet0 distributor0) "),
    PDDLTestCase(
        "r_unbalanced_paranthesis", "(unload hoist2 crate5 pallet5 distributor0)"
    ),
]

invalid_depot_sentences = [
    PDDLTestCase("undefined_object", "(lift moist0 crate0 pallet0 distributor0)"),
    PDDLTestCase("wrong_number_of_arguments", "(lift hoist0)"),
    PDDLTestCase("l_unbalanced_paranthesis", "(drive truck0 depot0 distributor0))"),
    PDDLTestCase(
        "unexisitng_operator",
        "(load-and-drive truck0 hoist0 crate0 pallet0 depot0 depot1)",
    ),
    PDDLTestCase("empty_paranthesis", "()"),
]

valid_satelite_sentences = [
    PDDLTestCase("simple_operator", "(switch-on instrument1 satelite3)"),
    PDDLTestCase(
        "multiple_actions",
        "(pick-up-and-stack b a) (pick-up-and-stack c b) (pick-up-and-stack d c)",
    ),
    PDDLTestCase(
        "complex_only",
        "(switch-on instrument1 satelite3) (turn-to satelite1 direction4 direction0)",
    ),
    PDDLTestCase(
        "take_image",
        "(take-image satelite1 direction4 instrument2 mode1)",
    ),
]

valid_satelite_prefixes = [
    PDDLTestCase("empty_string", ""),
    PDDLTestCase("one_action_spaced", "(switch-off instrument2 satelite3) "),
    PDDLTestCase(
        "r_unbalanced_paranthesis", "(calibrate satelite1 instrument2 direction4"
    ),
]

invalid_satelite_sentences = [
    PDDLTestCase("undefined_object", "(switch-on instrument8 satelite3)"),
    PDDLTestCase(
        "wrong_number_of_arguments", "(take-image satelite1 instrument2 mode1)"
    ),
    PDDLTestCase(
        "l_unbalanced_paranthesis", "(calibrate satelite1 instrument2 direction4))"
    ),
    PDDLTestCase("unexisitng_operator", "(turn satelite1 direction4 direction0)"),
    PDDLTestCase("empty_paranthesis", "()"),
]


TestCases = {
    "blocks": (
        valid_blocks_sentences,
        valid_blocks_sentences,
        invalid_blocks_sentences,
    ),
    #     "depot": (
    #         valid_depot_sentences,
    #         valid_depot_sentences,
    #         invalid_depot_sentences,
    #     ),
    #     "satelite": (
    #         valid_satelite_sentences,
    #         valid_satelite_sentences,
    #         invalid_satelite_sentences,
    #     ),
}


class Test_parsing_pddl_object(TestCase):
    def setUp(self):
        self.recognizers = {}
        for grammar_name in TestCases:
            with open(f"examples/grammars/PDDL/{grammar_name}.ebnf", "r") as file:
                input_text = file.read()

            parsed_grammar = parse_ebnf(input_text)
            print(parsed_grammar)

            start_rule_id = parsed_grammar.symbol_table["root"]

            self.recognizers[grammar_name] = StringRecognizer(
                parsed_grammar.grammar_encoding, start_rule_id
            )
        print("SetUp successfull!", flush=True)

    def test_valid_sentence(self):
        for grammar_name, recognizer in self.recognizers.items():
            valid_full, valid_prefix, invalid = TestCases[grammar_name]

            for PDDL_test_case in valid_full:
                fail_msg = (
                    f"{grammar_name.capitalize()}:"
                    + f"Failed on {PDDL_test_case.name}, {PDDL_test_case.PDDL}"
                )
                print(fail_msg, flush=True)
                self.assertEqual(
                    True,
                    recognizer._accept_string(PDDL_test_case.PDDL),
                    msg=fail_msg,
                )
            for PDDL_test_case in valid_prefix + invalid:
                fail_msg = (
                    f"{grammar_name.capitalize()}:"
                    + f"Failed on {PDDL_test_case.name}, {PDDL_test_case.PDDL}"
                )
                print(fail_msg, flush=True)
                self.assertEqual(
                    False,
                    recognizer._accept_string(PDDL_test_case.PDDL),
                    msg=fail_msg,
                )

    def test_valid_prefixes(self):
        for grammar_name, recognizer in self.recognizers.items():
            valid_full, valid_prefix, invalid = TestCases[grammar_name]
            for PDDL_test_case in valid_full + valid_prefix:
                fail_msg = (
                    f"{grammar_name.capitalize()}:"
                    + f"Failed on {PDDL_test_case.name}, {PDDL_test_case.PDDL}"
                )
                print(fail_msg, flush=True)
                self.assertEqual(
                    True,
                    recognizer._accept_prefix(PDDL_test_case.PDDL),
                    msg=fail_msg,
                )

            for PDDL_test_case in invalid:
                fail_msg = (
                    f"{grammar_name.capitalize()}:"
                    + f"Failed on {PDDL_test_case.name}, {PDDL_test_case.PDDL}"
                )
                print(fail_msg, flush=True)
                self.assertEqual(
                    False,
                    recognizer._accept_prefix(PDDL_test_case.PDDL),
                    msg=fail_msg,
                )
