import pytest
from transformers_cfg.parser import parse_ebnf
from transformers_cfg.recognizer import StringRecognizer
from dataclasses import dataclass


@dataclass
class MoleculeTestCase:
    name: str
    molecule: str


valid_smiles_sentences = [
    MoleculeTestCase("simple_atom", "C"),
    MoleculeTestCase("single_bond_no_hyphen", "CC"),
    MoleculeTestCase("double_bond", "C=O"),
    MoleculeTestCase("dot", "C.O"),
    MoleculeTestCase("radical", "CC(C)C"),
    MoleculeTestCase("isotope", "[14c]"),
    MoleculeTestCase("aromatic_no_hyphen", "C1CC1"),
    MoleculeTestCase("interleaved_cycle_explicit", "C1=CC=CC=C1"),
    MoleculeTestCase("interleaved_cycle_colon", "C1:C:C1"),
    MoleculeTestCase("interleaved_cycle_lower_case", "c1cc1"),
    MoleculeTestCase("cis_bond_right", "F/C=C\\F"),
    MoleculeTestCase("trans_bond_left", "F\\C=C\\F"),
    MoleculeTestCase("d_alanine", "C[C@H](C(=O)O)N"),
    MoleculeTestCase("l_alanine", "C[C@@H](C(=O)O)N"),
    MoleculeTestCase("nested_cycles", "C12(CCC1)CCC2"),
    MoleculeTestCase("charge", "[Cu+2].[O-]S(=O)[O-]"),
    MoleculeTestCase("mix_of_cases", "Cc(cc1)ccc1C#N"),
    MoleculeTestCase("mix_of_bonds_and_cycles", "C1CC/C=C1/C=C/C=C/C2=C(C)/CCC2"),
    MoleculeTestCase("wildcard", "Oc1c(*)cccc1"),
]

valid_smiles_prefixes = [
    MoleculeTestCase("empty_string", ""),
    MoleculeTestCase("simple_atom_dangling_bond", "C#"),
    MoleculeTestCase("unbalanced_paranthesis", "C(C(C)"),
    # Failure cases of SMILES in general
    # MoleculeTestCase("lowercase_outside_cycle", "c"),
    # MoleculeTestCase("unclosed cycle", "C1CC/C=C1/C=C/C=C/C2=C(C)/CCC"),
    # MoleculeTestCase("unclosed cycle", "C1CCC"),
]

invalid_smiles_sentences = [
    MoleculeTestCase("fake_atom", "L"),
    MoleculeTestCase("fake_molecule_in_brackets", "[Xx]"),
    MoleculeTestCase("bond_outside_parentheses", "CCC=(O)O"),
    MoleculeTestCase("double_double_bond", "C==C"),
    MoleculeTestCase("empty_paranthesis", "()"),
    MoleculeTestCase("invalid_charge", "[Cu+20].[O-]S(=O)(=O)[O-]"),
    # Failure cases of SMILES in general
    # MoleculeTestCase("two_bonds_same_atom", "C12C2CCC1"),
    # MoleculeTestCase("self-bond", "C11"),
]

valid_isocyanite_sentences = [
    MoleculeTestCase("short_isocyanate", "O=C=NCCCCCCN=C=O"),
    MoleculeTestCase("right_group", "CC1=C(C=C(C=C1)CN=C=O)N=C=O"),
    MoleculeTestCase("trans_bond_right", "Cc1ccc(cc1\\N=C=O)\\N=C=O"),
    MoleculeTestCase("group_radical", "O=C=NC1CCC(CC2CCC(CC2)N=C=O)CC1"),
    MoleculeTestCase("trans_bond_left", "O=C=N\\C1CC(C\\N=C=O)(CC(C1)(C)C)C"),
    MoleculeTestCase("trans_bond", "O=C=N\\CCCCCC/N=C=O"),
    MoleculeTestCase("group_radicals", "CCOC(C(N=C=O)CCCCN=C=O)=O"),
    MoleculeTestCase(
        "simple_atom", "O=C=NC1=CC=CC(CC2=CC=C(C=C2N=C=O)CC3=CC=C(C=C3)N=C=O)=C1"
    ),
    MoleculeTestCase(
        "single_bond_no_hyphen",
        "O=C=NC1=CC(CC2=C(C=C(C=C2)CC3=CC=C(C=C3N=C=O)CC4=CC=C(C=C4)N=C=O)N=C=O)=CC=C1",
    ),
    MoleculeTestCase(
        "double_bond",
        "O=C=NC1=CC=C(C=C1)CC2=CC=C(C=C2N=C=O)CC3=C(C=C(C=C3)CC4=CC=C(C=C4N=C=O)CC5=CC=C(C=C5)N=C=O)N=C=O",
    ),
    MoleculeTestCase("interleaved_cycle_explicit", "CC1(CC(CC(CN=C=O)(C1)C)N=C=O)C"),
    MoleculeTestCase("interleaved_cycle_colon", "CC1=C(C=C(C=C1)CN=C=O)N=C=O"),
    MoleculeTestCase("cycles", "O=C=N\\c1ccc(cc1)Cc2ccc(\\N=C=O)cc2"),
]

valid_acrylate_sentences = [
    MoleculeTestCase("simple_acrylate", "COC(=O)C=C"),
    MoleculeTestCase("simple_acrylate", "C=CC(=O)OC1=CC=CC=C1"),
    MoleculeTestCase("simple_acrylate_group_variation", "CC(=C)C(=O)OC1=CC=CC=C1"),
    MoleculeTestCase("", "C=CC(=O)OCCC1=CC=CC=C1"),
    MoleculeTestCase("", "CCC(C)OC(=O)C=C"),
    MoleculeTestCase("", "C=CC(=O)OC1=C(C(=C(C(=C1F)F)F)F)F"),
    MoleculeTestCase("", "CC(C)COC(=O)C(=C)C"),
    MoleculeTestCase("", "CCC(C)OC(=O)C(=C)C"),
    MoleculeTestCase("", "CCCOC(=O)C(=C)C"),
    MoleculeTestCase("", "CC1CC(CC(C1)(C)C)OC(=O)C(=C)C"),
    MoleculeTestCase("", "CCCOC(=O)C=C"),
    MoleculeTestCase("", "COCCOC(=O)C=C"),
    MoleculeTestCase("", "CC(=C)C(=O)OCCOC1=CC=CC=C1"),
    MoleculeTestCase("", "CCCCCCOC(=O)C=C"),
    MoleculeTestCase("", "CCCCOCCOC(=O)C(=C)C"),
    MoleculeTestCase("", "CC(=C)C(=O)OC"),
    MoleculeTestCase("", "CCCCOC(=O)C=C"),
    MoleculeTestCase("", "CCOCCOC(=O)C(=C)C"),
    MoleculeTestCase("", "CC(=C)C(=O)OC1CC2CCC1(C2(C)C)C"),
    MoleculeTestCase("", "CCCCC(CC)COC(=O)C(=C)C"),
    MoleculeTestCase("", "CC(C)(COCCCOC(=O)C=C)COCCCOC(=O)C=C"),
    MoleculeTestCase("", "C=CC(=O)OCCCCCCOC(=O)C=C"),
    MoleculeTestCase("", "C=CC(=O)OCC(CO)(COC(=O)C=C)COC(=O)C=C"),
    MoleculeTestCase("", "CCC(COCCCOC(=O)C=C)(COCCCOC(=O)C=C)COCCCOC(=O)C=C"),
    MoleculeTestCase("", "CCC(COCC(CC)(COC(=O)C=C)COC(=O)C=C)(COC(=O)C=C)COC(=O)C=C"),
    MoleculeTestCase(
        "", "C=CC(=O)OCC(CO)(COCC(COC(=O)C=C)(COC(=O)C=C)COC(=O)C=C)COC(=O)C=C"
    ),
    MoleculeTestCase(
        "", "C=CC(=O)OCC(COCC(COC(=O)C=C)(COC(=O)C=C)COC(=O)C=C)(COC(=O)C=C)COC(=O)C=C"
    ),
]

valid_chain_extender_sentences = [
    MoleculeTestCase("simplest_chain_extender", "OCCO"),
    MoleculeTestCase("", "OC(C)CCO"),
    MoleculeTestCase("", "OCCOCO"),
    MoleculeTestCase("", "OCCNC(=O)NCCCCCCNC(=O)NCCO"),
    MoleculeTestCase("", "OC(=O)C(N)CCCCN"),
    MoleculeTestCase("", "Oc1ccc(cc1)CCC(=O)OCCOC(=O)CCc1ccc(cc1)O"),
    MoleculeTestCase("", "OC(=O)C(N)CCN"),
    MoleculeTestCase("", "N1CCNCC1"),
    MoleculeTestCase("", "Nc1ccc(cc1)SSc2ccc(cc2)N"),
    MoleculeTestCase("", "Nc1ccc(cc1)Cc2ccc(cc2)N"),
]

TestCases = {
    "generic": (
        valid_smiles_sentences,
        valid_smiles_prefixes,
        invalid_smiles_sentences,
    ),
    "isocyanates": (
        valid_isocyanite_sentences,
        valid_smiles_sentences,
        invalid_smiles_sentences,
    ),
    "acrylates": (
        valid_acrylate_sentences,
        valid_smiles_sentences,
        invalid_smiles_sentences,
    ),
    "chain_extenders": (
        valid_chain_extender_sentences,
        valid_smiles_sentences,
        invalid_smiles_sentences,
    ),
}


@pytest.fixture(scope="module")
def recognizers():
    recognizers = {}
    for grammar_name in TestCases:
        with open(f"examples/grammars/SMILES/{grammar_name}.ebnf", "r") as file:
            input_text = file.read()
        parsed_grammar = parse_ebnf(input_text)
        start_rule_id = parsed_grammar.symbol_table["root"]
        recognizers[grammar_name] = StringRecognizer(
            parsed_grammar.grammar_encoding, start_rule_id
        )
    return recognizers


def test_valid_sentence(recognizers):
    for grammar_name, recognizer in recognizers.items():
        valid_full, valid_prefix, invalid = TestCases[grammar_name]

        for molecule_test_case in valid_full:
            fail_msg = (
                f"{grammar_name.capitalize()}:"
                + f"Failed on {molecule_test_case.name}, {molecule_test_case.molecule}"
            )
            assert (
                recognizer._accept_string(molecule_test_case.molecule) == True
            ), fail_msg

        for molecule_test_case in valid_prefix + invalid:
            fail_msg = (
                f"{grammar_name.capitalize()}:"
                + f"Failed on {molecule_test_case.name}, {molecule_test_case.molecule}"
            )
            assert (
                recognizer._accept_string(molecule_test_case.molecule) == False
            ), fail_msg


def test_valid_prefixes(recognizers):
    for grammar_name, recognizer in recognizers.items():
        valid_full, valid_prefix, invalid = TestCases[grammar_name]

        for molecule_test_case in valid_full + valid_prefix:
            fail_msg = (
                f"{grammar_name.capitalize()}:"
                + f"Failed on {molecule_test_case.name}, {molecule_test_case.molecule}"
            )
            assert (
                recognizer._accept_prefix(molecule_test_case.molecule) == True
            ), fail_msg

        for molecule_test_case in invalid:
            fail_msg = (
                f"{grammar_name.capitalize()}:"
                + f"Failed on {molecule_test_case.name}, {molecule_test_case.molecule}"
            )
            assert (
                recognizer._accept_prefix(molecule_test_case.molecule) == False
            ), fail_msg
