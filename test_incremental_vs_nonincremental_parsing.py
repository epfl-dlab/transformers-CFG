import timeit

import pandas as pd
from transformers_cfg.token_grammar_recognizer import IncrementalTokenRecognizer
from transformers import AutoTokenizer
import tqdm


model_id = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Test that we can load a JSON object
with open("examples/grammars/json.ebnf", "r") as file:
    input_text = file.read()
JsontokenRecognizer = IncrementalTokenRecognizer(
    grammar_str=input_text, start_rule_name="root", tokenizer=tokenizer
)

valid_json = '{"foo": "bar", "baz": "bat", "key": "value", "key2": "value2", "key3": "value3", "key4": "value4", "key5": "value5", "key6": "value6", "key7": "value7", "key8": "value8", "key9": "value9", "key10": "value10", "key11": "value11", "key12": "value12", "key13": "value13", "key14": "value14", "key15": "value15", "key16": "value16", "key17": "value17", "key18": "value18", "key19": "value19", "key20": "value20", "key21": "value21", "key22": "value22", "key23": "value23", "key24": "value24", "key25": "value25", "key26": "value26", "key27": "value27", "key28": "value28", "key29": "value29", "key30": "value30", "key31": "value31", "key32": "value32", "key33": "value33", "key34": "value34", "key35": "value35", "key36": "value36", "key37": "value37", "key38": "value38", "key39": "value39", "key40": "value40", "key41": "value41", "key42": "value42", "key43": "value43", "key44": "value44", "key45": "value45", "key46": "value46", "key47": "value47", "key48": "value48", "key49": "value49", "key50": "value50", "key51": "value51", "key52": "value52", "key53": "value53", "key54": "value54", "key55": "value55", "key56": "value56", "key57": "value57", "key58": "value58", "key59": "value59", "key60": "value60", "key61": "value61", "key62": "value62", "key63": "value63", "key64": "value64", "key65": "value65"}'
token_ids = tokenizer.encode(valid_json)

acc_state = JsontokenRecognizer._update_state_with_single_token_seq(
    token_ids, as_string=False
)


# Initialize list to hold data
data = {"Token Count": [], "incremental": [], "non-incremental": []}

# Incremental parsing
acc_state = None
for i, token_id in tqdm.tqdm(enumerate(token_ids), desc="Incremental parsing"):
    start_time = timeit.default_timer()
    acc_state = JsontokenRecognizer._update_state_with_single_token_seq(
        [token_id], as_string=False, parsing_state=acc_state
    )
    end_time = timeit.default_timer()
    # data.append({'Token Count': i + 1, 'Time (seconds)': end_time - start_time, 'Method': 'Incremental'})
    data["Token Count"].append(i + 1)
    data["incremental"].append(end_time - start_time)

# Non-incremental parsing
for i in tqdm.tqdm(range(len(token_ids)), desc="Non-incremental parsing"):
    start_time = timeit.default_timer()
    acc_state = JsontokenRecognizer._update_state_with_single_token_seq(
        token_ids[: i + 1], as_string=False
    )
    end_time = timeit.default_timer()
    data["non-incremental"].append(end_time - start_time)

# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("combined_parsing_times.csv", index=False)

print("Data saved to 'combined_parsing_times.csv'.")


# the json object is complete, so the stacks should be empty
assert acc_state.stacks == set() or acc_state.stacks == set(
    tuple()
), f"stacks: {acc_state.stacks}, not empty"
