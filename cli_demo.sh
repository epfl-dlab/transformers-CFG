

# generate C code
transformers-cfg-cli generate \
    -m "gpt2" \
    -g "examples/grammars/c.ebnf" \
    -p "#include <stdio.h>\n" \
    --use_4bit \
    --max_new_tokens 20 \
    --repetition_penalty 3.0


# generate relation extraction triples
transformers-cfg-cli generate \
    -m "gpt2" \
    -g "examples/grammars/cIE.ebnf" \
    -p "This is a valid json string for http request:" \
    --use_8bit \
    --max_new_tokens 60 \
    --repetition_penalty 1.1

# generate json object
transformers-cfg-cli generate \
    -m "gpt2" \
    -g "examples/grammars/json.ebnf" \
    -p "This is a valid json string for http request:" \
    --use_4bit \
    --max_new_tokens 60 \
    --repetition_penalty 1.1

# generate json array

transformers-cfg-cli generate \
    -m "gpt2" \
    -g "examples/grammars/json_arr.ebnf" \
    -p "This is my shopping list in json array:" \
    --use_4bit \
    --max_new_tokens 60 \
    --repetition_penalty 1.1

# generate calflow

transformers-cfg-cli generate \
    -m "gpt2" \
    -g "examples/grammars/calflow.ebnf" \
    -p 'Generate 3 CalFlow strings: 1.(Yield (toRecipient (CurrentUser))) 2.(Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (Event.subject_? (?= "choose the meeting"))))) 3.' \
    --use_4bit \
    --max_new_tokens 60 \
    --repetition_penalty 1.1
