
################
#
# JSON generation: object and array
#
################

# generate json object
transformers-cfg-cli generate \
    -m "microsoft/Phi-3-mini-4k-instruct" \
    -g "examples/grammars/json.ebnf" \
    -p "This is a valid json string for http request:" \
    --use_4bit \
    --max_new_tokens 60 \
    --repetition_penalty 1.1

# generate json array

transformers-cfg-cli generate \
    -m "microsoft/Phi-3-mini-4k-instruct" \
    -g "examples/grammars/json_arr.ebnf" \
    -p "Put my shopping list into a json array:" \
    --use_4bit \
    --max_new_tokens 60 \
    --repetition_penalty 1.1

################
#
# Code generation: Python, C
#
################

# generate C code
transformers-cfg-cli generate \
    -m "microsoft/Phi-3-mini-4k-instruct" \
    -g "examples/grammars/c.ebnf" \
    -p "#include <stdio.h>\n" \
    --use_4bit \
    --max_new_tokens 20 \
    --repetition_penalty 3.0

################
#
# NLP tasks: relation extraction
#
################

# generate relation extraction triples
transformers-cfg-cli generate \
    -m "microsoft/Phi-3-mini-4k-instruct" \
    -g "examples/grammars/cIE.ebnf" \
    -p "Extract relations from the following sentence: René Descartes was a French philosopher, scientist, and mathematician" \
    --use_8bit \
    --max_new_tokens 60 \
    --repetition_penalty 1.1


################
#
# Semantic parsing: CalFlow, GeoQuery, overnight, etc.
#
################

transformers-cfg-cli generate \
    -m "microsoft/Phi-3-mini-4k-instruct" \
    -g "examples/grammars/calflow.ebnf" \
    -p 'Generate 3 CalFlow strings: 1.(Yield (toRecipient (CurrentUser))) 2.(Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (Event.subject_? (?= "choose the meeting"))))) 3.' \
    --use_4bit \
    --max_new_tokens 60 \
    --repetition_penalty 1.1

transformers-cfg-cli generate \
    -m "microsoft/Phi-3-mini-4k-instruct" \
    -g "examples/grammars/geo_query.ebnf" \
    -p "Translate the following sentence into GeoQuery: What is the population of the largest city in California?" \
    --use_4bit \
    --max_new_tokens 60 \
    --repetition_penalty 1.1

transformers-cfg-cli generate \
    -m "microsoft/Phi-3-mini-4k-instruct" \
    -g "examples/grammars/overnight.ebnf" \
    -p """Translate natural language to DSL:
        Q: which brick is no wider than 3 inches
        A: listValue (filter (getProperty (singleton en.block) !type) (ensureNumericProperty width) <= (ensureNumericEntity 3 en.inch)))
        Q: which block is above block 1
        A: (listValue (filter (filter (getProperty (singleton en.block) !type) (reverse above) = en.block.block1) above = en.block.block1))
        Q: what block is longer than 3 inches
        A: """ \
    --use_4bit \
    --max_new_tokens 60 \
    --repetition_penalty 1.1



################
#
# Unicode support, Chinese, Emoji, etc.
#
################

transformers-cfg-cli generate \
    -m "microsoft/Phi-3-mini-4k-instruct" \
    -g "examples/grammars/chinese.ebnf" \
    -p "Translate the following sentence into Chinese: My neighbor is a very nice person. -> " \
    --use_4bit \
    --max_new_tokens 60 \
    --repetition_penalty 1.1


transformers-cfg-cli generate \
    -m "microsoft/Phi-3-mini-4k-instruct" \
    -g "examples/grammars/emoji.ebnf" \
    -p "Translate the following sentence into emoji: I am very happy today. -> " \
    --use_4bit \
    --max_new_tokens 60 \
    --repetition_penalty 1.1
