# Custom json grammars

You can use custom grammars to constrain the output of a language model to generate valid json objects. This is useful when you want to generate json objects for specific applications, such as http requests or shopping carts.

## Quickstart

There are multiple ways to represent json schemas. 
We provide recommendations on how to do this for two common formats: Typescript and json schemas.

<details>
<summary> Example of a Typescript schema for a Student object </summary>

```Typescript
interface Student{
 name: string;
 age: number;
 is_student : boolean;
 courses: string[];
}
```
</details>

<details>
<summary> Example of a json schema for a Student object </summary>

```json
{
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"},
        "is_student": {"type": "boolean"},
        "courses": {
            "type": "array",
            "items": { "type": "string"}
        }
    }
}
```
</details>


### From Typescript

To generate custom json grammars from Typescript schemas, you can use [this online tool](https://grammar.intrinsiclabs.ai/) or [this Typescript generator](https://github.com/IntrinsicLabsAI/gbnfgen) from Intrinsic AI. Then, simply copy paste the resulting grammar into a text file and use it with the `IncrementalGrammarConstraint`.


### From json schemas

Alternatively, you can generate custom json grammars from json format schemas using the `json_schema_to_grammar.py` script, analogous to [one in the lama.cpp repository](https://github.com/ggerganov/llama.cpp/blob/ab9a3240a9da941fdef5cd4a25f2b97c2f5a67aa/examples/json_schema_to_grammar.py). 


To generate a grammar from a json schema, run the following command:

```bash
python3 json_schema_to_grammar.py -i schemas/product_catalog.json -o grammars/product_catalog.ebnf
```
This script generates a grammar from a json schema file (see examples of json schemas in `/schemas` and the corresponding grammars in `/grammars`). The generated grammar is in the Extended Backus-Naur Form (EBNF) format and can be directly used with the `IncrementalGrammarConstraint`. 

Additional arguments allow to specify the property order of the json object as well as string formatting parameters.

