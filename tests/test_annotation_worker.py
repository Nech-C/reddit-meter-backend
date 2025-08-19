from app.llm_annotation.annotation_worker import build_prompt, parse_json


def test_parse_json():
        # Check standard json input
        standard_json_input = '{"joy": 1, "sadness": 2, "anger": 7, "fear": 3, "love": 1, "surprise": 5}'
        assert parse_json(standard_json_input) == {"joy": 1, "sadness": 2, "anger": 7, "fear": 3, "love": 1, "surprise": 5}

        # Check input with markdown codeblock and uppercase letter
        md_json_input = '```jSon{"joy": 1, "sadness": 2, "anger": 7, "fear": 3, "love": 1, "surprise": 5}```'
        assert parse_json(md_json_input) == {"joy": 1, "sadness": 2, "anger": 7, "fear": 3, "love": 1, "surprise": 5}
        
        # Check input with values out of bounds
        out_of_bounds = '{"joy": -1, "sadness": 2, "anger": 12, "fear": 3, "love": 0, "surprise": 5}'
        assert parse_json(out_of_bounds) == {"joy": 1, "sadness": 2, "anger": 10, "fear": 3, "love": 1, "surprise": 5}

        # Check all cases:
        all_cases = '```JSon{"joy": -1, "sadness": 2, "anger": 12, "fear": 3, "love": 0, "surprise": 5}```'
        assert parse_json(all_cases) == {"joy": 1, "sadness": 2, "anger": 10, "fear": 3, "love": 1, "surprise": 5}