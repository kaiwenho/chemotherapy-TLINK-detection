# Synthetic Data Generation Prompt

This document contains the exact prompt used with ChatGPT-4o-mini to generate synthetic training examples for the underrepresented relation types (BEGINS-ON and ENDS-ON).

## Generation Details

- **Model**: ChatGPT-4o-mini
- **Examples Generated**:
  - BEGINS-ON: 160 examples
  - ENDS-ON: 310 examples
- **Purpose**: Address class imbalance in chemotherapy temporal relation dataset

## Prompt Template

The following prompt was used, with relation type and examples adapted for each relation:

---

```
You are a clinical text generator helping create synthetic training data for a relation classification task. The task is to identify temporal relations between a chemotherapy treatment event (EVENT) and a time expression (TIMEX3) within a medical note.
Task Definition:
Generate realistic clinical note excerpts where:
1. The EVENT is a mention of chemotherapy treatment.
2. The TIMEX3 is a time expression that refers to when the chemotherapy began.
3. The relationship between the EVENT and the TIMEX3 is "[BEGINS-ON/ENDS-ON]".

Output Format:
Return the result as a JSON object with the following keys:
source_sentence: A realistic clinical sentence or paragraph containing the EVENT and TIMEX3.
subject_text: The exact span of text corresponding to the chemotherapy treatment (EVENT).
object_text: The exact span of text corresponding to the time expression (TIMEX3).
relation: Always set to "[BEGINS-ON/ENDS-ON]".

Constraints:
1. The EVENT must be a plausible chemotherapy regimen or drug.
2. The TIMEX3 must be a specific date or time period.
3. The source sentence must include enough context to clearly indicate that the chemotherapy [started/ended] at that time.
4. Use language to clarify the temporal relationship.

Example:
[3 real examples from the original dataset were provided here as context]

Now generate 5 diverse examples in the same JSON format, each with a different clinical situation and note style. Make sure each example uses a different sentence structure, clinical context, and phrasing of the time expression.
```

---

## Example Output

Here's an example of the expected JSON output format:

```json
{
    "source_sentence": "The chemotherapy regimen consisting of Docetaxel began its course in early July of this year, specifically on 07/01/2023.",
    "subject_text": "Docetaxel",
    "object_text": "07/01/2023",
    "relation": "BEGINS-ON"
}
```

## Notes

- Each generation request produced 5 synthetic examples
- Examples were manually reviewed for quality and realism
- The prompt included 3 real examples from the original dataset to provide context and style guidance
