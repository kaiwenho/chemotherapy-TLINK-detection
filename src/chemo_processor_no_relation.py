import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import spacy
import medspacy
import random

nlp = medspacy.load(enable=["medspacy_sentence_segmenter"])

class ChemotherapyDataProcessor:
    """Process chemotherapy temporal annotation data to extract has_date relations."""

    def __init__(self, no_relation_ratio: float = 0.3):
        self.valid_temporal_relations = {'BEGINS-ON', 'ENDS-ON', 'CONTAINS'}
        self.no_relation_ratio = no_relation_ratio

    def parse_chemotherapy_xml(self, xml_content: str) -> Dict:
        """Parse the chemotherapy temporal annotation XML file."""
        root = ET.fromstring(xml_content)

        entities = {}
        relations = []

        for entity in root.findall('.//entity'):
            entity_id = entity.find('id').text
            span = entity.find('span').text
            entity_type = entity.find('type').text
            start, end = map(int, span.split(','))

            properties = {}
            props_elem = entity.find('properties')
            if props_elem is not None:
                for prop in props_elem:
                    properties[prop.tag] = prop.text

            entities[entity_id] = {
                'span': (start, end),
                'type': entity_type,
                'properties': properties
            }

        for relation in root.findall('.//relation'):
            rel_type = relation.find('type').text
            properties = {}
            props_elem = relation.find('properties')
            if props_elem is not None:
                for prop in props_elem:
                    properties[prop.tag] = prop.text

            relations.append({'type': rel_type, 'properties': properties})

        return {'entities': entities, 'relations': relations}

    def extract_sentence_context(self, text: str, subject_span: Tuple[int, int],
                                object_span: Tuple[int, int]) -> Tuple[str, int, int]:
        """Extract the sentence containing both subject and object using smart tokenization."""
        return self._extract_sentence_with_medspacy(text, subject_span, object_span)

    def _extract_sentence_with_medspacy(self, text: str, subject_span: Tuple[int, int],
                                   object_span: Tuple[int, int]) -> Tuple[str, int, int]:
        """Extract sentence using medspacy."""
        doc = nlp(text)
        sentences = [s.text for s in doc.sents]

        # Find character positions for each sentence
        sentence_spans = []
        current_pos = 0

        for sentence in sentences:
            # Find the sentence in the text starting from current position
            sentence_start = text.find(sentence, current_pos)
            if sentence_start != -1:
                sentence_end = sentence_start + len(sentence)
                sentence_spans.append((sentence_start, sentence_end, sentence))
                current_pos = sentence_end
            else:
                # Fallback: try to handle whitespace differences
                sentence_clean = sentence.strip()
                sentence_start = text.find(sentence_clean, current_pos)
                if sentence_start != -1:
                    sentence_end = sentence_start + len(sentence_clean)
                    sentence_spans.append((sentence_start, sentence_end, sentence_clean))
                    current_pos = sentence_end

        # Find which sentence contains both spans
        subject_pos = subject_span[0]
        object_pos = object_span[0]

        for sent_start, sent_end, sentence in sentence_spans:
            if (sent_start <= subject_pos < sent_end and
                sent_start <= object_pos < sent_end):
                return sentence.strip(), sent_start, sent_end

        # If spans are in different sentences, find the range that covers both
        min_pos = min(subject_pos, object_pos)
        max_pos = max(subject_span[1], object_span[1])

        # Find sentences that overlap with this range
        covering_sentences = []
        for sent_start, sent_end, sentence in sentence_spans:
            if (sent_start <= max_pos and sent_end >= min_pos):
                covering_sentences.append((sent_start, sent_end, sentence))

        if covering_sentences:
            # Merge covering sentences
            first_start = covering_sentences[0][0]
            last_end = covering_sentences[-1][1]
            combined_text = text[first_start:last_end].strip()
            return combined_text, first_start, last_end

        # Fallback to simple method
        return self._extract_sentence_simple(text, subject_span, object_span)

    def _extract_sentence_simple(self, text: str, subject_span: Tuple[int, int],
                                object_span: Tuple[int, int]) -> Tuple[str, int, int]:
        """Simple sentence extraction fallback."""
        start_pos = min(subject_span[0], object_span[0])
        end_pos = max(subject_span[1], object_span[1])

        sentence_start = start_pos
        while sentence_start > 0 and text[sentence_start-1] not in '.!?':
            sentence_start -= 1

        sentence_end = end_pos
        while sentence_end < len(text) and text[sentence_end] not in '.!?':
            sentence_end += 1
        if sentence_end < len(text):
            sentence_end += 1

        while sentence_start < len(text) and text[sentence_start].isspace():
            sentence_start += 1
        while sentence_end > 0 and text[sentence_end-1].isspace():
            sentence_end -= 1

        sentence = text[sentence_start:sentence_end].strip()
        return sentence, sentence_start, sentence_end

    def extract_entity_context(self, text: str, entity_span: Tuple[int, int]) -> Tuple[str, int, int]:
        """Extract sentence context for a single entity using medspacy."""
        doc = nlp(text)
        sentences = [s.text for s in doc.sents]

        # Find character positions for each sentence
        sentence_spans = []
        current_pos = 0

        for sentence in sentences:
            sentence_start = text.find(sentence, current_pos)
            if sentence_start != -1:
                sentence_end = sentence_start + len(sentence)
                sentence_spans.append((sentence_start, sentence_end, sentence))
                current_pos = sentence_end

        # Find which sentence contains the entity
        entity_pos = entity_span[0]

        for sent_start, sent_end, sentence in sentence_spans:
            if sent_start <= entity_pos < sent_end:
                return sentence.strip(), sent_start, sent_end

        # Fallback to simple method for single entity
        start_pos = entity_span[0]
        end_pos = entity_span[1]

        sentence_start = start_pos
        while sentence_start > 0 and text[sentence_start-1] not in '.!?':
            sentence_start -= 1

        sentence_end = end_pos
        while sentence_end < len(text) and text[sentence_end] not in '.!?':
            sentence_end += 1
        if sentence_end < len(text):
            sentence_end += 1

        while sentence_start < len(text) and text[sentence_start].isspace():
            sentence_start += 1
        while sentence_end > 0 and text[sentence_end-1].isspace():
            sentence_end -= 1

        sentence = text[sentence_start:sentence_end].strip()
        return sentence, sentence_start, sentence_end

    def get_related_entity_pairs(self, relations: List[Dict]) -> Set[Tuple[str, str]]:
        """Get all pairs of entities that have explicit relations."""
        related_pairs = set()

        for relation in relations:
            if relation['type'] == 'TLINK':
                props = relation['properties']
                source_id = props.get('Source')
                target_id = props.get('Target')

                if source_id and target_id:
                    # Add both directions since we want to avoid generating
                    # no_relation examples for either direction
                    related_pairs.add((source_id, target_id))
                    related_pairs.add((target_id, source_id))

        return related_pairs

    def generate_no_relation_examples(self, note_text: str, entities: Dict,
                                    relations: List[Dict], num_examples: int) -> List[Dict]:
        """Generate no_relation examples by pairing EVENT and TIMEX3 entities without explicit relations."""
        # Get all EVENT and TIMEX3 entities
        event_entities = {eid: edata for eid, edata in entities.items()
                         if edata['type'] == 'EVENT'}
        timex3_entities = {eid: edata for eid, edata in entities.items()
                          if edata['type'] == 'TIMEX3'}

        # Get pairs that already have relations
        related_pairs = self.get_related_entity_pairs(relations)

        # Generate all possible pairs (both directions)
        all_possible_pairs = []

        # EVENT -> TIMEX3 pairs
        for event_id in event_entities:
            for timex3_id in timex3_entities:
                if (event_id, timex3_id) not in related_pairs:
                    all_possible_pairs.append((event_id, timex3_id, 'EVENT_TIMEX3'))

        # TIMEX3 -> EVENT pairs
        for timex3_id in timex3_entities:
            for event_id in event_entities:
                if (timex3_id, event_id) not in related_pairs:
                    all_possible_pairs.append((timex3_id, event_id, 'TIMEX3_EVENT'))

        # Filter pairs that are too far apart (optional heuristic)
        MAX_DISTANCE = 500  # characters

        filtered_pairs = []
        for subject_id, object_id, pair_type in all_possible_pairs:
            subject_span = entities[subject_id]['span']
            object_span = entities[object_id]['span']

            # Calculate distance between entities
            distance = min(abs(subject_span[0] - object_span[1]),
                          abs(object_span[0] - subject_span[1]))

            if distance <= MAX_DISTANCE:
                filtered_pairs.append((subject_id, object_id, pair_type))

        # Randomly sample the requested number of examples
        if len(filtered_pairs) > num_examples:
            selected_pairs = random.sample(filtered_pairs, num_examples)
        else:
            selected_pairs = filtered_pairs

        # Create no_relation examples
        no_relation_examples = []
        for subject_id, object_id, pair_type in selected_pairs:
            subject_data = entities[subject_id]
            object_data = entities[object_id]

            subject_span = subject_data['span']
            object_span = object_data['span']

            subject_text = note_text[subject_span[0]:subject_span[1]].strip()
            object_text = note_text[object_span[0]:object_span[1]].strip()

            # Determine subject and object classes based on pair type
            if pair_type == 'EVENT_TIMEX3':
                subject_class = 1  # EVENT
                object_class = 2   # TIMEX3
            else:  # TIMEX3_EVENT
                subject_class = 2  # TIMEX3
                object_class = 1   # EVENT

            # Extract sentence context
            try:
                sentence, sent_start, sent_end = self.extract_sentence_context(
                    note_text, subject_span, object_span)

                no_relation_examples.append({
                    'note': note_text,
                    'relation': 'no_relation',
                    'subject_text': subject_text,
                    'subject_start': subject_span[0],
                    'subject_end': subject_span[1],
                    'pred_text': '',
                    'pred_start': -1,
                    'pred_end': -1,
                    'object_text': object_text,
                    'object_start': object_span[0],
                    'object_end': object_span[1],
                    'source_sentence': sentence,
                    'source_sentence_start': sent_start,
                    'source_sentence_end': sent_end,
                    'subject_class': subject_class,
                    'object_class': object_class
                })
            except Exception as e:
                print(f"Error generating no_relation example: {e}")
                continue

        return no_relation_examples

    def extract_entities_for_ner(self, note_text: str, entities: Dict) -> List[Dict]:
        """Extract entities for NER training grouped by sentences."""
        doc = nlp(note_text)
        sentences = list(doc.sents)
        sentence_examples = []

        for sent in sentences:
            sent_start = sent.start_char
            sent_end = sent.end_char
            sent_text = sent.text.strip()

            if not sent_text:
                continue

            sentence_entities = []
            for entity_id, entity_data in entities.items():
                entity_span = entity_data['span']
                entity_start, entity_end = entity_span
                entity_type = entity_data['type']

                # Skip DOCTIME entities as they're not relevant for has_date relations
                if entity_type in ['DOCTIME', 'SECTIONTIME']:
                    continue

                # Check if entity is within this sentence
                if sent_start <= entity_start < sent_end and sent_start <= entity_end <= sent_end:
                    entity_text = note_text[entity_start:entity_end].strip()

                    # Determine entity class based on type
                    if entity_type == 'EVENT':
                        entity_class = 1
                    elif entity_type == 'TIMEX3':
                        entity_class = 2
                    else:
                        entity_class = 0

                    # Adjust entity positions relative to sentence start
                    relative_start = entity_start - sent_start
                    relative_end = entity_end - sent_start

                    sentence_entities.append({
                        "text": entity_text,
                        "start": relative_start,
                        "end": relative_end,
                        "type": entity_type,
                        "class": entity_class
                    })

            # Only create example if sentence has entities
            if sentence_entities:
                sentence_examples.append({
                    "source_sentence": sent_text,
                    "source_sentence_start": sent_start,
                    "entities": sentence_entities
                })

        return sentence_examples

    def process_files(self, note_text: str, xml_content: str) -> Tuple[List[Dict], List[Dict]]:
        """Process a single note and its annotations. Returns (relations, entities)."""
        try:
            parsed_data = self.parse_chemotherapy_xml(xml_content)
            entities = parsed_data['entities']
            relations = parsed_data['relations']

            # Extract text for each entity
            entity_texts = {}
            for entity_id, entity_data in entities.items():
                start, end = entity_data['span']
                entity_texts[entity_id] = note_text[start:end].strip()

            # Extract positive relations
            has_date_relations = []
            for relation in relations:
                if relation['type'] == 'TLINK':
                    props = relation['properties']
                    source_id = props.get('Source')
                    target_id = props.get('Target')
                    rel_type = props.get('Type')

                    # EVENT -> TIMEX3 relations
                    if (source_id in entities and entities[source_id]['type'] == 'EVENT' and
                        target_id in entities and entities[target_id]['type'] == 'TIMEX3'):

                        source_text = entity_texts[source_id]
                        target_text = entity_texts[target_id]
                        source_span = entities[source_id]['span']
                        target_span = entities[target_id]['span']

                        sentence, sent_start, sent_end = self.extract_sentence_context(
                            note_text, source_span, target_span)

                        has_date_relations.append({
                            'note': note_text,
                            'relation': rel_type,
                            'subject_text': source_text,
                            'subject_start': source_span[0],
                            'subject_end': source_span[1],
                            'pred_text': '',
                            'pred_start': -1,
                            'pred_end': -1,
                            'object_text': target_text,
                            'object_start': target_span[0],
                            'object_end': target_span[1],
                            'source_sentence': sentence,
                            'source_sentence_start': sent_start,
                            'source_sentence_end': sent_end,
                            'subject_class': 1,
                            'object_class': 2
                        })

                    # TIMEX3 -> EVENT relations (reverse)
                    if (source_id in entities and entities[source_id]['type'] == 'TIMEX3' and
                        target_id in entities and entities[target_id]['type'] == 'EVENT'):

                        source_text = entity_texts[source_id]
                        target_text = entity_texts[target_id]
                        source_span = entities[source_id]['span']
                        target_span = entities[target_id]['span']

                        sentence, sent_start, sent_end = self.extract_sentence_context(
                            note_text, source_span, target_span)

                        has_date_relations.append({
                            'note': note_text,
                            'relation': rel_type,
                            'subject_text': source_text,
                            'subject_start': source_span[0],
                            'subject_end': source_span[1],
                            'pred_text': '',
                            'pred_start': -1,
                            'pred_end': -1,
                            'object_text': target_text,
                            'object_start': target_span[0],
                            'object_end': target_span[1],
                            'source_sentence': sentence,
                            'source_sentence_start': sent_start,
                            'source_sentence_end': sent_end,
                            'subject_class': 2,
                            'object_class': 1
                        })

            # Generate no_relation examples
            if self.no_relation_ratio > 0:
                num_no_relation = int(len(has_date_relations) * self.no_relation_ratio)
                if num_no_relation > 0:
                    no_relation_examples = self.generate_no_relation_examples(
                        note_text, entities, relations, num_no_relation)
                    has_date_relations.extend(no_relation_examples)

            # Extract entities for NER training
            entity_examples = self.extract_entities_for_ner(note_text, entities)

            return has_date_relations, entity_examples

        except Exception as e:
            print(f"Error processing data: {e}")
            return [], []


def parse_directory(directory_path: str, processor: ChemotherapyDataProcessor) -> Tuple[List[Dict], List[Dict]]:
    """Parse a directory with exactly 2 files: one text file and one XML annotation file."""
    directory_path = Path(directory_path)
    all_files = [f for f in directory_path.iterdir() if f.is_file()]

    if len(all_files) != 2:
        print(f"Error: Expected 2 files, found {len(all_files)}")
        return [], []

    # Identify text and XML files
    txt_file = None
    xml_file = None

    for file_path in all_files:
        if file_path.name.endswith('.xml'):
            xml_file = file_path
        else:
            txt_file = file_path

    if not txt_file or not xml_file:
        print(f"Error: Could not identify text and XML files")
        return [], []

    # Process files
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            note_text = f.read()

        with open(xml_file, 'r', encoding='utf-8') as f:
            xml_content = f.read()

        relations, entities = processor.process_files(note_text, xml_content)

        if relations:
            print(f"Processed {txt_file.name}: {len(relations)} relations, {len(entities)} entities")
        else:
            print(f"Processed {txt_file.name}: No relations found, but extracted {len(entities)} entities for NER")

        return relations, entities

    except Exception as e:
        print(f"Error processing {directory_path}: {e}")
        return [], []


def parse_all_directories(base_path: str, split: str, processor: ChemotherapyDataProcessor) -> Tuple[List[Dict], List[Dict]]:
    """Parse all patient directories in Gold_PairWise_Annotations."""
    base_path = Path(base_path)
    all_relations = []
    all_entities = []

    # Navigate to Gold_PairWise_Annotations
    annotations_dir = base_path / "Gold_PairWise_Annotations"

    if not annotations_dir.exists():
        print(f"Error: {annotations_dir} not found!")
        return all_relations, all_entities

    # Iterate through cancer types
    cancer_types = ['breast', 'melanoma', 'ovarian']

    for cancer_type in cancer_types:
        cancer_dir = annotations_dir / cancer_type
        if not cancer_dir.exists():
            print(f"Warning: {cancer_dir} not found, skipping...")
            continue

        split_dir = cancer_dir / split
        if not split_dir.exists():
            print(f"Warning: {split_dir} not found, skipping...")
            continue

        print(f"Processing {cancer_type} - {split}")

        # Find all patient directories in this split
        for patient_dir in split_dir.iterdir():
            if patient_dir.is_dir():
                print(f"  Processing patient: {patient_dir.name}")

                # Each patient directory contains multiple report directories
                for report_dir in patient_dir.iterdir():
                    if report_dir.is_dir():
                        relations, entities = parse_directory(str(report_dir), processor)
                        all_relations.extend(relations)
                        all_entities.extend(entities)

    print(f"Total relations found in {split}: {len(all_relations)}")
    print(f"Total entities found in {split}: {len(all_entities)}")
    return all_relations, all_entities


def save_relations(relations: List[Dict], output_file: str):
    """Save relations to JSONL format."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for relation in relations:
            json.dump(relation, f, ensure_ascii=False)
            f.write('\n')
    print(f"Saved {len(relations)} relations to {output_file}")


def save_entities(entities: List[Dict], output_file: str):
    """Save entities to JSONL format for NER training."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for entity in entities:
            json.dump(entity, f, ensure_ascii=False)
            f.write('\n')
    print(f"Saved {len(entities)} entities to {output_file}")


def print_entity_statistics(entities: List[Dict], split_name: str):
    """Print statistics about the extracted entities."""
    print(f"\n{split_name} Entity Statistics:")

    total_sentences = len(entities)
    total_entities = sum(len(example['entities']) for example in entities)

    # Count by entity type and class
    type_counts = {}
    class_counts = {}

    for example in entities:
        for entity in example['entities']:
            entity_type = entity.get('type', 'unknown')
            entity_class = entity.get('class', 0)

            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
            class_counts[entity_class] = class_counts.get(entity_class, 0) + 1

    print(f"  Total sentences with entities: {total_sentences}")
    print(f"  Total entities: {total_entities}")
    if total_sentences > 0:
        print(f"  Avg entities per sentence: {total_entities/total_sentences:.2f}")
    print(f"  By type: {dict(type_counts)}")
    print(f"  By class: {dict(class_counts)}")

    # Sample sentences
    print(f"  Sample sentences:")
    for i, example in enumerate(entities[:3]):
        entities_info = [f"'{e['text']}' ({e['type']})" for e in example['entities']]
        print(f"    {i+1}. \"{example['source_sentence'][:60]}...\"")
        print(f"       Entities: {', '.join(entities_info)}")
        print()


def process_dataset(base_directory: str, no_relation_ratio: float = 0.3):
    """Process entire dataset and save results."""
    print(f"Processing dataset with no_relation_ratio={no_relation_ratio}")

    # Create processor with specified no_relation ratio
    processor = ChemotherapyDataProcessor(no_relation_ratio=no_relation_ratio)

    # Process train and dev splits
    train_relations, train_entities = parse_all_directories(base_directory, 'train', processor)
    dev_relations, dev_entities = parse_all_directories(base_directory, 'dev', processor)

    # Save results
    save_relations(train_relations, "chemo_train_relations.jsonl")
    save_relations(dev_relations, "chemo_dev_relations.jsonl")
    save_entities(train_entities, "train_entities.jsonl")
    save_entities(dev_entities, "dev_entities.jsonl")

    # Print relation type distribution
    print(f"\nRelation Type Distribution:")
    relation_counts = {}
    all_relations = train_relations + dev_relations
    for rel in all_relations:
        rel_type = rel['relation']
        relation_counts[rel_type] = relation_counts.get(rel_type, 0) + 1

    for rel_type, count in sorted(relation_counts.items()):
        print(f"  {rel_type}: {count}")

    # Print final results
    print(f"\nFinal Results:")
    print(f"Training relations: {len(train_relations)}")
    print(f"Development relations: {len(dev_relations)}")
    print(f"Training entities: {len(train_entities)} sentences")
    print(f"Development entities: {len(dev_entities)} sentences")
    print(f"Total relations: {len(train_relations) + len(dev_relations)}")
    print(f"Total entities: {len(train_entities) + len(dev_entities)} sentences")

    # Print entity statistics
    print_entity_statistics(train_entities, "Training")
    print_entity_statistics(dev_entities, "Development")

    return (train_relations, dev_relations), (train_entities, dev_entities)


if __name__ == "__main__":
    # Example usage
    base_directory = "/path/to/your/dataset"  # Update this path

    # Process with default 30% no_relation examples
    process_dataset(base_directory)

    # Or process with custom ratio
    # process_dataset(base_directory, no_relation_ratio=0.5)

    # Or process without negative examples (original behavior)
    # process_dataset(base_directory, no_relation_ratio=0.0)
