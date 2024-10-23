from datasets import load_dataset, Dataset
from utils.enums import SpecialTokens, SupportedDatasets
import torch
import random
from tqdm import tqdm

class SFCDatasetLoader:
    def __init__(self, dataset_name, model, task_prompt='', clean_system_prompt='', corrupted_system_prompt='', split="train", 
                 return_message_objects=False, num_samples=None, local_dataset=False, base_folder_path='./data'):
        self.dataset = self.load_supported_dataset(dataset_name, split, local_dataset, base_folder_path)
        self.dataset_name = dataset_name
        self.task_prompt = task_prompt

        self.clean_system_prompt = clean_system_prompt
        self.corrupted_system_prompt = corrupted_system_prompt

        if not clean_system_prompt:
            print('WARNING: Clean system prompt not provided.')

        if not corrupted_system_prompt:
            print('WARNING: Corrupted system prompt not provided.')

        self.return_message_objects = return_message_objects
        self.model = model
        self.special_tokens_tensor = self._get_special_tokens_tensor()

        # Sample the dataset if num_samples is specified
        if num_samples is not None and num_samples < len(self.dataset):
            self.dataset = self.dataset.select(random.sample(range(len(self.dataset)), num_samples))

    def _get_max_prompt_length(self):
        if self.dataset_name == SupportedDatasets.COMMONSENSE_QA:
            return 180
        elif self.dataset_name == SupportedDatasets.VERB_AGREEMENT:
            return 30

    def _get_special_tokens_tensor(self, selected_special_tokens=None):
        """
        Returns a tensor of selected special tokens from the tokenizer and custom tokens ['user', 'model'].
        If `selected_special_tokens` is None, all tokens from the given set are included.
        
        Parameters:
        - selected_special_tokens (list[SpecialTokens], optional): A list of special tokens to include. 
                                                                If None, all are included by default.
        
        Returns:
        - torch.Tensor: Tensor of selected special tokens' IDs.
        """

        # Step 2: Set default tokens if none are provided
        if selected_special_tokens is None:
            selected_special_tokens = [
                SpecialTokens.BOS, SpecialTokens.EOS, SpecialTokens.UNK, SpecialTokens.PAD,
                SpecialTokens.ADDITIONAL, SpecialTokens.ROLE
            ]

        # Step 3: Mapping from Enum values to actual tokenizer attributes
        special_token_mapping = {
            SpecialTokens.BOS: self.model.tokenizer.bos_token_id,
            SpecialTokens.EOS: self.model.tokenizer.eos_token_id,
            SpecialTokens.UNK: self.model.tokenizer.unk_token_id,
            SpecialTokens.PAD: self.model.tokenizer.pad_token_id,
            SpecialTokens.ADDITIONAL: [self.model.tokenizer.convert_tokens_to_ids(token) 
                                    for token in self.model.tokenizer.additional_special_tokens],
            SpecialTokens.ROLE: [self.model.tokenizer.convert_tokens_to_ids(token) 
                                for token in ['user', 'model']] 
        }

        # Step 4: Collect the token IDs based on the selection
        selected_token_ids = []
        for token_enum in selected_special_tokens:
            token_value = special_token_mapping.get(token_enum)
            if token_value is not None:
                if isinstance(token_value, list):
                    selected_token_ids.extend(token_value)
                else:
                    selected_token_ids.append(token_value)

        # Step 5: Convert to tensor and return
        return torch.tensor(selected_token_ids, device=self.model.cfg.device)

    def get_special_tokens_mask(self, tokens, selected_special_tokens=None):
        """Return the special tokens tensor."""
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, device=self.model.cfg.device)

        special_tokens = self._get_special_tokens_tensor(selected_special_tokens)
        special_token_mask = torch.where(torch.isin(tokens, special_tokens), 1, 0)

        return special_token_mask

    @staticmethod
    def load_supported_dataset(dataset_name, split, local_dataset, base_folder_path):
        """Load a supported dataset from Hugging Face by name."""
        if not isinstance(dataset_name, SupportedDatasets):
            raise ValueError(f"{dataset_name} is not a supported dataset. Choose from {list(SupportedDatasets)}")

        if local_dataset:
            local_dataset_name = str(base_folder_path / dataset_name.value)
            return load_dataset('json', data_files=local_dataset_name, split='train')

        dataset_hf = load_dataset(dataset_name.value)  # Load dataset using HF datasets library
        if split not in dataset_hf:
            raise ValueError(f"Split '{split}' not found in the dataset. Available splits: {list(dataset_hf.keys())}")

        return dataset_hf[split]  # Return the selected split (e.g., 'train')
    
    def apply_chat_template_and_tokenize(self, prompt, tokenize=True, apply_chat_template=True, prepend_generation_prefix=False):
        tokenizer = self.model.tokenizer
        max_length = self._get_max_prompt_length()
        special_token_mask = None

        # Apply chat template if required
        if apply_chat_template:
            conversation = [
                {"role": "user", "content": prompt}
            ]

            prompt = tokenizer.apply_chat_template(
                conversation, 
                tokenize=False, 
                continue_final_message = False if prepend_generation_prefix else True,
                add_generation_prompt = prepend_generation_prefix
            )
        
        # Apply padding when manually tokenizing
        if tokenize:
            if apply_chat_template:
                # Tokenize using the tokenizer with padding and truncation
                tokenized = tokenizer(
                    prompt, 
                    add_special_tokens=False,
                    padding='max_length',        # Pad to max_length
                    truncation=True,             # Truncate to max_length
                    max_length=max_length,
                    return_special_tokens_mask=False
                )
                tokenized['special_tokens_mask'] = self.get_special_tokens_mask(tokenized['input_ids'])
            else:
                # Tokenize using the tokenizer with padding and truncation, and add special tokens
                tokenized = tokenizer(
                    prompt, 
                    add_special_tokens=True,
                    padding='max_length',        # Pad to max_length
                    truncation=True,             # Truncate to max_length
                    max_length=max_length,
                    return_special_tokens_mask=True
                )
                
            prompt = tokenized["input_ids"]  # Padded input IDs
            special_token_mask = torch.tensor(tokenized["special_tokens_mask"])  # Mask for special tokens

        # Convert prompt to tensor if required
        # prompt = torch.tensor(prompt, device=self.model.cfg.device)
        return prompt, special_token_mask


    def get_formatted_prompt(self, item, system_prompt, task_prompt):
        if self.dataset_name == SupportedDatasets.COMMONSENSE_QA:
            choices = [
                f"{label}) {text}" 
                for label, text in zip(item['choices']['label'], item['choices']['text'])
            ]
            question_with_choices = f"{item['question']}\n" + "\n".join(choices)

            prompt = (
                f"{system_prompt} Now, here's the user's question:"
                f'\n"{question_with_choices}"'
                f'\n{task_prompt}"'
            )
            return prompt
        elif self.dataset_name == SupportedDatasets.VERB_AGREEMENT:
            if system_prompt == self.clean_system_prompt:
                return item['clean_prefix']
            elif system_prompt == self.corrupted_system_prompt:
                return item['patch_prefix']
    
    def get_clean_answer(self, item, prompt, tokenize=True):
        if self.dataset_name == SupportedDatasets.COMMONSENSE_QA:
            answer = item['answerKey']
        elif self.dataset_name == SupportedDatasets.VERB_AGREEMENT:
            answer = item['clean_answer']

        try:
            # Find answer pos as the first token before padding in the prompt
            answer_pos = prompt.index(self.model.tokenizer.pad_token_id) - 1
        except ValueError: # If this doesn't work, either the prompt is not tokenizer or it's too long
            # In which case it's enough to provide the last token position
            answer_pos = - 1

        if tokenize:
            answer = self.model.to_single_token(answer)

        return answer, answer_pos
    
    def get_corrupted_answer(self, item, prompt, tokenize=True):
        if self.dataset_name == SupportedDatasets.COMMONSENSE_QA:
            correct_answer = item['answerKey']
            answer = [option for option in ['A', 'B', 'C', 'D', 'E'] if option != correct_answer]
        elif self.dataset_name == SupportedDatasets.VERB_AGREEMENT:
            answer = item['patch_answer']

        try:
            # Find answer pos as the first token before padding in the prompt
            answer_pos = prompt.index(self.model.tokenizer.pad_token_id) - 1
        except ValueError: # If this doesn't work, either the prompt is not tokenizer or it's too long
            # In which case it's enough to provide the last token position
            answer_pos = - 1

        if tokenize:
            if isinstance(answer, list):
                answer = [self.model.to_single_token(option) for option in answer]
            else:
                answer = self.model.to_single_token(answer)

        return answer, answer_pos

    def get_masks(self, special_token_mask):
        first_non_special_token_idx = torch.nonzero(special_token_mask == 0, as_tuple=False)[0].item()
        control_sequence_length = first_non_special_token_idx + 1

        attention_mask = torch.ones_like(special_token_mask)
        reversed_special_mask = torch.flip(special_token_mask, [0])

        first_padding_pos_reversed = torch.nonzero(reversed_special_mask == 0, as_tuple=False)[0].item() - 1
        first_padding_pos = len(special_token_mask) - 1 - first_padding_pos_reversed

        attention_mask[first_padding_pos:] = 0

        return control_sequence_length, attention_mask
            
    def get_clean_sample(self, item, tokenize, apply_chat_template, prepend_generation_prefix=False):
        """Process each example from the dataset with padding when tokenizing."""

        prompt = self.get_formatted_prompt(item, system_prompt=self.clean_system_prompt, task_prompt=self.task_prompt)

        prompt, special_token_mask = self.apply_chat_template_and_tokenize(prompt, tokenize=tokenize, 
                                                                           apply_chat_template=apply_chat_template, 
                                                                           prepend_generation_prefix=prepend_generation_prefix)

        # Construct the answer key
        clean_answer, clean_answer_pos = self.get_clean_answer(item, prompt, tokenize=tokenize)

        # Wrap the prompt in message objects if required
        if self.return_message_objects:
            prompt = {"role": "user", "content": prompt}

        # Prepare the result dictionary
        result_dict = {
            "prompt": prompt,
            "answer": clean_answer,
            "answer_pos": clean_answer_pos
        }

        # Include special_token_mask if tokenization was applied
        if tokenize:
            result_dict["special_token_mask"] = special_token_mask

            control_sequence_length, attention_mask = self.get_masks(special_token_mask)
            result_dict["control_sequence_length"] = control_sequence_length
            result_dict["attention_mask"] = attention_mask

        return result_dict

    def get_corrupted_sample(self, item, tokenize, apply_chat_template, prepend_generation_prefix=False):
        """Process each example from the dataset with padding when tokenizing."""

        prompt = self.get_formatted_prompt(item, system_prompt=self.corrupted_system_prompt, task_prompt=self.task_prompt)

        prompt, special_token_mask = self.apply_chat_template_and_tokenize(prompt, tokenize=tokenize, 
                                                                           apply_chat_template=apply_chat_template, 
                                                                           prepend_generation_prefix=prepend_generation_prefix)
        
        # Construct the answer key
        corrupted_answer, corrupted_answer_pos = self.get_corrupted_answer(item, prompt, tokenize=tokenize)

        # Wrap the prompt in message objects if required
        if self.return_message_objects:
            prompt = {"role": "user", "content": prompt}

        # Prepare the result dictionary
        result_dict = {
            "prompt": prompt,
            "answer": corrupted_answer,
            "answer_pos": corrupted_answer_pos
        }

        # Include special_token_mask if tokenization was applied
        if tokenize:
            result_dict["special_token_mask"] = special_token_mask

            control_sequence_length, attention_mask = self.get_masks(special_token_mask)
            result_dict["control_sequence_length"] = control_sequence_length
            result_dict["attention_mask"] = attention_mask

        return result_dict

    def get_clean_corrupted_datasets(self, tokenize=True, apply_chat_template=True, prepend_generation_prefix=False, pt=True):
        """
        Refactored method to return PyTorch tensors if the 'pt' parameter is set to True (default).
        """

        clean_samples = []
        corrupted_samples = []

        # Iterate through the dataset manually
        for item in tqdm(self.dataset):
            # Process the example
            clean_sample = self.get_clean_sample(item, tokenize=tokenize, prepend_generation_prefix=prepend_generation_prefix,
                                                apply_chat_template=apply_chat_template)
            corrupted_sample = self.get_corrupted_sample(item, tokenize=tokenize, prepend_generation_prefix=prepend_generation_prefix,
                                                        apply_chat_template=apply_chat_template)

            clean_samples.append(clean_sample)
            corrupted_samples.append(corrupted_sample)

        # Convert list of dictionaries into dictionaries suitable for Hugging Face Datasets or PyTorch tensors
        clean_hf_dict = {
            "prompt": [entry["prompt"] for entry in clean_samples],
            "answer": [entry["answer"] for entry in clean_samples],
            "answer_pos": [entry["answer_pos"] for entry in clean_samples]
        }
        corrupted_hf_dict = {
            "prompt": [entry["prompt"] for entry in corrupted_samples],
            "answer": [entry["answer"] for entry in corrupted_samples],
            "answer_pos": [entry["answer_pos"] for entry in corrupted_samples]
        }

        if tokenize:
            clean_hf_dict["special_token_mask"] = [entry["special_token_mask"] for entry in clean_samples]
            corrupted_hf_dict["special_token_mask"] = [entry["special_token_mask"] for entry in corrupted_samples]

            clean_hf_dict["control_sequence_length"] = [entry["control_sequence_length"] for entry in clean_samples]
            corrupted_hf_dict["control_sequence_length"] = [entry["control_sequence_length"] for entry in corrupted_samples]

            clean_hf_dict["attention_mask"] = [entry["attention_mask"] for entry in clean_samples]
            corrupted_hf_dict["attention_mask"] = [entry["attention_mask"] for entry in corrupted_samples]

        if pt:
            for dataset_dict in [clean_hf_dict, corrupted_hf_dict]:
                dataset_dict['prompt'] = torch.tensor(dataset_dict['prompt'], device=self.model.cfg.device)
                dataset_dict['answer'] = torch.tensor(dataset_dict['answer'], device=self.model.cfg.device)
                dataset_dict['answer_pos'] = torch.tensor(dataset_dict['answer_pos'], device=self.model.cfg.device)

                if tokenize:
                    dataset_dict['special_token_mask'] = torch.stack(dataset_dict['special_token_mask'], dim=0)
                    dataset_dict['control_sequence_length'] = torch.tensor(dataset_dict['control_sequence_length'], device=self.model.cfg.device)
                    dataset_dict['attention_mask'] = torch.stack(dataset_dict['attention_mask'], dim=0)

            return clean_hf_dict, corrupted_hf_dict
        else:
            # If not using PyTorch tensors, return as Hugging Face Datasets
            clean_hf_dataset = Dataset.from_dict(clean_hf_dict)
            corrupted_hf_dataset = Dataset.from_dict(corrupted_hf_dict)

            return clean_hf_dataset, corrupted_hf_dataset
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# for i in range(3):
    # clean_sample = clean_dataset[i]
    # corrupted_sample = corrupted_dataset[i]

    # clean_prompt = model.to_string(clean_sample['prompt'])
    # corrupted_prompt = model.to_string(corrupted_sample['prompt'])

    # clean_prompt_str = model.to_str_tokens(clean_prompt, prepend_bos=False)
    # corrupted_prompt_str = model.to_str_tokens(corrupted_prompt, prepend_bos=False)

    # clean_answer_pos = clean_sample['answer_pos']
    # corrupted_answer_pos = corrupted_sample['answer_pos']

    # clean_answer_str = model.to_string(clean_sample['answer'])
    # corrupted_answer_str = model.to_string(corrupted_sample['answer'])

    # print('Clean prompt around answer:', clean_prompt_str[:clean_answer_pos+5] 
    #       if clean_answer_pos != -1 else clean_prompt_str)
    # print('Clean prompt answer token:', clean_prompt_str[clean_answer_pos])
    
    # print('Corrupted prompt around answer:', corrupted_prompt_str[:corrupted_answer_pos+5] 
    #         if corrupted_answer_pos != -1 else corrupted_prompt_str)
    # print('Corrupted prompt answer token:', corrupted_prompt_str[corrupted_answer_pos])

    # print('Clean prompt answer:', clean_answer_str)
    # print('Corrupted prompt answer:', corrupted_answer_str)

#     #Print the special token and attention mask
#     print('Clean')
#     print(clean_sample['special_token_mask'])
#     print(clean_sample['attention_mask'])
    
#     print('Corrupted')
#     print(corrupted_sample['special_token_mask'])
#     print(corrupted_sample['attention_mask'])

    # print()