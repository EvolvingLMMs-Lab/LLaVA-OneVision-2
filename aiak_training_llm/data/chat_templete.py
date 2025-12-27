# Copyright 2024 the LlamaFactory team.
# Copyright (c) 2024, AIAK team. All rights reserved.
# This code was adopted from https://github.com/hiyouga/LLaMA-Factory
# and the source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

"""Chat templates

About chat templates, can see more info at
https://huggingface.co/docs/transformers/main/en/chat_templating#templates-for-chat-models.
"""

import re
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Type, Dict, List, Optional, Sequence, Set, Tuple, Union

from aiak_training_llm.utils.constants import DataRoles
from .mm_plugin import MMPlugin, Qwen2VLPlugin


if TYPE_CHECKING:
    from aiak_training_llm.tokenizer import AutoTokenizerFromHF


SlotsType = Sequence[Union[str, Set[str], Dict[str, str]]]


@dataclass
class Formatter(ABC):
    """Base class of all formatters."""
    slots: SlotsType = field(default_factory=list)

    @abstractmethod
    def apply(self, **kwargs) -> SlotsType:
        """Apply the formatter to the given arguments"""
        raise NotImplementedError


@dataclass
class EmptyFormatter(Formatter):
    """An empty formatter that does nothing"""
    def __post_init__(self):
        has_placeholder = False
        for slot in filter(lambda s: isinstance(s, str), self.slots):
            if re.search(r"\{\{[a-zA-Z_][a-zA-Z0-9_]*\}\}", slot):
                has_placeholder = True

        if has_placeholder:
            raise ValueError("Empty formatter should not contain any placeholder.")

    def apply(self, **kwargs) -> SlotsType:
        """Apply the formatter to the given arguments"""
        return self.slots


@dataclass
class StringFormatter(Formatter):
    """String formatter"""
    def __post_init__(self):
        has_placeholder = False
        for slot in filter(lambda s: isinstance(s, str), self.slots):
            if re.search(r"\{\{[a-zA-Z_][a-zA-Z0-9_]*\}\}", slot):
                has_placeholder = True

        if not has_placeholder:
            raise ValueError("A placeholder is required in the string formatter.")

    def apply(self, **kwargs) -> SlotsType:
        """Apply the formatter to the given arguments"""
        elements = []
        for slot in self.slots:
            if isinstance(slot, str):
                for name, value in kwargs.items():
                    if not isinstance(value, str):
                        raise RuntimeError("Expected a string, got {}".format(value))

                    slot = slot.replace("{{" + name + "}}", value, 1)
                elements.append(slot)
            elif isinstance(slot, (dict, set)):
                elements.append(slot)
            else:
                raise RuntimeError("Input must be string, set[str] or dict[str, str], got {}".format(type(slot)))

        return elements


@dataclass
class ChatTemplate:
    """ChatTemplate class."""
    format_user: Optional[Formatter] = None
    format_assistant: Optional[Formatter] = None
    format_system: Optional[Formatter] = None
    format_separator: Optional[Formatter] = None
    format_prefix: Optional[Formatter] = None
    default_system: str = ""
    stop_words: List[str] = field(default_factory=list)
    efficient_eos: bool = False
    replace_eos: bool = False
    mm_plugin: Optional[MMPlugin] = None

    def __post_init__(self):
        if self.format_user is None:
            self.format_user = StringFormatter(slots=["{{content}}"])
        
        # if efficient_eos=true, we will not add eos_token among the multiple turns,
        # and it will be added in the end of the last response.
        eos_slots = [] if self.efficient_eos else [{"eos_token"}]
        if self.format_assistant is None:
            self.format_assistant = StringFormatter(slots=["{{content}}"] + eos_slots)

        if self.format_system is None:
            self.format_system = StringFormatter(slots=["{{content}}"])

        if self.format_separator is None:
            self.format_separator = EmptyFormatter()
        
        if self.format_prefix is None:
            self.format_prefix = EmptyFormatter()

    def encode_multiturn(
        self,
        tokenizer: "AutoTokenizerFromHF",
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
    ) -> List[Tuple[List[int], List[int]]]:
        """
        Returns multiple pairs of token ids representing prompts and responses respectively.
        """

        if len(messages) % 2 != 0:
            system = messages[0]['content']
            messages = messages[1:]

        encoded_messages = self._encode(tokenizer, messages, system)
        # ZXW: missing "/n"
        if int(os.environ.get("FILL_TOKEN_198",0))==1:
            encoded_messages[-1].append(198)
        return [(encoded_messages[i], encoded_messages[i + 1]) for i in range(0, len(encoded_messages), 2)]

    def encode_oneturn(
        self,
        tokenizer: "AutoTokenizerFromHF",
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
    ) -> Tuple[List[int], List[int]]:
        """
        Returns a single pair of token ids representing prompt and response respectively.
        """
        encoded_messages = self._encode(tokenizer, messages, system)
        prompt_ids = []
        for encoded_ids in encoded_messages[:-1]:
            prompt_ids += encoded_ids

        answer_ids = encoded_messages[-1]
        return prompt_ids, answer_ids

    def _encode(
        self,
        tokenizer: "AutoTokenizerFromHF",
        messages: Sequence[Dict[str, str]],
        system: Optional[str],
    ) -> List[List[int]]:
        """
        Encodes formatted inputs to pairs of token ids.
        Turn 0: prefix + system + query     resp
        Turn t: sep + query                 resp
        """
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []

            if i == 0:
                elements += self.format_prefix.apply()
                if system:
                    elements += self.format_system.apply(content=system)

            elif i > 0 and i % 2 == 0:
                elements += self.format_separator.apply()

            if message["role"] == DataRoles.USER:
                elements += self.format_user.apply(content=message["content"], idx=str(i // 2))
            elif message["role"] == DataRoles.ASSISTANT:
                elements += self.format_assistant.apply(content=message["content"])
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))

            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return encoded_messages

    def _convert_elements_to_ids(
        self,
        tokenizer: "AutoTokenizerFromHF",
        elements: "SlotsType",
    ) -> List[int]:
        """
        Converts elements to token ids.
        """
        token_ids = []
        for elem in elements:
            if isinstance(elem, str):
                if len(elem) != 0:
                    token_ids += tokenizer.tokenize(elem, add_special_tokens=False)

            elif isinstance(elem, dict):
                token_ids += [tokenizer.convert_tokens_to_ids(elem.get("token"))]

            elif isinstance(elem, set):
                if "bos_token" in elem and tokenizer.bos is not None:
                    token_ids += [tokenizer.bos]

                elif "eos_token" in elem and tokenizer.eos is not None:
                    token_ids += [tokenizer.eos]

            else:
                raise ValueError("Input must be string, set[str] or dict[str, str], got {}".format(type(elem)))

        return token_ids

    @classmethod
    def from_name(cls, name: str) -> "ChatTemplate":
        """build template."""
        return MAPPING_NAME_TO_TEMPLATE.get(name, None)


@dataclass
class Llama2Template(ChatTemplate):
    """LLaMA-2 Template"""
    def _encode(
        self,
        tokenizer: "AutoTokenizerFromHF",
        messages: Sequence[Dict[str, str]],
        system: str,
    ) -> List[List[int]]:
        """
        Encodes formatted inputs to pairs of token ids.
        Turn 0: prefix + system + query    resp
        Turn t: sep + query                resp
        """
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []
            
            system_text = ""

            if i == 0:
                elements += self.format_prefix.apply()
                if system:
                    system_text = self.format_system.apply(content=system)[0]
            
            if i > 0 and i % 2 == 0:
                elements += self.format_separator.apply()

            if message["role"] == DataRoles.USER:
                elements += self.format_user.apply(content=system_text + message["content"])
            elif message["role"] == DataRoles.ASSISTANT:
                elements += self.format_assistant.apply(content=message["content"])
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))

            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return encoded_messages


MAPPING_NAME_TO_TEMPLATE: Dict[str, ChatTemplate] = {}


def _register_chat_template(
    name: str,
    cls: Type[ChatTemplate] = ChatTemplate,
    format_user: Optional[Formatter] = None,
    format_assistant: Optional[Formatter] = None,
    format_system: Optional[Formatter] = None,
    format_separator: Optional[Formatter] = None,
    format_prefix: Optional[Formatter] = None,
    default_system: str = "",
    stop_words: Sequence[str] = [],
    efficient_eos: bool = False,
    replace_eos: bool = False,
    mm_plugin: Optional[MMPlugin] = None,
) -> None:
    """
    Registers a chat template.

    To add the following chat template:
    ```
    [HUMAN]:
    user prompt here
    [AI]:
    model response here

    [HUMAN]:
    user prompt here
    [AI]:
    model response here
    ```

    The corresponding code should be:
    ```
    _register_chat_template(
        name="custom",
        format_user=StringFormatter(slots=["[HUMAN]:\n{{content}}\n[AI]:\n"]),
        format_separator=EmptyFormatter(slots=["\n\n"]),
        efficient_eos=True,
    )
    ```
    """
    if name in MAPPING_NAME_TO_TEMPLATE:
        raise ValueError(f"Cannot register duplicate template with name {name}.")
    
    MAPPING_NAME_TO_TEMPLATE[name] = cls(
        format_user=format_user,
        format_assistant=format_assistant,
        format_system=format_system,
        format_separator=format_separator,
        format_prefix=format_prefix,
        default_system=default_system,
        stop_words=stop_words,
        efficient_eos=efficient_eos,
        replace_eos=replace_eos,
        mm_plugin=mm_plugin,
    )
    

def get_support_templates() -> List[str]:
    """
    Returns a list of supported chat templates.
    """
    return list(MAPPING_NAME_TO_TEMPLATE.keys())
    

_register_chat_template(
    name="empty",
    efficient_eos=True,
)


_register_chat_template(
    name="default",
    format_user=StringFormatter(slots=["Human: {{content}}\nAssistant:"]),
    format_system=StringFormatter(slots=["{{content}}\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
)


_register_chat_template(
    name="alpaca",
    format_user=StringFormatter(slots=["### Instruction:\n{{content}}\n\n### Response:\n"]),
    format_separator=EmptyFormatter(slots=["\n\n"]),
    default_system=(
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
    ),
)


_register_chat_template(
    name="baichuan",
    format_user=StringFormatter(slots=[{"token": "<reserved_102>"}, "{{content}}", {"token": "<reserved_103>"}]),
    efficient_eos=True,
)


_register_chat_template(
    name="baichuan2",
    format_user=StringFormatter(slots=["<reserved_106>{{content}}<reserved_107>"]),
    efficient_eos=True,
)


_register_chat_template(
    name="llama2",
    cls=Llama2Template,
    format_user=StringFormatter(slots=[{"bos_token"}, "[INST] {{content}} [/INST]"]),
    format_system=StringFormatter(slots=["<<SYS>>\n{{content}}\n<</SYS>>\n\n"]),
)


_register_chat_template(
    name="llama2_zh",
    cls=Llama2Template,
    format_user=StringFormatter(slots=[{"bos_token"}, "[INST] {{content}} [/INST]"]),
    format_system=StringFormatter(slots=["<<SYS>>\n{{content}}\n<</SYS>>\n\n"]),
    default_system="You are a helpful assistant. 你是一个乐于助人的助手。",
)


_register_chat_template(
    name="llama3",
    format_user=StringFormatter(
        slots=[
            (
                "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_system=StringFormatter(slots=["<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|eot_id|>"],
    replace_eos=True,
)

_register_chat_template(
    name="llama3.1",
    format_user=StringFormatter(
        slots=[
            (
                "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_system=StringFormatter(slots=["<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|eot_id|>"],
    replace_eos=True,
)


_register_chat_template(
    name="mistral",
    format_user=StringFormatter(slots=["[INST] {{content}} [/INST]"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
)


_register_chat_template(
    name="qwen",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    default_system="You are a helpful assistant.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
)

_register_chat_template(
    name="qwen2-vl",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    default_system="You are a helpful assistant.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
    mm_plugin=Qwen2VLPlugin(image_token="<|image_pad|>", video_token="<|video_pad|>"),
)

_register_chat_template(
    name="deepseek",
    format_user=StringFormatter(slots=["User: {{content}}\n\nAssistant:"]),
    format_system=StringFormatter(slots=["{{content}}\n\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
)

_register_chat_template(
    name="deepseek3",
    format_user=StringFormatter(slots=["<｜User｜>{{content}}<｜Assistant｜>"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
)
