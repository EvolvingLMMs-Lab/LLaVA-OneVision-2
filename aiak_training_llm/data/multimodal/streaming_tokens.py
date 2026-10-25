"""Canonical list of the special tokens streaming samples are written with.

Kept dependency-free so both the training encoder and the standalone tokenizer
tooling can import it: the list must not drift between what training tokenizes
and what an exported checkpoint ships.

Each entry must survive tokenization as exactly one id. Registering them is what
makes that true; until then ``<silence>`` is four subword pieces and the state
tokens cannot be weighted, counted or sampled as single symbols.

ORDER IS PART OF THE CONTRACT. Ids are handed out in the order below, so a
checkpoint trained against this list depends on it. Append new tokens at the end
of their group; never reorder or remove an existing entry.
"""

#: The two assistant states, and the pair the focal-loss option balances against
#: each other: ``<silence>`` for a frame with nothing to say, ``<response>`` for
#: the turn that speaks - a running caption and an answer to a question are the
#: same act, so they share one marker. ``_sample_state_token_alpha`` computes an
#: inverse-frequency alpha over exactly these two classes, so this tuple must
#: stay length 2.
STREAMING_STATE_TOKENS = (
    "<silence>",
    "<response>",
)

#: User-side memory blocks: the task being answered, and the reply the model
#: gave most recently. A historical ``<response>`` turn is deliberately left
#: bare, which makes these blocks the only path by which that content reaches
#: the model. They carry no outer envelope - each block delimits itself.
STREAMING_MEMORY_TOKENS = (
    "<question_memory>",
    "</question_memory>",
    "<answer_memory>",
    "</answer_memory>",
)

#: The caption spelling of ``<response>``. A caption turn is a speaking turn, but it
#: carries its own marker so a run can weight the two apart: ``SEGMENT_MARKERS`` accepts
#: both, and ``--state-token-focal-captioning`` folds this one into the speaking class.
#: It cannot join ``STREAMING_STATE_TOKENS`` - that tuple must stay length 2 - but it is
#: registered here, before the task markers, because that is the order the trained
#: checkpoints assign ids in (``<captioning>`` 248083, ``<caption_begin>`` 248084).
STREAMING_CAPTION_TOKENS = (
    "<captioning>",
)

#: Task markers placed in the user turn before the first frame. They name the job for
#: the whole sample, so unlike the memory blocks they carry no payload and appear once.
#: ``<caption_begin>`` asks for event-level streaming captioning; it replaces the fixed
#: instruction that used to be wrapped in ``<question_memory>``, which now always holds
#: a real question. Never supervised: it sits on the user side.
STREAMING_TASK_TOKENS = (
    "<caption_begin>",
)

#: Everything a streaming checkpoint must have registered, in a stable order.
STREAMING_SPECIAL_TOKENS = (STREAMING_STATE_TOKENS + STREAMING_MEMORY_TOKENS
                            + STREAMING_CAPTION_TOKENS + STREAMING_TASK_TOKENS)
