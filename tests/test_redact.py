# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Secret redaction tests: known token formats are scrubbed at the adapter
choke point (make_trajectory) and survive no false positives on clean text.

Fake tokens are assembled by concatenation so this file itself does not match
secret-scanning patterns (GitHub push protection scans committed test files too).
"""

import json

from aqa.agent.adapters.base import make_trajectory
from aqa.agent.redact import redact_text

GHP = "ghp_" + "A" * 32 + "1234"
HF = "hf_" + "a" * 30
SK = "sk-" + "abcd" * 5
AWS = "AKIA" + "I" * 16
GL = "glpat-" + "x" * 18


def test_redact_text_known_tokens():
    assert redact_text(f"url is https://{GHP}@github.com/x/y.git") == "url is https://***REDACTED***@github.com/x/y.git"
    assert redact_text(f"token {HF}") == "token ***REDACTED***"
    assert redact_text(f"key {SK}") == "key ***REDACTED***"
    assert redact_text(f"aws {AWS}") == "aws ***REDACTED***"
    assert redact_text(f"gl {GL}") == "gl ***REDACTED***"


def test_redact_text_url_userinfo():
    secret = "s3cr3t" + "pass"
    out = redact_text(f"git clone https://oauth2:{secret}@gitlab.com/grp/proj.git")
    assert secret not in out
    assert "https://oauth2:***REDACTED***@gitlab.com" in out


def test_redact_text_clean_text_untouched():
    clean = "auto-round --model_name foo --scheme W4A16 --ignore_layers lm_head,embed_tokens"
    assert redact_text(clean) == clean
    # 短串不误伤（不到 token 长度阈值）
    short = "hf_short sk-123 ghp_abc"
    assert redact_text(short) == short
    # 自然语言中的 "sk-" 子串（mask-probability-derived 这类）不误伤
    natural = "use mask-probability-derived boxes for scoring"
    assert redact_text(natural) == natural


def test_make_trajectory_redacts_steps():
    traj = make_trajectory(
        session_id="s1",
        agent_name="kimi",
        steps=[
            {"source": "user", "message": f"token is {HF}"},
            {"source": "agent", "message": "ok",
             "tool_calls": [{"tool_call_id": "c1", "function_name": "Bash",
                             "arguments": {"command": f"echo {SK}"}}],
             "observation": {"results": [{"source_call_id": "c1",
                                          "content": f"aws {AWS}"}]},
             "reasoning_content": f"uses {GL}"},
        ],
    )
    blob = json.dumps(traj)
    for token in (GHP, HF, SK, AWS, GL):
        assert token not in blob
    assert blob.count("***REDACTED***") == 4
    # 标识字段不动
    assert traj["session_id"] == "s1" and traj["agent"]["name"] == "kimi"
