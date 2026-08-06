"""`ab-pipeline_zh.md` §5 的 prompt 全文必须与 `vlm_audit.py` 逐字一致。

为什么要有这个测试：服务器上**没有 Claude**，维护者是照着 `ab-pipeline_zh.md`
理解「模型到底收到了什么」的。改了代码里的 prompt 却忘了改文档，服务器就会
按过时的文本去解读产出——而这种错在结果里**看不出来**，只会让人把模型的行为
归因错。文档里的 §5 是渲染后的样子（源码里经 `.format` 消费的把 `{}` 写成
`{{}}`），所以比对前先把双大括号还原。

新增 prompt 时：在 `ab-pipeline_zh.md` §5 加一节 `### 5.x 说明 \\`NAME\\`` +
一个 ```text 代码块，本测试会自动带上它。
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "recaption", "scripts"))
import vlm_audit  # noqa: E402

DOC = os.path.join(os.path.dirname(__file__), "..", "recaption", "docs",
                   "ab-pipeline_zh.md")

# ### 5.x <随便什么说明> `NAME_PROMPT`（...）  紧跟一个 ```text ... ``` 块
SECTION = re.compile(
    r"^### 5\.\d+[^\n`]*`(?P<name>[A-Z_]+_PROMPT)`[^\n]*\n+```text\n"
    r"(?P<body>.*?)\n```", re.M | re.S)


def _doc_prompts():
    with open(DOC, encoding="utf-8") as f:
        return {m.group("name"): m.group("body") for m in SECTION.finditer(f.read())}


def _rendered(prompt):
    """源码 -> 文档展示形态：`.format` 用的双大括号还原成单个。"""
    return prompt.replace("{{", "{").replace("}}", "}").strip()


def test_doc_section_5_is_parseable():
    """§5 至少要能解析出这几条——正则挂了会让下面的比对静默通过。"""
    found = _doc_prompts()
    assert {"RECAP_PROMPT", "SLOT_RECAP_PROMPT", "FUSE_LONG_PROMPT",
            "FUSE_SHORT_PROMPT"} <= set(found), sorted(found)


def test_every_documented_prompt_matches_the_code():
    for name, body in _doc_prompts().items():
        assert hasattr(vlm_audit, name), (
            f"文档 §5 写着 {name}，但 vlm_audit.py 里没有——"
            f"prompt 被删/改名了就把文档那一节也删掉")
        assert body.strip() == _rendered(getattr(vlm_audit, name)), (
            f"{name} 文档与代码不一致：改了 vlm_audit.py 就要同步更新 "
            f"recaption/docs/ab-pipeline_zh.md §5（服务器只读文档）")


def test_live_prompts_are_all_documented():
    """反向：链路上在用的 prompt 不许缺文档。"""
    documented = set(_doc_prompts())
    for name in ("RECAP_PROMPT", "SLOT_RECAP_PROMPT", "FUSE_LONG_PROMPT",
                 "FUSE_SHORT_PROMPT", "DERIVE_PROMPT", "CONDENSE_PROMPT",
                 "REFINE_PROMPT"):
        assert name in documented, f"{name} 在代码里用着，却没写进 §5"
