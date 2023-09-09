#!python3
import os
from tqdm.auto import tqdm


def create_tag(tag: str, tag_path: str):
    if len(tag) <= 1:
        return
    tag_name = os.path.join(tag_path, tag + ".md")
    if os.path.exists(tag_name):
        return
    with open(tag_name, "w") as fo:
        content = f"""---
layout: tagpage
title: "Reading Note on Deep Learning"
tag: {tag}
---
"""
        fo.write(content)


def canonize_tag(line: str) -> str:
    tags = line[len("tags:"):].split(" ")
    mapping = mapping = {
        "volume": "nerf",
        "textimage": "text2image",
        "dalle": "dall-e",
        "3d-face-reconstruction": "3d",
    }
    new_tags = set()
    for t in tags:
        t = t.lower()
        if t in mapping:
            t = mapping[t]
        new_tags.add(t.replace("\n", ""))
    return "tags: " + " ".join(new_tags) + "\n"


def replace_line_text(post_file: str):
    with open(post_file, "r") as fi:
        lines = fi.readlines()
    for i, line in enumerate(lines):
        if not line.startswith("tags:"):
            continue
        lines[i] = canonize_tag(line)
        break
    with open(post_file, "w") as fo:
        fo.writelines(lines)


def main(post_path: str = "_posts", tag_path: str = "tag"):
    root_path = os.getcwd()
    post_path = os.path.join(root_path, post_path)
    tag_path = os.path.join(root_path, tag_path)
    processed_tags = set()
    for name in tqdm(os.listdir(post_path)):
        if not name.endswith(".md"):
            continue
        post_file = os.path.join(post_path, name)
        replace_line_text(post_file)
        with open(post_file, "r") as fi:
            for line in fi.readlines():
                if not line.startswith("tags:"):
                    continue
                tags = line[len("tags:"):].split(" ")
                # create tag file
                for t in tqdm(tags, leave=False):
                    if t.endswith("\n"):
                        t = t[:-1]
                    if t in processed_tags:
                        continue
                    create_tag(t, tag_path)
                    processed_tags.add(t)
                break


if __name__ == "__main__":
    main()