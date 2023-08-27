---
layout: post
title: Set up Obsidian to Work with Zotero
tags:  obsidian zotero github jekyii
---

This is my set up to enable Obsidian work with Zotero to export Zotero note and publish to my [Github website](https://zhangtemplar.github.io/). You could create your own website on Github using [JekyII](https://github.com/barryclark/jekyll-now), which requires you to create markdown files with a specific front matter format. Then Github will publish your markdown files to html.

To start please make sure you have installed [[obsidian-zotero-integration](https://github.com/mgmeyers/obsidian-zotero-integration)](https://github.com/mgmeyers/obsidian-zotero-integration) and followed the set ups there. Then you could use my template described below.
# Plugin Setting
This is my configuration of obsidian-zotero-integration:
![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/obsidian1.png)
![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/obsidian2.png)
![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/obsidian3.png)
# Template
Please change `https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/` to your own link.
```
---
layout: post
title: {{title}}
tags: {% if tags.length > 0 -%}{% for t in tags %} {{t.tag | lower | replace(" ", "-")}}{%- endfor %}{% endif %}
---

> {{abstractNote}}

# Notes
{% persist "annotations" %}
{% set annotations = annotations | filterby("date", "dateafter", lastImportDate) -%}
{% if annotations.length > 0 %}

{%- for annotation in annotations %}
{% if annotation.color !== "#ffd400" %}
>[!quote{% if annotation.color %}|{{annotation.color}}{% endif %}] {{calloutHeader(annotation.color)}}
>{%- endif -%}{% if annotation.imageRelativePath %}
![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/{{annotation.imageRelativePath}}) {% endif %}{% if annotation.annotatedText %}
{{annotation.annotatedText}} [(p. {{annotation.pageLabel}})](zotero://open-pdf/library/items/{{annotation.attachment.itemKey}}?page={{annotation.pageLabel}}&annotation={{annotation.id}}){%- endif %}{%- if annotation.comment%}
%%{{annotation.comment}}%%{%- endif %}{%- endfor %}{% endif %} {% endpersist %}
```

# How to Use
1. open command palette to select `Zotero Intergation: import #1`;
2. in the prompt, put your cite key of paper you want to import from Zotero;
3. after note generated, rename the note as `YYYY-mm-dd-title`;
4. optionally, update the contents of notes as you wish;
5. commit the diff to Github to publish: `git add .` and `git commit -a -m "some message"`.
