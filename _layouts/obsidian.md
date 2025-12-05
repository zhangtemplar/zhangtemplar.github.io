---
layout: post
title: {{title | replace(":", "") | replace(";", "")}}
tags: {% if tags.length > 0 -%}{% for t in tags %} {{t.tag | lower | replace(" ", "-")}}{%- endfor %}{% endif %}
---

> {{abstractNote}}

# Notes
{% persist "annotations" %}
{% set annotations = annotations | filterby("date", "dateafter", lastImportDate) -%}
{% if annotations.length > 0 %}

{%- for annotation in annotations %}
{% if annotation.color !== "#ffd400" %}
{%- if annotation.color == "#ff6666" or annotation.color == "#f19837" %}
>[!quote|{{annotation.color}}] Important
{%- elif annotation.color == "#5fb236" %}
>[!quote|{{annotation.color}}] Question
{%- elif annotation.color == "#2ea8e5" %}
>[!quote|{{annotation.color}}] Definition
{%- elif annotation.color == "#a28ae5" %}
>[!quote|{{annotation.color}}] Comment
{%- else %}
>[!quote{% if annotation.color %}|{{annotation.color}}{% endif %}] Highlight
{%- endif %}
>{%- endif -%}{% if annotation.imageRelativePath %}
![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/{{annotation.imageRelativePath}}) {% endif %}{% if annotation.annotatedText %}
{{annotation.annotatedText}} [(p. {{annotation.pageLabel}})](zotero://open-pdf/library/items/{{annotation.attachment.itemKey}}?page={{annotation.pageLabel}}&annotation={{annotation.id}}){%- endif %}{%- if annotation.comment%}
%%{{annotation.comment}}%%{%- endif %}{%- endfor %}{% endif %} {% endpersist %}