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
![[{{annotation.imageRelativePath}}]] {% endif %}{% if annotation.annotatedText %}
{{annotation.annotatedText}} [(p. {{annotation.pageLabel}})](zotero://open-pdf/library/items/{{annotation.attachment.itemKey}}?page={{annotation.pageLabel}}&annotation={{annotation.id}}){%- endif %}{%- if annotation.comment%}
%%{{annotation.comment}}%%{%- endif %}{%- endfor %}{% endif %} {% endpersist %}