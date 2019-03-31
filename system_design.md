---
layout: page
title: "System Design Note"
permalink: "/system_design/"
---

<ul>
  {% for sd in site.system_design %}
    <li>
      <a href="{{ sd.url }}">{{ sd.title }}</a>
      - {{ sd.headline }}
    </li>
  {% endfor %}
</ul>