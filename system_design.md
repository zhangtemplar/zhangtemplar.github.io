---
layout: page
title: "System Design Note"
permalink: "/system_design/"
---

<ul>
  {% for sd in site.system_design %}
    <li>
      <a href="{{ site.baseurl }}{{ sd.url }}">{{ sd.title }}</a>

      <div class="entry">
        {{ sd.excerpt }}
      </div>

      <a href="{{ site.baseurl }}{{ sd.url }}" class="read-more">Read More</a>
    </li>
  {% endfor %}
</ul>