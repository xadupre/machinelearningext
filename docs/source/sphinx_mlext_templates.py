"""
Template for documentation.
"""

index_template = """
Type of Components
==================

.. contents::
    :local:    
{% for vk in sorted_kinds %}
    {{vk[1]}}{% endfor %}
"""

kind_template = """

{{title}}
{{"=" * len(title)}}

.. contents::
    :local:    
{% for fname in fnames %}
    {{fname}}{% endfor %}
"""

component_template = """

{{title}}
{{"=" * len(title)}}

**Aliases**

{{aliases}}

**Description**

*{{kind}}*

{{summary}}

**Parameters**

.. list-table::
    :widths: 5 5 15
    :header-rows: 1
    * - Name
      - Short name
      - Default
      - Description    
{% for kv in sorted_params %}
    * - {{kv["Name"]}}
      - {{kv["ShortName"]}}
      - {{kv["Default"]}}
      - {{kv["Description"]}}{% endfor %}

"""