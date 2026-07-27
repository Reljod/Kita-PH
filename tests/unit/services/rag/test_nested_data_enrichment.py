"""Tests for app.services.rag.nested_data_enrichment_service.

This turns LlamaParse's flat page/item output back into the document's nested
shape, then emits one leaf per value with a `json_path` locating it in that
tree. Retrieval resolves candidates by replaying the same build and looking
their path up, so the paths this produces have to be stable and unique --
a collision silently merges two unrelated values in the knowledge base.
"""

from __future__ import annotations

import pytest

from app.services.rag.nested_data_enrichment_service import (
    NestedDataEnrichmentService,
    get_nested_value,
    is_heading,
    set_nested_value,
    slugify,
)

FILE_ID = "file_1"
ORG_ID = "org_test_0001"


@pytest.fixture
def service() -> NestedDataEnrichmentService:
    return NestedDataEnrichmentService()


def a_page(items, page=1) -> dict:
    return {"page": page, "items": items}


def build(service, items, page=1):
    return service.build_hierarchy_and_leaves(
        parse_result={"pages": [a_page(items, page)]}, file_id=FILE_ID, org_id=ORG_ID
    )


# --- helpers --------------------------------------------------------------


class TestSlugify:
    def test_text_is_lowercased_and_joined(self):
        assert slugify("Employee Handbook") == "employee_handbook"

    def test_punctuation_is_dropped(self):
        assert slugify("Q&A: What's new?") == "qa_whats_new"

    def test_html_tags_are_stripped(self):
        """LlamaParse emits markup inside headings; the tag names would
        otherwise become part of the path."""
        assert slugify("<b>Bold</b> heading") == "bold_heading"

    def test_runs_of_separators_collapse(self):
        assert slugify("a   -  b") == "a_b"

    def test_leading_and_trailing_separators_are_trimmed(self):
        assert slugify("  spaced  ") == "spaced"

    def test_empty_text_yields_nothing(self):
        assert slugify("") == ""

    def test_text_with_no_usable_characters_yields_nothing(self):
        assert slugify("!!!") == ""


class TestIsHeading:
    @pytest.mark.parametrize("text", ["# Title", "## Sub", "###   Deep"])
    def test_markdown_headings_are_recognised(self, text):
        assert is_heading(text) is True

    @pytest.mark.parametrize("text", ["Title", "#NoSpace", "", "  "])
    def test_other_text_is_not(self, text):
        assert is_heading(text) is False

    def test_a_non_string_is_not_a_heading(self):
        """Items arrive from JSON, so a value can be a number or None."""
        assert is_heading(None) is False and is_heading(42) is False


class TestNestedValues:
    def test_a_value_is_set_at_a_path(self):
        tree = {}
        set_nested_value(tree, ["a", "b"], "k", 1)
        assert tree == {"a": {"b": {"k": 1}}}

    def test_an_empty_path_sets_at_the_root(self):
        tree = {}
        set_nested_value(tree, [], "k", 1)
        assert tree == {"k": 1}

    def test_a_scalar_in_the_way_is_replaced(self):
        """Two leaves can claim the same slug when one is a value and the
        other a section; the section has to win or the path breaks."""
        tree = {"a": "scalar"}
        set_nested_value(tree, ["a", "b"], "k", 1)
        assert tree == {"a": {"b": {"k": 1}}}

    def test_a_value_is_retrieved(self):
        assert get_nested_value({"a": {"b": 1}}, ["a", "b"]) == 1

    def test_a_sub_tree_is_retrieved(self):
        assert get_nested_value({"a": {"b": 1}}, ["a"]) == {"b": 1}

    def test_a_missing_path_yields_nothing(self):
        assert get_nested_value({"a": 1}, ["b"]) is None

    def test_descending_through_a_scalar_yields_nothing(self):
        assert get_nested_value({"a": 1}, ["a", "b"]) is None


# --- building the hierarchy -----------------------------------------------


class TestBuildHierarchy:
    def test_no_pages_yields_nothing(self, service):
        assert service.build_hierarchy_and_leaves({}, FILE_ID, ORG_ID) == ({}, [])

    def test_pages_can_be_nested_under_items(self, service):
        """LlamaParse has emitted both shapes across versions."""
        result = {"items": {"pages": [a_page([{"type": "text", "value": "hello"}])]}}
        _, leaves = service.build_hierarchy_and_leaves(result, FILE_ID, ORG_ID)
        assert len(leaves) == 1

    def test_a_text_item_becomes_a_leaf(self, service):
        _, leaves = build(service, [{"type": "text", "value": "hello"}])
        assert leaves[0]["content"] == "hello"

    def test_a_leaf_carries_its_tenancy(self, service):
        _, leaves = build(service, [{"type": "text", "value": "hello"}])
        assert leaves[0]["org_id"] == ORG_ID and leaves[0]["file_id"] == FILE_ID

    def test_an_empty_item_is_skipped(self, service):
        _, leaves = build(service, [{"type": "text", "value": "   "}])
        assert leaves == []

    def test_an_item_with_neither_value_nor_markdown_is_skipped(self, service):
        _, leaves = build(service, [{"type": "text"}])
        assert leaves == []

    def test_markdown_stands_in_for_a_missing_value(self, service):
        _, leaves = build(service, [{"type": "text", "md": "from markdown"}])
        assert leaves[0]["content"] == "from markdown"

    def test_a_heading_scopes_the_leaves_under_it(self, service):
        tree, leaves = build(
            service,
            [
                {"type": "heading", "value": "Benefits", "level": 1},
                {"type": "text", "value": "free coffee"},
            ],
        )
        assert leaves[0]["json_path"].startswith("benefits.")
        assert "benefits" in tree

    def test_the_heading_trail_is_recorded(self, service):
        """This is what tells the model which section an answer came from."""
        _, leaves = build(
            service,
            [
                {"type": "heading", "value": "Handbook", "level": 1},
                {"type": "heading", "value": "Benefits", "level": 2},
                {"type": "text", "value": "free coffee"},
            ],
        )
        assert leaves[0]["heading_text"] == "Handbook > Benefits"

    def test_a_shallower_heading_closes_the_deeper_ones(self, service):
        """Otherwise a later top-level section inherits the previous
        section's subheadings."""
        _, leaves = build(
            service,
            [
                {"type": "heading", "value": "One", "level": 1},
                {"type": "heading", "value": "Deep", "level": 2},
                {"type": "heading", "value": "Two", "level": 1},
                {"type": "text", "value": "content"},
            ],
        )
        assert leaves[0]["heading_text"] == "Two"

    def test_a_markdown_heading_is_recognised_without_a_type(self, service):
        _, leaves = build(
            service,
            [
                {"type": "text", "md": "## Benefits"},
                {"type": "text", "value": "free coffee"},
            ],
        )
        assert leaves[0]["heading_text"] == "Benefits"

    def test_the_heading_level_is_inferred_from_the_hashes(self, service):
        _, leaves = build(
            service,
            [
                {"type": "text", "md": "# One"},
                {"type": "text", "md": "## Two"},
                {"type": "text", "value": "content"},
            ],
        )
        assert leaves[0]["heading_text"] == "One > Two"

    def test_an_empty_heading_is_skipped(self, service):
        _, leaves = build(
            service,
            [
                {"type": "heading", "value": "  "},
                {"type": "text", "value": "content"},
            ],
        )
        assert leaves[0]["heading_text"] == ""

    def test_a_heading_of_only_punctuation_still_gets_a_path(self, service):
        """slugify returns nothing for it, but a leaf with no path segment
        would collide with everything at the root."""
        _, leaves = build(
            service,
            [
                {"type": "heading", "value": "???", "level": 1},
                {"type": "text", "value": "content"},
            ],
        )
        assert leaves[0]["json_path"].startswith("section_1.")

    def test_the_page_number_is_carried(self, service):
        """Citations point at a page; losing it makes an answer unverifiable."""
        _, leaves = build(service, [{"type": "text", "value": "x"}], page=7)
        assert leaves[0]["page"] == 7


class TestKeyValueLeaves:
    def test_a_colon_separated_line_becomes_a_named_key(self, service):
        """ "Start date: 2024-01-01" is far more useful keyed as start_date
        than as item_0."""
        _, leaves = build(
            service, [{"type": "text", "value": "Start Date: 2024-01-01"}]
        )
        assert leaves[0]["json_path"] == "start_date"
        assert leaves[0]["content"] == "2024-01-01"

    def test_a_url_is_not_split_on_its_scheme(self, service):
        _, leaves = build(service, [{"type": "text", "value": "https://kita.ph/docs"}])
        assert leaves[0]["content"] == "https://kita.ph/docs"

    def test_a_long_prefix_is_not_treated_as_a_key(self, service):
        """A whole sentence before a colon is prose, not a field name."""
        long_key = "x" * 45
        _, leaves = build(service, [{"type": "text", "value": f"{long_key}: value"}])
        assert leaves[0]["json_path"] == "item_0"

    def test_a_multiline_prefix_is_not_treated_as_a_key(self, service):
        _, leaves = build(service, [{"type": "text", "value": "a\nb: value"}])
        assert leaves[0]["json_path"] == "item_0"

    def test_a_colon_with_nothing_after_it_is_not_a_key(self, service):
        _, leaves = build(service, [{"type": "text", "value": "Trailing:"}])
        assert leaves[0]["json_path"] == "item_0"


class TestPathUniqueness:
    def test_sequential_items_get_distinct_paths(self, service):
        """Two leaves sharing a path overwrite each other in the tree, so one
        of the document's values silently disappears."""
        _, leaves = build(
            service,
            [
                {"type": "text", "value": "first"},
                {"type": "text", "value": "second"},
            ],
        )
        assert leaves[0]["json_path"] != leaves[1]["json_path"]

    def test_the_counter_is_scoped_per_section(self, service):
        _, leaves = build(
            service,
            [
                {"type": "heading", "value": "One", "level": 1},
                {"type": "text", "value": "a"},
                {"type": "heading", "value": "Two", "level": 1},
                {"type": "text", "value": "b"},
            ],
        )
        assert leaves[0]["json_path"] == "one.item_0"
        assert leaves[1]["json_path"] == "two.item_0"

    def test_a_table_is_keyed_as_a_table(self, service):
        _, leaves = build(service, [{"type": "table", "value": "| a | b |"}])
        assert leaves[0]["json_path"] == "table_0"

    def test_every_leaf_resolves_in_the_tree_it_built(self, service):
        """Retrieval replays this build and looks each path up; one that does
        not resolve drops the candidate."""
        tree, leaves = build(
            service,
            [
                {"type": "heading", "value": "Benefits", "level": 1},
                {"type": "text", "value": "Coffee: free"},
                {"type": "text", "value": "plain line"},
            ],
        )
        for leaf in leaves:
            assert get_nested_value(tree, leaf["json_path"].split(".")) is not None


class TestListItems:
    def test_each_list_entry_becomes_its_own_leaf(self, service):
        _, leaves = build(
            service,
            [
                {
                    "type": "list",
                    "text": "Perks",
                    "items": [{"value": "coffee"}, {"value": "gym"}],
                }
            ],
        )
        assert sorted(leaf["content"] for leaf in leaves) == ["coffee", "gym"]

    def test_the_list_title_becomes_a_heading(self, service):
        _, leaves = build(
            service,
            [{"type": "list", "text": "Perks", "items": [{"value": "coffee"}]}],
        )
        assert leaves[0]["heading_text"] == "Perks"

    def test_the_list_title_is_popped_afterwards(self, service):
        """Leaving it on the stack would nest every later item under the
        list's heading."""
        _, leaves = build(
            service,
            [
                {"type": "list", "text": "Perks", "items": [{"value": "coffee"}]},
                {"type": "text", "value": "after"},
            ],
        )
        assert leaves[-1]["heading_text"] == ""

    def test_list_markers_are_stripped_from_the_title(self, service):
        _, leaves = build(
            service,
            [{"type": "list", "text": "1. Perks", "items": [{"value": "coffee"}]}],
        )
        assert leaves[0]["heading_text"] == "Perks"

    def test_a_list_without_a_title_still_yields_leaves(self, service):
        _, leaves = build(service, [{"type": "list", "items": [{"value": "coffee"}]}])
        assert leaves[0]["content"] == "coffee"

    def test_empty_entries_are_skipped(self, service):
        _, leaves = build(
            service,
            [{"type": "list", "items": [{"value": ""}, {"value": "coffee"}]}],
        )
        assert len(leaves) == 1

    def test_an_entry_inherits_the_lists_coordinates(self, service):
        """Sub-items often carry no bbox of their own, and a citation with no
        location cannot be highlighted in the source document."""
        _, leaves = build(
            service,
            [
                {
                    "type": "list",
                    "bbox": [{"x": 1, "y": 2}],
                    "items": [{"value": "coffee"}],
                }
            ],
        )
        assert leaves[0]["location"] == {"x": 1, "y": 2}


class TestLocation:
    def test_a_bounding_box_list_takes_its_first_entry(self, service):
        _, leaves = build(
            service, [{"type": "text", "value": "x", "bbox": [{"x": 1}, {"x": 2}]}]
        )
        assert leaves[0]["location"] == {"x": 1}

    def test_a_bounding_box_mapping_is_used_directly(self, service):
        _, leaves = build(service, [{"type": "text", "value": "x", "bbox": {"x": 1}}])
        assert leaves[0]["location"] == {"x": 1}

    def test_no_bounding_box_is_accepted(self, service):
        _, leaves = build(service, [{"type": "text", "value": "x"}])
        assert leaves[0]["location"] is None

    def test_an_empty_bounding_box_list_is_accepted(self, service):
        _, leaves = build(service, [{"type": "text", "value": "x", "bbox": []}])
        assert leaves[0]["location"] is None


class TestMarkdownFallback:
    def test_markdown_is_parsed_when_items_are_missing(self, service):
        """Some parse results carry only rendered markdown; without this the
        whole page would be dropped."""
        result = {"pages": [{"page": 1, "markdown": "# Benefits\nfree coffee"}]}
        _, leaves = service.build_hierarchy_and_leaves(result, FILE_ID, ORG_ID)
        assert leaves[0]["heading_text"] == "Benefits"
        assert leaves[0]["content"] == "free coffee"

    def test_blank_lines_are_ignored(self, service):
        result = {"pages": [{"page": 1, "markdown": "one\n\n\ntwo"}]}
        _, leaves = service.build_hierarchy_and_leaves(result, FILE_ID, ORG_ID)
        assert len(leaves) == 2

    def test_heading_levels_are_preserved(self, service):
        result = {"pages": [{"page": 1, "markdown": "# A\n## B\ntext"}]}
        _, leaves = service.build_hierarchy_and_leaves(result, FILE_ID, ORG_ID)
        assert leaves[0]["heading_text"] == "A > B"

    def test_a_page_with_neither_items_nor_markdown_yields_nothing(self, service):
        result = {"pages": [{"page": 1}]}
        assert service.build_hierarchy_and_leaves(result, FILE_ID, ORG_ID) == ({}, [])


class TestIndexedText:
    def test_the_indexed_text_carries_the_path(self, service):
        """Search runs on this field, so the path is what lets a query match
        a value by the name of the field holding it."""
        _, leaves = build(service, [{"type": "text", "value": "Start Date: 2024"}])
        assert leaves[0]["text"] == "start_date: 2024"

    def test_the_readable_text_carries_the_heading_trail(self, service):
        _, leaves = build(
            service,
            [
                {"type": "heading", "value": "Benefits", "level": 1},
                {"type": "text", "value": "free coffee"},
            ],
        )
        assert leaves[0]["heading_to_text"] == "Benefits > free coffee"

    def test_a_leaf_with_no_heading_reads_as_its_value(self, service):
        _, leaves = build(service, [{"type": "text", "value": "orphan"}])
        assert leaves[0]["heading_to_text"] == "orphan"
