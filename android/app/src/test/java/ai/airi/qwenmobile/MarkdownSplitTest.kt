package ai.airi.qwenmobile

import org.junit.Assert.assertEquals
import org.junit.Test

/** The stable head of a streamed Markdown text. */
class MarkdownSplitTest {
    @Test
    fun aTextWithoutAParagraphBreakHasNoStableHead() {
        assertEquals(0, Markdown.stableSplit(""))
        assertEquals(0, Markdown.stableSplit("one paragraph that still grows"))
        assertEquals(0, Markdown.stableSplit("line one\nline two"))
    }

    @Test
    fun theSplitIsAtTheLastParagraphBreak() {
        assertEquals(3, Markdown.stableSplit("a\n\nb"))
        val text = "first\n\nsecond\n\nthird still grows"
        assertEquals(text.indexOf("third"), Markdown.stableSplit(text))
        assertEquals("first\n\nsecond\n\n", text.substring(0, Markdown.stableSplit(text)))
    }

    @Test
    fun aBreakAtTheEndWaitsForTheNextLine() {
        assertEquals(0, Markdown.stableSplit("a\n\n"))
        assertEquals(0, Markdown.stableSplit("a\n\n   "))
    }

    @Test
    fun twoBlankLinesAreOneBreak() {
        val text = "a\n\n\nb"
        assertEquals(text.indexOf('b'), Markdown.stableSplit(text))
    }

    @Test
    fun anIndentedLineAfterTheBreakBelongsToTheHead() {
        assertEquals(0, Markdown.stableSplit("- item\n\n  continued"))
        assertEquals(0, Markdown.stableSplit("para\n\n    code"))
        assertEquals(0, Markdown.stableSplit("para\n\n\tcode"))
        val text = "a\n\nb\n\n  indented"
        assertEquals(text.indexOf('b'), Markdown.stableSplit(text))
    }

    @Test
    fun aBlankLineInsideAFencedCodeBlockIsNotABreak() {
        val text = "intro\n\n```python\nx = 1\n\ny = 2\n```\n\nafter"
        assertEquals(text.indexOf("after"), Markdown.stableSplit(text))
        val tildes = "intro\n\n~~~\nx\n\ny\n~~~\n\nafter"
        assertEquals(tildes.indexOf("after"), Markdown.stableSplit(tildes))
    }

    @Test
    fun anOpenFenceKeepsTheSplitBeforeIt() {
        val text = "intro\n\n```\nx = 1\n\ny = 2"
        assertEquals(text.indexOf("```"), Markdown.stableSplit(text))
    }

    @Test
    fun aShorterFenceDoesNotCloseALongerOne() {
        val text = "intro\n\n````\n```\n\nstill code\n````\n\nafter"
        assertEquals(text.indexOf("after"), Markdown.stableSplit(text))
    }

    @Test
    fun aFenceWithAnIndentationOfFourSpacesIsCode() {
        val text = "intro\n\n    ```\ncode\n\nafter"
        assertEquals(text.indexOf("after"), Markdown.stableSplit(text))
    }

    @Test
    fun aListItemAfterTheBreakStartsTheTail() {
        val text = "1. one\n\n2. two"
        assertEquals(text.indexOf("2."), Markdown.stableSplit(text))
    }
}
