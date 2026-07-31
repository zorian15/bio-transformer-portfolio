// MathJax configuration for the metric definitions in the writeups.
//
// pymdownx.arithmatex(generic: true) rewrites $...$ and $$...$$ into \(...\)
// and \[...\], so MathJax only has to be told which delimiters to scan for and
// which elements to skip.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    // Do not typeset inside code blocks; a subscript in a shell snippet is not math.
    skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"],
    ignoreHtmlClass: "highlight",
  },
};
